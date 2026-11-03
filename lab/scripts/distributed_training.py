"""Frozen full SFT/DPO worker; launch with torchrun, max_restarts=0, timeout 7200.

External admission reserves both physical GPU UUIDs before setting
CUDA_VISIBLE_DEVICES. This worker never selects or reserves GPUs itself.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import socket
import time

from lab.artifacts import atomic_json
from lab.launches import file_hash, preflight, resolve_sft_parent
from lab.distributed_training import (completed_checkpoint, distributed_metadata,
                                      publish_checkpoint, train_full)
from lab.training import validate_full_recipe


def _rank_zero_call(operation, rank):
    """Deliver validation failures to peers before they enter the next collective."""
    import torch.distributed as dist
    result = [None]
    if rank == 0:
        try:
            result[0] = dict(value=operation())
        except Exception as error:
            result[0] = dict(error_type=type(error).__name__, error=str(error))
    dist.broadcast_object_list(result, src=0)
    if 'error' in result[0]:
        raise ValueError(result[0]['error'])
    return result[0]['value']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--expected-config-sha256', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resolve-parent', action='store_true')
    args = parser.parse_args(argv)
    if file_hash(args.config) != args.expected_config_sha256:
        raise ValueError('frozen distributed config hash mismatch')
    if not args.execute:
        raise ValueError('explicit --execute is required for this training worker')
    config = json.loads(args.config.read_text())
    edge = config['trainer']
    if edge not in {'sft', 'dpo'}:
        raise ValueError('distributed worker supports only full SFT/DPO')
    if config['recipe']['device'] != 'cuda:0':
        raise ValueError('canonical distributed recipe device must be cuda:0')
    # JSON runtime configurations must never activate the in-memory fixture path.
    if {'model', 'token_ids', 'reference_model'} & config['inputs'].keys():
        raise ValueError('runtime requires frozen local model/token inputs')
    rank, local_rank, world_size = (int(os.environ[name]) for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    if world_size != 2 or rank not in (0, 1) or local_rank != rank:
        raise ValueError('this worker requires two ranks on one node')
    import torch
    import torch.distributed as dist
    if not torch.cuda.is_available() or torch.cuda.device_count() != world_size:
        raise RuntimeError('exactly two externally admitted visible CUDA GPUs are required')
    torch.cuda.set_device(local_rank)
    torch.cuda.reset_peak_memory_stats(local_rank)
    metadata = distributed_metadata(validate_full_recipe(config['recipe'], edge), world_size, 'nccl')
    canonical_recipe = {**config['recipe'], 'distributed': metadata, 'inputs': config['inputs']}
    profile_path = args.profile if rank == 0 else args.profile.with_name(args.profile.name + '.rank' + str(rank) + '.json')
    started = time.monotonic()
    phase_started, current_phase = started, 'CHECKING'
    profile = dict(status=current_phase, model_id=canonical_recipe['model_id'],
        config_path=str(args.config.resolve()), config_sha256=args.expected_config_sha256,
        host=socket.gethostname(), pid=os.getpid(), rank=rank, local_rank=local_rank,
        started_utc=datetime.now(timezone.utc).isoformat(), steps_completed=0,
        gpu_name=torch.cuda.get_device_name(local_rank),
        gpu_uuid=str(getattr(torch.cuda.get_device_properties(local_rank), 'uuid', 'unavailable')),
        torch_version=torch.__version__,
        distributed=metadata, phase_seconds={})
    group_initialized = False
    profile_owned = False
    def publish():
        profile['elapsed_seconds'] = time.monotonic() - started
        profile['current_allocated_bytes'] = torch.cuda.memory_allocated(local_rank)
        profile['current_reserved_bytes'] = torch.cuda.memory_reserved(local_rank)
        atomic_json(profile_path, profile)
    def progress(phase, details):
        nonlocal phase_started, current_phase
        now = time.monotonic()
        if phase != current_phase:
            profile['phase_seconds'][current_phase.lower()] = now - phase_started
            current_phase, phase_started = phase, now
        profile.update(status=phase, **details)
        # Keep last-progress and synchronized step timing visible without an IO
        # operation for every optimizer update on the shared filesystem.
        step = details.get('steps_completed', 0)
        if phase != 'TRAINING' or step <= 1 or step % 8 == 0 or step == canonical_recipe['steps']:
            profile['last_progress_utc'] = datetime.now(timezone.utc).isoformat()
            publish()
        if rank == 0 and phase == 'TRAINING' and (step == 1 or step % 16 == 0):
            print(json.dumps(dict(status=phase, model_id=canonical_recipe['model_id'], rank=rank, **details), sort_keys=True), flush=True)
    try:
        dist.init_process_group('nccl', timeout=timedelta(seconds=180), device_id=torch.device('cuda', local_rank))
        group_initialized = True
        def check_profile():
            if args.profile.exists() or args.profile.with_name(args.profile.name + '.rank1.json').exists():
                raise FileExistsError('use a new profile path for each launch attempt')
        _rank_zero_call(check_profile, rank)
        profile_owned = True
        publish()
        if args.resolve_parent:
            config = _rank_zero_call(lambda: resolve_sft_parent(config, args.data_root), rank)
            canonical_recipe = {**config['recipe'], 'distributed': metadata, 'inputs': config['inputs']}
            def save_resolved():
                resolved = args.profile.parent / 'resolved-configs' / (canonical_recipe['model_id'] + '.json')
                if resolved.exists():
                    if json.loads(resolved.read_text()) != config:
                        raise ValueError('resolved config differs from existing immutable config')
                else:
                    atomic_json(resolved, config)
                return dict(resolved_config_path=str(resolved), resolved_config_sha256=file_hash(resolved),
                            actual_parent_key=config['inputs']['local_parent']['model_key'])
            profile.update(_rank_zero_call(save_resolved, rank))
            publish()
        # All source, license, split, token, mask and preprocessing evidence is
        # checked by the existing preflight; each replica loader then rechecks
        # the actual frozen inputs and 1.7B header/parameter cap.
        _rank_zero_call(lambda: preflight(config, args.data_root, load_model=False), rank)
        output_root = args.data_root / 'lab'
        skip = completed_checkpoint(canonical_recipe, output_root, resume=args.resume, rank=rank)
        bundle = None
        if not skip:
            runtime_recipe = {**canonical_recipe, 'device': 'cuda:' + str(local_rank)}
            runtime_inputs = {**config['inputs'], 'root': args.data_root}
            bundle = train_full(runtime_recipe, runtime_inputs, edge, rank=rank, world_size=world_size, progress=progress)
            if rank == 0:
                # Published device is canonical; actual per-rank devices are in
                # the explicit distributed hardware record.
                bundle.training['device'] = canonical_recipe['device']
                profile['distributed'] = bundle.training['distributed']
                profile['synchronized_step_seconds'] = bundle.training['synchronized_step_seconds']
        # Every rank follows the same resume branch and finishes collectives
        # before rank zero performs potentially slow NFS checkpoint serialization.
        dist.destroy_process_group()
        group_initialized = False
        torch.cuda.empty_cache()
        if rank == 0:
            if not skip:
                progress('SERIALIZING', {})
                output = publish_checkpoint(canonical_recipe, output_root, bundle, rank=rank, resume=args.resume)
            else:
                output = output_root / canonical_recipe['model_id']
            profile.update(output=str(output), resumed_completed_output=skip,
                output_steps_completed=canonical_recipe['steps'],
                training_json_sha256=file_hash(output / 'training.json'))
            progress('COMPLETE', {})
            print(json.dumps(profile, sort_keys=True), flush=True)
        else:
            progress('WORKER_COMPLETE', dict(resumed_completed_output=skip))
    except BaseException as error:
        if profile_owned:
            profile.update(error_type=type(error).__name__, error=str(error))
            progress('FAILED', {})
        raise
    finally:
        if group_initialized:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
