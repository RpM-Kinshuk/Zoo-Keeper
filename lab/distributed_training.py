"""Full SFT/DPO with deterministic global batches and one DDP replica per rank.

Callers own process-group initialization/destruction. Only rank zero receives a
CPU export bundle; publication is deliberately separate from GPU training.
"""
from dataclasses import dataclass
from pathlib import Path
import time

from .artifacts import atomic_json, write_checkpoint
from .training import _full_inputs, validate_full_recipe

PARTITION_POLICY = 'contiguous_global_step_batch_v1'


@dataclass
class ExportBundle:
    model: object
    training: dict


def distributed_metadata(recipe, world_size, backend):
    if type(world_size) is not int or world_size < 1:
        raise ValueError('world_size must be a positive integer')
    batch = recipe['effective_batch']
    if batch % world_size:
        raise ValueError('effective_batch must be divisible by world_size')
    metadata = dict(world_size=world_size, backend=backend, global_batch=batch,
                    local_batch=batch // world_size, partition_policy=PARTITION_POLICY)
    if 'distributed' in recipe and recipe['distributed'] != metadata:
        raise ValueError('frozen distributed metadata differs from runtime')
    return metadata


def completion_logps(logits, tokens, mask):
    """Unnormalized sequence log probability over the frozen completion suffix."""
    logps = logits[:, :-1].float().log_softmax(-1).gather(-1, tokens[:, 1:, None]).squeeze(-1)
    return (logps * mask[:, 1:]).sum(-1)


def completed_checkpoint(recipe, output_root, *, resume, rank):
    """Rank zero validates an existing atomic output, broadcasting skip OR error.

    Every rank participates, including on an invalid resume. This check happens
    before loading replicas, and never invokes the training writer.
    """
    import json
    import torch.distributed as dist
    result = [None]
    if rank == 0:
        try:
            target = Path(output_root) / recipe['model_id']
            exists = target.exists() or target.is_symlink()
            if exists:
                def forbidden(_):
                    raise RuntimeError('completed output unexpectedly disappeared')
                target = write_checkpoint(recipe['model_id'], output_root, recipe, forbidden, resume=resume)
                training = json.loads((target / 'training.json').read_text())
                if (training.get('steps_completed') != recipe['steps']
                        or training.get('full_parameters') is not True):
                    raise ValueError('resume mismatch or incomplete training')
            result[0] = dict(skip=exists)
        except Exception as error:
            result[0] = dict(error_type=type(error).__name__, error=str(error))
    if dist.is_initialized():
        dist.broadcast_object_list(result, src=0)
    if 'error' in result[0]:
        raise ValueError(result[0]['error'])
    return result[0]['skip']


def train_full(recipe, inputs, edge, *, rank, world_size, progress=None):
    """Train equal contiguous shards of each original deterministic step batch.

    DDP averages replica gradients, so equal-size local mean losses produce the
    original global mean objective without changing steps, LR or global batch.
    DPO concatenates chosen/rejected rows into ONE synchronized policy forward.
    """
    import copy
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    recipe = validate_full_recipe(recipe, edge)
    # Reject impossible shards before touching model/data inputs or collectives.
    backend = dist.get_backend() if dist.is_initialized() else ('nccl' if recipe['device'].startswith('cuda:') else 'gloo')
    metadata = distributed_metadata(recipe, world_size, backend)
    if not 0 <= rank < world_size:
        raise ValueError('rank is outside world_size')
    if world_size > 1 and (not dist.is_initialized() or dist.get_rank() != rank or dist.get_world_size() != world_size):
        raise ValueError('initialized process group must match rank and world_size')
    def report(phase, **details):
        if progress is not None:
            progress(phase, details)
    report('LOADING', distributed=metadata)
    load_started = time.monotonic()
    torch.manual_seed(recipe.get('training_seed', recipe['seed']))
    model, tokens, preference = _full_inputs(recipe, inputs, edge)
    model.config._name_or_path = ''
    model.config.use_cache = False
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
    reference = None
    if edge == 'dpo':
        reference = inputs.get('reference_model')
        if reference is None:
            reference = copy.deepcopy(model)
        pointers = {p.data_ptr() for p in model.parameters()}
        if reference is model or any(p.data_ptr() in pointers for p in reference.parameters()):
            raise ValueError('DPO reference must not share policy tensors')
        policy_state, reference_state = model.state_dict(), reference.state_dict()
        if (policy_state.keys() != reference_state.keys()
                or any(not torch.equal(v, reference_state[k].to(v)) for k, v in policy_state.items())):
            raise ValueError('DPO reference must exactly match the SFT parent')
        # State-dict values retain device storage; release them before CPU export.
        del policy_state, reference_state
        reference.to(device=next(model.parameters()).device, dtype=torch.float32)
        reference.requires_grad_(False).eval()
    model.float().requires_grad_(True).train()
    reference_before = None if reference is None else {k: p._version for k, p in reference.named_parameters()}
    matrix_name, matrix = next((name, p) for name, p in model.named_parameters()
                               if p.ndim == 2 and '.mlp.' in name)
    matrix_before = matrix.detach().float().clone()
    device = next(model.parameters()).device
    distributed_model = (DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None,
                         broadcast_buffers=False) if world_size > 1 else model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['learning_rate'], weight_decay=0)
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    synchronize()
    loading_seconds = time.monotonic() - load_started
    losses, step_seconds = [], []
    report('TRAINING', loading_seconds=loading_seconds, steps_completed=0)
    for step in range(recipe['steps']):
        synchronize()
        step_started = time.monotonic()
        start = step * recipe['effective_batch'] + rank * metadata['local_batch']
        indexes = torch.arange(start, start + metadata['local_batch']) % len(tokens)
        chosen = tokens[indexes].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=recipe['torch_dtype'] == 'bfloat16'):
            if edge == 'sft':
                loss = distributed_model(input_ids=chosen, labels=chosen).loss
            else:
                rejected, chosen_mask, rejected_mask = [value[indexes].to(device) for value in preference]
                rows = torch.cat((chosen, rejected))
                masks = torch.cat((chosen_mask, rejected_mask))
                logps = completion_logps(distributed_model(input_ids=rows).logits, rows, masks)
                chosen_logps, rejected_logps = logps.chunk(2)
                policy_ratio = chosen_logps - rejected_logps
                with torch.no_grad():
                    frozen_logps = completion_logps(reference(input_ids=rows).logits, rows, masks)
                    frozen_chosen, frozen_rejected = frozen_logps.chunk(2)
                    reference_ratio = frozen_chosen - frozen_rejected
                loss = -torch.nn.functional.logsigmoid(recipe['beta'] * (policy_ratio - reference_ratio)).mean()
        finite = torch.isfinite(loss.detach()).to(torch.int32)
        if world_size > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise ValueError('nonfinite full training loss on a rank')
        loss.backward()
        optimizer.step()
        global_loss = loss.detach().double()
        if world_size > 1:
            dist.all_reduce(global_loss)
            global_loss /= world_size
        synchronize()
        elapsed = torch.tensor(time.monotonic() - step_started, device=device, dtype=torch.float64)
        if world_size > 1:
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        losses.append(global_loss.item())
        step_seconds.append(elapsed.item())
        report('TRAINING', steps_completed=step + 1, global_loss=losses[-1],
               last_step_seconds=step_seconds[-1], training_seconds=sum(step_seconds))
    unchanged = reference is None or all(p._version == reference_before[k] for k, p in reference.named_parameters())
    valid = torch.tensor(int(unchanged), device=device)
    if world_size > 1:
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    if not valid.item():
        raise ValueError('frozen DPO reference changed')
    updated_matrix = dict(name=matrix_name,
        delta_norm_before_export=float(torch.linalg.vector_norm(matrix.detach().float() - matrix_before)))
    peaks = dict(rank=rank, device=str(device),
        gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        gpu_uuid=str(getattr(torch.cuda.get_device_properties(device), 'uuid', 'unavailable')) if device.type == 'cuda' else None,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0)
    all_peaks = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(all_peaks, peaks)
    else:
        all_peaks[0] = peaks
    # No distributed operation is needed after this point. Drop all device
    # activations, optimizer states, reference tensors and DDP buckets before IO.
    report('CPU_EXPORT', steps_completed=recipe['steps'])
    export_started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    del optimizer, distributed_model, matrix, matrix_before
    if edge == 'dpo':
        del rows, masks, logps, chosen_logps, rejected_logps, policy_ratio
        del frozen_logps, frozen_chosen, frozen_rejected, reference_ratio
        del rejected, chosen_mask, rejected_mask
    del loss, global_loss, chosen, finite, elapsed, valid
    # Runtime-created models/references are owned here. The optional fixture
    # reference remains caller-owned and is moved off GPU as well.
    if reference is not None:
        reference.cpu()
    del reference
    model.cpu()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    export_seconds = time.monotonic() - export_started
    report('CPU_READY', cpu_export_seconds=export_seconds)
    if rank != 0:
        return None
    training = dict(steps_completed=recipe['steps'], losses=losses,
        training_seed=recipe.get('training_seed', recipe['seed']), sample_ids=inputs['sample_ids'],
        heldout_ids=inputs['heldout_ids'], full_parameters=True, updated_matrix=updated_matrix,
        preprocessing=inputs.get('data_pin', {}).get('preprocessing', {'fixture': 'synthetic-tokens-v1'}),
        reference_unchanged=unchanged if edge == 'dpo' else None,
        reference_parent=recipe['parents'][0] if edge == 'dpo' else None,
        loss_implementation='torch_logsigmoid_dpo' if edge == 'dpo' else 'causal_cross_entropy',
        device=recipe['device'], torch_dtype=recipe['torch_dtype'], parameter_dtype='float32', output_dtype='bfloat16',
        distributed={**metadata, 'per_rank_peaks': all_peaks},
        phase_seconds=dict(loading=loading_seconds, training=sum(step_seconds), cpu_export=export_seconds),
        synchronized_step_seconds=step_seconds)
    return ExportBundle(model, training)


def publish_checkpoint(recipe, output_root, bundle, *, rank, resume=False):
    """Only rank zero serializes the CPU bundle through the atomic writer."""
    if rank != 0:
        return None
    if bundle is None:
        raise ValueError('rank zero export bundle is required')
    def writer(stage):
        import torch
        bundle.model.to(dtype=torch.bfloat16)
        bundle.model.config.use_cache = True
        bundle.model.save_pretrained(stage, safe_serialization=True)
        training = {**bundle.training, 'primary_weights': sorted(p.name for p in stage.glob('*.safetensors'))}
        atomic_json(stage / 'training.json', training)
    return write_checkpoint(recipe['model_id'], output_root, recipe, writer, resume=resume)
