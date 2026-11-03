"""Real CPU/Gloo equivalence checks for deterministic full-parameter DDP."""
import copy
from datetime import timedelta
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.lab.test_full_tuning import memory_inputs, recipe
from lab import distributed_training as distributed


def _baseline(edge):
    """Independent global-batch AdamW calculation, without DDP or its helpers."""
    inputs = memory_inputs(edge)
    model = inputs['model'].float()
    model.config.use_cache = False
    model.train()
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0)
    losses = []
    for step in range(2):
        ids = torch.arange(step * 4, (step + 1) * 4)
        chosen = inputs['token_ids'][ids]
        optimizer.zero_grad(set_to_none=True)
        if edge == 'sft':
            loss = model(input_ids=chosen, labels=chosen).loss
        else:
            rejected = inputs['rejected_token_ids'][ids]
            def scores(network, rows, mask):
                logits = network(input_ids=rows).logits[:, :-1].float()
                logps = logits.log_softmax(-1).gather(-1, rows[:, 1:, None]).squeeze(-1)
                return (logps * mask[:, 1:]).sum(-1)
            cm, rm = inputs['chosen_mask'][ids], inputs['rejected_mask'][ids]
            policy = scores(model, chosen, cm) - scores(model, rejected, rm)
            with torch.no_grad():
                frozen = scores(reference, chosen, cm) - scores(reference, rejected, rm)
            loss = -torch.nn.functional.logsigmoid(.1 * (policy - frozen)).mean()
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    return model.state_dict(), losses


def _worker(rank, rendezvous, directory, edge):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        inputs = memory_inputs(edge)
        reference = None
        if edge == 'dpo':
            reference = copy.deepcopy(inputs['model'])
            before = {k: v.clone() for k, v in reference.state_dict().items()}
            inputs['reference_model'] = reference
        bundle = distributed.train_full(recipe(edge), inputs, edge, rank=rank, world_size=2)
        if rank == 0:
            expected, losses = _baseline(edge)
            assert bundle is not None
            for key, value in bundle.model.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=2e-5, atol=2e-7)
            assert bundle.training['losses'] == pytest.approx(losses, abs=1e-6)
            assert bundle.training['distributed']['global_batch'] == 4
            assert bundle.training['distributed']['local_batch'] == 2
            assert len(bundle.training['distributed']['per_rank_peaks']) == 2
        else:
            assert bundle is None
        if reference is not None:
            assert all(not p.requires_grad for p in reference.parameters())
            for key, value in reference.state_dict().items():
                assert torch.equal(value, before[key])
        # A nonzero rank cannot create a checkpoint, even when called explicitly.
        distributed.publish_checkpoint(recipe(edge), Path(directory) / 'forbidden', bundle, rank=1)
        assert not (Path(directory) / 'forbidden').exists()
        if rank == 0:
            distributed.publish_checkpoint(recipe(edge), Path(directory), bundle, rank=0)
        dist.barrier()
        assert distributed.completed_checkpoint(recipe(edge), Path(directory), resume=True, rank=rank)
        with pytest.raises(ValueError, match='resume|exists'):
            distributed.completed_checkpoint(recipe(edge), Path(directory), resume=False, rank=rank)
        with pytest.raises(ValueError, match='resume|mismatch'):
            distributed.completed_checkpoint({**recipe(edge), 'seed': 2}, Path(directory), resume=True, rank=rank)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('edge', ['sft', 'dpo'])
def test_two_rank_global_loss_update_frozen_reference_and_collective_resume(tmp_path, edge):
    # Catches wrong/no sharding, omitted DDP averaging, local-only loss reporting,
    # double-forward reducer misuse, reference mutation and publication races.
    mp.spawn(_worker, args=(str(tmp_path / 'store'), str(tmp_path / 'output'), edge),
             nprocs=2, join=True)
    report = json.loads((tmp_path / 'output' / ('tiny-' + edge) / 'training.json').read_text())
    assert report['steps_completed'] == 2
    assert report['reference_unchanged'] is (True if edge == 'dpo' else None)
    assert report['parameter_dtype'] == 'float32' and report['output_dtype'] == 'bfloat16'


def test_uneven_batch_is_rejected_before_loading_any_inputs():
    with pytest.raises(ValueError, match='divisible'):
        distributed.train_full({**recipe(), 'effective_batch': 3}, {}, 'sft', rank=0, world_size=2)


def test_completion_logps_exclude_prompt_predictions():
    logits = torch.zeros(1, 4, 3)
    logits[0, 0, 1] = 20  # Large prompt likelihood must be excluded.
    tokens = torch.tensor([[0, 1, 2, 0]])
    mask = torch.tensor([[0, 0, 1, 1]])
    result = distributed.completion_logps(logits, tokens, mask)
    torch.testing.assert_close(result, torch.tensor([-2 * __import__('math').log(3)]))


def test_worker_cli_requires_explicit_execution_and_config_hash(tmp_path):
    from lab.scripts.distributed_training import main
    config = tmp_path / 'config.json'
    config.write_text('{}')
    args = ['--config', str(config), '--data-root', str(tmp_path),
            '--profile', str(tmp_path / 'profile.json'), '--expected-config-sha256', '0' * 64]
    with pytest.raises(ValueError, match='hash mismatch'):
        main(args + ['--execute'])
    from lab.launches import file_hash
    args[-1] = file_hash(config)
    with pytest.raises(ValueError, match='execute'):
        main(args)


def test_frozen_distribution_and_runtime_fixture_bypass_are_rejected(tmp_path):
    with pytest.raises(ValueError, match='metadata'):
        distributed.train_full({**recipe(), 'distributed': {'world_size': 7}}, {}, 'sft', rank=0, world_size=1)
    from lab.launches import file_hash
    from lab.scripts.distributed_training import main
    config = tmp_path / 'unsafe.json'
    config.write_text(json.dumps(dict(trainer='sft', recipe={**recipe(), 'device': 'cuda:0'},
                                     inputs={'model': 'fixture', 'token_ids': 'fixture'})))
    with pytest.raises(ValueError, match='frozen local'):
        main(['--config', str(config), '--data-root', str(tmp_path), '--profile', str(tmp_path / 'profile'),
              '--expected-config-sha256', file_hash(config), '--execute'])


def test_distributed_loader_rechecks_real_frozen_bytes_and_preprocessing(tmp_path):
    from tests.lab.test_lora import frozen_inputs
    base, inputs = frozen_inputs(tmp_path)
    r = {**recipe(), 'parent_identities': base['parent_identities']}
    with pytest.raises(ValueError, match='preprocessing'):
        distributed.train_full(r, inputs, 'sft', rank=0, world_size=1)
    inputs['data_pin']['preprocessing'] = dict(format='causal_tokens_v1', loss='all_tokens', padding='none')
    inputs['model_pin']['weight_sha256']['model.safetensors'] = '0' * 64
    with pytest.raises(ValueError, match='weight hash'):
        distributed.train_full(r, inputs, 'sft', rank=0, world_size=1)
