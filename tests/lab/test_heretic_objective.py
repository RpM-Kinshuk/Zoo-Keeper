"""Exact installed refusal scoring and a fixed safe base-tokenizer template."""
import importlib.util
from types import SimpleNamespace

import pytest
import torch


def adapter():
    assert importlib.util.find_spec('lab.heretic_objective') is not None, 'stock objective adapter missing'
    from lab import heretic_objective
    return heretic_objective


class GenerationBoundary:
    """Controlled generation outputs; the installed evaluator/scoring remains real."""
    def __init__(self, baseline_refusal=True):
        from heretic.config import Settings
        self.settings = Settings.model_construct(model='', print_responses=False)
        self.model = object()
        self.calls = 0
        self.baseline_refusal = baseline_refusal
    def get_logprobs_batched(self, prompts):
        self.calls += 1
        return torch.tensor([[0.5, 0.5] if self.calls == 1 else [0.75, 0.25]]).log()
    def get_responses_batched(self, prompts, **kwargs):
        return ['I cannot help.' if self.baseline_refusal and self.calls == 1 else 'A short response.']


def records():
    return [dict(sample_id='train-good', role='group_0', text='Add two numbers.'),
            dict(sample_id='train-bad', role='group_1', text='A labeled test input.')]


def test_callback_uses_actual_installed_kl_and_refusal_objective():
    safe = adapter(); wrapped = GenerationBoundary()
    callback = safe.StockRefusalObjective(wrapped, training_records=records(), heldout_ids=['heldout'])
    scores = callback(wrapped.model, records())
    assert scores[0] == pytest.approx(0.143841036, abs=1e-7)
    assert scores[1] == 0.
    assert callback.baseline_refusals == 1
    assert callback.last_metrics['kl_divergence'] == pytest.approx(0.143841036, abs=1e-7)


def test_callback_preserves_stock_baseline_zero_absolute_count():
    safe = adapter(); wrapped = GenerationBoundary(baseline_refusal=False)
    callback = safe.StockRefusalObjective(wrapped, training_records=records(), heldout_ids=[])
    wrapped.get_responses_batched = lambda *a, **kw: ['Sorry.']
    scores = callback(wrapped.model, records())
    assert callback.baseline_refusals == 0
    assert scores[1] == 1.
    assert callback.last_metrics['refusals'] == 1


def test_callback_rejects_heldout_records_before_baseline_generation():
    safe = adapter(); wrapped = GenerationBoundary()
    with pytest.raises(ValueError, match='overlap'):
        safe.StockRefusalObjective(wrapped, training_records=records(), heldout_ids=['train-good'])
    assert wrapped.calls == 0


def test_callback_rejects_changed_records_before_trial_generation():
    safe = adapter(); wrapped = GenerationBoundary()
    callback = safe.StockRefusalObjective(wrapped, training_records=records(), heldout_ids=[])
    changed = records(); changed[0]['text'] = 'Different input.'
    with pytest.raises(ValueError, match='frozen'):
        callback(wrapped.model, changed)
    assert wrapped.calls == 1


def test_fallback_template_preserves_system_user_and_assistant_roles():
    safe = adapter()
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]')))
    safe.configure_tokenizer(tokenizer)
    rendered = tokenizer.apply_chat_template([{'role': 'system', 'content': 'System text.'},
        {'role': 'user', 'content': 'User text.'}], tokenize=False, add_generation_prompt=True)
    assert rendered == 'system: System text.\nuser: User text.\nassistant: '
    assert tokenizer.padding_side == 'left'


def test_prepare_rejects_direction_objective_overlap_before_model_work():
    safe = adapter()
    with pytest.raises(ValueError, match='overlap'):
        safe.prepare_stock_run(None, None, runtime_config={}, direction_records=records(),
                               objective_records=records(), heldout_ids=[])


def test_stock_direction_orthogonalization_uses_benign_residual_mean():
    safe = adapter()
    from heretic.config import Settings
    wrapped = SimpleNamespace(settings=Settings.model_construct(model='', orthogonalize_direction=True))
    means = iter([torch.tensor([[1., 0., 0.]]), torch.tensor([[2., 2., 0.]])])
    wrapped.get_residuals_mean = lambda prompts: next(means)
    directions = safe.stock_residual_directions(wrapped, training_records=records(), heldout_ids=[])
    torch.testing.assert_close(directions, torch.tensor([[0., 1., 0.]]))


def test_callback_preserves_stock_below_target_kl_branch():
    safe = adapter(); wrapped = GenerationBoundary()
    callback = safe.StockRefusalObjective(wrapped, training_records=records(), heldout_ids=[])
    wrapped.get_logprobs_batched = lambda prompts: torch.tensor([[0.501, 0.499]]).log()
    scores = callback(wrapped.model, records())
    assert 0 < callback.last_metrics['kl_divergence'] < 0.01
    assert scores == pytest.approx((0.01, 1.0))


def test_real_cpu_stock_prepare_and_two_trials_never_enter_upstream_loaders(tmp_path, monkeypatch):
    safe = adapter()
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (PreTrainedTokenizerFast, LlamaConfig, LlamaForCausalLM,
                              AutoModelForCausalLM, AutoTokenizer, AutoProcessor, PretrainedConfig)
    from heretic.model import Model
    from lab.heretic_safe import optimize_loaded_model, merge_loaded_model
    def forbidden(*args, **kwargs):
        raise AssertionError('upstream model/tokenizer/config loading was entered')
    for cls in (AutoModelForCausalLM, AutoTokenizer, AutoProcessor):
        monkeypatch.setattr(cls, 'from_pretrained', forbidden)
    monkeypatch.setattr(PretrainedConfig, 'get_config_dict', forbidden)
    monkeypatch.setattr(Model, '__init__', forbidden)
    monkeypatch.setattr(Model, 'reset_model', forbidden)
    monkeypatch.setattr(Model, 'get_merged_model', forbidden)
    vocabulary = {'[PAD]': 0, '[EOS]': 1, '[BOS]': 2, '[UNK]': 3,
                  'system': 4, 'user': 5, 'assistant': 6, ':': 7,
                  'Add': 8, 'two': 9, 'numbers': 10, 'Count': 11, 'three': 12,
                  'symbols': 13, 'Read': 14, 'four': 15, 'words': 16, 'List': 17,
                  'five': 18, 'letters': 19, '.': 20}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]',
        eos_token='[EOS]', bos_token='[BOS]', unk_token='[UNK]')
    torch.manual_seed(20261028)
    model = LlamaForCausalLM(LlamaConfig(hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        vocab_size=64, max_position_embeddings=128, tie_word_embeddings=True,
        bos_token_id=2, eos_token_id=1, pad_token_id=0)).float().eval()
    direction = [dict(sample_id='direction-good', role='group_0', text='Add two numbers.'),
                 dict(sample_id='direction-bad', role='group_1', text='Count three symbols.')]
    objective_records = [dict(sample_id='objective-good', role='group_0', text='Read four words.'),
                         dict(sample_id='objective-bad', role='group_1', text='List five letters.')]
    config = dict(seed=20261028, batch_size=2, max_response_length=2, quantization='none',
        response_prefix='', row_normalization='full', full_normalization_lora_rank=3,
        orthogonalize_direction=True, winsorization_quantile=1., offload_outputs_to_cpu=True,
        system_prompt='You are a helpful assistant.')
    wrapped, directions, objective, metadata = safe.prepare_stock_run(model, tokenizer,
        runtime_config=config, direction_records=direction, objective_records=objective_records,
        heldout_ids=['validation'])
    assert directions.shape == (4, 32) and torch.isfinite(directions).all()
    assert directions[0].norm() == 0  # Shared suffix embedding is unused by upstream.
    assert (directions[1:].norm(dim=1) > 0).all()
    assert metadata['fallback_chat_template'] is True
    report = optimize_loaded_model(wrapped, directions, objective=objective,
        objective_id=metadata['objective_id'], training_records=objective_records,
        heldout_ids=['validation'], seed=20261028, n_trials=2, n_startup_trials=1)
    assert len(report['optimization_trace']) == len(objective.metrics_trace) == 2
    assert all(torch.isfinite(torch.tensor(row['objectives'])).all() for row in report['optimization_trace'])
    merged = merge_loaded_model(wrapped)
    merged.save_pretrained(tmp_path, safe_serialization=True)
    assert (tmp_path / 'model.safetensors').is_file()
    assert not list(tmp_path.rglob('*.bin'))
