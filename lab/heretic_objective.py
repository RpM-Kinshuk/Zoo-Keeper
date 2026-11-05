"""Stock installed-Heretic refusal objective over explicit frozen local prompts."""
from .artifacts import content_hash

FALLBACK_CHAT_TEMPLATE = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
FALLBACK_CHAT_TEMPLATE_ID = 'explicit_system_user_assistant_roles_v1'


def configure_tokenizer(tokenizer):
    """Use only an existing local tokenizer and a fixed trusted fallback template."""
    if not tokenizer.chat_template:
        tokenizer.chat_template = FALLBACK_CHAT_TEMPLATE
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return dict(chat_template_sha256=content_hash(tokenizer.chat_template),
                fallback_chat_template=tokenizer.chat_template == FALLBACK_CHAT_TEMPLATE)


def _records(records, heldout_ids):
    ids = [r['sample_id'] for r in records]
    if set(ids) & set(heldout_ids):
        raise ValueError('transform-training and heldout sample IDs overlap')
    if (not ids or len(ids) != len(set(ids))
            or {r.get('role') for r in records} != {'group_0', 'group_1'}
            or any(not isinstance(r.get('text'), str) or not r['text'].strip() for r in records)):
        raise ValueError('unique frozen prompt IDs, text and both group roles required')
    return ids


def _prompts(wrapped, records, role):
    from heretic.utils import Prompt
    return [Prompt(system=wrapped.settings.system_prompt, user=r['text'])
            for r in records if r['role'] == role]


class StockRefusalObjective:
    """Bypass only upstream dataset loading, retain Evaluator.get_score unchanged.

    Baseline distributions and refusal counts use the explicit transform-training
    objective slice. No raw prompts, distributions, or completions enter reports.
    A zero-refusal baseline uses upstream's absolute refusal count score.
    """
    def __init__(self, wrapped, *, training_records, heldout_ids):
        from heretic.evaluator import Evaluator
        self.training_sample_ids = _records(training_records, heldout_ids)
        self.records_hash = content_hash(training_records)
        wrapped.settings.print_responses = False
        self.evaluator = object.__new__(Evaluator)
        self.evaluator.settings = wrapped.settings
        self.evaluator.model = wrapped
        self.evaluator.good_prompts = _prompts(wrapped, training_records, 'group_0')
        self.evaluator.bad_prompts = _prompts(wrapped, training_records, 'group_1')
        self.evaluator.base_logprobs = wrapped.get_logprobs_batched(self.evaluator.good_prompts)
        self.evaluator.base_refusals = self.evaluator.count_refusals()
        self.baseline_refusals = self.evaluator.base_refusals
        self.last_metrics = None
        self.metrics_trace = []

    def __call__(self, model, training_records):
        if model is not self.evaluator.model.model or content_hash(training_records) != self.records_hash:
            raise ValueError('objective must use the same model and frozen training records')
        scores, kl_divergence, refusals = self.evaluator.get_score()
        self.last_metrics = dict(kl_divergence=kl_divergence, refusals=refusals,
                                 base_refusals=self.baseline_refusals)
        self.metrics_trace.append(dict(self.last_metrics))
        return scores


def stock_residual_directions(wrapped, *, training_records, heldout_ids):
    """Use upstream last-position residual means and stock orthogonalization."""
    import torch
    import torch.nn.functional as F
    _records(training_records, heldout_ids)
    with torch.no_grad():
        good = wrapped.get_residuals_mean(_prompts(wrapped, training_records, 'group_0'))
        bad = wrapped.get_residuals_mean(_prompts(wrapped, training_records, 'group_1'))
        directions = F.normalize(bad - good, p=2, dim=1)
        if wrapped.settings.orthogonalize_direction:
            good = F.normalize(good, p=2, dim=1)
            directions = F.normalize(directions - (directions * good).sum(dim=1, keepdim=True) * good,
                                     p=2, dim=1)
    if not torch.isfinite(directions).all() or (directions[1:].norm(dim=1) <= 0).any():
        raise ValueError('stock residual extraction produced a zero/nonfinite direction')
    return directions


def prepare_stock_run(model, tokenizer, *, runtime_config, direction_records,
                      objective_records, heldout_ids):
    """Bind safely preloaded inputs, collect stock directions and baseline scores.

    The caller must verify local model/tokenizer pins and frozen record hashes
    before entering this API. It does not load files, download, choose prompts,
    optimize on heldout data, or execute repository code.
    """
    from heretic.config import Settings
    from .heretic_safe import wrap_loaded_model
    direction_ids = _records(direction_records, heldout_ids)
    objective_ids = _records(objective_records, heldout_ids)
    if set(direction_ids) & set(objective_ids):
        raise ValueError('direction and objective training sample IDs overlap')
    if (runtime_config.get('batch_size', 0) < 1 or runtime_config.get('max_response_length', 0) < 1
            or runtime_config.get('quantization') != 'none'
            or runtime_config.get('response_prefix') != ''
            or runtime_config.get('row_normalization') != 'full'
            or runtime_config.get('full_normalization_lora_rank') != 3):
        raise ValueError('explicit bounded unquantized stock runtime configuration required')
    template = configure_tokenizer(tokenizer)
    wrapped = wrap_loaded_model(model, seed=runtime_config['seed'], row_normalization='full')
    selected = {k: v for k, v in runtime_config.items() if k in Settings.model_fields}
    wrapped.settings = Settings.model_construct(model='', print_responses=False, **selected)
    wrapped.tokenizer = tokenizer
    wrapped.processor = None
    directions = stock_residual_directions(wrapped, training_records=direction_records,
                                          heldout_ids=heldout_ids)
    objective = StockRefusalObjective(wrapped, training_records=objective_records,
                                     heldout_ids=heldout_ids)
    metadata = dict(template, objective_id='installed_Evaluator.get_score_stock_refusal_and_benign_KL',
        direction_sample_ids=direction_ids, objective_sample_ids=objective_ids,
        baseline_refusals=objective.baseline_refusals,
        baseline_zero_rule='absolute_refusal_count; same as installed Evaluator.get_score',
        validation_used_for_optimization=False)
    return wrapped, directions, objective, metadata
