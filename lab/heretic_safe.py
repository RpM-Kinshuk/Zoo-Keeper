"""Installed Heretic's weight algorithm on an already verified builtin model.

This module never calls Heretic's constructor, reset/reload, tokenizer, processor
or merged-model loader. The caller owns safe pinned loading and the objective.
An installed algorithm primitive alone is not a completed Heretic optimization.
"""
from importlib.metadata import version
import math


def wrap_loaded_model(model, *, seed, row_normalization):
    """Attach upstream's LoRA algorithm without entering any upstream loader."""
    import torch
    from transformers import PreTrainedModel
    from heretic.config import RowNormalization, Settings
    from heretic.model import Model
    if version('heretic-llm') != '1.4.0':
        raise RuntimeError('Heretic installed version differs from audited 1.4.0')
    if not isinstance(model, PreTrainedModel) or not type(model).__module__.startswith('transformers.models.'):
        raise ValueError('Heretic requires an already safely loaded builtin Transformers model')
    if model.config.model_type not in {'llama', 'qwen2', 'qwen3', 'olmo2'}:
        raise ValueError('unsupported builtin residual architecture')
    normalization = RowNormalization(row_normalization)
    # No CLI/env/TOML processing or dataset defaults are used by these primitives.
    settings = Settings.model_construct(model='', seed=seed, row_normalization=normalization)
    wrapped = object.__new__(Model)
    wrapped.settings = settings
    wrapped.model = model
    wrapped.dtype = model.dtype
    wrapped.needs_reload = False
    torch.manual_seed(seed)
    model.eval().requires_grad_(False)
    wrapped._apply_lora()
    return wrapped


def apply_trial(wrapped, directions, *, direction_index, parameters):
    """Reset only in-memory adapters and call the actual installed algorithm."""
    import torch
    from heretic.model import AbliterationParameters
    layers = len(wrapped.get_layers())
    width = wrapped.model.config.hidden_size
    # Row0 is the embedding residual; a shared chat suffix makes it zero.
    # Upstream only consumes decoder directions at indices layer+1.
    if (directions.shape != (layers + 1, width) or not directions.is_floating_point()
            or not torch.isfinite(directions).all() or (directions[1:].norm(dim=1) <= 0).any()):
        raise ValueError('invalid embedding/decoder residual directions')
    if direction_index is not None and (not math.isfinite(direction_index)
                                       or not 0 <= direction_index < layers - 1):
        raise ValueError('global direction index cannot interpolate outside decoder layers')
    if set(parameters) != set(wrapped.get_abliterable_components()):
        raise ValueError('parameters must cover every installed Heretic component')
    selected = {}
    for component, values in parameters.items():
        params = AbliterationParameters(**values)
        if (not all(math.isfinite(v) for v in values.values())
                or params.min_weight_distance <= 0):
            raise ValueError('invalid Heretic per-component settings')
        selected[component] = params
    with torch.no_grad():
        # Deliberately avoid upstream reset_model(), which may reload unsafe inputs.
        for name, module in wrapped.model.named_modules():
            if 'lora_B' in name and hasattr(module, 'weight'):
                module.weight.zero_()
        wrapped.abliterate(directions.float(), direction_index, selected)


def merge_loaded_model(wrapped):
    """Merge the nonquantized in-memory PEFT model; no reload or file IO."""
    if getattr(wrapped.model, 'is_loaded_in_4bit', False) or getattr(wrapped.model, 'is_loaded_in_8bit', False):
        raise ValueError('quantized Heretic export is unsupported')
    # get_merged_model() can reload in upstream; this path cannot.
    merged = wrapped.model.merge_and_unload(safe_merge=True)
    merged.config._name_or_path = ''
    return merged


def optimize_loaded_model(wrapped, directions, *, objective, objective_id,
                          training_records, heldout_ids, seed, n_trials,
                          n_startup_trials=10):
    """Search upstream's exact parameter space with an explicitly named objective.

    Only transform-training records enter the callback. This generic interface
    chooses no refusal or sentiment semantics. Selection minimizes objective 0,
    then objective 1, then trial number; upstream's interactive trial menu is
    deliberately replaced by this recorded deterministic selection rule.
    """
    from dataclasses import asdict
    import random
    import numpy as np
    import optuna
    import torch
    from heretic.model import AbliterationParameters
    ids = [row['sample_id'] for row in training_records]
    if set(ids) & set(heldout_ids):
        raise ValueError('transform-training and heldout sample IDs overlap')
    if (not ids or len(ids) != len(set(ids)) or not callable(objective)
            or not isinstance(objective_id, str) or not objective_id.strip()
            or type(seed) is not int or type(n_trials) is not int or n_trials < 1
            or type(n_startup_trials) is not int or n_startup_trials < 1):
        raise ValueError('explicit objective, unique training records, seed and positive trials required')
    if {row.get('role') for row in training_records} != {'group_0', 'group_1'}:
        raise ValueError('training records require both frozen group roles')
    last = len(wrapped.get_layers()) - 1
    if last < 2:
        raise ValueError('Heretic 1.4.0 search space requires at least three decoder layers')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    def evaluate(trial):
        scope = trial.suggest_categorical('direction_scope', ['global', 'per layer'])
        index = trial.suggest_float('direction_index', 0.4 * last, 0.9 * last)
        index = None if scope == 'per layer' else index
        params = {}
        for component in wrapped.get_abliterable_components():
            maximum = trial.suggest_float(component + '.max_weight', 0.8, 1.5)
            position = trial.suggest_float(component + '.max_weight_position', 0.6 * last, 1.0 * last)
            minimum_fraction = trial.suggest_float(component + '.min_weight', 0., 1.)
            distance = trial.suggest_float(component + '.min_weight_distance', 1., 0.6 * last)
            params[component] = asdict(AbliterationParameters(maximum, position, minimum_fraction * maximum, distance))
        trial.set_user_attr('direction_index', index)
        trial.set_user_attr('per_component_settings', params)
        apply_trial(wrapped, directions, direction_index=index, parameters=params)
        with torch.no_grad():
            scores = tuple(float(v) for v in objective(wrapped.model, training_records))
        if len(scores) != 2 or not all(math.isfinite(v) for v in scores):
            raise ValueError('objective must return two finite minimizing scores')
        return scores
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(
        n_startup_trials=n_startup_trials, n_ei_candidates=128, multivariate=True, seed=seed),
        directions=['minimize', 'minimize'])
    study.optimize(evaluate, n_trials=n_trials, n_jobs=1)
    selected = min(study.trials, key=lambda t: (*t.values, t.number))
    params = selected.user_attrs['per_component_settings']
    index = selected.user_attrs['direction_index']
    apply_trial(wrapped, directions, direction_index=index, parameters=params)
    per_layer = {}
    for layer in range(last + 1):
        per_layer[str(layer)] = {}
        for component, settings in params.items():
            distance = abs(layer - settings['max_weight_position'])
            active = distance <= settings['min_weight_distance']
            weight = (settings['max_weight'] + distance / settings['min_weight_distance'] *
                      (settings['min_weight'] - settings['max_weight'])) if active else 0.
            per_layer[str(layer)][component] = dict(weight=weight, direction_index=index,
                                                   direction_layer=layer + 1 if index is None else None)
    return dict(package='heretic-llm', pinned_version='1.4.0',
        algorithm='installed_Model.abliterate', objective_id=objective_id,
        selection_rule='lexicographic_minimize_first_then_second_then_trial_number',
        seed=seed, n_trials=n_trials, n_startup_trials=n_startup_trials,
        sampler=dict(name='TPESampler', multivariate=True, n_ei_candidates=128),
        training_sample_ids=ids, heldout_used_for_optimization=False,
        selected_trial=selected.number, direction_index=index, per_component_settings=params,
        per_layer_settings=per_layer,
        optimization_trace=[dict(trial=t.number, parameters=t.params, objectives=t.values,
            direction_index=t.user_attrs['direction_index'],
            per_component_settings=t.user_attrs['per_component_settings']) for t in study.trials])
