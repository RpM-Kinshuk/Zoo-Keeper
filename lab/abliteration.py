"""Frozen-data residual projection and a fail-closed pinned Heretic boundary."""
import hashlib
from importlib.metadata import version
from pathlib import Path

from .artifacts import atomic_json, content_hash, _portable
from .schema import contained_path

HERETIC_VERSION = '1.4.0'
SUPPORTED_FAMILIES = {'llama', 'qwen2', 'qwen3', 'olmo2'}


def _unit(direction, width):
    import torch
    r = direction.detach().float()
    if r.ndim != 1 or r.numel() != width or not torch.isfinite(r).all() or r.norm() <= 0:
        raise ValueError('invalid or zero residual direction')
    return r / r.norm()


def classic_edit(state: dict, directions: dict, modules: dict) -> dict:
    """Project declared residual writers in fp32; axis describes output channels.

    Native torch Linear weights have output channels on axis 0. An explicit
    axis 1 handles transposed storage without guessing from square matrices.
    Biases are separate one-dimensional entries with residual_axis=0.
    """
    import torch
    if not modules or not directions:
        raise ValueError('nonempty residual writers and directions are required')
    result = {name: tensor.clone() for name, tensor in state.items()}
    for name, spec in modules.items():
        if name not in state or spec.get('layer') not in directions:
            raise ValueError(f'{name}: missing writer or layer direction')
        w = state[name]
        axis = spec.get('residual_axis')
        if w.ndim not in (1, 2) or type(axis) is not int or axis not in range(w.ndim):
            raise ValueError(f'{name}: unsupported residual writer orientation')
        if not w.is_floating_point() or not torch.isfinite(w).all():
            raise ValueError(f'{name}: nonfinite or nonfloating writer')
        r = _unit(directions[spec['layer']], w.shape[axis]).to(w.device)
        oriented = w.detach().float().movedim(axis, 0)
        flat = oriented.reshape(len(r), -1)
        projected = flat - r[:, None] * (r @ flat)[None, :]
        # Retain fp32 through projection. The caller explicitly chooses export dtype.
        result[name] = projected.reshape(oriented.shape).movedim(0, axis).contiguous()
    return result


def mean_directions(residuals: dict, labels: list) -> dict:
    """Normalized group-1 minus group-0 means, with no fitted heldout data."""
    import torch
    if set(labels) != {0, 1}:
        raise ValueError('direction extraction needs both labeled groups 0 and 1')
    result = {}
    for layer, values in residuals.items():
        if values.ndim != 2 or len(values) != len(labels) or not torch.isfinite(values).all():
            raise ValueError(f'{layer}: invalid direction residual observations')
        group = torch.tensor(labels, device=values.device)
        delta = values[group == 1].float().mean(0) - values[group == 0].float().mean(0)
        result[layer] = _unit(delta, values.shape[1])
    if not result:
        raise ValueError('no residual directions were collected')
    return result


def validate_recipe(recipe):
    value = {'device': 'cpu', 'torch_dtype': 'bfloat16',
             'heretic_config': {'n_trials': 200, 'quantization': 'none'}, **recipe}
    if value.get('edge_type') != 'abliteration' or value.get('method') not in {'classic', 'heretic'}:
        raise ValueError('expected classic or heretic abliteration recipe')
    if len(value.get('parents', [])) != 1 or type(value.get('seed')) is not int:
        raise ValueError('abliteration requires one pinned parent and an explicit seed')
    import re
    if not re.fullmatch(r'cpu|cuda:[0-9]+', value['device']):
        raise ValueError('device must be cpu or an explicitly allocated cuda index')
    if value['torch_dtype'] not in {'float32', 'bfloat16'}:
        raise ValueError('unsupported abliteration export dtype')
    if value['method'] == 'heretic' and not isinstance(value.get('heretic_config'), dict):
        raise ValueError('Heretic requires its actual selected config')
    return value


def _partition(inputs):
    required = {'sample_ids', 'source_training_ids', 'heldout_ids', 'validation_ids', 'test_ids', 'direction_labels'}
    if not required <= inputs.keys():
        raise ValueError('frozen direction training and heldout split IDs/labels are required')
    ids = inputs['sample_ids']
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('direction sample IDs must be nonempty and unique')
    for key in ('heldout_ids', 'validation_ids', 'test_ids', 'source_training_ids'):
        if set(ids) & set(inputs[key]):
            raise ValueError('direction training and ' + key + ' overlap')
    if set(inputs['source_training_ids']) & set(inputs['heldout_ids']):
        raise ValueError('source training and heldout IDs overlap')
    if set(inputs['validation_ids']) & set(inputs['test_ids']):
        raise ValueError('validation and test IDs overlap')
    if not (set(inputs['validation_ids']) | set(inputs['test_ids'])) <= set(inputs['heldout_ids']):
        raise ValueError('validation/test IDs must belong to the frozen heldout set')
    labels = inputs['direction_labels']
    if len(labels) != len(ids) or any(type(v) is not int for v in labels) or set(labels) != {0, 1}:
        raise ValueError('direction labels must contain both groups 0 and 1 for every sample')


def installed_heretic_runner(recipe, inputs, output):
    """Reject the installed unsafe API before it can load or download anything.

    Inspected upstream heretic-llm 1.4.0 model.Model.__init__, reload and
    config.Settings. Local paths are supported, but the model loader has no
    local_files_only/use_safetensors enforcement and may enable remote code.
    Its tokenizer and processor loaders have the same locality gap. We never
    invoke the interactive CLI (which also exposes download/upload actions).
    https://github.com/p-e-w/heretic
    """
    installed = version('heretic-llm')
    if installed != HERETIC_VERSION:
        raise RuntimeError('Heretic installed version differs from audited 1.4.0; safe adapter requires review')
    raise RuntimeError('Heretic 1.4.0 safe local output unavailable: model loader lacks '
                       'local_files_only=True and use_safetensors=True enforcement, and '
                       'trust_remote_code can become True on reload; no weights were opened')


def _writers(model):
    import torch
    if model.config.model_type not in SUPPORTED_FAMILIES:
        raise NotImplementedError('unsupported residual writer architecture: ' + model.config.model_type)
    layers = getattr(getattr(model, 'model', None), 'layers', None)
    if layers is None:
        raise NotImplementedError('model.layers residual layout is unsupported')
    result = {}
    for index, layer in enumerate(layers):
        for path in ('self_attn.o_proj', 'mlp.down_proj'):
            module = layer.get_submodule(path)
            name = f'model.layers.{index}.{path}'
            if not isinstance(module, torch.nn.Linear) or module.out_features != model.config.hidden_size:
                raise NotImplementedError(name + ': unsupported residual writer')
            result[name + '.weight'] = dict(layer=index, residual_axis=0)
            if module.bias is not None:
                result[name + '.bias'] = dict(layer=index, residual_axis=0)
    return layers, result


def _validation_tokens(inputs):
    import torch
    from safetensors.torch import load_file
    if 'validation_tokens_path' not in inputs:
        return None
    path = contained_path(Path(inputs['root']), inputs['validation_tokens_path'], 'validation tokens')
    if path.suffix != '.safetensors' or path.is_symlink():
        raise ValueError('validation tokens require a local safetensors file')
    if hashlib.sha256(path.read_bytes()).hexdigest() != inputs.get('validation_tokens_sha256'):
        raise ValueError('frozen validation token hash mismatch')
    tokens = load_file(str(path))['input_ids']
    if tokens.ndim != 2 or tokens.dtype != torch.long or len(tokens) != len(inputs['validation_ids']) or not len(tokens):
        raise ValueError('validation tokens differ from frozen validation IDs')
    return tokens


def run_abliteration(recipe: dict, inputs: dict, output: Path) -> None:
    """Write into write_checkpoint staging; never touch test tokens or the Hub."""
    import torch
    from safetensors.torch import save_file
    selected = validate_recipe(recipe)
    _partition(inputs)
    output = Path(output)
    if selected['method'] == 'heretic':
        injected = inputs.get('heretic_runner')
        if injected is None:
            # Refuse before any unsafe upstream loading. No fake classic fallback.
            installed_heretic_runner(selected, inputs, output)
        output.mkdir(parents=True, exist_ok=True)
        # An injected test runner gets only training IDs/labels, never validation,
        # test data, credentials, or runtime paths. It cannot claim a real output.
        report = injected(recipe, {k: inputs[k] for k in ('sample_ids', 'direction_labels')}, output)
        if (not isinstance(report, dict) or report.get('config') != selected['heretic_config']
                or not report.get('optimization_trace') or not report.get('per_layer_settings')
                or report.get('synthetic') is not True or report.get('runner_kind') != 'injected_test'
                or not (output / 'model.safetensors').is_file()):
            raise ValueError('Heretic test runner requires safe output, exact config, per-layer settings and optimization trace')
        _portable(report)
        atomic_json(output / 'abliteration.json', dict(report, method='heretic',
            package='heretic-llm', pinned_version=HERETIC_VERSION,
            direction_sample_ids=inputs['sample_ids'], validation_status='not_run_synthetic_dispatch',
            quality_claim='none; injected dispatch fixture is not Heretic optimization'))
        return
    from .training import _frozen_inputs
    preprocessing = inputs.get('data_pin', {}).get('preprocessing', {})
    if (preprocessing.get('format') != 'abliteration_tokens_v1' or preprocessing.get('padding') != 'none'
            or preprocessing.get('residual_position') != 'last_token'
            or preprocessing.get('direction_labels_sha256') != content_hash(inputs['direction_labels'])):
        raise ValueError('frozen abliteration preprocessing and direction label hash are required')
    validation = _validation_tokens(inputs)
    model, tokens = _frozen_inputs(selected, inputs)
    if tokens.ndim != 2 or tokens.shape[1] < 1 or tokens.min() < 0 or tokens.max() >= model.config.vocab_size:
        raise ValueError('invalid frozen direction token shape or vocabulary')
    if validation is not None and (validation.min() < 0 or validation.max() >= model.config.vocab_size):
        raise ValueError('invalid frozen validation vocabulary')
    model.eval().requires_grad_(False)
    layers, writers = _writers(model)
    # Float32 projection/checkpoint export is available for precise algebra tests.
    model.to(dtype=torch.float32)
    device = next(model.parameters()).device
    observations = {index: [] for index in range(len(layers))}
    handles = []
    def hook(index):
        def collect(module, args, result):
            value = result[0] if isinstance(result, tuple) else result
            observations[index].append(value[:, -1].detach().cpu().float())
        return collect
    try:
        handles = [layer.register_forward_hook(hook(index)) for index, layer in enumerate(layers)]
        with torch.no_grad():
            for batch in tokens.split(8):
                model(input_ids=batch.to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    directions = mean_directions({i: torch.cat(rows) for i, rows in observations.items()}, inputs['direction_labels'])
    with torch.no_grad():
        before = model(input_ids=validation.to(device)).logits.float().cpu() if validation is not None else None
    state = model.state_dict()
    edited = classic_edit(state, directions, writers)
    model.load_state_dict(edited)
    with torch.no_grad():
        after = model(input_ids=validation.to(device)).logits.float().cpu() if validation is not None else None
    metrics = None if before is None else dict(max_abs_logit_drift=float((after - before).abs().max()),
                                               rms_logit_drift=float((after - before).square().mean().sqrt()))
    model.config._name_or_path = ''
    model.to(dtype=getattr(torch, selected['torch_dtype']))
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    diagnostics = output / 'diagnostics'; diagnostics.mkdir()
    save_file({str(i): r.cpu().contiguous() for i, r in directions.items()}, str(diagnostics / 'directions.safetensors'))
    atomic_json(output / 'abliteration.json', dict(method='classic', algorithm='normalized_mean_difference_projection_v1',
        formula='(I-r*r.T)W', writers=writers, direction_sample_ids=inputs['sample_ids'],
        labels_sha256=content_hash(inputs['direction_labels']),
        source_training_ids_sha256=content_hash(inputs['source_training_ids']), residual_position='last_token_after_decoder_block',
        validation_status='reported_only' if metrics is not None else 'not_run_no_frozen_validation_tokens',
        validation_metrics=metrics, validation_ids=inputs['validation_ids'], test_data_used=False,
        primary_weights=['model.safetensors'], secondary_weights=['diagnostics/directions.safetensors'],
        secondary_reason='direction diagnostics; excluded from effective checkpoint tensor set',
        skipped_tensors={name: 'not a residual-writing output projection or bias' for name in state if name not in writers},
        export_dtype=selected['torch_dtype'], quality_claim='none; projection is not a refusal-quality test'))
