"""Bounded LoRA training with frozen parents and complete safe checkpoints."""
import hashlib
import json
import re
from pathlib import Path

from .artifacts import atomic_json, content_hash
from .pins import verify_pin
from .schema import contained_path

DEFAULTS = dict(steps=128, effective_batch=16, sequence_length=512,
                learning_rate=2e-4, optimizer='adamw', scheduler='constant',
                dropout=0, rslora=False, device='cpu', torch_dtype='bfloat16')
# OLMo2's installed native implementation uses these separate projections too.
TARGETS = {name: ['q_proj', 'v_proj'] for name in ('llama', 'qwen2', 'qwen3', 'olmo2')}


def validate_recipe(recipe):
    value = {**DEFAULTS, **recipe}
    for field in ('steps', 'effective_batch', 'sequence_length', 'rank'):
        if type(value.get(field)) is not int or value[field] < 1:
            raise ValueError(f'{field}: expected a positive integer')
    if value.get('alpha') != value['rank'] or value['dropout'] != 0 or value['rslora']:
        raise ValueError('LoRA requires alpha=rank, zero dropout and no rslora')
    if value['optimizer'] != 'adamw' or value['scheduler'] != 'constant':
        raise ValueError('unsupported optimizer or scheduler')
    import math
    if not math.isfinite(value['learning_rate']) or value['learning_rate'] <= 0:
        raise ValueError('learning_rate must be finite and positive')
    if not re.fullmatch(r'cpu|cuda:[0-9]+', value['device']):
        raise ValueError('device must be cpu or an explicitly allocated cuda index')
    if value['torch_dtype'] not in {'float32', 'bfloat16'}:
        raise ValueError('torch_dtype must be float32 or bfloat16')
    return value


def merge_adapter(base_state: dict, adapter_state: dict, config: dict) -> dict:
    """Merge explicit PEFT A/B matrices in fp32; reject missing/unmatched factors."""
    rank, alpha = config['rank'], config['alpha']
    if rank <= 0 or config.get('rslora', False):
        raise ValueError('invalid rank or unsupported rslora')
    normalized = {}
    for key, tensor in adapter_state.items():
        key = key.removeprefix('base_model.model.').replace('.default.weight', '.weight')
        if key in normalized:
            raise ValueError('duplicate adapter factor')
        normalized[key] = tensor
    result = {key: tensor.float().clone() for key, tensor in base_state.items()}
    consumed = set()
    for key in normalized:
        if not key.endswith('.lora_A.weight'):
            continue
        stem = key.removesuffix('.lora_A.weight')
        b_key, target = stem + '.lora_B.weight', stem + '.weight'
        if b_key not in normalized or target not in result:
            raise ValueError(f'{stem}: unmatched adapter factor or target')
        a, b = normalized[key].float(), normalized[b_key].float()
        if (a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank
                or tuple(result[target].shape) != (b.shape[0], a.shape[1])):
            raise ValueError(f'{stem}: invalid adapter factor shape')
        result[target] += (alpha / rank) * (b @ a)
        consumed.update((key, b_key))
    if consumed != set(normalized) or not consumed:
        raise ValueError('unmatched or empty adapter factors')
    return result


def _frozen_inputs(recipe, inputs, *, model_loader=None, model_kwargs=None):
    """Resolve only prepared local safetensors inputs with frozen evidence."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM
    if 'model' in inputs and 'token_ids' in inputs:
        # Explicit in-memory inputs are for the small offline fixture only.
        model, tokens = inputs['model'], inputs['token_ids']
        if (model.config.hidden_size != 32 or model.config.num_hidden_layers != 2
                or model.config.vocab_size != 64 or tuple(tokens.shape) != (8, 8)
                or recipe['steps'] != 2 or recipe.get('rank', 8) != 8):
            raise ValueError('in-memory inputs are restricted to the tiny fixture')
        return model.to(device=recipe.get('device', 'cpu')), tokens
    required = {'root', 'model_path', 'tokens_path', 'model_pin', 'data_pin', 'sample_ids', 'heldout_ids', 'tokens_sha256'}
    if not required <= inputs.keys():
        raise ValueError('frozen pinned model and dataset inputs are required; dataset license/splits may be pending')
    for key in ('model_pin', 'data_pin'):
        verify_pin(inputs[key])
    parent = recipe['parents'][0]
    pin = inputs['model_pin']
    local = inputs.get('local_parent')
    if local:
        from .schema import LOCAL_KEY
        identity = recipe['parent_identities'][parent]
        match = LOCAL_KEY.fullmatch(identity)
        if not match or match.group(1) != parent or local.get('model_key') != identity:
            raise ValueError('local SFT parent manifest identity mismatch')
        pin = local
    elif recipe['parent_identities'][parent] != pin['repo_id'] + '@' + pin['revision']:
        raise ValueError('pinned parent identity mismatch')
    if not inputs['sample_ids'] or len(set(inputs['sample_ids'])) != len(inputs['sample_ids']):
        raise ValueError('frozen training sample IDs must be unique and nonempty')
    if set(inputs['sample_ids']) & set(inputs['heldout_ids']):
        raise ValueError('training and heldout sample IDs overlap')
    if content_hash(inputs['sample_ids']) != inputs['data_pin']['selected_index_sha256']:
        raise ValueError('frozen selected index hash mismatch')
    root = Path(inputs['root'])
    model_path = contained_path(root, inputs['model_path'], 'model')
    token_path = contained_path(root, inputs['tokens_path'], 'tokens')
    def file_hash(path):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()
    for path in model_path.rglob('*'):
        contained_path(model_path, path.relative_to(model_path).as_posix(), 'model')
        if path.is_symlink():
            raise ValueError('symlinks are not permitted in pinned local inputs')
    config_path = model_path / 'config.json'
    if local:
        provenance = json.loads((model_path / 'recipe.json').read_text())
        lineage = json.loads((model_path / 'lineage.json').read_text())
        from .artifacts import _lineage
        source = provenance.get('recipe', {})
        training = json.loads((model_path / 'training.json').read_text())
        if (lineage.get('name') != 'lab/' + parent or lineage.get('relation') != 'sft'
                or provenance.get('model_id') != parent
                or provenance.get('config_hash') != content_hash(source)
                or source.get('edge_type') != 'sft'
                or lineage != _lineage(parent, source)
                or provenance.get('input_hash') != content_hash(dict(
                    parents=source['parent_identities'], inputs=source.get('inputs', {})))
                or training.get('steps_completed', 0) < 1
                or training.get('steps_completed') != source.get('steps')
                or training.get('full_parameters') is not True):
            raise ValueError('DPO requires a completed SFT parent with valid provenance')
        if source.get('inputs', {}).get('model_pin') != inputs['model_pin']:
            raise ValueError('DPO source pin differs from the SFT parent source')
    if content_hash(json.loads(config_path.read_text())) != pin['config_sha256']:
        raise ValueError('pinned model canonical config hash mismatch')
    if pin.get('config_file_sha256') and file_hash(config_path) != pin['config_file_sha256']:
        raise ValueError('pinned model source config hash mismatch')
    expected = pin.get('weight_sha256')
    if not isinstance(expected, dict) or not expected:
        raise ValueError('missing weight provenance: expected pinned weight_sha256 mapping')
    for filename, digest in expected.items():
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('invalid pinned weight provenance hash')
        path = contained_path(model_path, filename, 'weight')
        if path.suffix != '.safetensors' or not path.is_file():
            raise ValueError('missing pinned safetensors weight file')
        if file_hash(path) != digest:
            raise ValueError(f'{filename}: pinned weight hash mismatch')
    actual = {p.relative_to(model_path).as_posix() for p in model_path.rglob('*.safetensors')}
    if actual != set(expected):
        raise ValueError('local weight file set differs from pinned weight provenance')
    index_path = model_path / 'model.safetensors.index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map = index.get('weight_map')
        if not isinstance(weight_map, dict) or not weight_map or set(weight_map.values()) != set(expected):
            raise ValueError('sharding index file list differs from pinned weights')
        if 'model.safetensors' in expected:
            raise ValueError('ambiguous monolithic and sharded weight layouts')
    elif set(expected) != {'model.safetensors'}:
        raise ValueError('sharded weights require a sharding index')
    if recipe.get('max_parameters'):
        from safetensors import safe_open
        count = 0
        for filename in expected:
            with safe_open(str(model_path / filename), framework='pt') as weights:
                for name in weights.keys():
                    import math
                    count += math.prod(weights.get_slice(name).get_shape())
        if count > recipe['max_parameters']:
            raise ValueError('full tuning is capped at 1.7B parameters (weight headers)')
    if file_hash(token_path) != inputs['tokens_sha256']:
        raise ValueError('frozen token file hash mismatch')
    tokens = load_file(str(token_path))['input_ids']
    if tokens.dtype != torch.long or len(tokens) != len(inputs['sample_ids']):
        raise ValueError('frozen token rows differ from selected sample IDs')
    # No Hub access or implicit downloads, no pickle weights or remote code.
    model = (model_loader or AutoModelForCausalLM).from_pretrained(model_path, **(model_kwargs or {}), local_files_only=True,
                                                use_safetensors=True, trust_remote_code=False,
                                                dtype=getattr(torch, recipe.get('torch_dtype', 'bfloat16')))
    return model.to(device=recipe.get('device', 'cpu')), tokens


def train_lora(recipe: dict, inputs: dict, output: Path) -> None:
    """Write into write_checkpoint's staging directory; never publish directly."""
    import torch
    recipe = validate_recipe(recipe)
    torch.manual_seed(recipe['seed'])
    model, tokens = _frozen_inputs(recipe, inputs)
    targets = recipe.get('target_modules', TARGETS.get(model.config.model_type))
    if not targets or any(target not in TARGETS.get(model.config.model_type, []) for target in targets):
        raise ValueError('unsupported architecture or target modules')
    names = dict(model.named_modules())
    for target in targets:
        matching = [module for name, module in names.items() if name.endswith('.' + target)]
        if not matching or any(not isinstance(module, torch.nn.Linear) for module in matching):
            raise ValueError(f'unknown or unsupported target {target}')
    if (tokens.ndim != 2 or tokens.shape[1] != recipe['sequence_length'] or tokens.dtype != torch.long
            or tokens.min() < 0 or tokens.max() >= model.config.vocab_size):
        raise ValueError('invalid frozen token shape, dtype or vocabulary')
    model.config._name_or_path = ''
    model.requires_grad_(False)
    adapted = _local_peft_model(model, dict(r=recipe['rank'], lora_alpha=recipe['alpha'],
                              lora_dropout=0, target_modules=targets, bias='none', task_type='CAUSAL_LM'))
    parameters = dict(adapted.named_parameters())
    trainable = [p for name, p in parameters.items() if p.requires_grad]
    if not trainable or any(p.requires_grad != ('lora_A.' in name or 'lora_B.' in name)
                            for name, p in parameters.items()):
        raise ValueError('only declared adapters may be trainable')
    frozen_versions = {name: p._version for name, p in parameters.items() if not p.requires_grad}
    optimizer = torch.optim.AdamW(trainable, lr=recipe['learning_rate'], weight_decay=0)
    adapted.train()
    adapted.config.use_cache = False
    actual_device = str(next(adapted.parameters()).device)
    actual_dtype = str(next(adapted.parameters()).dtype).removeprefix('torch.')
    losses = []
    for step in range(recipe['steps']):
        indexes = torch.arange(step * recipe['effective_batch'], (step + 1) * recipe['effective_batch']) % len(tokens)
        batch = tokens[indexes].to(next(adapted.parameters()).device)
        optimizer.zero_grad(set_to_none=True)
        loss = adapted(input_ids=batch, labels=batch).loss
        if not torch.isfinite(loss):
            raise ValueError('nonfinite training loss')
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    unchanged = all(parameters[name]._version == version for name, version in frozen_versions.items())
    if not unchanged:
        raise ValueError('frozen parent tensors changed during adapter training')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    adapted.save_pretrained(output / 'adapter', safe_serialization=True, save_embedding_layers=False)
    merged = adapted.merge_and_unload(safe_merge=True)
    merged.to(dtype=torch.bfloat16)
    merged.config.use_cache = True
    # Remove runtime source paths from the portable model config.
    merged.config._name_or_path = ''
    merged.save_pretrained(output, safe_serialization=True)
    atomic_json(output / 'training.json', dict(steps_completed=recipe['steps'], losses=losses,
                base_tensors_unchanged=unchanged, primary_weights=['model.safetensors'],
                secondary_weights=['adapter/adapter_model.safetensors'],
                secondary_reason='adapter factors for provenance; exclude from effective checkpoint tensor set',
                resolved_target_modules=targets, device=actual_device, torch_dtype=actual_dtype))


FULL_DEFAULTS = {
    'sft': dict(steps=256, effective_batch=16, sequence_length=512, learning_rate=2e-5),
    'dpo': dict(steps=128, effective_batch=8, sequence_length=512, learning_rate=5e-6, beta=0.1),
}


def validate_full_recipe(recipe, edge):
    import math
    value = validate_recipe({**FULL_DEFAULTS[edge], **recipe, 'rank': 8, 'alpha': 8})
    for key in ('rank', 'alpha', 'rslora'):
        value.pop(key)
    if value.get('edge_type', edge) != edge:
        raise ValueError('wrong full tuning edge type')
    count = value.get('base_num_parameters')
    if count is not None and (type(count) is not int or count < 1 or count > 1_700_000_000):
        raise ValueError('full tuning is capped at 1.7B parameters')
    if edge == 'dpo' and (not math.isfinite(value['beta']) or value['beta'] <= 0):
        raise ValueError('beta must be finite and positive')
    value['max_parameters'] = 1_700_000_000
    return value


def _full_inputs(recipe, inputs, edge):
    import torch
    from safetensors.torch import load_file
    if not inputs.get('sample_ids') or len(set(inputs['sample_ids'])) != len(inputs['sample_ids']):
        raise ValueError('frozen training sample IDs are required and must be unique')
    if 'heldout_ids' not in inputs:
        raise ValueError('frozen heldout sample IDs are required')
    if set(inputs['sample_ids']) & set(inputs['heldout_ids']):
        raise ValueError('training and heldout sample IDs overlap')
    tiny = 'model' in inputs
    if edge == 'dpo':
        if tiny and inputs.get('parent_relation') != 'sft':
            raise ValueError('DPO requires its matching SFT parent')
        if not tiny and not inputs.get('local_parent'):
            raise ValueError('DPO requires a pinned local SFT parent manifest')
    if not tiny:
        preprocessing = inputs.get('data_pin', {}).get('preprocessing', {})
        expected = dict(format='causal_tokens_v1', loss='all_tokens', padding='none') if edge == 'sft' else dict(
            format='preference_tokens_v1', loss='completion_only', padding='none')
        if any(preprocessing.get(k) != v for k, v in expected.items()):
            raise ValueError('explicit frozen token/template preprocessing is required')
    model, tokens = _frozen_inputs(recipe, inputs)
    if sum(p.numel() for p in model.parameters()) > recipe['max_parameters']:
        raise ValueError('full tuning is capped at 1.7B parameters')
    if (tokens.ndim != 2 or tokens.shape[1] != recipe['sequence_length']
            or len(tokens) != len(inputs['sample_ids']) or tokens.dtype != torch.long
            or tokens.numel() == 0 or tokens.min() < 0 or tokens.max() >= model.config.vocab_size):
        raise ValueError('invalid frozen token shape, dtype or vocabulary')
    preference = None
    if edge == 'dpo':
        data = inputs if tiny else load_file(str(contained_path(Path(inputs['root']), inputs['tokens_path'], 'tokens')))
        rejected = data['rejected_token_ids' if tiny else 'rejected_input_ids']
        masks = [data['chosen_mask'], data['rejected_mask']]
        if (rejected.shape != tokens.shape or rejected.dtype != torch.long
                or rejected.min() < 0 or rejected.max() >= model.config.vocab_size):
            raise ValueError('invalid frozen rejected token rows')
        for mask in masks:
            if (mask.shape != tokens.shape or not torch.all((mask == 0) | (mask == 1))
                    or not torch.all(mask[:, 0] == 0) or not torch.all(mask[:, 1:].sum(1) > 0)):
                raise ValueError('invalid completion mask')
        # Both completions must follow the same nonempty prompt, without masked holes.
        for index in range(len(tokens)):
            starts = [int(torch.nonzero(mask[index], as_tuple=False)[0]) for mask in masks]
            if starts[0] != starts[1] or not torch.equal(tokens[index, :starts[0]], rejected[index, :starts[1]]):
                raise ValueError('chosen/rejected prompts must match')
            if any(not torch.all(mask[index, start:] == 1) for mask, start in zip(masks, starts)):
                raise ValueError('completion masks must be contiguous suffixes')
        preference = (rejected, *masks)
    return model, tokens, preference


def _completion_logps(model, tokens, mask):
    import torch
    logits = model(input_ids=tokens).logits[:, :-1].float()
    logps = torch.log_softmax(logits, dim=-1).gather(-1, tokens[:, 1:, None]).squeeze(-1)
    return (logps * mask[:, 1:]).sum(dim=-1)


def _train_full(recipe, inputs, output, edge):
    import copy
    import torch
    import torch.nn.functional as F
    recipe = validate_full_recipe(recipe, edge)
    torch.manual_seed(recipe.get('training_seed', recipe['seed']))
    model, tokens, preference = _full_inputs(recipe, inputs, edge)
    model.config._name_or_path = ''
    model.config.use_cache = False
    # Disable architecture dropout explicitly in both policy and reference.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
    reference, reference_before = None, None
    if edge == 'dpo':
        reference = inputs.get('reference_model')
        if reference is None:
            reference = copy.deepcopy(model)
        policy_pointers = {p.data_ptr() for p in model.parameters()}
        if reference is model or any(p.data_ptr() in policy_pointers for p in reference.parameters()):
            raise ValueError('DPO reference must not share policy tensors')
        policy_state, reference_state = model.state_dict(), reference.state_dict()
        if (policy_state.keys() != reference_state.keys()
                or any(not torch.equal(v, reference_state[k].to(v)) for k, v in policy_state.items())):
            raise ValueError('DPO reference must exactly match the SFT parent')
        reference.to(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
        reference.requires_grad_(False)
        reference.eval()
        # Mutation versions detect writes without a third full weight copy.
        reference_before = {k: p._version for k, p in reference.named_parameters()}
    # Keep full-parameter optimizer updates in fp32 at the specified small LRs.
    # Requested bf16 controls autocast compute and all checkpoints remain bf16.
    compute_dtype = recipe['torch_dtype']
    model.float()
    if reference is not None:
        reference.float()
        reference_before = {k: p._version for k, p in reference.named_parameters()}
    model.requires_grad_(True)
    matrix_name, matrix = next((name, p) for name, p in model.named_parameters()
                               if p.ndim == 2 and '.mlp.' in name)
    matrix_before = matrix.detach().float().clone()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['learning_rate'], weight_decay=0)
    device = next(model.parameters()).device
    dtype = str(next(model.parameters()).dtype).removeprefix('torch.')
    losses = []
    for step in range(recipe['steps']):
        indexes = torch.arange(step * recipe['effective_batch'], (step + 1) * recipe['effective_batch']) % len(tokens)
        chosen = tokens[indexes].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=compute_dtype == 'bfloat16'):
            if edge == 'sft':
                loss = model(input_ids=chosen, labels=chosen).loss
            else:
                rejected, chosen_mask, rejected_mask = [value[indexes].to(device) for value in preference]
                policy_ratio = _completion_logps(model, chosen, chosen_mask) - _completion_logps(model, rejected, rejected_mask)
                with torch.no_grad():
                    reference_ratio = _completion_logps(reference, chosen, chosen_mask) - _completion_logps(reference, rejected, rejected_mask)
                loss = -F.logsigmoid(recipe['beta'] * (policy_ratio - reference_ratio)).mean()
        if not torch.isfinite(loss):
            raise ValueError('nonfinite full training loss')
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    unchanged = reference is None or all(p._version == reference_before[k] for k, p in reference.named_parameters())
    if not unchanged:
        raise ValueError('frozen DPO reference changed')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    updated_matrix = dict(name=matrix_name,
        delta_norm_before_export=float(torch.linalg.vector_norm(matrix.detach().float() - matrix_before)))
    model.to(dtype=torch.bfloat16)
    model.config.use_cache = True
    model.save_pretrained(output, safe_serialization=True)
    atomic_json(output / 'training.json', dict(steps_completed=recipe['steps'], losses=losses,
        training_seed=recipe.get('training_seed', recipe['seed']),
        sample_ids=inputs['sample_ids'], heldout_ids=inputs['heldout_ids'], full_parameters=True,
        updated_matrix=updated_matrix,
        preprocessing=inputs.get('data_pin', {}).get('preprocessing', {'fixture': 'synthetic-tokens-v1'}),
        reference_unchanged=unchanged if reference is not None else None,
        reference_parent=recipe['parents'][0] if edge == 'dpo' else None,
        loss_implementation='torch_logsigmoid_dpo' if edge == 'dpo' else 'causal_cross_entropy',
        primary_weights=sorted(p.name for p in output.glob('*.safetensors')), device=str(device),
        torch_dtype=compute_dtype, parameter_dtype=dtype, output_dtype='bfloat16'))


def train_sft(recipe: dict, inputs: dict, output: Path) -> None:
    """Full-parameter causal SFT over explicitly frozen, unpadded token rows."""
    _train_full(recipe, inputs, output, 'sft')


def train_dpo(recipe: dict, inputs: dict, output: Path) -> None:
    """Standard completion-only DPO; direct torch loss avoids an uninstalled TRL."""
    _train_full(recipe, inputs, output, 'dpo')


def validate_decision_recipe(recipe):
    """Bounded two-label sequence classification; embedded factors retain LoRA truth."""
    embedded = recipe.get('embedded_adapter', False)
    edge = 'lora' if embedded else 'sft'
    if recipe.get('edge_type', edge) != edge:
        raise ValueError('decision edge_type must reflect actual tuning')
    value = validate_full_recipe({**recipe, 'edge_type': 'sft'}, 'sft')
    value['edge_type'] = edge
    names = value.get('label_names')
    if not isinstance(names, list) or len(names) != 2 or len(set(names)) != 2 or any(not isinstance(n, str) or not n for n in names):
        raise ValueError('two distinct label names are required')
    if type(value.get('pad_token_id')) is not int or value['pad_token_id'] < 0:
        raise ValueError('explicit pad_token_id is required')
    if embedded:
        if recipe.get('rank') != 8 or recipe.get('alpha') != 8:
            raise ValueError('embedded decision adapter requires rank=alpha=8')
        value.update(rank=8, alpha=8)
    return value


def _local_peft_model(model, adapter):
    """Keep optional PEFT quantization imports from fetching Hub kernels."""
    import os
    from huggingface_hub import constants
    previous = constants.HF_HUB_OFFLINE
    environment = os.environ.get('HF_HUB_OFFLINE')
    constants.HF_HUB_OFFLINE = True
    os.environ['HF_HUB_OFFLINE'] = '1'
    try:
        from peft import LoraConfig, get_peft_model
        return get_peft_model(model, LoraConfig(**adapter))
    finally:
        constants.HF_HUB_OFFLINE = previous
        if environment is None:
            os.environ.pop('HF_HUB_OFFLINE', None)
        else:
            os.environ['HF_HUB_OFFLINE'] = environment


def load_decision_checkpoint(path):
    """Rebuild native installed Transformers/PEFT classes from safe local weights."""
    from transformers import AutoConfig, AutoModelForSequenceClassification
    from safetensors.torch import load_file
    path = Path(path)
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    metadata = path / 'embedded_adapter.json'
    if not metadata.exists():
        return AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True,
            trust_remote_code=False, use_safetensors=True)
    adapter = json.loads(metadata.read_text())
    if (adapter.get('r') != 8 or adapter.get('lora_alpha') != 8
            or adapter.get('target_modules') != ['q_proj', 'v_proj']
            or adapter.get('task_type') != 'SEQ_CLS' or adapter.get('use_rslora') is not False):
        raise ValueError('unsupported embedded adapter metadata')
    model = AutoModelForSequenceClassification.from_config(config, trust_remote_code=False)
    model = _local_peft_model(model, adapter)
    model.load_state_dict(load_file(str(path / 'model.safetensors')), strict=True)
    model.eval()
    return model


def train_decision(recipe: dict, inputs: dict, output: Path) -> None:
    """Train full classification models or a full checkpoint with embedded PEFT factors."""
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForSequenceClassification
    recipe = validate_decision_recipe(recipe)
    torch.manual_seed(recipe['seed'])
    tiny = 'model' in inputs
    if not inputs.get('sample_ids') or 'heldout_ids' not in inputs:
        raise ValueError('frozen sample and heldout IDs are required')
    if len(set(inputs['sample_ids'])) != len(inputs['sample_ids']) or set(inputs['sample_ids']) & set(inputs['heldout_ids']):
        raise ValueError('frozen sample IDs overlap or are duplicated')
    if not tiny:
        preprocessing = inputs.get('data_pin', {}).get('preprocessing', {})
        if preprocessing != dict(format='classification_tokens_v1', padding='right', label_names=recipe['label_names'], pad_token_id=recipe['pad_token_id']):
            raise ValueError('frozen classification preprocessing is required')
    model, tokens = _frozen_inputs(recipe, inputs, model_loader=AutoModelForSequenceClassification,
        model_kwargs=dict(num_labels=2, id2label=dict(enumerate(recipe['label_names'])),
                          label2id={name: i for i, name in enumerate(recipe['label_names'])},
                          pad_token_id=recipe['pad_token_id']))
    if model.config.model_type not in {'llama', 'qwen2', 'qwen3', 'modernbert'}:
        raise ValueError('unsupported decision architecture')
    if sum(p.numel() for p in model.parameters()) > recipe['max_parameters']:
        raise ValueError('full tuning is capped at 1.7B parameters')
    data = inputs if tiny else load_file(str(contained_path(Path(inputs['root']), inputs['tokens_path'], 'tokens')))
    labels, mask = data['labels'], data['attention_mask']
    if (tokens.ndim != 2 or tokens.shape != (len(inputs['sample_ids']), recipe['sequence_length'])
            or tokens.dtype != torch.long or tokens.min() < 0 or tokens.max() >= model.config.vocab_size):
        raise ValueError('invalid frozen classification tokens')
    if labels.shape != (len(tokens),) or labels.dtype != torch.long or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError('invalid two-class labels')
    if (mask.shape != tokens.shape or not torch.all((mask == 0) | (mask == 1))
            or not torch.all(mask[:, 0] == 1) or torch.any(mask[:, 1:] > mask[:, :-1])
            or torch.any(tokens[mask == 0] != recipe['pad_token_id'])
            or torch.any(tokens[mask == 1] == recipe['pad_token_id'])):
        raise ValueError('invalid right padding or attention mask')
    model.config.num_labels = 2
    model.config.id2label = dict(enumerate(recipe['label_names']))
    model.config.label2id = {n: i for i, n in model.config.id2label.items()}
    model.config.pad_token_id = recipe['pad_token_id']
    model.config.problem_type = 'single_label_classification'
    model.config._name_or_path = ''
    model.config.architectures = [type(model).__name__]
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
    model.float()
    model.config.dtype = torch.float32
    model.requires_grad_(True)
    embedded = recipe.get('embedded_adapter', False)
    adapter_config = None
    if embedded:
        if model.config.model_type != 'qwen3' or recipe.get('target_modules') != ['q_proj', 'v_proj']:
            raise ValueError('embedded decision factors require qwen3 q_proj/v_proj')
        model.requires_grad_(False)
        adapter_config = dict(r=8, lora_alpha=8, lora_dropout=0, target_modules=['q_proj', 'v_proj'],
                              bias='none', task_type='SEQ_CLS', use_rslora=False)
        model = _local_peft_model(model, adapter_config)
    frozen = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=recipe['learning_rate'], weight_decay=0)
    model.train()
    losses = []
    for step in range(recipe['steps']):
        indices = torch.arange(step * recipe['effective_batch'], (step + 1) * recipe['effective_batch']) % len(tokens)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=recipe['torch_dtype'] == 'bfloat16'):
            loss = model(input_ids=tokens[indices].to(device), attention_mask=mask[indices].to(device), labels=labels[indices].to(device)).loss
        if not torch.isfinite(loss):
            raise ValueError('nonfinite classification loss')
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    if any(p._version != frozen[n] for n, p in model.named_parameters() if n in frozen):
        raise ValueError('frozen decision base changed')
    model.eval()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if embedded:
        model.config.save_pretrained(output)
        # Clone aliases into a single complete selected file; never adapter-only.
        save_file({n: t.detach().cpu().contiguous().clone() for n, t in model.state_dict().items()}, str(output / 'model.safetensors'), metadata={'format': 'pt'})
        atomic_json(output / 'embedded_adapter.json', adapter_config)
    else:
        model.save_pretrained(output, safe_serialization=True)
    atomic_json(output / 'training.json', dict(steps_completed=recipe['steps'], losses=losses,
        full_parameters=not embedded, base_tensors_unchanged=True if embedded else None,
        label_names=recipe['label_names'], architectures=model.config.architectures,
        pooling=getattr(model.config, 'classifier_pooling', 'last_nonpad_token'),
        pad_token_id=recipe['pad_token_id'], primary_weights=['model.safetensors'],
        sample_ids=inputs['sample_ids'], heldout_ids=inputs['heldout_ids'], device=str(device),
        torch_dtype=recipe['torch_dtype'], output_dtype='float32',
        adapter_scale=1.0 if embedded else None, adapter_config=adapter_config))
    if tiny:
        inputs['effective_model'] = model
