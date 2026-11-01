"""Audited symmetries for the four explicitly reviewed lab decoder layouts.

Only value channels rotate; Q/K, RoPE, residual coordinates and nonlinear gates
never rotate. Perturbations are bounded edits, not an evasion claim.
"""
import hashlib

import torch

METHODS = ('mlp_permutation', 'rescaling', 'value_rotation', 'targeted_perturbation')
ADAPTERS = {'llama', 'qwen2', 'qwen3', 'olmo2'}


def _layout(state, config, recipe):
    family = config.get('model_type')
    if family not in ADAPTERS:
        raise NotImplementedError(f'{family}: unsupported model adapter')
    if recipe.get('method') not in METHODS:
        raise NotImplementedError(f"{recipe.get('method')}: unsupported transform")
    if config.get('hidden_act', 'silu') != 'silu':
        raise NotImplementedError('mlp.gate_proj: only silu gated MLP supported')
    hidden, inter = config['hidden_size'], config['intermediate_size']
    heads = config['num_attention_heads']
    kv = config.get('num_key_value_heads', heads)
    dim = config.get('head_dim') or hidden // heads
    if min(hidden, inter, heads, kv, dim, config['num_hidden_layers']) <= 0 or heads % kv:
        raise NotImplementedError('self_attn.v_proj: unsupported GQA dimensions')
    required = {'model.embed_tokens.weight': (config['vocab_size'], hidden),
                'model.norm.weight': (hidden,)}
    allowed = dict(required, **{'lm_head.weight': (config['vocab_size'], hidden)})
    projections = []
    for i in range(config['num_hidden_layers']):
        prefix = f'model.layers.{i}.'
        shapes = {'mlp.gate_proj.weight': (inter, hidden), 'mlp.up_proj.weight': (inter, hidden),
                  'mlp.down_proj.weight': (hidden, inter), 'self_attn.q_proj.weight': (heads * dim, hidden),
                  'self_attn.k_proj.weight': (kv * dim, hidden), 'self_attn.v_proj.weight': (kv * dim, hidden),
                  'self_attn.o_proj.weight': (hidden, heads * dim)}
        for name, shape in shapes.items():
            required[prefix + name] = shape
            allowed[prefix + name] = shape
            # Affine terms transform on their output axis; downstream biases stay fixed.
            allowed[prefix + name.replace('.weight', '.bias')] = (shape[0],)
        norms = (['post_attention_layernorm', 'post_feedforward_layernorm'] if family == 'olmo2'
                 else ['input_layernorm', 'post_attention_layernorm'])
        if family in {'qwen3', 'olmo2'}:
            for name in ['self_attn.q_norm', 'self_attn.k_norm']:
                width = (heads if name.endswith('q_norm') else kv) * dim if family == 'olmo2' else dim
                required[prefix + name + '.weight'] = (width,)
                allowed[prefix + name + '.weight'] = (width,)
        for name in norms:
            required[prefix + name + '.weight'] = (hidden,)
            allowed[prefix + name + '.weight'] = (hidden,)
        projections.extend(prefix + 'self_attn.' + p + '_proj.weight' for p in ['q', 'k', 'v', 'o'])
    for name in required:
        if name not in state:
            raise NotImplementedError(name + ': missing required module')
    for name, tensor in state.items():
        if name not in allowed or tuple(tensor.shape) != allowed[name]:
            raise NotImplementedError(name + ': unsupported tensor/layout')
        if tensor.dtype not in {torch.float32, torch.float64, torch.bfloat16, torch.float16}:
            raise NotImplementedError(name + ': unsupported tensor dtype')
        if not torch.isfinite(tensor).all():
            raise ValueError(name + ': nonfinite tensor')
    if not config.get('tie_word_embeddings', False) and 'lm_head.weight' not in state:
        raise NotImplementedError('lm_head.weight: missing untied output')
    if config.get('tie_word_embeddings', False) and 'lm_head.weight' in state:
        if not torch.equal(state['lm_head.weight'], state['model.embed_tokens.weight']):
            raise NotImplementedError('lm_head.weight: inconsistent tied weights')
    if recipe['method'] == 'targeted_perturbation':
        if recipe.get('relative_frobenius_norm', 1e-3) != 1e-3 or recipe.get('target_matrices', 4) != 4:
            raise ValueError('targeted_perturbation requires four projections and relative norm 1e-3')
        targets = recipe.get('targets', projections[:4])
        if len(targets) != 4 or len(set(targets)) != 4 or any(n not in projections for n in targets):
            raise NotImplementedError('targeted_perturbation: four distinct reviewed attention projections required')
    return hidden, inter, heads, kv, dim, projections


def _transform(state, model_config, recipe):
    hidden, inter, heads, kv, dim, projections = _layout(state, model_config, recipe)
    generator = torch.Generator(device='cpu').manual_seed(int(recipe['seed']))
    # Keep aliasing of tied storage, while leaving every input tensor untouched.
    clones = {}
    result = {}
    for name, tensor in state.items():
        identity = (tensor.device, tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()))
        if identity not in clones:
            clones[identity] = tensor.clone()
        result[name] = clones[identity]
    if model_config.get('tie_word_embeddings', False) and 'lm_head.weight' in result:
        result['lm_head.weight'] = result['model.embed_tokens.weight']
    audit = dict(method=recipe['method'], seed=int(recipe['seed']),
                 function_preserving_exact_arithmetic=recipe['method'] != 'targeted_perturbation')
    def put(name, value):
        result[name] = value.to(device=state[name].device, dtype=state[name].dtype)
    if recipe['method'] == 'targeted_perturbation':
        audit['perturbations'] = {}
        for name in recipe.get('targets', projections[:4]):
            weight = state[name].detach().cpu().double()
            delta = torch.outer(torch.randn(weight.shape[0], generator=generator, dtype=torch.float64),
                                torch.randn(weight.shape[1], generator=generator, dtype=torch.float64))
            norm = weight.norm()
            if norm == 0:
                raise ValueError(name + ': relative perturbation of zero matrix undefined')
            delta *= 1e-3 * norm / delta.norm()
            put(name, weight + delta)
            audit['perturbations'][name] = dict(
                rank=1, precast_relative_norm=(delta.norm() / norm).item(),
                postcast_relative_norm=((result[name].detach().cpu().double() - weight).norm() / norm).item())
        audit['algebra'] = 'W + rank1_delta; four attention projections; no function-preservation or evasion claim'
    else:
        for i in range(model_config['num_hidden_layers']):
            prefix = f'model.layers.{i}.'
            if recipe['method'] == 'mlp_permutation':
                permutation = torch.randperm(inter, generator=generator)
                for projection in ['gate_proj', 'up_proj']:
                    for suffix in ['weight', 'bias']:
                        name = prefix + 'mlp.' + projection + '.' + suffix
                        if name in state:
                            put(name, state[name].index_select(0, permutation.to(state[name].device)))
                name = prefix + 'mlp.down_proj.weight'
                put(name, state[name].index_select(1, permutation.to(state[name].device)))
            elif recipe['method'] == 'rescaling':
                scale = 2. ** torch.randint(-2, 3, (inter,), generator=generator).double()
                for suffix in ['weight', 'bias']:
                    name = prefix + 'mlp.up_proj.' + suffix
                    if name in state:
                        multiplier = scale[:, None] if suffix == 'weight' else scale
                        put(name, state[name].detach().cpu().double() * multiplier)
                name = prefix + 'mlp.down_proj.weight'
                put(name, state[name].detach().cpu().double() / scale[None, :])
            else:
                v_name = prefix + 'self_attn.v_proj.weight'
                o_name = prefix + 'self_attn.o_proj.weight'
                v = state[v_name].detach().cpu().double().reshape(kv, dim, hidden)
                o = state[o_name].detach().cpu().double().reshape(hidden, heads, dim)
                rotations = torch.linalg.qr(torch.randn(kv, dim, dim, generator=generator, dtype=torch.float64)).Q
                # V' = R V and O'_h = O_h R^T for every query head in that KV group.
                put(v_name, (rotations @ v).reshape(kv * dim, hidden))
                inverse = rotations.repeat_interleave(heads // kv, dim=0).transpose(-1, -2)
                put(o_name, torch.einsum('hnd,nde->hne', o, inverse).reshape(hidden, heads * dim))
                name = prefix + 'self_attn.v_proj.bias'
                if name in state:
                    bias = state[name].detach().cpu().double().reshape(kv, dim, 1)
                    put(name, (rotations @ bias).flatten())
        audit['algebra'] = {
            'mlp_permutation': 'gate/up rows and biases P; down columns P inverse',
            'rescaling': 'up rows and bias times positive powers of two; down inverse; gate untouched',
            'value_rotation': 'per KV head V and bias R; all grouped O blocks R transpose; Q/K untouched',
        }[recipe['method']]
    return result, audit


def transform_state(state: dict, model_config: dict, recipe: dict) -> dict:
    """Return an independent transformed state; reject unreviewed layouts first."""
    return _transform(state, model_config, recipe)[0]


def check_equivalence(original, transformed, token_batches: list) -> dict:
    """Measure frozen-token logit drift with the normative fixture tolerances."""
    if not token_batches:
        raise ValueError('equivalence requires nonempty frozen validation tokens')
    dtype = next(original.parameters()).dtype
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-5
    digest = hashlib.sha256()
    passed, max_abs, max_rel, count = True, 0., 0., 0
    training = original.training, transformed.training
    original.eval()
    transformed.eval()
    try:
        with torch.inference_mode():
            for tokens in token_batches:
                if tokens.dtype != torch.long or tokens.ndim != 2 or not tokens.numel():
                    raise ValueError('validation tokens must be nonempty rank-two int64 batches')
                digest.update(str(tuple(tokens.shape)).encode())
                digest.update(tokens.detach().cpu().contiguous().numpy().tobytes())
                first = original(tokens.to(next(original.parameters()).device)).logits.float()
                second = transformed(tokens.to(next(transformed.parameters()).device)).logits.float().to(first.device)
                if first.shape != second.shape:
                    raise ValueError('logit shape mismatch')
                difference = (first - second).abs()
                passed &= bool(torch.allclose(first, second, atol=tolerance, rtol=tolerance))
                max_abs = max(max_abs, difference.max().item())
                max_rel = max(max_rel, (difference / first.abs().clamp_min(1e-12)).max().item())
                count += tokens.numel()
    finally:
        original.train(training[0])
        transformed.train(training[1])
    return dict(passed=passed, dtype=str(dtype).removeprefix('torch.'), atol=tolerance, rtol=tolerance,
                max_abs_error=max_abs, max_relative_error=max_rel, token_count=count,
                tokens_sha256=digest.hexdigest(), scope='frozen validation tokens only; not a global scientific guarantee')
