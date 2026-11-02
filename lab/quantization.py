"""Signed per-row quantize/dequantize simulation with pinned safe inputs.

Prepared input metadata uses path/model_key/config_sha256/config_file_sha256/
weight_sha256 plus a supplied extractor manifest (repo_id, revision, files).
This verifies supplied identities; it never computes a local identity.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

from .artifacts import _lineage, atomic_json, content_hash
from .schema import contained_path


def _bits(bits):
    if type(bits) is not int or bits not in (2, 4, 8):
        raise ValueError('bits must be one of 2, 4, 8')


def simulate_quantization(weight, bits):
    """Return fp32 simulated values; checkpoint serialization casts to bf16."""
    import torch
    _bits(bits)
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError('expected a 2D floating matrix')
    value = weight.detach().float()
    if not torch.isfinite(value).all():
        raise ValueError('nonfinite matrix weights')
    qmax = 2 ** (bits - 1) - 1
    scale = value.abs().amax(dim=1, keepdim=True) / qmax
    divisor = torch.where(scale == 0, torch.ones_like(scale), scale)
    return torch.round(value / divisor).clamp(-qmax, qmax) * scale


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_source(recipe, inputs):
    """Preflight portable names, immutable identity and bytes before tensor reads."""
    _bits(recipe.get('bits'))
    if recipe.get('edge_type') != 'quantization_sim' or len(recipe.get('parents', [])) != 1:
        raise ValueError('quantization requires exactly one direct parent')
    _lineage(recipe['model_id'], recipe)
    parent = recipe['parents'][0]
    if set(recipe.get('inputs', {})) != {parent}:
        raise ValueError('missing frozen parent input metadata')
    pin = recipe['inputs'][parent]
    identity = recipe['parent_identities'][parent]
    manifest = pin.get('manifest', {})
    if pin.get('model_key') != identity or manifest.get('repo_id', '')+'@'+manifest.get('revision', '') != identity:
        raise ValueError('supplied parent manifest identity mismatch')
    expected = pin.get('weight_sha256', {})
    files = manifest.get('files', [])
    if not isinstance(files, list) or not files or len({f['filename'] for f in files}) != len(files):
        raise ValueError('missing or duplicate manifest selected files')
    manifest_weights = {f['filename']: f.get('sha256') for f in files if f['filename'].endswith('.safetensors')}
    if not expected or expected != manifest_weights:
        raise ValueError('weight hashes must match supplied immutable parent manifest')
    path = contained_path(inputs['root'], pin['path'], parent)
    if path.is_symlink() or not path.is_dir():
        raise ValueError('missing or unsafe parent directory')
    for file in path.rglob('*'):
        contained_path(path, file.relative_to(path).as_posix(), parent)
        if file.is_symlink() or file.suffix in {'.py','.bin','.pt','.pth','.pkl','.pickle'}:
            raise ValueError('unsafe parent file')
    config_path = path/'config.json'
    config = json.loads(config_path.read_text())
    if content_hash(config) != pin.get('config_sha256') or _sha(config_path) != pin.get('config_file_sha256'):
        raise ValueError('pinned configuration hash mismatch')
    if config.get('quantization_config') or config.get('auto_map'):
        raise ValueError('packed quantization or remote-code configuration is unsupported')
    actual = {f.relative_to(path).as_posix() for f in path.rglob('*.safetensors')}
    if set(expected) != actual:
        raise ValueError('selected weight file set mismatch')
    for filename, digest in expected.items():
        file = contained_path(path, filename, parent)
        if '/' in filename or not re.fullmatch('[0-9a-f]{64}', str(digest)) or _sha(file) != digest:
            raise ValueError('pinned weight hash mismatch')
    index = path/'model.safetensors.index.json'
    mapping = None
    if index.exists():
        mapping = json.loads(index.read_text()).get('weight_map', {})
        if not mapping or set(mapping.values()) != actual or 'model.safetensors' in actual:
            raise ValueError('ambiguous or incorrect shard index')
    elif actual != {'model.safetensors'}:
        raise ValueError('sharded checkpoint requires an index')
    return path, config, expected, mapping


def run_quantization(recipe: dict, inputs: dict, output: Path) -> None:
    """Write only to the caller's atomic checkpoint staging directory.

    Iterate one source shard and one tensor at a time. Memory includes the current
    output shard; it never includes the complete multi-shard checkpoint.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForCausalLM
    path, config, expected, index = checked_source(recipe, inputs)
    location, shapes = {}, {}
    for filename in sorted(expected):
        with safe_open(str(path/filename), framework='pt', device='cpu') as stream:
            for name in stream.keys():
                if name in location:
                    raise ValueError('duplicate tensor across shards: '+name)
                location[name] = filename
                shapes[name] = tuple(stream.get_slice(name).get_shape())
    if not location or (index is not None and index != location):
        raise ValueError('empty weights or shard index tensor map mismatch')
    # A meta-device skeleton finds architecture-specific ties without loading weights.
    model_config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    with torch.device('meta'):
        skeleton = AutoModelForCausalLM.from_config(model_config, trust_remote_code=False)
        skeleton.tie_weights()
    groups, parameter_names = {}, set()
    for name, parameter in skeleton.named_parameters(remove_duplicate=False):
        parameter_names.add(name)
        groups.setdefault(id(parameter), []).append(name)
        if name in shapes and shapes[name] != tuple(parameter.shape):
            raise ValueError('checkpoint shape mismatch: '+name)
    tied = [names for names in groups.values() if len(names) > 1]
    for names in groups.values():
        if not any(name in location for name in names):
            raise ValueError('missing effective parameter: '+names[0])
    def tensor(name):
        with safe_open(str(path/location[name]), framework='pt', device='cpu') as stream:
            return stream.get_tensor(name)
    for names in tied:
        present = [name for name in names if name in location]
        if not present:
            raise ValueError('missing tied effective weights')
        first = tensor(present[0])
        for name in present[1:]:
            if not torch.equal(first, tensor(name)):
                raise ValueError('conflicting tied effective weights: '+name)
    del skeleton
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    numerator, denominator, matrices = 0., 0., 0
    preserved_masks = {}
    total_size = 0
    for filename in sorted(expected):
        result = {}
        with safe_open(str(path/filename), framework='pt', device='cpu') as stream:
            for name in sorted(stream.keys()):
                source = stream.get_tensor(name)
                legacy_mask = re.fullmatch(r'gpt_neox\.layers\.(0|[1-9][0-9]*)\.attention\.masked_bias', name)
                if (config.get('model_type') == 'gpt_neox' and legacy_mask
                        and int(legacy_mask.group(1)) < config['num_hidden_layers']
                        and name not in parameter_names and source.is_floating_point()
                        and source.ndim == 0 and bool(torch.isneginf(source))):
                    # Old GPTNeoX checkpoints serialize this structural causal
                    # mask sentinel. It is not a weight or part of the norm.
                    transformed = source.contiguous()
                    preserved_masks[name] = dict(dtype=str(source.dtype).removeprefix('torch.'), value='-inf')
                elif source.is_floating_point():
                    if not torch.isfinite(source).all():
                        raise ValueError('nonfinite source tensor: '+name)
                    transformed = simulate_quantization(source, recipe['bits']) if source.ndim == 2 else source
                    matrices += int(source.ndim == 2)
                    transformed = transformed.to(torch.bfloat16).contiguous()
                    # Measure saved-value change, including the requested bf16 cast.
                    denominator += source.double().square().sum().item()
                    numerator += (transformed.double()-source.double()).square().sum().item()
                else:
                    transformed = source.contiguous()
                result[name] = transformed
                total_size += transformed.numel()*transformed.element_size()
        save_file(result, str(output/filename), metadata={'format':'pt'})
        del result
    config['_name_or_path'] = ''
    config['dtype'] = 'bfloat16'
    config['torch_dtype'] = 'bfloat16'
    from .artifacts import _portable
    _portable(config)
    atomic_json(output/'config.json', config)
    if index is not None:
        atomic_json(output/'model.safetensors.index.json', {'metadata':{'total_size':total_size}, 'weight_map':index})
    # Copy only safe tokenizer data. No source code or executable formats.
    for filename in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt','tokenizer.model']:
        source = path/filename
        if source.is_file():
            if source.suffix == '.json':
                value = json.loads(source.read_text())
                value.pop('name_or_path', None)
                value.pop('auto_map', None)
                # Vocabulary and added-token strings (e.g. "/") are data,
                # not provenance paths. Model config remains checked above;
                # executable tokenizer metadata is still removed here.
                atomic_json(output/filename, value)
            else:
                shutil.copyfile(source, output/filename)
    atomic_json(output/'quantization.json', dict(quantizer='symmetric_signed_per_row_round_ties_even',
        bits=recipe['bits'], scaling_axis=1, qmax=2**(recipe['bits']-1)-1,
        format='safetensors', output_dtype='bfloat16', input_hashes=expected,
        matrices=matrices, tied_alias_groups=tied, preserved_legacy_attention_masks=preserved_masks,
        relative_weight_change=math.sqrt(numerator/denominator) if denominator else 0.,
        metric=('relative Frobenius change of finite saved floating values including bf16 cast; legacy mask sentinels excluded'
                if preserved_masks else 'relative Frobenius change of all saved floating values including bf16 cast')))
