"""Portable mergekit recipes and checked local safetensors execution."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import yaml

from .artifacts import _lineage, _portable, atomic_json, content_hash
from .schema import HUB_PATH, LOCAL_KEY, contained_path, validate_id


METHODS = {'linear', 'slerp', 'ties', 'dare_ties'}


def _yaml(recipe):
    models = []
    for index, parent in enumerate(recipe['parents']):
        entry = {'model': parent}
        if 'configured_weights' in recipe:
            entry['parameters'] = {'weight': recipe['configured_weights'][index]}
            if 'density' in recipe:
                entry['parameters']['density'] = recipe['density']
        models.append(entry)
    config = {'merge_method': recipe['method'], 'models': models, 'dtype': 'float32'}
    # mergekit calls SLERP's first interpolation endpoint its base. The lab's
    # common reference remains a separate provenance role, never a third input.
    if recipe['method'] == 'slerp':
        config['base_model'] = recipe['parents'][0]
    elif recipe['method'] in {'ties', 'dare_ties'}:
        config['base_model'] = recipe['reference']
    if recipe['method'] == 'slerp':
        config['parameters'] = {'t': recipe['t']}
    return yaml.safe_dump(config, sort_keys=True)


def prepare_recipe(recipe):
    recipe = dict(recipe)
    _portable(recipe)
    validate_id(recipe['model_id'])
    parents = recipe['parents']
    if recipe.get('edge_type') != 'merge' or recipe.get('method') not in METHODS:
        raise ValueError('unsupported merge recipe')
    if not 2 <= len(parents) <= 4 or len(set(parents)) != len(parents):
        raise ValueError('merge requires two to four distinct parents')
    for name in [*parents, recipe['reference']]:
        validate_id(name)
    if recipe['reference'] in parents:
        raise ValueError('common reference must have its own base role')
    if recipe['method'] == 'slerp':
        if len(parents) != 2:
            raise ValueError('SLERP requires two parents')
        values = recipe['t'] if isinstance(recipe['t'], list) else [recipe['t']]
        if not values or any(not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in values):
            raise ValueError('invalid interpolation schedule')
    else:
        weights = recipe.get('configured_weights', [])
        if len(weights) != len(parents) or any(not isinstance(w, (int, float)) or not 0 <= w <= 1 for w in weights) or abs(sum(weights)-1) > 1e-8:
            raise ValueError('invalid configured weights')
    if recipe['method'] in {'ties','dare_ties'}:
        if not 0 < recipe.get('density', 0) <= 1:
            raise ValueError('invalid sparsification density')
        recipe['coefficient_semantics'] = 'configured_task_weight'
    elif recipe['method'] == 'slerp':
        recipe['coefficient_semantics'] = 'nonlinear_interpolation_t_in_params'
    else:
        recipe['coefficient_semantics'] = 'linear_weight'
    recipe['mergekit_version'] = importlib.metadata.version('mergekit')
    source = _yaml(recipe)
    if recipe.get('mergekit_yaml', source) != source:
        raise ValueError('mergekit YAML differs from recipe')
    recipe['mergekit_yaml'] = source
    yaml_hash = hashlib.sha256(source.encode()).hexdigest()
    if recipe.get('mergekit_yaml_sha256', yaml_hash) != yaml_hash:
        raise ValueError('mergekit YAML hash mismatch')
    recipe['mergekit_yaml_sha256'] = yaml_hash
    return recipe


def mergekit_configs(inventory: dict, pins: dict) -> dict[str, str]:
    """IDs in these exact portable YAML bytes are resolved only at invocation."""
    return {row['model_id']: prepare_recipe(row)['mergekit_yaml'] for row in inventory['merge']}


def _shape_map(path):
    from safetensors import safe_open
    shapes = {}
    for file in sorted(path.glob('*.safetensors')):
        with safe_open(file, framework='pt', device='cpu') as stream:
            for key in stream.keys():
                if key in shapes:
                    raise ValueError('duplicate tensor across shards: ' + key)
                shapes[key] = tuple(stream.get_slice(key).get_shape())
    return shapes


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _architecture_config(config):
    # Local config constructors resolve default fields and legacy RoPE aliases
    # exactly as the installed model loader does, without fetching repo code.
    from transformers import AutoConfig
    ignored = {'_name_or_path', 'torch_dtype', 'dtype', 'architectures', 'transformers_version'}
    fields = {k: v for k, v in config.items() if k not in ignored}
    model_type = fields.pop('model_type')
    normalized = AutoConfig.for_model(model_type, **fields).to_dict()
    return {k: v for k, v in normalized.items() if k not in ignored}


def _checked_inputs(recipe, root):
    _lineage(recipe['model_id'], recipe)
    names = recipe['parents'] + [recipe['reference']]
    if set(recipe.get('inputs', {})) != set(names):
        raise ValueError('missing frozen parent inputs')
    paths, shapes, configs = {}, [], []
    synthetic = recipe.get('synthetic_fixture', False)
    for name in names:
        pin = recipe['inputs'][name]
        identity = recipe['parent_identities'][name]
        if pin.get('model_key') != identity:
            raise ValueError(name + ': pinned parent identity mismatch')
        match = LOCAL_KEY.fullmatch(identity)
        if match and match.group(1) != name:
            raise ValueError(name + ': local identity mismatch')
        path = contained_path(root, pin['path'], name)
        if not path.is_dir():
            raise ValueError(name + ': missing parent checkpoint')
        for file in path.rglob('*'):
            contained_path(path, file.relative_to(path).as_posix(), name)
            if file.is_symlink() or file.suffix in {'.bin','.pt','.pth','.pkl','.pickle','.py'}:
                raise ValueError(name + ': unsafe parent file')
        config = json.loads((path/'config.json').read_text())
        if content_hash(config) != pin.get('config_sha256') or _sha(path/'config.json') != pin.get('config_file_sha256'):
            raise ValueError(name + ': pinned config hash mismatch')
        training = {}
        if not synthetic and name in recipe['parents']:
            training = json.loads((path/'training.json').read_text())
        expected = pin.get('weight_sha256', {})
        actual = {f.name for f in path.glob('*.safetensors')}
        secondary = {f.relative_to(path).as_posix() for f in path.rglob('*.safetensors')} - actual
        if secondary != set(training.get('secondary_weights', [])):
            raise ValueError(name + ': undeclared secondary safetensors files')
        if not expected or set(expected) != actual:
            raise ValueError(name + ': pinned weight file set mismatch')
        for filename, digest in expected.items():
            if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest) or _sha(contained_path(path, filename, name)) != digest:
                raise ValueError(name + ': pinned weight hash mismatch')
        index = path/'model.safetensors.index.json'
        if index.exists():
            mapping = json.loads(index.read_text()).get('weight_map', {})
            if not mapping or set(mapping.values()) != actual:
                raise ValueError(name + ': shard index file set mismatch')
        shape = _shape_map(path)
        if not shape:
            raise ValueError(name + ': empty checkpoint')
        if synthetic:
            matrix_fixture = (config == {'model_type':'llama','num_hidden_layers':2,'hidden_size':4}
                              and len(shape) == 2 and all(s == (4,4) for s in shape.values()))
            model_fixture = (config.get('model_type') == 'llama' and config.get('num_hidden_layers') == 2
                             and config.get('hidden_size') == 8 and config.get('vocab_size') == 16
                             and len(shape) <= 32 and all(all(d <= 32 for d in dims) for dims in shape.values()))
            if not (matrix_fixture or model_fixture):
                raise ValueError('synthetic execution is restricted to the small two-layer fixture')
        elif name in recipe['parents']:
            if not match:
                raise ValueError(name + ': trained LoRA parent requires extractor local identity')
            provenance = json.loads((path/'recipe.json').read_text())
            source = provenance['recipe']
            lineage = json.loads((path/'lineage.json').read_text())
            if (provenance.get('model_id') != name or source.get('edge_type') != 'lora'
                    or source.get('parents') != [recipe['reference']]
                    or source.get('anchor') != recipe['reference']
                    or training.get('steps_completed', 0) < 1
                    or training.get('steps_completed') != source.get('steps')
                    or provenance.get('input_hash') != content_hash({'parents':source['parent_identities'], 'inputs':source.get('inputs',{})})
                    or provenance.get('config_hash') != content_hash(source)
                    or lineage != _lineage(name, source)
                    or source['parent_identities'][recipe['reference']] != recipe['parent_identities'][recipe['reference']]):
                raise ValueError(name + ': requires completed same-base LoRA checkpoint')
        elif not HUB_PATH.fullmatch(identity):
            raise ValueError(name + ': real reference must have a pinned Hub identity')
        else:
            from .pins import verify_pin
            verify_pin(pin)
            if identity != pin['repo_id'] + '@' + pin['revision']:
                raise ValueError(name + ': reference identity differs from verified pin')
        paths[name] = path
        shapes.append(shape)
        configs.append({k:v for k,v in config.items() if k not in {'_name_or_path','torch_dtype','dtype','architectures','transformers_version'}})
    if any(shape != shapes[0] for shape in shapes[1:]):
        raise ValueError('incompatible parent tensor names or shapes')
    if any(config != configs[0] for config in configs[1:]):
        # Identical dictionaries already satisfy the existing contract, including
        # the deliberately minimal matrix fixtures. Normalize only export drift.
        configs = [_architecture_config(config) for config in configs]
        if any(config != configs[0] for config in configs[1:]):
            raise ValueError('incompatible parent architecture configs')
    return paths


def merge_states(recipe, parents, base):
    """Execute installed mergekit kernels on the small offline fixture only."""
    import torch
    from mergekit.merge_methods import get
    from mergekit.merge_methods.base import TensorGroup
    states = parents + [base]
    if any(set(state) != set(base) for state in states):
        raise ValueError('incompatible tensor names')
    if any(state[k].shape != base[k].shape for state in states for k in base):
        raise ValueError('incompatible tensor shapes')
    result = {}
    keys = sorted(base)
    layer_ids = sorted({int(m.group(1)) for k in keys if (m := re.search(r'\.layers\.(\d+)\.', k))})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(recipe.get('sparsification_seed', recipe['seed']))
        for key in keys:
            method = get(recipe['method'])
            ids = recipe['parents']
            tensors = [state[key] for state in parents]
            parameters = {}
            base_index = 0 if recipe['method'] == 'slerp' else None
            if recipe['method'] in {'ties','dare_ties'}:
                base_index = len(tensors)
                tensors.append(base[key])
                ids = ids + ['__reference__']
                parameters['density'] = recipe['density']
            if recipe['method'] == 'slerp':
                t = recipe['t']
                if isinstance(t, list):
                    m = re.search(r'\.layers\.(\d+)\.', key)
                    position = layer_ids.index(int(m.group(1))) / max(1, len(layer_ids)-1) if m else 0
                    index = position*(len(t)-1)
                    lo = int(index)
                    t = t[lo] + (t[min(lo+1,len(t)-1)]-t[lo])*(index-lo)
                parameters['t'] = t
            else:
                parameters['weight'] = recipe['configured_weights']
            group = TensorGroup.from_tensors(tensors, ids=ids, base_index=base_index, name=key)
            result[key] = method([group], parameters=parameters)[0].contiguous()
    return result


def run_merge(recipe: dict, lab_root: Path, output: Path) -> None:
    recipe = prepare_recipe(recipe)
    paths = _checked_inputs(recipe, Path(lab_root))
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('merge output must be an empty staging directory')
    with tempfile.TemporaryDirectory(prefix='lab-merge-') as temporary:
        temporary = Path(temporary)
        produced = temporary/'result'
        produced.mkdir()
        if recipe.get('synthetic_fixture') and recipe.get('execution_backend') != 'mergekit_yaml':
            from safetensors.torch import load_file, save_file
            def state(name):
                return {k:v for f in sorted(paths[name].rglob('*.safetensors')) for k,v in load_file(str(f)).items()}
            merged = merge_states(recipe, [state(p) for p in recipe['parents']], state(recipe['reference']))
            save_file(merged, str(produced/'model.safetensors'))
            shutil.copyfile(paths[recipe['reference']]/'config.json', produced/'config.json')
        else:
            config = yaml.safe_load(recipe['mergekit_yaml'])
            if 'base_model' in config:
                config['base_model'] = str(paths[config['base_model']].resolve())
            for entry, parent in zip(config['models'], recipe['parents']):
                entry['model'] = str(paths[parent].resolve())
            invocation = temporary/'invocation.yaml'
            invocation.write_text(yaml.safe_dump(config, sort_keys=True))
            subprocess.run([sys.executable, '-m', 'mergekit.scripts.run_yaml', str(invocation), str(produced),
                            '--random-seed', str(recipe.get('sparsification_seed', recipe['seed'])),
                            '--safe-serialization', '--no-trust-remote-code', '--no-lazy-unpickle',
                            '--no-write-model-card', '--num-threads', '1',
                            *(['--no-copy-tokenizer'] if recipe.get('synthetic_fixture') else [])],
                           check=True, env=_offline_env(), capture_output=True, text=True)
        if not list(produced.glob('*.safetensors')):
            raise ValueError('mergekit produced no safe checkpoint')
        output.mkdir(parents=True, exist_ok=True)
        # mergekit's generated YAML/card may contain runtime paths. Keep only safe
        # loadable checkpoint files; preserve the portable source in provenance.
        for file in produced.iterdir():
            if file.is_symlink() or file.suffix in {'.bin','.pt','.pth','.pickle','.pkl','.py'}:
                raise ValueError('unsafe mergekit output')
            if file.is_file() and file.suffix == '.json':
                value = json.loads(file.read_text())
                if isinstance(value, dict):
                    for key in ('_name_or_path', 'name_or_path'):
                        if key in value:
                            value[key] = ''
                checked = value
                if (file.name == 'tokenizer.json' and isinstance(value, dict)
                        and isinstance(value.get('model'), dict) and 'merges' in value['model']):
                    rules = value['model']['merges']
                    valid = isinstance(rules, list) and all(
                        isinstance(rule, str) or (isinstance(rule, list) and len(rule) == 2
                            and all(isinstance(token, str) for token in rule)) for rule in rules)
                    if not valid:
                        raise ValueError('tokenizer BPE merges must be token strings or pairs')
                    # BPE rules contain literal slash/backslash tokens. Validate
                    # all metadata, retaining these lexical rules unchanged.
                    checked = {**value, 'model': {k:v for k,v in value['model'].items() if k != 'merges'}}
                _portable(checked)
                atomic_json(output/file.name, value)
            elif file.is_file() and file.suffix in {'.safetensors','.model','.txt'}:
                shutil.copyfile(file, output/file.name)
        atomic_json(output/'merge.json', {'mergekit_version':recipe['mergekit_version'],
            'config_sha256':content_hash(recipe), 'yaml_sha256':recipe['mergekit_yaml_sha256'],
            'source_yaml':recipe['mergekit_yaml'],
            'weight_selection':'top-level effective safetensors; declared nested adapter factors excluded', 'seed':recipe.get('sparsification_seed',recipe['seed'])})


def _offline_env():
    import os
    names = ('PATH', 'HOME', 'TMPDIR', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'CUDA_VISIBLE_DEVICES')
    return {**{name:os.environ[name] for name in names if name in os.environ},
            'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1'}
