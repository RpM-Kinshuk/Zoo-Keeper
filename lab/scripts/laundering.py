"""Sixteen public transforms; real batch is a HUMAN launch, offline tiny is local."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

from lab.artifacts import content_hash, write_checkpoint
from lab.registry import expand_inventory
from lab.schema import LAB_SCHEMA_VERSION, LOCAL_KEY, canonical_json, contained_path
from lab.transforms import ADAPTERS, METHODS, _transform, check_equivalence


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_source(inputs, root, parent, identity):
    """Validate pinned local files and an extractor local key before loading tensors."""
    import torch
    from transformers import AutoModelForCausalLM
    from lab.pins import verify_pin
    root = Path(root)
    source = contained_path(root, inputs['model_path'], parent)
    if source.is_symlink() or not source.is_dir():
        raise ValueError('model source must be a local directory')
    for path in source.rglob('*'):
        contained_path(source, path.relative_to(source).as_posix(), parent)
        if path.is_symlink():
            raise ValueError('source symlinks are unsupported')
    pin = inputs['model_pin']
    verify_pin(pin)
    local = inputs.get('local_parent')
    if local is not None:
        match = LOCAL_KEY.fullmatch(identity)
        if not match or match.group(1) != parent or local.get('model_key') != identity:
            raise ValueError('local parent manifest identity mismatch')
        manifest_path = contained_path(root, inputs['local_manifest_path'], parent)
        if _file_hash(manifest_path) != inputs.get('local_manifest_sha256'):
            raise ValueError('local manifest sidecar hash mismatch')
        manifest = json.loads(manifest_path.read_text())
        if manifest != local:
            raise ValueError('local manifest sidecar content mismatch')
        # The extractor's key is authoritative; never invent another identity algorithm.
        evidence = local
    else:
        if identity != pin['repo_id'] + '@' + pin['revision']:
            raise ValueError('pinned parent identity mismatch')
        evidence = pin
    config_path = source / 'config.json'
    config = json.loads(config_path.read_text())
    if config.get('model_type') not in ADAPTERS:
        raise NotImplementedError(str(config.get('model_type')) + ': unsupported model adapter')
    if pin.get('repo_id') == 'Qwen/Qwen2.5-3B':
        raise ValueError('Qwen2.5-3B is excluded from the lab')
    if content_hash(config) != evidence.get('config_sha256'):
        raise ValueError('source config hash mismatch')
    if evidence.get('config_file_sha256') and _file_hash(config_path) != evidence['config_file_sha256']:
        raise ValueError('source config file hash mismatch')
    expected = evidence.get('weight_sha256')
    if not isinstance(expected, dict) or not expected:
        raise ValueError('source weight hash mapping required')
    actual = {p.relative_to(source).as_posix() for p in source.rglob('*.safetensors') if p.is_file()}
    if actual != set(expected):
        raise ValueError('source weight file set differs from hash sidecar')
    for name, digest in expected.items():
        path = contained_path(source, name, parent)
        if path.suffix != '.safetensors' or not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('source requires safetensors and valid SHA256 hashes')
        if _file_hash(path) != digest:
            raise ValueError(name + ': source weight hash mismatch')
    index_path = source / 'model.safetensors.index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text())
        mapping = index.get('weight_map')
        if not isinstance(mapping, dict) or not mapping or set(mapping.values()) != actual or 'model.safetensors' in actual:
            raise ValueError('source sharding index differs from hash sidecar')
    elif actual != {'model.safetensors'}:
        raise ValueError('source sharded weights require an index')
    # Config/model dispatch remains local and never executes repository code.
    model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True,
                trust_remote_code=False, use_safetensors=True, dtype=torch.float32).eval()
    return model


def _validation_tokens(inputs, root):
    import torch
    from safetensors.torch import load_file
    path = contained_path(Path(root), inputs['validation_tokens_path'], 'validation tokens')
    if path.suffix != '.safetensors' or path.is_symlink() or _file_hash(path) != inputs['validation_tokens_sha256']:
        raise ValueError('frozen validation token hash mismatch')
    sample_ids = inputs['validation_sample_ids']
    if not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError('frozen validation sample IDs must be unique and nonempty')
    for split in ['training_sample_ids', 'direction_sample_ids', 'scoring_sample_ids']:
        if split not in inputs or set(sample_ids) & set(inputs[split]):
            raise ValueError('missing or overlapping frozen validation split: ' + split)
    if content_hash(sample_ids) != inputs['validation_sample_ids_sha256']:
        raise ValueError('frozen validation sample hash mismatch')
    tokens = load_file(str(path))['input_ids']
    if tokens.dtype != torch.long or tokens.ndim != 2 or tokens.shape[0] != len(sample_ids):
        raise ValueError('frozen validation tokens differ from selected sample IDs')
    return [tokens[i:i + 2] for i in range(0, len(tokens), 2)]


def _audit(model, recipe, batches):
    import torch
    model = model.float().eval()
    transformed = copy.deepcopy(model)
    fp_state, audit = _transform(model.state_dict(), model.config.to_dict(), recipe)
    transformed.load_state_dict(fp_state, strict=True)
    del fp_state
    audit['fp32'] = check_equivalence(model, transformed, batches)
    # Cast the actual fp32 edit for deployment, and record what survived casting.
    baseline_bf = copy.deepcopy(model).to(torch.bfloat16)
    transformed.to(torch.bfloat16)
    audit['bfloat16'] = check_equivalence(baseline_bf, transformed, batches)
    if recipe['method'] == 'targeted_perturbation':
        baseline = baseline_bf.state_dict()
        result = transformed.state_dict()
        for name, norms in audit['perturbations'].items():
            weight = baseline[name].double()
            norms['postcast_relative_norm'] = ((result[name].double() - weight).norm() / weight.norm()).item()
        audit['preservation_required'] = False
    elif not audit['fp32']['passed'] or not audit['bfloat16']['passed']:
        raise ValueError('function-preserving transform failed frozen-token equivalence checks')
    return transformed, audit


def _save(root, recipe, model, resume):
    def writer(output):
        model.save_pretrained(output, safe_serialization=True)
        # Verify the serialized checkpoint and ties through the supported local loader.
        from transformers import AutoModelForCausalLM
        reloaded = AutoModelForCausalLM.from_pretrained(output, local_files_only=True,
                     use_safetensors=True, trust_remote_code=False)
        if model.config.tie_word_embeddings:
            if reloaded.get_input_embeddings().weight.data_ptr() != reloaded.get_output_embeddings().weight.data_ptr():
                raise ValueError('serialized tied embeddings did not reload tied')
    return write_checkpoint(recipe['model_id'], root, recipe, writer, resume=resume)


def _tiny(config, root, resume):
    import torch
    from transformers import Qwen2Config, AutoModelForCausalLM
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261028)
        cfg = Qwen2Config(hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
                max_position_embeddings=32, tie_word_embeddings=True,
                bos_token_id=1, eos_token_id=2, pad_token_id=0)
        model = AutoModelForCausalLM.from_config(cfg).eval()
    batches = [torch.tensor([[1, 5, 9, 2], [3, 8, 2, 7]], dtype=torch.long)]
    fixture = dict(type='local-qwen2-gqa-v1', seed=20261028, config=json.loads(cfg.to_json_string()))
    ids = []
    for index, method in enumerate(METHODS):
        recipe = dict(model_id='tiny-' + method, method=method, seed=20261028 + index,
                edge_type='laundering', parents=['tiny-base'],
                parent_identities={'tiny-base': 'lab/tiny-base@local:' + content_hash(fixture)},
                inputs={'fixture': fixture}, requested_config=config)
        edited, audit = _audit(model, recipe, batches)
        recipe['audit'] = audit
        _save(root, recipe, edited, resume)
        ids.append(recipe['model_id'])
    return dict(status='COMPLETE_SYNTHETIC', public_batch_launched=False, model_ids=ids)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--lab-root', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--tiny', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if config.get('schema_version') != LAB_SCHEMA_VERSION:
        raise ValueError('unsupported laundering schema')
    if config.get('relative_frobenius_norm', 1e-3) != 1e-3 or config.get('target_matrices', 4) != 4:
        raise ValueError('laundering requires four projections with relative norm 1e-3')
    if config.get('torch_dtype', 'bfloat16') != 'bfloat16' or config.get('device', 'cpu') != 'cpu':
        raise ValueError('this audited runner requires CPU validation and bf16 output')
    recipes = expand_inventory({})['laundering']
    if args.dry_run:
        print(canonical_json(dict(recipes=recipes, launch='HUMAN', config_sha256=content_hash(config),
                  inputs_state=config.get('inputs_state'), estimate=config.get('estimate'))))
        return 0
    if args.tiny:
        print(canonical_json(_tiny(config, args.lab_root, args.resume)))
        return 0
    prepared = config.get('prepared_inputs', {})
    if config.get('inputs_state') != 'frozen' or any(r['model_id'] not in prepared for r in recipes):
        raise ValueError('real batch requires verified frozen source hash sidecars and validation tokens (HUMAN launch)')
    # Validate all prepared metadata before any checkpoint publication.
    for row in recipes:
        data = prepared[row['model_id']]
        if data.get('parent_identity') is None:
            raise ValueError(row['model_id'] + ': missing pinned/local parent identity')
    for row in recipes:
        data = prepared[row['model_id']]
        recipe = dict(row, inputs=data, requested_config=config,
                      parent_identities={row['parents'][0]: data['parent_identity']})
        model = load_source(data, args.lab_root, row['parents'][0], data['parent_identity'])
        batches = _validation_tokens(data, args.lab_root)
        edited, audit = _audit(model, recipe, batches)
        recipe['audit'] = audit
        _save(args.lab_root, recipe, edited, args.resume)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
