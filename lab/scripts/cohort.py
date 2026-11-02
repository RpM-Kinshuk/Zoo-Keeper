"""Bounded real-weight cohort using reviewed quantization and transform writers.

Run preparation with network permission; generation itself is strictly offline.
All runtime paths are relative to ZOO_DATA. No training or quality claim is made.
"""
import argparse
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time

from lab.artifacts import atomic_json, content_hash, write_checkpoint
from lab.pins import verify_pin
from lab.quantization import checked_source, run_quantization

RUN = 'runs/track-b-real-cohort-v1'
PROBES = [
    'The small red boat crossed the quiet lake before sunset.',
    'A careful reader compares evidence before reaching a conclusion.',
    'Count the four stones beside the open garden gate.',
    'Water freezes when the temperature falls below its freezing point.',
]
TOKEN_FILES = ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']
SAFE_FILES = TOKEN_FILES + ['config.json']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def state(root):
    path = root / RUN / 'ready.json'
    return json.loads(path.read_text()) if path.exists() else dict(schema_version='1', sources=[], outputs=[])


def publish_ready(root, kind, row):
    value = state(root)
    existing = [r for r in value[kind] if r['model_id'] == row['model_id']]
    if existing and existing != [row]:
        raise ValueError('ready entry differs')
    if not existing:
        value[kind].append(row)
    atomic_json(root / RUN / 'ready.json', value)


def verify_source(root, pin, rel):
    path = root / 'lab' / rel
    if sha(path / 'config.json') != pin['config_file_sha256']:
        raise ValueError('configuration bytes differ')
    if content_hash(json.loads((path / 'config.json').read_text())) != pin['config_sha256']:
        raise ValueError('configuration content differs')
    tokens = {n: sha(path / n) for n in TOKEN_FILES if (path / n).exists()}
    if content_hash(tokens) != pin['tokenizer_sha256']:
        raise ValueError('tokenizer hash differs')
    evidence = Path(pin['license_path'])
    if not evidence.exists():
        evidence = root / pin['license_path']
    if sha(evidence) != pin['license_sha256']:
        raise ValueError('license evidence differs')
    expected = pin['weight_sha256']
    if not expected or set(expected) != {p.name for p in path.glob('*.safetensors')}:
        raise ValueError('weight set differs')
    for name, digest in expected.items():
        if sha(path / name) != digest:
            raise ValueError('weight bytes differ')
    marker = dict(model_key=pin['repo_id'] + '@' + pin['revision'],
                  weight_sha256=expected, config_file_sha256=pin['config_file_sha256'],
                  tokenizer_sha256=pin['tokenizer_sha256'], verified=True)
    atomic_json(path / 'track-b-source.json', marker)
    return marker


def prepare(root, labels, pins_override=None):
    from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache
    pins = json.loads(Path('lab/pins.json').read_text())
    if pins_override:
        pins.update(json.loads(Path(pins_override).read_text()))
    api = HfApi(token=os.environ.get('HF_TOKEN'))
    failures = []
    for label in labels:
        pin = dict(pins[label])
        verify_pin(pin)
        try:
            # Existing pins with full SHA256 need no remote metadata lookup.
            if not pin['weight_sha256']:
                info = api.model_info(pin['repo_id'], revision=pin['revision'], files_metadata=True)
                if info.sha != pin['revision']:
                    raise ValueError('metadata revision differs')
                pin['weight_sha256'] = {s.rfilename: s.lfs.sha256 for s in info.siblings
                    if s.rfilename.endswith('.safetensors') and '/' not in s.rfilename and s.lfs}
                atomic_json(root / RUN / (label + '-hub-metadata.json'), dict(
                    repo_id=pin['repo_id'], revision=info.sha,
                    safetensors=pin['weight_sha256'],
                    filenames=[s.rfilename for s in info.siblings]))
                if not pin['weight_sha256']:
                    raise ValueError('no safetensors at exact pinned revision')
            rel = 'inputs/cohort-v1-' + label + '-' + pin['revision'][:12]
            dest = root / 'lab' / rel
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                stage = Path(tempfile.mkdtemp(prefix='.cohort-source-', dir=dest.parent))
                try:
                    for name in [*pin['weight_sha256'], *SAFE_FILES]:
                        cached = try_to_load_from_cache(pin['repo_id'], name, revision=pin['revision'])
                        if isinstance(cached, str):
                            shutil.copyfile(cached, stage / name)
                            continue
                        try:
                            source = hf_hub_download(pin['repo_id'], name, revision=pin['revision'], token=True,
                                                     local_dir=stage)
                        except Exception as error:
                            if getattr(getattr(error, 'response', None), 'status_code', None) == 404 and name not in pin['weight_sha256'] and name != 'config.json':
                                continue
                            raise RuntimeError('allowlisted download failed') from None
                        if Path(source) != stage / name:
                            shutil.copyfile(source, stage / name)
                    shutil.rmtree(stage / '.cache', ignore_errors=True)
                    os.rename(stage, dest)
                finally:
                    if stage.exists():
                        shutil.rmtree(stage)
            marker = verify_source(root, pin, rel)
            atomic_json(root / RUN / (label + '-verified-pin.json'), pin)
            seed = pin.get('seed')
            row = dict(model_id=label, path='lab/' + rel, root_id=label, kind='source', parents=[],
                repo_id=pin['repo_id'], revision=pin['revision'], license=pin['license'],
                edge_type=None, synthetic=False, complete=True, weight_sha256=pin['weight_sha256'],
                split=('development' if seed in [1, 2] else 'heldout_test') if seed else 'descriptive_smol',
                marker='lab/' + rel + '/track-b-source.json')
            publish_ready(root, 'sources', row)
            print(json.dumps(dict(phase='source_verified', **row)), flush=True)
        except Exception as error:
            failures.append(dict(model_id=label, error_type=type(error).__name__, status='BLOCKED_SAFE_INPUT'))
            print(json.dumps(failures[-1]), flush=True)
    atomic_json(root / RUN / 'source-preparation.json', dict(failures=failures))


def freeze(root):
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    recipes = []
    for row in state(root)['sources']:
        label = row['model_id']
        pin_path = root / RUN / (label + '-verified-pin.json')
        if not pin_path.exists():
            continue  # separately frozen original OLMo runner owns that source
        pin = json.loads(pin_path.read_text())
        rel = Path(row['path']).relative_to('lab').as_posix()
        identity = pin['repo_id'] + '@' + pin['revision']
        prepared = dict(path=rel, model_key=identity, config_sha256=pin['config_sha256'],
            config_file_sha256=pin['config_file_sha256'], weight_sha256=pin['weight_sha256'],
            manifest=dict(repo_id=pin['repo_id'], revision=pin['revision'],
                files=[dict(filename=n, sha256=h) for n, h in sorted(pin['weight_sha256'].items())]))
        stem = 'cohort-v1-pythia-14m-seed' + str(pin['seed']) if 'seed' in pin else 'cohort-v1-smol-135'
        for bits in ([8, 4] if 'seed' in pin else [8, 4, 2]):
            recipe = dict(model_id=stem + '-q' + str(bits), edge_type='quantization_sim', bits=bits,
                parents=[label], parent_identities={label: identity}, inputs={label: prepared},
                cohort='track-b-real-cohort-v1', synthetic=False)
            checked_source(recipe, {'root': root / 'lab'})
            recipes.append(recipe)
        if label != 'smol-135':
            continue
        tokenizer = AutoTokenizer.from_pretrained(root / row['path'], local_files_only=True, trust_remote_code=False)
        token_rows = [tokenizer(text, add_special_tokens=False)['input_ids'][:16] for text in PROBES]
        width = min(map(len, token_rows))
        tokens = torch.tensor([r[:width] for r in token_rows], dtype=torch.long)
        probe_rel = RUN + '/validation-probes.safetensors'
        if not (root / probe_rel).exists():
            save_file({'input_ids': tokens}, str(root / probe_rel))
        sample_ids = ['cohort-v1-authored-validation-' + str(i) for i in range(len(PROBES))]
        data = dict(model_path=rel, model_pin=pin, parent_identity=identity,
            validation_tokens_path='inputs/cohort-v1-validation-probes.safetensors',
            validation_tokens_sha256=sha(root / probe_rel), validation_sample_ids=sample_ids,
            validation_sample_ids_sha256=content_hash(sample_ids), training_sample_ids=[],
            direction_sample_ids=[], scoring_sample_ids=[],
            probe_scope='self-authored fixed validation text; logits equivalence only; no quality dataset')
        # The existing validation reader resolves paths against its supplied root.
        token_dest = root / 'lab/inputs/cohort-v1-validation-probes.safetensors'
        if not token_dest.exists():
            shutil.copyfile(root / probe_rel, token_dest)
        for i, method in enumerate(['mlp_permutation', 'rescaling', 'targeted_perturbation']):
            recipes.append(dict(model_id=stem + '-' + method, edge_type='laundering', method=method,
                seed=20261101 + i, parents=[label], parent_identities={label: identity}, inputs=data,
                relative_frobenius_norm=0.001, target_matrices=4, cohort='track-b-real-cohort-v1', synthetic=False))
    code_hashes = {name: sha(Path(name)) for name in ['lab/quantization.py', 'lab/transforms.py',
        'lab/scripts/laundering.py', 'lab/artifacts.py']}
    plan = dict(schema_version='1', sources=state(root)['sources'], recipes=recipes,
        probe_texts=PROBES, probe_texts_sha256=content_hash(PROBES), torch_threads=4,
        runner_sha256=sha(Path(__file__)), planned_outputs=14,
        generator_sha256=code_hashes,
        limits=dict(weight_models=100, wall_seconds=7000, gpu_hours=0),
        scope='scoped actual safetensors only; no training; no quality or evasion claims')
    path = root / RUN / 'frozen.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        path = root / RUN / ('frozen-' + content_hash(plan) + '.json')
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('frozen cohort differs')
    atomic_json(path, plan)
    runner = root / RUN / ('runner-' + plan['runner_sha256'] + '.py')
    if not runner.exists():
        shutil.copyfile(Path(__file__), runner)
    if not (root / RUN / 'runner.py').exists():
        shutil.copyfile(Path(__file__), root / RUN / 'runner.py')
    atomic_json(root / RUN / 'active-frozen.json', dict(path=path.relative_to(root).as_posix(), sha256=sha(path)))
    return plan


def generate(root):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from huggingface_hub import constants
    constants.HF_HUB_OFFLINE = True
    import torch
    torch.set_num_threads(4)
    from transformers import AutoModelForCausalLM
    from lab.scripts.laundering import load_source, _validation_tokens, _audit, _save
    plan = freeze(root)
    begin = time.monotonic()
    sources = {r['model_id']: r for r in plan['sources']}
    timings = []
    failures = []
    for recipe in plan['recipes']:
        start = time.monotonic()
        if start - begin > 5400:
            raise ValueError('bounded generation time exceeded')
        parent = recipe['parents'][0]
        source = sources[parent]
        existing = next((r for r in state(root)['outputs'] if r['model_id'] == recipe['model_id']), None)
        if existing:
            target = root / existing['path']
            saved = json.loads((target / 'recipe.json').read_text())['recipe']
            if any(saved.get(k) != v for k, v in recipe.items()):
                raise ValueError('ready output recipe differs')
            if any(sha(target / n) != h for n, h in existing['weight_sha256'].items()):
                raise ValueError('ready output weights differ')
            continue
        failure_path = root / RUN / (recipe['model_id'] + '-failure.json')
        if failure_path.exists():
            failures.append(json.loads(failure_path.read_text()))
            continue
        if recipe['edge_type'] == 'quantization_sim':
            target = write_checkpoint(recipe['model_id'], root / 'lab', recipe,
                lambda stage: run_quantization(recipe, {'root': root / 'lab'}, stage), resume=True)
        else:
            data = recipe['inputs']
            model = load_source(data, root / 'lab', parent, recipe['parent_identities'][parent])
            model.config._name_or_path = ''
            try:
                edited, audit = _audit(model, recipe, _validation_tokens(data, root / 'lab'))
            except ValueError as error:
                if str(error) != 'function-preserving transform failed frozen-token equivalence checks':
                    raise
                failure = dict(model_id=recipe['model_id'], status='REJECTED_EQUIVALENCE',
                    method=recipe['method'], published=False, tolerances_unchanged=True)
                atomic_json(failure_path, failure)
                failures.append(failure)
                del model
                gc.collect()
                continue
            audited = dict(recipe, audit=audit)
            target = _save(root / 'lab', audited, edited, True)
            atomic_json(root / RUN / (recipe['model_id'] + '-audit.json'), audit)
            del model, edited
            gc.collect()
        model = AutoModelForCausalLM.from_pretrained(target, local_files_only=True,
            trust_remote_code=False, use_safetensors=True, dtype=torch.bfloat16)
        parameters = sum(p.numel() for p in model.parameters())
        del model
        gc.collect()
        row = dict(model_id=recipe['model_id'], path=target.relative_to(root).as_posix(),
            root_id=parent, kind='derived', parents=[parent], repo_id=None, revision=None,
            license=source['license'], edge_type=recipe['edge_type'], synthetic=False, complete=True,
            split=source['split'], parameters_reloaded=parameters,
            weight_sha256={p.name: sha(p) for p in target.glob('*.safetensors')},
            weight_bytes=sum(p.stat().st_size for p in target.glob('*.safetensors')),
            seconds=round(time.monotonic() - start, 3))
        row.update({k: recipe[k] for k in ['bits', 'method'] if k in recipe})
        publish_ready(root, 'outputs', row)
        timings.append(row['seconds'])
        atomic_json(root / RUN / 'progress.json', dict(completed=len(timings),
            available_planned=len(plan['recipes']), seconds=round(time.monotonic() - begin, 3),
            projection_seconds=sum(timings) / len(timings) * len(plan['recipes'])))
        print(json.dumps(dict(phase='output_verified', **row)), flush=True)
    ids = {r['model_id'] for r in plan['recipes']}
    actual = len([r for r in state(root)['outputs'] if r['model_id'] in ids])
    atomic_json(root / RUN / 'summary.json', dict(status='COMPLETE_AVAILABLE_SAFE_INPUTS',
        actual_outputs=actual, planned_outputs=14, sources=len(plan['sources']), failures=failures,
        elapsed_seconds=round(time.monotonic() - begin, 3), gpu_hours=0,
        runner_sha256=sha(Path(__file__)), frozen= json.loads((root / RUN / 'active-frozen.json').read_text())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--pins-override', type=Path)
    parser.add_argument('--labels', nargs='+', default=['smol-135'] + ['poly-size-slot-0-seed-slot-' + str(i) for i in range(4)])
    args = parser.parse_args()
    root = Path(os.environ['ZOO_DATA'])
    (root / RUN).mkdir(parents=True, exist_ok=True)
    if args.prepare:
        prepare(root, args.labels, args.pins_override)
    if args.generate:
        generate(root)


if __name__ == '__main__':
    logging.disable(logging.CRITICAL)
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(status='FAILED', error_type=type(error).__name__)), flush=True)
        raise SystemExit(2)
