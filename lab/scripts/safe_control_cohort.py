"""Successor cohort from existing official safetensors conversion pull requests.

Acquisition is public HTTPS only; generation uses reviewed offline CPU writers.
Neither pickle weights nor new conversion jobs are allowed.
"""
import argparse
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from urllib.parse import quote

sys.path.insert(0, str(Path.cwd()))
from lab.artifacts import atomic_json, content_hash, write_checkpoint
from lab.pins import verify_pin

RUN = 'runs/track-b-safe-control-cohort-v1'
KNOWN = {1: 'd69651876749bd56b006b4d7ac3b10f8bcd2c976',
         2: '05775313619da467a1f6410c3b222bf8295eadb3'}
KNOWN_WEIGHT = {2: '5f83b47a3b7dab9a408f739cf9e6799f47090f186680a6f7c5e8fb7a03a61b22'}
TOKEN_FILES = ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']
FILES = ['README.md', 'config.json', *TOKEN_FILES, 'model.safetensors']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def immutable(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('immutable run evidence differs: ' + path.name)
    if not path.exists():
        atomic_json(path, value)


def snapshot(root):
    run = root / RUN
    run.mkdir(parents=True, exist_ok=True)
    target = run / ('runner-' + sha(Path(__file__)) + '.py')
    if not target.exists():
        shutil.copyfile(Path(__file__), target)
    return target


def acquire(root):
    import requests
    from huggingface_hub import try_to_load_from_cache
    begin = time.monotonic()
    run = root / RUN
    runner = snapshot(root)
    session = requests.Session()
    # No credential environment variables, authorization headers, or .env reads.
    session.headers['User-Agent'] = 'Zoo-Keeper-safe-control-cohort-v1'
    pins = json.loads(Path('lab/pins.json').read_text())
    selected, availability = [], []
    criterion = dict(candidate_order=list(range(1, 10)), select_first=4,
        requirements=['existing PR1 authored by SFconvertbot', 'immutable commit',
            'single safetensors with immutable LFS SHA256',
            'original pinned config/tokenizers/README match', 'actual downloaded bytes verified'],
        split_rule='first two selected seeds development; last two selected seeds heldout_test',
        independent_seed_evidence='existing approved Polypythias pretraining roster in original pins',
        selection_uses_scores=False, conversion_triggered=False, pickle_allowed=False,
        expected_source_models=4, expected_outputs=8,
        runner_sha256=sha(runner), network='public HTTPS requests only, no token')
    immutable(run / 'availability-criterion.json', criterion)

    def get(url, *, stream=False):
        response = session.get(url, timeout=(15, 90), stream=stream)
        response.raise_for_status()
        return response

    for seed in range(1, 10):
        if len(selected) == 4:
            break
        if time.monotonic() - begin > 5400:
            raise ValueError('bounded acquisition time exceeded')
        original_id = 'poly-size-slot-0-seed-slot-' + str(seed - 1)
        original = pins[original_id]
        verify_pin(original)
        repo = original['repo_id']
        label = 'safe-control-v1-pythia-14m-seed' + str(seed)
        evidence = run / 'publisher' / ('seed' + str(seed))
        evidence.mkdir(parents=True, exist_ok=True)
        try:
            pr = get('https://huggingface.co/api/models/' + repo + '/discussions/1').json()
            atomic_json(evidence / 'pr1.json', pr)
            author = pr.get('author', {})
            author = author.get('name') if isinstance(author, dict) else author
            if author != 'SFconvertbot' or pr.get('isPullRequest') is not True:
                raise ValueError('PR1 is not an existing official conversion request')
            revision = KNOWN.get(seed, 'refs/pr/1')
            info = get('https://huggingface.co/api/models/' + repo + '/revision/' +
                       quote(revision, safe='') + '?blobs=true').json()
            atomic_json(evidence / 'converted-metadata.json', info)
            commit = info.get('sha', '')
            if not re.fullmatch('[a-f0-9]{40}', commit) or (seed in KNOWN and commit != KNOWN[seed]):
                raise ValueError('conversion revision did not resolve to expected immutable commit')
            siblings = {s['rfilename']: s for s in info.get('siblings', [])}
            safe = {name: value for name, value in siblings.items() if name.endswith('.safetensors')}
            if set(safe) != {'model.safetensors'}:
                raise ValueError('conversion does not have exactly one allowlisted safetensors file')
            weight_info = safe['model.safetensors']
            weight_sha = weight_info.get('lfs', {}).get('sha256', '')
            weight_size = weight_info.get('size')
            if not re.fullmatch('[a-f0-9]{64}', weight_sha) or not isinstance(weight_size, int) or not 0 < weight_size < 100_000_000:
                raise ValueError('missing safe weight SHA256 or exceeds tiny-model byte bound')
            if seed in KNOWN_WEIGHT and weight_sha != KNOWN_WEIGHT[seed]:
                raise ValueError('known converted weight digest differs')
            if any(name not in siblings for name in FILES):
                raise ValueError('missing allowlisted public metadata')
            rel = 'lab/inputs/' + label + '-' + commit[:12]
            dest = root / rel
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                stage = Path(tempfile.mkdtemp(prefix='.safe-control-source-', dir=dest.parent))
                try:
                    for name in FILES:
                        cached = try_to_load_from_cache(repo, name, revision=commit)
                        if isinstance(cached, str):
                            shutil.copyfile(cached, stage / name)
                        else:
                            response = get('https://huggingface.co/' + repo + '/resolve/' + commit + '/' + name,
                                           stream=True)
                            maximum = weight_size if name.endswith('.safetensors') else 20_000_000
                            downloaded = 0
                            with (stage / name).open('wb') as target:
                                for block in response.iter_content(1024 * 1024):
                                    downloaded += len(block)
                                    if downloaded > maximum:
                                        raise ValueError('allowlisted file exceeds frozen byte bound')
                                    target.write(block)
                            response.close()
                    os.rename(stage, dest)
                finally:
                    if stage.exists():
                        shutil.rmtree(stage)
            config = json.loads((dest / 'config.json').read_text())
            if sha(dest / 'config.json') != original['config_file_sha256'] or content_hash(config) != original['config_sha256']:
                raise ValueError('conversion changed original pinned configuration')
            if config.get('model_type') != 'gpt_neox' or config.get('auto_map') or config.get('quantization_config'):
                raise ValueError('unsafe or unexpected model configuration')
            if content_hash({name: sha(dest / name) for name in TOKEN_FILES}) != original['tokenizer_sha256']:
                raise ValueError('conversion changed original pinned tokenizer')
            if sha(dest / 'README.md') != original['license_sha256']:
                raise ValueError('conversion README differs from approved publisher license evidence')
            if sha(dest / 'model.safetensors') != weight_sha or (dest / 'model.safetensors').stat().st_size != weight_size:
                raise ValueError('downloaded safe weight bytes differ from converted Hub metadata')
            if any(p.is_symlink() or p.suffix in {'.py', '.bin', '.pt', '.pth', '.pkl', '.pickle'} for p in dest.rglob('*')):
                raise ValueError('unsafe local input files')
            pin = dict(original, model_id=label, revision=commit,
                weight_sha256={'model.safetensors': weight_sha},
                license_path=rel + '/README.md', original_model_id=original_id,
                original_revision=original['revision'], conversion_ref='refs/pr/1',
                conversion_author=author, conversion_url='https://huggingface.co/' + repo + '/discussions/1',
                converted_commit_url='https://huggingface.co/' + repo + '/commit/' + commit,
                cohort_only=True)
            verify_pin(pin)
            immutable(run / (label + '-verified-pin.json'), pin)
            marker = dict(model_key=repo + '@' + commit, weight_sha256=pin['weight_sha256'],
                config_file_sha256=original['config_file_sha256'], config_sha256=original['config_sha256'],
                tokenizer_sha256=original['tokenizer_sha256'], original_model_id=original_id,
                original_revision=original['revision'], conversion_author=author,
                conversion_ref='refs/pr/1', verified=True)
            immutable(dest / 'track-b-source.json', marker)
            row = dict(model_id=label, root_id=label, original_model_id=original_id,
                path=rel, kind='source', parents=[], repo_id=repo, revision=commit,
                original_revision=original['revision'], license=pin['license'],
                config_file_sha256=original['config_file_sha256'], config_sha256=original['config_sha256'],
                tokenizer_sha256=original['tokenizer_sha256'], weight_sha256=pin['weight_sha256'],
                weight_bytes=weight_size, seed=seed, edge_type=None, synthetic=False, complete=True,
                marker=rel + '/track-b-source.json', independence_evidence=original['independence_evidence'],
                conversion_author=author, conversion_ref='refs/pr/1')
            selected.append(row)
            availability.append(dict(seed=seed, status='VERIFIED_SAFE_CONVERSION', revision=commit))
            print(json.dumps(dict(phase='source_verified', **row)), flush=True)
        except Exception as error:
            availability.append(dict(seed=seed, status='UNAVAILABLE_VERIFIED_SAFE_CONVERSION', error_type=type(error).__name__))
            print(json.dumps(dict(phase='availability', **availability[-1])), flush=True)
        atomic_json(run / 'availability.json', availability)
    if len(selected) != 4:
        atomic_json(run / 'acquisition-status.json', dict(status='BLOCKED_FEWER_THAN_FOUR_SAFE_CONTROLS',
            verified_sources=len(selected), availability=availability, elapsed_seconds=time.monotonic() - begin))
        return
    for index, row in enumerate(selected):
        row['split'] = 'development' if index < 2 else 'heldout_test'
    immutable(run / 'selection.json', dict(sources=selected, criterion_sha256=sha(run / 'availability-criterion.json'),
        selected_seeds=[r['seed'] for r in selected], root_group_split_frozen_before_generation=True))
    immutable(run / 'ready.json', dict(schema_version='1', sources=selected, outputs=[]))
    immutable(run / 'pins-override.json', {r['model_id']: json.loads((run / (r['model_id'] + '-verified-pin.json')).read_text()) for r in selected})
    atomic_json(run / 'acquisition-status.json', dict(status='COMPLETE_FOUR_SAFE_CONTROLS',
        verified_sources=4, elapsed_seconds=round(time.monotonic() - begin, 3),
        selected_seeds=[r['seed'] for r in selected], runner_sha256=sha(runner)))
    print(json.dumps(dict(phase='selection_frozen', selected_seeds=[r['seed'] for r in selected],
        roots=[dict(model_id=r['model_id'], revision=r['revision'], split=r['split']) for r in selected])), flush=True)


def generate(root):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from huggingface_hub import constants
    constants.HF_HUB_OFFLINE = True
    import torch
    torch.set_num_threads(4)
    from transformers import AutoModelForCausalLM
    from lab.quantization import checked_source, run_quantization
    begin = time.monotonic()
    run = root / RUN
    runner = snapshot(root)
    selection = json.loads((run / 'selection.json').read_text())
    if len(selection['sources']) != 4 or selection['root_group_split_frozen_before_generation'] is not True:
        raise ValueError('four availability-selected root groups must be frozen before generation')
    sources = selection['sources']
    recipes = []
    for source in sources:
        parent = source['model_id']
        identity = source['repo_id'] + '@' + source['revision']
        prepared = dict(path=Path(source['path']).relative_to('lab').as_posix(), model_key=identity,
            config_file_sha256=source['config_file_sha256'], config_sha256=source['config_sha256'],
            weight_sha256=source['weight_sha256'], manifest=dict(repo_id=source['repo_id'],
                revision=source['revision'], files=[dict(filename=n, sha256=h) for n, h in source['weight_sha256'].items()]))
        for bits in [8, 4]:
            recipe = dict(model_id=parent + '-q' + str(bits), edge_type='quantization_sim', bits=bits,
                parents=[parent], parent_identities={parent: identity}, inputs={parent: prepared},
                cohort='track-b-safe-control-cohort-v1', root_id=parent, split=source['split'], synthetic=False,
                original_model_id=source['original_model_id'], conversion_author=source['conversion_author'])
            checked_source(recipe, {'root': root / 'lab'})
            recipes.append(recipe)
    frozen = dict(schema_version='1', sources=sources, recipes=recipes, expected_outputs=8,
        expected_unique_weight_models=12, torch_threads=4, gpu_hours=0, runner_sha256=sha(runner),
        selection_sha256=sha(run / 'selection.json'),
        generator_sha256={n: sha(Path(n)) for n in ['lab/quantization.py', 'lab/artifacts.py']},
        limits=dict(weight_models=100, wall_seconds=7000), selection_uses_scores=False,
        no_training=True, no_pickle=True, no_conversion_triggered=True)
    frozen_path = run / 'frozen.json'
    if frozen_path.exists() and json.loads(frozen_path.read_text()) != frozen:
        frozen['supersedes_frozen_sha256'] = sha(frozen_path)
        correction = run / 'generator-correction.json'
        if not correction.exists():
            raise ValueError('successor generator freeze requires explicit correction evidence')
        frozen['generator_correction_sha256'] = sha(correction)
        frozen_path = run / ('frozen-successor-' + content_hash(frozen) + '.json')
    immutable(frozen_path, frozen)
    atomic_json(run / 'active-frozen.json', dict(path=frozen_path.relative_to(root).as_posix(),
        sha256=sha(frozen_path)))
    for recipe in recipes:
        start = time.monotonic()
        if start - begin > 5400:
            raise ValueError('bounded generation time exceeded')
        ready = json.loads((run / 'ready.json').read_text())
        if any(r['model_id'] == recipe['model_id'] for r in ready['outputs']):
            continue
        target = write_checkpoint(recipe['model_id'], root / 'lab', recipe,
            lambda stage: run_quantization(recipe, {'root': root / 'lab'}, stage), resume=True)
        model = AutoModelForCausalLM.from_pretrained(target, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=torch.bfloat16)
        parameters = sum(p.numel() for p in model.parameters())
        del model
        gc.collect()
        source = next(s for s in sources if s['model_id'] == recipe['parents'][0])
        row = dict(model_id=recipe['model_id'], root_id=recipe['root_id'], path=target.relative_to(root).as_posix(),
            kind='derived', parents=recipe['parents'], repo_id=None, revision=None, license=source['license'],
            edge_type='quantization_sim', bits=recipe['bits'], synthetic=False, complete=True,
            split=recipe['split'], seed=source['seed'], parameters_reloaded=parameters,
            weight_sha256={p.name: sha(p) for p in target.glob('*.safetensors')},
            weight_bytes=sum(p.stat().st_size for p in target.glob('*.safetensors')),
            seconds=round(time.monotonic() - start, 3))
        ready['outputs'].append(row)
        atomic_json(run / 'ready.json', ready)
        elapsed = time.monotonic() - begin
        atomic_json(run / 'progress.json', dict(completed_outputs=len(ready['outputs']), expected_outputs=8,
            elapsed_seconds=round(elapsed, 3), projected_total_seconds=round(elapsed / len(ready['outputs']) * 8, 3)))
        print(json.dumps(dict(phase='output_verified', **row)), flush=True)
    ready = json.loads((run / 'ready.json').read_text())
    atomic_json(run / 'generation-status.json', dict(status='COMPLETE_REAL_SAFE_CONTROLS',
        actual_sources=4, actual_outputs=len(ready['outputs']), planned_outputs=8,
        selected_seeds=[s['seed'] for s in sources], synthetic=False, training_performed=False,
        gpu_hours=0, torch_threads=4, elapsed_seconds=round(time.monotonic() - begin, 3),
        frozen_path=frozen_path.relative_to(root).as_posix(), frozen_sha256=sha(frozen_path),
        ready_sha256=sha(run / 'ready.json'), runner_sha256=sha(runner)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acquire', action='store_true')
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--data-root', type=Path)
    args = parser.parse_args()
    data_root = args.data_root or os.environ.get('ZOO_DATA')
    if not data_root:
        parser.error('--data-root or ZOO_DATA is required')
    root = Path(data_root)
    if args.acquire:
        acquire(root)
    if args.generate:
        generate(root)


if __name__ == '__main__':
    logging.disable(logging.CRITICAL)
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(status='FAILED', error_type=type(error).__name__)), flush=True)
        raise SystemExit(2)
