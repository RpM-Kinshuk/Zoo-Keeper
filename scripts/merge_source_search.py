"""Fixed, offline pair-ranking diagnostic; no historical-source decision gate."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import platform
import socket
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'Zoo-Keeper'), str(ROOT/'Model-Zoo')]


def rank_pairs(vectors, target):
    """Preserve the full LS objective in QR; return every unconstrained pair fit."""
    keys = sorted(vectors)
    if len(keys) < 2:
        return []
    target = np.asarray(target, dtype=np.float64)
    matrix = np.column_stack([vectors[k] for k in keys]+[target]).astype(np.float64)
    if matrix.ndim != 2 or len(target) == 0 or not np.isfinite(matrix).all():
        raise ValueError('finite nonempty aligned vectors required')
    target_norm = np.linalg.norm(target)
    if target_norm == 0:
        raise ValueError('zero update target')
    reduced = np.linalg.qr(matrix, mode='r')
    # Pair LS on N original rows uses eps*max(N,2), even after QR reduction.
    rcond = np.finfo(np.float64).eps*max(len(target), 2)
    rows = []
    for i, j in itertools.combinations(range(len(keys)), 2):
        x, y = reduced[:, [i, j]], reduced[:, -1]
        coefficients, _, rank, singular = np.linalg.lstsq(x, y, rcond=rcond)
        rows.append(dict(pair=[keys[i], keys[j]], coefficients=coefficients.tolist(),
            relative_residual=float(np.linalg.norm(y-x@coefficients)/target_norm),
            rank=int(rank), condition_number=float(singular[0]/singular[-1]) if singular[-1] else None))
    return sorted(rows, key=lambda r: (r['relative_residual'], r['pair']))


def source_pools(keys, parents, inventories):
    removed = [set(), {inventories[parents[0]]}, {inventories[parents[1]]},
               {inventories[p] for p in parents}]
    return {name: sorted(k for k in keys if inventories[k] not in excluded)
            for name, excluded in zip(['present', 'remove_first', 'remove_second', 'remove_both'], removed)}


def control_fingerprint(row, ledger):
    if row.get('row_block_fingerprint') in ledger:
        return row['row_block_fingerprint']
    full = Path(row['full_weight_fingerprint'])
    stem = full.name.removesuffix('.t2.fingerprint.safetensors').removesuffix('.full_weights.fingerprint.safetensors')
    matches = [name for name in ledger if Path(name).name in
               [stem+'.t1.fingerprint.safetensors', stem+'.row_blocks.fingerprint.safetensors']]
    if len(matches) > 1:
        raise ValueError('ambiguous saved control fingerprints')
    return matches[0] if matches else None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out):
    from lab.artifacts import atomic_json
    from lineage.upstream import load_records
    from lineage.records import view_diagnostics
    from zoo.schema import arch_signature
    import lineage.records, lineage.upstream
    import evidence.fingerprint, evidence.compatibility
    import yaml

    assert not (out/'freeze.json').exists(), 'preserve frozen inputs'
    files = [Path(__file__), out/'PROTOCOL.md', Path(lineage.records.__file__),
             Path(lineage.upstream.__file__), Path(evidence.fingerprint.__file__),
             Path(evidence.compatibility.__file__), ROOT/'Model-Zoo/zoo/schema.py']
    formal_path = ROOT/'runs/track-b-pivot-unblock-a-20261105/active-roster-v2/artifact-map166.json'
    formal = json.loads(formal_path.read_text())['rows']; files.append(formal_path)
    ledger_path = ROOT/'runs/track-b-existing-artifacts-20261108/inputs.json'
    ledger = json.loads(ledger_path.read_text()); files.append(ledger_path)
    edges_path = formal_path.parent/'edges.csv'
    with edges_path.open() as handle:
        edges = list(csv.DictReader(handle))
    files.append(edges_path)
    parents_by_child = {}
    for edge in edges:
        parents_by_child.setdefault(edge['child'], set()).add(edge['parent'])
    def ancestors(model_id):
        seen, todo = set(), list(parents_by_child.get(model_id, []))
        while todo:
            parent = todo.pop()
            if parent not in seen:
                seen.add(parent); todo.extend(parents_by_child.get(parent, []))
        return seen
    control_truth = {}
    indices = {}; recipes = {}; all_entries = {}; inventories = {}; metadata = {}; gaps = []
    for family, lane, dirname in [('smol-360', 'smol360', 'recipes'), ('q25-05', 'qwen', 'recipes-v2-frozen'), ('q3-06', 'qwen', 'recipes-v2-frozen')]:
        run = ROOT/f'runs/track-b-formal-{lane}-merges-v1'
        ip = run/'evidence/record-index.json'; files.append(ip)
        entries = json.loads(ip.read_text())
        rps = [run/dirname/f'{family}-merge-linear-{t}.json' for t in ['025', '05', '075']]
        rs = [json.loads(p.read_text()) for p in rps]; files += rps
        ref = rs[1]['parent_identities'][family]
        arch = next(e['arch'] for e in entries if e['key'] == ref)
        selected = [e for e in entries if e['arch'] == arch]
        indices[family] = dict(reference=ref, pool_keys=sorted(e['key'] for e in selected if e['key'] != ref))
        recipes[family] = rs
        for r in rs:
            published = ROOT/'lab'/r['model_id']/'recipe.json'
            pub = json.loads(published.read_text()); pub = pub.get('recipe', pub)
            assert yaml.safe_load(r['mergekit_yaml']) == yaml.safe_load(pub['mergekit_yaml'])
            files.append(published)
            assert r['method'] == 'linear' and r['parents'] == rs[1]['parents']
            for parent in r['parents']+[family]:
                expected = r['inputs'][parent]['weight_sha256']
                entry = next(e for e in selected if e['key'] == r['parent_identities'][parent])
                manifest = json.loads((ROOT/entry['manifest']).read_text())
                assert expected == {f['filename']: f['sha256'] for f in manifest['files'] if f['filename'].endswith('.safetensors')}
        weights = [np.array([r['coefficients'][p] for p in r['parents']]) for r in rs]
        assert np.array_equal(weights[1], [.5, .5])
        assert np.array_equal((weights[0]+weights[2])*.5, weights[1])
        for suffix in ['sft-s0', 'dpo-s0']:
            row = next((r for r in formal if r['model_id'] == family+'-'+suffix), None)
            fp = control_fingerprint(row, ledger) if row is not None else None
            if fp is None:
                gaps.append(dict(model_id=family+'-'+suffix, reason='no_saved_row_fingerprint')); continue
            truth = [e for e in edges if e['child'] == row['model_id']]
            assert truth and {e['edge_type'] for e in truth} == {suffix.split('-')[0]}
            assert family in ancestors(row['model_id'])
            control_truth[row['key']] = truth
            assert sha(ROOT/fp) == ledger[fp]
            mp = ROOT/row['manifest']; assert sha(mp) == row['manifest_sha256']
            manifest = json.loads(mp.read_text())
            # Use the same original architecture contract as the merge index.
            assert arch_signature(manifest['config'], []) == arch
            selected.append(dict(key=row['key'], arch=arch, manifest=row['manifest'], fingerprint=fp))
        for e in selected:
            if e['key'] in all_entries:
                assert all_entries[e['key']] == e
            all_entries[e['key']] = e
            for field in ['manifest', 'fingerprint']:
                assert (ROOT/e[field]).is_file(); files.append(ROOT/e[field])
            manifest = json.loads((ROOT/e['manifest']).read_text())
            inventory = sorted((f['filename'], f['size'], f['sha256']) for f in manifest['files'] if f['filename'].endswith('.safetensors'))
            assert inventory and all(len(h) == 64 for _, _, h in inventory)
            inventories[e['key']] = hashlib.sha256(json.dumps(inventory).encode()).hexdigest()
            metadata[e['key']] = dict(weight_inventory=inventory)
    atomic_json(out/'record-index.json', list(all_entries.values()))
    files.append(out/'record-index.json')
    records = {r.key: r for r in load_records(out/'record-index.json', ROOT/'runs', data_root=ROOT)}
    cases = []
    for family, info in indices.items():
        rs = recipes[family]; ref = info['reference']; query = 'lab/'+rs[1]['model_id']+'@'
        query = next(k for k in records if k.startswith(query))
        for key in [query]+[k for k in records if k.startswith('lab/'+family+'-') and any(k.split('@')[0].endswith(s) for s in ['sft-s0', 'dpo-s0'])]:
            excluded = [k for k in info['pool_keys'] if inventories[k] == inventories[key]]
            mid = key.split('@')[0].removeprefix('lab/')
            descendants = [k for k in info['pool_keys'] if mid in ancestors(k.split('@')[0].removeprefix('lab/'))]
            pool = [k for k in info['pool_keys'] if k not in excluded and k not in descendants]
            common = set(range(len(records[key].names))); diagnostics = {}
            for k in [ref]+pool:
                pairs, reasons = view_diagnostics(records[key], records[k], k=64)
                common &= {i for i, _ in pairs}
                if reasons: diagnostics[k] = reasons
            names = sorted(records[key].names[i] for i in common)
            is_merge = key == query
            parents = [rs[1]['parent_identities'][p] for p in rs[1]['parents']] if is_merge else []
            witnesses = [next(k for k in records if k.startswith('lab/'+r['model_id']+'@')) for r in [rs[0], rs[2]]] if is_merge else []
            pools = source_pools(pool, parents, inventories) if is_merge else {'control': sorted(pool)}
            cases.append(dict(family=family, query=key, kind='linear_merge' if is_merge else 'nonmerge_control', reference=ref,
                parents=parents, alternative_pair=witnesses, pools=pools, query_alias_exclusions=excluded,
                descendant_exclusions=descendants, common_names=names, requested_views=len(records[key].names), diagnostics=diagnostics))
    for k, record in records.items():
        metadata[k].update(settings={s: record.meta[s] for s in ['tier', 'evidence_version', 'row_block', 'sketch_seed', 'sketch_k']})
    freeze = dict(cases=cases, control_truth=control_truth, inventories=inventories, metadata=metadata, gaps=gaps,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in files},
        exposure='All three roots and K3 constructions already exposed; diagnostic development, no heldout relabeling.',
        policy=dict(weight_reads=0, network_reads=0, gpu_devices=[], cpu_threads=1, max_seconds=1800),
        decision='No calibrated historical-source decisions; pair ranking and fixed-witness geometry only.')
    atomic_json(out/'freeze.json', freeze)
    print('prepared', len(cases), 'queries', len(records), 'saved fingerprints', flush=True)


def run(out):
    from lab.artifacts import atomic_json
    from lineage.upstream import load_records
    from lineage.records import sketch_block
    import importlib.metadata
    import os

    assert (out/'prospective-review.md').exists(), 'independent review required'
    assert not (out/'results.json').exists(), 'completed diagnostic is immutable'
    freeze = json.loads((out/'freeze.json').read_text())
    started = time.monotonic()
    for name, expected in freeze['source_sha256'].items():
        assert sha(ROOT/name) == expected, name
    records = {r.key: r for r in load_records(out/'record-index.json', ROOT/'runs', data_root=ROOT)}
    runtime = dict(hostname=socket.gethostname(), owned_pid=os.getpid(), gpu_devices=[], python=platform.python_version(),
                   packages={n: importlib.metadata.version(n) for n in ['numpy', 'torch', 'safetensors']})
    atomic_json(out/'runtime.json', runtime)
    results = []
    for case in freeze['cases']:
        assert time.monotonic()-started < 1700
        keys = sorted({case['reference'], case['query']} | {k for pool in case['pools'].values() for k in pool})
        arrays = {}
        for key in keys:
            rec = records[key]; by_name = {n: i for i, n in enumerate(rec.names)}
            arrays[key] = np.concatenate([sketch_block(rec, by_name[n], 64).ravel() for n in case['common_names']])
        base = arrays[case['reference']]; raw = arrays[case['query']]; y = raw-base
        vectors = {k: v-base for k, v in arrays.items() if k not in [case['reference'], case['query']]}
        fullkeys = sorted(vectors)
        all_fits = rank_pairs(vectors, y)
        pools = []
        for name, candidates in case['pools'].items():
            # Filtering already-computed pair scores preserves the common panel and LS target.
            allowed = set(candidates); fits = [r for r in all_fits if set(r['pair']) <= allowed]
            nearest = sorted(candidates, key=lambda k: (np.linalg.norm(arrays[k]-raw), k))[:2]
            true_fit = next((r for r in fits if set(r['pair']) == set(case['parents'])), None)
            rank_range = None if true_fit is None else [1+sum(r['relative_residual'] < true_fit['relative_residual'] for r in fits), sum(r['relative_residual'] <= true_fit['relative_residual'] for r in fits)]
            pools.append(dict(pool=name, candidates=candidates, pair_count=len(fits), fits=fits, nearest_two=nearest,
                nearest_ingredient_overlap=len(set(nearest)&set(case['parents'])) if case['parents'] else None,
                best_ingredient_overlap=len(set(fits[0]['pair'])&set(case['parents'])) if case['parents'] and fits else None,
                true_pair_rank_range=rank_range, true_pair_fit=true_fit,
                best_wrong_fit=next((r for r in fits if set(r['pair']) != set(case['parents'])), None),
                historical_source_decision='not_calibrated'))
        witnesses = {}
        for label, pair in [('documented', case['parents']), ('alternative', case['alternative_pair'])]:
            if not pair: continue
            prediction = .5*arrays[pair[0]]+.5*arrays[pair[1]]
            error = prediction-raw
            witnesses[label] = dict(pair=pair, fixed_coefficients=[.5, .5],
                relative_update_residual=float(np.linalg.norm(error)/np.linalg.norm(y)),
                relative_raw_residual=float(np.linalg.norm(error)/np.linalg.norm(raw)),
                max_abs_sketch_error=float(np.abs(error).max()),
                max_epsilon_scaled_error=float(np.max(np.abs(error)/(np.finfo(np.float32).eps*np.maximum(1, np.abs(raw))))))
        results.append(dict(query=case['query'], family=case['family'], kind=case['kind'], pools=pools, fixed_witnesses=witnesses,
            views_shared=len(case['common_names']), views_requested=case['requested_views'],
            update_relative_to_base=float(np.linalg.norm(y)/np.linalg.norm(base)), n_coordinates=len(y)))
        atomic_json(out/'progress.json', dict(completed_queries=len(results)))
        print(case['family'], case['kind'], 'pair_rank', pools[0]['true_pair_rank_range'], 'best', pools[0]['fits'][0]['pair'], flush=True)
    atomic_json(out/'results.json', dict(queries=results, elapsed_seconds=time.monotonic()-started,
        freeze_sha256=sha(out/'freeze.json'), runtime=runtime, weight_reads=0, network_reads=0,
        source_decision_metrics='NOT MEASURED: no calibrated historical-source gate', interpretation=freeze['decision']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args(); out = ROOT/'runs/merge-source-search-20261112'
    out.mkdir(exist_ok=True)
    (prepare if args.prepare else run)(out)


if __name__ == '__main__':
    main()
