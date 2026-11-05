"""Consume A's completed L12 artifacts in a new, frozen descriptive experiment.

No model weights, network, fitting, training, or old report writes. T2 is
compared through the extractor's native statistic; the T1 engine is unchanged.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from evidence.compare import pair_similarity
from evidence import READABLE_EVIDENCE_VERSIONS
from evidence.compatibility import canonical_mode, companion_spectra_path
from evidence.fingerprint import load_fingerprint
from safetensors import safe_open
from evidence.sketch import SKETCH_SEED
from lab.artifacts import atomic_json, content_hash
from lab.chains import chain_paths
from lab.registry import validate_tables
from lab.schema import MODEL_COLUMNS, EDGE_COLUMNS, canonical_json
from lab.score import score_lab
from lineage.engine import configuration, infer
from lineage.evaluate import scorer_identity
from lineage.io import write_edges
from lineage.lab_report import (bind_artifact, common_distances, freeze_json,
                                training_input_index, validate_measurement)
from lineage.tree import pivot_flags, tree_diagnostics
from lineage.upstream import load_records
from zoo.schema import LINEAGE_COLUMNS
from zoo.tables import build_tables
from zoo.retrieval import build_retrieval


MODES = {'row-blocks-b128-k128': ('row_blocks_128', 128),
         'row-blocks-b256-k64': ('row_blocks_256', 64),
         'row-blocks-b256-k128': ('row_blocks_256', 128),
         'row-blocks-b512-k128': ('row_blocks_512', 128),
         'full-weights-k64': ('full_weight_fingerprint', 64),
         'full-weights-k128': ('full_weight_fingerprint', 128)}
COHORTS = ('track-b-real-cohort-v1', 'track-b-safe-control-cohort-v1',
           'track-b-training-gpu-v1')
MEASUREMENTS = {'original': ('row_blocks', 256, 64),
                'row_blocks_128': ('row_blocks_128', 128, 128),
                'row_blocks_256': ('row_blocks', 256, 128),
                'row_blocks_512': ('row_blocks_512', 512, 128),
                'full_weight_fingerprint': ('full_weights', 256, 128)}
DELIVERY_FIELDS = ('manifest', 'full_weight_fingerprint', 'full_weight_spectra',
                   'row_blocks_128', 'row_blocks_256', 'row_blocks_512')


def normalize_delivery_entry(entry):
    """Adapt legacy input columns without changing the source index or files."""
    result = dict(entry)
    for old, new in (('fingerprint_t2', 'full_weight_fingerprint'),
                     ('spectra_h5', 'full_weight_spectra'),
                     ('t1_b128', 'row_blocks_128'), ('t1_b256', 'row_blocks_256'),
                     ('t1_b512', 'row_blocks_512')):
        if old in result:
            if new in result and result[new] != result[old]:
                raise ValueError('Conflicting legacy and descriptive delivery columns')
            result[new] = result.pop(old)
    if Path(result['full_weight_spectra']) != companion_spectra_path(result['full_weight_fingerprint']):
        raise ValueError('full-weight sidecar differs from bound delivery')
    return result


def read_chain_delivery(directory, bind):
    """Read either chain index generation with its frozen source records."""
    directory = Path(directory)
    index = directory/'artifact-index.csv'
    if not index.exists():
        index = directory/'artifact-index3.csv'
    if not index.exists():
        return None
    prefix = 'full-weights' if (directory/'full-weights-freeze.json').exists() else 't2'
    bind(directory/(prefix+'-freeze.json'))
    bind(directory/(prefix+'-progress.json'))
    return pd.read_csv(bind(index), keep_default_na=False)


def freeze_measurement(path, *, key, measurement):
    """Freeze the readable source version, never the current writer default."""
    with safe_open(path, framework='numpy') as handle:
        meta = handle.metadata()
    version = meta.get('evidence_version')
    if version not in READABLE_EVIDENCE_VERSIONS:
        raise ValueError('unsupported source evidence version')
    mode, block, size = MEASUREMENTS[measurement]
    contract = dict(tier=mode, row_block=block, sketch_k=size,
                    sketch_seed=SKETCH_SEED, evidence_version=version)
    if meta.get('repo_id', '')+'@'+meta.get('revision', '') != key:
        raise ValueError('fingerprint identity differs from frozen delivery')
    actual = dict(tier=canonical_mode(meta.get('tier')), evidence_version=version,
                  **{f: json.loads(meta.get(f, 'null')) for f in ('row_block', 'sketch_k', 'sketch_seed')})
    if actual != contract:
        raise ValueError('fingerprint measurement settings differ from declared mode')
    return contract


def validate_frozen_measurement(fp, *, key, contract):
    validate_measurement(fp, key=key, **contract)


def scoped_tables(rows, identities):
    models, edges = [], []
    for name, row in sorted(rows.items()):
        models.append(dict(model_id=name, kind='base' if row.get('repo_id') else 'derived',
            path=identities[name] if row.get('repo_id') else row['path'].removeprefix('lab/'),
            base_family=row['root_id'], license=row['license']))
        for parent in row.get('parents', []):
            edges.append(dict(child=name, parent=parent, edge_type=row['edge_type'], role='parent',
                              coefficient='', params_json=canonical_json({'scope': 'observed_construction'})))
    models, edges = pd.DataFrame(models, columns=MODEL_COLUMNS), pd.DataFrame(edges, columns=EDGE_COLUMNS)
    validate_tables(models, edges)
    return models, edges


def universe_for(ids, rows, keys, *, independent=False):
    parents = {name: set(row.get('parents', [])) for name, row in rows.items()}
    def ancestors(name):
        found, pending = set(), list(parents[name])
        while pending:
            parent = pending.pop()
            if parent not in found:
                found.add(parent); pending.extend(parents[parent])
        return found
    return pd.DataFrame([dict(query_key=keys[q], candidate_key=keys[c], eligible=True,
        truth_parent=c in parents[q], truth_ancestor=c in ancestors(q),
        truth_same_root=rows[q]['root_id'] == rows[c]['root_id'],
        truth_tier='verified_hub' if independent else 'lab_exact')
        for q in ids for c in ids if q != c])


def similarity_predictions(ids, keys, fingerprints, k, mode):
    if len({fingerprints[name].meta.get('evidence_version') for name in ids}) != 1:
        raise ValueError('cannot compare different source evidence versions')
    predictions = []
    for query in ids:
        for candidate in ids:
            if query == candidate:
                continue
            result = pair_similarity(fingerprints[query], fingerprints[candidate], sketch_k=k)
            if result['median_cosine'] is not None:
                predictions.append(dict(child_key=keys[query], parent_key=keys[candidate],
                    estimand='same_root', edge_type='unknown', evidence_tier='detected',
                    evidence=canonical_json({'statistic': 'median_cosine', 'mode': mode,
                                             'n_shared': result['n_shared']}),
                    score=result['median_cosine'], p_value=None, calibrated=False,
                    method_version='p1-0-' + mode))
    return pd.DataFrame(predictions, columns=LINEAGE_COLUMNS)


def run(root, delivery, out, chain_run=None):
    root, delivery, out = Path(root).resolve(), Path(delivery).resolve(), Path(out).resolve()
    if not delivery.is_relative_to(root) or not out.is_relative_to(root):
        raise ValueError('delivery and output must remain inside data root')
    if out.exists():
        raise FileExistsError('new evaluation requires a new output directory')
    out.mkdir(parents=True)
    proofs = {}
    def bind(path, expected=None):
        relative = str(Path(path).relative_to(root)) if Path(path).is_absolute() else str(path)
        proof = bind_artifact(root, relative, expected=expected)
        if relative in proofs and proofs[relative] != proof:
            raise ValueError('input changed while preparing freeze')
        proofs[relative] = proof
        return root / relative
    rows, old_progress = {}, {}
    for cohort in COHORTS:
        run_root = root/'runs'/cohort
        ready = json.loads(bind(run_root/'ready.json').read_text())
        progress = json.loads(bind(run_root/'evidence/progress.json').read_text())
        for row in ready['sources'] + ready['outputs']:
            name = row['model_id']
            if name in rows and (rows[name]['root_id'] != row['root_id']
                                 or rows[name]['weight_sha256'] != row['weight_sha256']):
                raise ValueError('inconsistent repeated ready identity')
            rows[name] = row
        for measured in progress['rows']:
            old_progress[measured['model_id']] = measured
    controls = json.loads(bind(root/'runs/track-b-safe-control-cohort-v1/pre-score-protocol.json').read_text())
    control_ready = root/'runs/track-b-safe-control-cohort-v1/ready.json'
    if proofs[str(control_ready.relative_to(root))]['sha256'] != controls['ready_sha256']:
        raise ValueError('original held-out control inventory changed')
    pins = json.loads((Path(__file__).parents[1]/'pins.json').read_text())
    for row in rows.values():
        if row.get('original_model_id'):
            pin = pins[row['original_model_id']]
            if pin.get('reference_type') != 'polypythias' or pin['repo_id'] != row['repo_id'] or pin['seed'] != row['seed']:
                raise ValueError('independent control lacks original published seed evidence')
    index_path = delivery/'artifact-index.csv'
    if not index_path.exists():
        index_path = delivery/'artifact-index43.csv'
    a_index = pd.read_csv(bind(index_path), keep_default_na=False)
    if a_index.model_id.duplicated().any() or a_index.model_key.duplicated().any():
        raise ValueError('duplicate delivery identities')
    entries = {r['model_id']: normalize_delivery_entry(r) for r in a_index.to_dict('records') if r['model_id'] in rows}
    if set(entries) != set(rows):
        raise ValueError('delivery does not cover the frozen ready cohort')
    delivery_ids = set(entries)
    for name, entry in entries.items():
        if entry['model_key'] != old_progress[name]['key']:
            raise ValueError('A delivery and completed T1 identity differ')
        expected_parents = [old_progress[n]['key'] for n in rows[name].get('parents', [])]
        if sorted(json.loads(entry['parent_keys'])) != sorted(expected_parents) or entry['root_id'] != rows[name]['root_id']:
            raise ValueError('delivery and verified construction parents/root differ')
        for field in DELIVERY_FIELDS:
            bind(entry[field])
        item = old_progress[name]
        bind(item['manifest'], item['manifest_sha256'])
        bind(item['fingerprint'], item['fingerprint_sha256'])
    chain_ids = []
    if chain_run:
        chain_run = Path(chain_run).resolve()
        ready = json.loads(bind(chain_run/'ready.json').read_text())
        progress = json.loads(bind(chain_run/'evidence/progress.json').read_text())
        for row in ready['sources'] + ready['outputs']:
            if row['model_id'] not in rows:
                rows[row['model_id']] = row; chain_ids.append(row['model_id'])
        for measured in progress['rows']:
            if measured['model_id'] in chain_ids:
                old_progress[measured['model_id']] = measured
                bind(measured['manifest'], measured['manifest_sha256'])
                bind(measured['fingerprint'], measured['fingerprint_sha256'])
        if not set(chain_ids) <= set(old_progress):
            raise ValueError('chain outputs lack completed evidence')
        bind(chain_run/'frozen.json')
        extra = read_chain_delivery(chain_run, bind)
        if extra is not None:
            if extra.model_id.duplicated().any() or set(extra.model_id) != set(chain_ids):
                raise ValueError('chain T2 index differs from the new ready endpoints')
            for entry in extra.to_dict('records'):
                name = entry['model_id']
                if entry['model_key'] != old_progress[name]['key']:
                    raise ValueError('chain T1/T2 identities differ')
                expected_parents = [old_progress[n]['key'] for n in rows[name]['parents']]
                if sorted(json.loads(entry['parent_keys'])) != sorted(expected_parents):
                    raise ValueError('chain T2 construction parent differs')
                entry = normalize_delivery_entry(entry)
                for field in DELIVERY_FIELDS:
                    bind(entry[field])
                entries[name] = entry
    keys = {name: item['key'] for name, item in old_progress.items()}
    manifests = {name: json.loads((root/item['manifest']).read_text()) for name,item in old_progress.items()}
    measurement_sources = {}
    for name in rows:
        sources = {'original': old_progress[name]['fingerprint']}
        if name in entries:
            sources.update({field: entries[name][field] for field in MEASUREMENTS if field != 'original'})
        for measurement, relative in sources.items():
            path = bind(relative)
            contract = freeze_measurement(path, key=keys[name], measurement=measurement)
            if relative in measurement_sources and measurement_sources[relative] != contract:
                raise ValueError('conflicting frozen measurement contracts')
            measurement_sources[relative] = contract
    models, edges = scoped_tables(rows, keys)
    training_ids = sorted(name for name in rows if name == 'smol-135' or name.startswith('smol-135-gpu-v1-'))
    training_index = training_input_index(training_ids, keys, old_progress, manifests)
    repository = Path(__file__).parents[2]
    source_files = [Path(__file__), Path(__file__).parents[1]/'pins.json']
    for package in ('lineage', 'lab'):
        source_files.extend(sorted((repository/package).glob('*.py')))
    scorer_components, runtime = scorer_identity()
    runtime.update({name: importlib.metadata.version(name)
                    for name in ('scipy', 'safetensors', 'h5py', 'networkx')})
    upstream_sources = {name: hashlib.sha256(Path(importlib.util.find_spec(name).origin).read_bytes()).hexdigest()
                        for name in ('evidence', 'evidence.fingerprint', 'evidence.compatibility', 'evidence.compare', 'evidence.sketch',
                                     'zoo.schema', 'zoo.tables', 'zoo.retrieval')}
    source_revisions = {name: subprocess.check_output(
        ['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        for name, path in (('zoo_keeper', repository),
                           ('model_zoo', Path(importlib.util.find_spec('evidence').origin).parents[1]))}
    freeze = dict(schema_version='2', artifacts=proofs, sources={str(p.relative_to(Path(__file__).parents[2])):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}, development_threshold=.75,
        scorer_components=scorer_components, runtime_identity=runtime, upstream_sources=upstream_sources,
        training_input_index=training_index, measurement_contract=MEASUREMENTS,
        measurement_sources=measurement_sources, delivery_inputs=entries, source_revisions=source_revisions,
        threshold_origin='unchanged predeclared descriptive threshold; no tuning',
        root_split=controls['root_split'], modes=MODES, sketch_distance='mean squared Frobenius over one common view set',
        quartet_tolerances=[1e-6, 1e-8, 1e-10], chain_models=chain_ids,
        scope='new descriptive reuse of fixed public cohorts; full lab/pool/baselines incomplete')
    freeze_json(out/'pre-score-freeze.json', freeze)
    fp_cache = {}
    def read_fp(name, relative, measurement='original'):
        if relative not in fp_cache:
            bind_artifact(root, relative, expected=proofs[relative]['sha256'])
            fp = load_fingerprint(root/relative)
            fp_cache[relative] = fp
        fp = fp_cache[relative]
        validate_frozen_measurement(fp, key=keys[name], contract=measurement_sources[relative])
        return fp
    models.to_csv(out/'models.csv', index=False); edges.to_csv(out/'edges.csv', index=False)
    started = time.monotonic()
    training_run = root/'runs/track-b-training-gpu-v1'
    atomic_json(out/'training-12/input-index.json', training_index)
    records = load_records(out/'training-12/input-index.json', training_run/'evidence', data_root=root)
    if {record.key for record in records} != {keys[n] for n in training_ids}:
        raise ValueError('loaded training universe differs from frozen entries')
    for record in records:
        item = next(row for row in training_index if row['key'] == record.key)
        for field in ('manifest', 'fingerprint'):
            if record.meta[field+'_sha256'] != proofs[item[field]]['sha256']:
                raise ValueError('loaded training artifact differs from freeze')
        validate_frozen_measurement(record.meta['upstream_fingerprint'], key=record.key,
            contract=measurement_sources[item['fingerprint']])
    # The real engine needs the same upstream retrieval index as its CLI.
    # Build it from the explicit files only, never a live-directory scan.
    private_manifests = out/'training-12/input-manifests'
    private_manifests.mkdir()
    for item in training_index:
        source = root/item['manifest']; data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != proofs[item['manifest']]['sha256']:
            raise ValueError('training manifest changed before retrieval construction')
        (private_manifests/source.name).write_bytes(data)
    tables = build_tables(private_manifests, [root/item['fingerprint'] for item in training_index], data_root=root)
    if set(tables['models'].model_key) != {item['key'] for item in training_index}:
        raise ValueError('retrieval universe differs from frozen training inputs')
    duplicates = dict(zip(tables['models'].model_key, tables['models'].dup_group))
    with tempfile.TemporaryDirectory(prefix='lineage-frozen-retrieval-') as temporary:
        build_retrieval(Path(temporary), tables['models'], tables['views'], root)
        for record in records:
            record.meta.update(zoo_dir=temporary, zoo_index_size=len(records),
                               duplicate_group=duplicates[record.key], duplicate_scope='all_measured_T1_blocks')
        predictions, diagnostics = infer(records, config=configuration({}))
    write_edges(predictions, out/'training-12/edges.parquet', provenance=diagnostics)
    train_universe = universe_for(training_ids, rows, keys)
    train_report = score_lab(models[models.model_id.isin(training_ids)], edges[edges.child.isin(training_ids)],
        predictions, [manifests[n] for n in training_ids], train_universe, threshold=.75)
    atomic_json(out/'training-12/metrics.json', train_report)
    train_universe.to_parquet(out/'training-12/universe.parquet', index=False)
    control_ids = sorted(n for n in rows if n.startswith('safe-control-v1-pythia-14m-seed'))
    results, similarity_frames = {}, {}
    for mode, (field, k) in MODES.items():
        fingerprints = {n: read_fp(n, entries[n][field], field) for n in control_ids}
        for n, fp in fingerprints.items():
            if fp.meta['repo_id']+'@'+fp.meta['revision'] != keys[n] or fp.meta['sketch_k'] < k:
                raise ValueError('fingerprint identity/size differs from frozen delivery')
        predictions = similarity_predictions(control_ids, keys, fingerprints, k, mode)
        similarity_frames[mode] = predictions
        results[mode] = {}
        for partition in ('development', 'test'):
            ids = [n for n in control_ids if controls['root_split'][rows[n]['root_id']] == partition]
            universe = universe_for(ids, rows, keys, independent=True)
            subset = predictions[predictions.child_key.isin([keys[n] for n in ids]) &
                                 predictions.parent_key.isin([keys[n] for n in ids])]
            result = score_lab(models[models.model_id.isin(ids)], edges[edges.child.isin(ids)], subset,
                               [manifests[n] for n in ids], universe, threshold=.75)
            results[mode][partition] = result
            directory = out/'controls'/mode/partition; directory.mkdir(parents=True)
            atomic_json(directory/'metrics.json', result)
            subset.to_parquet(directory/'predictions.parquet', index=False)
            universe.to_parquet(directory/'universe.parquet', index=False)
    left, right = similarity_frames['row-blocks-b256-k64'], similarity_frames['full-weights-k64']
    joined = left.merge(right, on=['child_key', 'parent_key'], suffixes=('_t1', '_t2'))
    # Same partition only; never borrow a test root for a development reference.
    split_by_key = {keys[n]: controls['root_split'][rows[n]['root_id']] for n in control_ids}
    joined = joined[[split_by_key[q] == split_by_key[c] for q,c in zip(joined.child_key,joined.parent_key)]]
    def correlation(frame):
        if len(frame) < 2 or frame.score_t1.nunique() < 2 or frame.score_t2.nunique() < 2:
            return None
        value = spearmanr(frame.score_t1, frame.score_t2).statistic
        return float(value) if np.isfinite(value) else None
    roots_by_key = {keys[n]: rows[n]['root_id'] for n in control_ids}
    related = joined[[roots_by_key[q] == roots_by_key[c] for q,c in zip(joined.child_key,joined.parent_key)]]
    comparison = dict(n_pairs=len(joined), all_pairs_spearman=correlation(joined),
        n_related_pairs=len(related), related_pair_spearman=correlation(related),
        heldout_same_root_auroc_row_blocks=results['row-blocks-b256-k64']['test']['tiers']['verified_hub']['same_root']['auroc'],
        heldout_same_root_auroc_full_weights=results['full-weights-k64']['test']['tiers']['verified_hub']['same_root']['auroc'],
        status='SMALL_SCOPE_DIAGNOSTIC', full_g0='NOT_EVALUABLE')
    trees = {}
    for root_id in ('smol-135', 'olmo-1b'):
        base_ids = sorted(n for n,r in rows.items() if r['root_id'] == root_id and n in delivery_ids)
        modes = {'original-row-blocks-k64': {n: read_fp(n, old_progress[n]['fingerprint']) for n in base_ids},
                 'full-weights-k128': {n: read_fp(n, entries[n]['full_weight_fingerprint'], 'full_weight_fingerprint') for n in base_ids}}
        if root_id == 'smol-135' and chain_ids:
            ids = sorted(n for n,r in rows.items() if r['root_id'] == root_id)
            modes['expanded-chain-row-blocks-k64'] = {n: read_fp(n, old_progress[n]['fingerprint']) for n in ids}
            if set(ids) <= set(entries):
                modes['expanded-chain-full-weights-k128'] = {n: read_fp(n, entries[n]['full_weight_fingerprint'], 'full_weight_fingerprint') for n in ids}
        for mode, fingerprints in modes.items():
            ids = sorted(fingerprints); k = 128 if mode.endswith('weights-k128') else 64
            distance, views = common_distances([fingerprints[n] for n in ids], k=k)
            truth = [(r.parent, r.child) for r in edges.itertuples() if r.parent in ids and r.child in ids]
            per_tolerance = {}
            for tolerance in (1e-6, 1e-8, 1e-10):
                result = tree_diagnostics(distance, ids, truth, predicted_edges=None, tolerance=tolerance)
                result['flags_for_person'] = pivot_flags(four_point=result['four_point_violation_fraction'], normalized_rf=None)
                per_tolerance[str(tolerance)] = result
            name = root_id+'-'+mode
            np.savez_compressed(out/(name+'-distances.npz'), distances=distance, labels=np.array(ids))
            trees[name] = dict(models=len(ids), common_view_count=len(views), common_views=views,
                diagnostics=per_tolerance, normalized_rf=None, direction_status='unresolved; no truth-oriented prediction supplied')
    atomic_json(out/'tree-diagnostics.json', trees)
    chain_paths_found = [p for p in chain_paths(models, edges) if len(p) >= 4]
    report = dict(status='REAL_SCOPED_DESCRIPTIVE', models=len(rows), edges=len(edges),
        training_models=len(training_ids), control_models=len(control_ids),
        ready_full_weight_models=len(entries),
        row_block_full_weight_comparison=comparison, trees=trees, depth_three_paths=chain_paths_found,
        new_chain_models=chain_ids, formal_chain_credit=0, gate_status='NOT_EVALUABLE',
        unresolved_dependencies=['formal generated cohort', 'safe pinned source weights/tokenization',
            'large reference pool', 'matched measured baselines', 'validated asymmetric direction'],
        additional_A_code_needed=False, elapsed_seconds=time.monotonic()-started,
        input_file_bytes=sum(p['size_bytes'] for p in proofs.values()),
        cost_status='incomplete: artifact sizes do not meter repeated reads/full extraction/verification IO',
        freeze_sha256=bind_artifact(root, str((out/'pre-score-freeze.json').relative_to(root)))['sha256'])
    atomic_json(out/'report.json', report)
    lines=['# L12 delivery evaluation and real tree diagnostics', '',
        f"{len(rows)} ready real models; {len(training_ids)} training-cohort models rescored; {len(control_ids)} frozen independent-seed control models.",
        'All original evaluation reports and held-out root assignments remain unchanged. No scores were used to select thresholds or cohorts.', '',
        f"Held-out same-root AUROC: T1={comparison['heldout_same_root_auroc_row_blocks']}, T2={comparison['heldout_same_root_auroc_full_weights']}; related-pair median-cosine Spearman={comparison['related_pair_spearman']} over{len(related)} ordered related pairs. All-pair Spearman={comparison['all_pairs_spearman']} over{len(joined)} within-partition pairs.",
        'These are small-scope relatedness diagnostics, not calibrated parent-direction or full G0/P1 evidence.', '',
        '| Real family / measurement | Nodes | Common views | Four-point violations / quartets at frozen1e-6 |',
        '| --- | ---: | ---: | --- |']
    for name, value in trees.items():
        diagnostic=value['diagnostics']['1e-06']
        lines.append(f"| {name} | {value['models']} | {value['common_view_count']} | {diagnostic['four_point_violations']} / {diagnostic['four_point_quartets']} |")
    lines+=['', 'Tolerance sensitivity1e-8/1e-10 is reported separately; no threshold was relaxed. Distances use one identical view/coordinate intersection per family and are stored as NPZ.',
        'Direction remains unresolved; nRF is null because no asymmetric assembled prediction exists. Truth edges only label diagnostics and never orient predictions. Pivot flags are reported for a person; this scoped experiment does not decide the formal E6 pivot.',
        f'{len(chain_paths_found)} depth-three scoped paths now exist; formal chain quota credit remains0.',
        ('The three new chain endpoints also have T2; expanded-chain T1 and T2 use identical20-node cohorts.'
         if chain_ids and set(chain_ids) <= set(entries) else
         'New chain endpoints lack T2 in this freeze; the original A T2 comparison covers33 ready inputs.'),
        'No further A code change is required. Remaining full-evaluation dependencies are generated formal checkpoints, readable-safe pinned source/tokenizer inputs, a larger reference pool, matched baselines and validated asymmetric direction.',
        'Input artifact sizes are recorded; total extraction/verification/repeated-read IO is incomplete. Free T1/k128 artifacts share their T2 extraction-pass counters and do not establish a standalone T1 byte saving. Release staging remains deferred.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=report['status'], models=len(rows), chains=len(chain_paths_found), report=str(out/'REPORT.md'))), flush=True)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'delivery', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--chain-run', type=Path)
    args=parser.parse_args(argv)
    run(args.data_root, args.delivery, args.out, args.chain_run)


if __name__ == '__main__': main()
