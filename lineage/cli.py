"""Offline inference over existing fingerprints; no checkpoint loading."""
import argparse
import json
from pathlib import Path
import tempfile

from .engine import configuration, digest, infer
from .io import write_edges
from .testing import make_synthetic_pool, replace_record
from .upstream import load_records


def _publish(records, args, config, calibration):
    edges, diagnostics = infer(records, config=config, calibration=calibration)
    provenance = dict(method_version=config['method_version'], config=config,
                      method_hash=diagnostics['method_hash'], input_hash=diagnostics['input_hash'],
                      calibration_hash=digest(calibration), diagnostics=diagnostics)
    out = Path(args.out) / 'edges.parquet' if args.command == 'dry-run' else Path(args.out)
    if out.exists() or out.is_symlink():
        if not args.resume:
            raise FileExistsError('output exists; use --resume for matching inputs')
        previous = json.loads((out.resolve().parent / 'provenance.json').read_text())
        for key in ('method_hash', 'input_hash', 'calibration_hash'):
            if previous.get(key) != provenance[key]:
                raise ValueError('resume rejected: changed ' + key)
        print(json.dumps({'status': 'resumed', 'out': str(out), 'edges': len(edges)}))
        return 0
    write_edges(edges, out, provenance=provenance)
    print(json.dumps({'status': 'synthetic' if args.command == 'dry-run' else 'complete',
                      'out': str(out), 'edges': len(edges), 'calibrated': bool(edges.calibrated.any())}))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('evaluate', 'freeze'):
        sub = commands.add_parser(command)
        sub.add_argument('--config', type=Path, required=True)
        sub.add_argument('--out', type=Path, required=True)
        if command == 'freeze':
            sub.add_argument('--inputs', type=Path, required=True,
                             help='JSON object mapping input names to local paths')
        else:
            for name in ('predictions', 'truth', 'universe', 'costs'):
                sub.add_argument('--' + name, type=Path, required=True)
            for name in ('models', 'edges', 'manifests'):
                sub.add_argument('--' + name, type=Path)
    for command in ('infer', 'dry-run'):
        sub = commands.add_parser(command)
        sub.add_argument('--config', type=Path)
        sub.add_argument('--calibration', type=Path)
        sub.add_argument('--out', type=Path, required=True)
        sub.add_argument('--resume', action='store_true')
        if command == 'infer':
            sub.add_argument('--manifests', type=Path, required=True)
            sub.add_argument('--fingerprints', type=Path, required=True)
            sub.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command in ('evaluate', 'freeze'):
        return _evaluate_command(args)
    config = configuration(json.loads(args.config.read_text()) if args.config else {})
    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    if args.command == 'dry-run':
        config['direction_method_hash'] = config['direction_method_hash'] or 'a' * 64
        records = make_synthetic_pool().records
        child, parent = records[3], records[0]
        records[3] = replace_record(child, direction_evidence=dict(parent=parent.key,
            status='asymmetric', score=.8, provenance='development_frozen',
            method_hash=config['direction_method_hash']))
        return _publish(records, args, config, calibration)
    root = args.data_root.resolve()
    if not args.manifests.is_dir() or not args.fingerprints.is_dir():
        raise ValueError('infer requires manifest and fingerprint directories')
    records = load_records(args.manifests, args.fingerprints, data_root=root)
    # Build only the existing-fingerprint retrieval matrices, in private scratch.
    # The temporary index is used within this lifetime and is never a checkpoint cache.
    from zoo.retrieval import build_retrieval
    from zoo.tables import build_tables
    with tempfile.TemporaryDirectory(prefix='lineage-retrieval-') as temporary:
        tables = build_tables(args.manifests, [args.fingerprints], data_root=root)
        build_retrieval(Path(temporary), tables['models'], tables['views'], root)
        for record in records:
            record.meta.update(zoo_dir=temporary, zoo_index_size=len(records))
        return _publish(records, args, config, calibration)


def _evaluate_command(args):
    import pandas as pd
    from .evaluate import canonical, evaluate, freeze, verify_freeze
    config = json.loads(args.config.read_text())
    if args.command == 'freeze':
        protocol = freeze(config, json.loads(args.inputs.read_text()), args.out)
        print(json.dumps({'status': 'frozen', 'out': str(args.out),
                          'method_hash': protocol['method_hash']}))
        return 0
    inputs = {name: getattr(args, name) for name in ('predictions', 'truth', 'universe', 'costs')}
    lab_inputs = {name: getattr(args, name) or config.get('lab_' + name)
                  for name in ('models', 'edges', 'manifests')}
    if any(lab_inputs.values()) and not all(lab_inputs.values()):
        raise ValueError('public lab scoring requires models, edges, and manifests together')
    inputs.update({name: path for name, path in lab_inputs.items() if path})
    verify_freeze(config, inputs, args.out)
    for name in ('metrics.json', 'curves.parquet', 'coverage.parquet'):
        if (args.out / name).exists():
            raise FileExistsError('evaluation artifact already exists: ' + name)

    def read_table(path, *, registry=False):
        path = Path(path)
        return pd.read_csv(path, keep_default_na=not registry) if path.suffix == '.csv' else pd.read_parquet(path)

    tables = {name: read_table(path) for name, path in inputs.items()
              if name in ('predictions', 'truth', 'universe', 'costs')}
    report = evaluate(**tables, config=config)
    if all(lab_inputs.values()):
        from lab.score import score_lab
        manifests_path = Path(lab_inputs['manifests'])
        if manifests_path.is_dir():
            manifests = [json.loads(path.read_text()) for path in sorted(manifests_path.glob('*.json'))]
        else:
            manifests = json.loads(manifests_path.read_text())
        lab_universe = tables['universe'].copy()
        # Explicit sidecars make disagreement with complete construction truth
        # fail in the public scorer instead of silently producing two reports.
        tier_lookup = {}
        label_lookup = {}
        for row in tables['truth'].to_dict('records'):
            pair = row['query_key'], row['candidate_key']
            previous = tier_lookup.setdefault(pair, row['truth_tier'])
            if previous != row['truth_tier']:
                raise ValueError('public scorer requires one truth tier per universe pair')
            label_lookup[pair + (row['estimand'],)] = row['label']
        for index, row in lab_universe.iterrows():
            pair = row['query_key'], row['candidate_key']
            if pair in tier_lookup:
                old_tier = row.get('truth_tier')
                if pd.notna(old_tier) and old_tier != tier_lookup[pair]:
                    raise ValueError('truth tier conflicts with universe sidecar')
                lab_universe.loc[index, 'truth_tier'] = tier_lookup[pair]
            for estimand in ('parent', 'ancestor', 'same_root'):
                label = label_lookup.get(pair + (estimand,))
                if pd.notna(label):
                    column = 'truth_' + estimand
                    old_label = row.get(column)
                    if pd.notna(old_label) and old_label != bool(label):
                        raise ValueError('truth label conflicts with universe sidecar')
                    lab_universe.loc[index, column] = bool(label)
        report['lab_metrics'] = score_lab(read_table(lab_inputs['models'], registry=True),
            read_table(lab_inputs['edges'], registry=True), tables['predictions'], manifests,
            lab_universe, threshold=config['deployment_threshold'])
    report.pop('curves').to_parquet(args.out / 'curves.parquet', index=False)
    report.pop('coverage').to_parquet(args.out / 'coverage.parquet', index=False)
    with (args.out / 'metrics.json').open('x') as stream:
        stream.write(canonical(report) + '\n')
    print(json.dumps({'status': 'complete', 'out': str(args.out), 'gate': report['gate']['status']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
