"""Metadata-only lab commands; recipe execution lives in lab.scripts."""
import argparse
import json
import os
from pathlib import Path

import pandas as pd

from .artifacts import atomic_json
from .registry import (expand_inventory, materialize_tables, read_tables,
                       validate_tables, verify_tables, write_tables)
from .schema import LAB_SCHEMA_VERSION, canonical_json


def _load_tables(args):
    if args.tables_root:
        return read_tables(args.tables_root)
    return (pd.read_csv(args.models, keep_default_na=False),
            pd.read_csv(args.edges, keep_default_na=False))


def _table_arguments(parser):
    parser.add_argument('--models', type=Path, default=Path(__file__).parent / 'models.csv')
    parser.add_argument('--edges', type=Path, default=Path(__file__).parent / 'edges.csv')
    parser.add_argument('--tables-root', type=Path, help='Read a versioned table pair through current.json')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    validate = commands.add_parser('validate')
    _table_arguments(validate)
    inventory = commands.add_parser('inventory')
    inventory.add_argument('--config', type=Path, default=Path(__file__).parent / 'configs/inventory.json')
    inventory.add_argument('--pins', type=Path)
    inventory.add_argument('--pin', action='store_true', help='Resolve selected metadata inputs')
    inventory.add_argument('--pin-negatives', action='store_true', help='Pin the explicit independent roster')
    inventory.add_argument('--data-config', type=Path)
    inventory.add_argument('--output', type=Path,
                           help='Inventory JSON, or versioned table directory when --pins is given')
    verify = commands.add_parser('verify')
    _table_arguments(verify)
    verify.add_argument('--lab-root', type=Path, required=True)
    verify.add_argument('--minimum', action='store_true', help='Report every public minimum without launching jobs')
    # The remaining commands are registered now; their modules belong to later tasks.
    export = commands.add_parser('export-models')
    _table_arguments(export)
    export.add_argument('--lab-root', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    score = commands.add_parser('score')
    _table_arguments(score)
    score.add_argument('--predictions', type=Path, required=True)
    score.add_argument('--manifests', type=Path, required=True)
    score.add_argument('--universe', type=Path, required=True,
                       help='Explicit pairs with any external truth labels already joined')
    score.add_argument('--config', type=Path, default=Path(__file__).parent / 'configs/scoring.json')
    score.add_argument('--threshold', type=float, help='Frozen development threshold; overrides config')
    score.add_argument('--output', type=Path, required=True)
    commands.add_parser('release-stage', add_help=False)
    args, extra = parser.parse_known_args(argv)
    if args.command == 'release-stage':
        parser.error(f'{args.command} requires its later genealogy lab implementation task')
    if extra:
        parser.error('unrecognized arguments: ' + ' '.join(extra))
    try:
        if args.command == 'score':
            from .score import score_lab

            def strict_json(path):
                def reject_constant(value):
                    raise ValueError('nonfinite JSON constant: ' + value)
                return json.loads(path.read_text(), parse_constant=reject_constant)

            config = strict_json(args.config)
            if (not isinstance(config, dict) or config.get('schema_version') != '1'
                    or set(config) - {'schema_version', 'threshold'}):
                raise ValueError('unsupported scoring config')
            if not args.manifests.is_dir():
                raise ValueError('manifests must be an existing directory')
            manifests = [strict_json(path) for path in sorted(args.manifests.glob('*.json'))]
            def read_evaluation(path):
                return pd.read_csv(path) if path.suffix == '.csv' else pd.read_parquet(path)
            models, edges = _load_tables(args)
            report = score_lab(models, edges, read_evaluation(args.predictions), manifests,
                               read_evaluation(args.universe),
                               threshold=args.threshold if args.threshold is not None else config.get('threshold'))
            atomic_json(args.output, report)
            print(canonical_json({'status': 'SCORED', 'output': str(args.output)}))
            return 0
        if args.command == 'export-models':
            from .export import export_models
            models, edges = _load_tables(args)
            validate_tables(models, edges)
            export_models(models, args.lab_root, args.output)
            print(canonical_json({'status': 'EXPORTED', 'models': len(models),
                                  'invocation_only': True, 'remove_after_manifest': True}))
            return 0
        if args.command == 'inventory':
            if args.pin_negatives:
                from .negatives import resolve_independents
                from .pins import metadata_callbacks
                if args.pin or not os.environ.get('ZOO_DATA') or not args.output:
                    raise ValueError('--pin-negatives requires ZOO_DATA, --output, and no --pin')
                config = json.loads(args.config.read_text())
                sizes = (14, 31, 70, 160, 410)
                entries = [{'model_id': f"poly-size-slot-{sizes.index(r['size_m'])}-seed-slot-{r['seed']-1}",
                            'repo_id': r['repo_id']} for r in config['roster']]
                entries += [r for r in config['siblings'] if r.get('repo_id')]
                data_root = Path(os.environ['ZOO_DATA']).resolve()
                if not args.output.resolve().is_relative_to(data_root):
                    raise ValueError('negative pin output must live under ZOO_DATA')
                api, download = metadata_callbacks(data_root, entries)
                pins = resolve_independents(config, api=api, download_text=download)
                atomic_json(args.output, pins)
                n_pinned = sum(p['status'] == 'pinned' for p in pins)
                complete = n_pinned == len(pins)
                print(canonical_json({'status': 'PINNED' if complete else 'INCOMPLETE',
                                      'inputs': len(pins), 'pinned': n_pinned}))
                return 0 if complete else 1
            if args.pin:
                from .pins import pin_config
                if not os.environ.get('ZOO_DATA') or not args.output:
                    raise ValueError('--pin requires ZOO_DATA and --output')
                config = json.loads(args.config.read_text())
                if args.data_config:
                    config['entries'].extend(json.loads(args.data_config.read_text())['entries'])
                pins = pin_config(config, args.output, data_root=Path(os.environ['ZOO_DATA']))
                print(canonical_json({'status': 'PINNED' if all(p['status'] == 'pinned' for p in pins.values()) else 'INCOMPLETE',
                                      'inputs': len(pins), 'pinned': sum(p['status'] == 'pinned' for p in pins.values())}))
                return 0 if all(p['status'] == 'pinned' for p in pins.values()) else 1
            expanded = expand_inventory(json.loads(args.config.read_text()))
            unresolved = [{'model_id': row['model_id'], 'reason': row['unresolved_reason']}
                          for row in expanded['bases'] + expanded['references']]
            if args.pins:
                models, edges = materialize_tables(expanded, json.loads(args.pins.read_text()))
                if not args.output:
                    raise ValueError('--output table directory is required with --pins')
                write_tables(models, edges, args.output)
                report = {'schema_version': LAB_SCHEMA_VERSION, 'models': len(models),
                          'edges': len(edges), 'status': 'MATERIALIZED'}
            else:
                report = {'schema_version': LAB_SCHEMA_VERSION, 'status': 'PLANNED',
                          'counts': {key: len(rows) for key, rows in expanded.items()},
                          'unresolved': sorted(unresolved, key=lambda row: row['model_id']),
                          'inventory': expanded}
                if args.output:
                    atomic_json(args.output, report)
        else:
            models, edges = _load_tables(args)
            validate_tables(models, edges)
            if args.command == 'verify':
                report = verify_tables(models, edges, args.lab_root)
                if args.minimum:
                    from .chains import verify_minimum
                    from .schema import HUB_PATH
                    pinned = set(models.loc[models.path.map(lambda p: bool(HUB_PATH.fullmatch(p))), 'model_id'])
                    report['minimum'] = verify_minimum(models, edges, set(report['completed_ids']) | pinned)
                    report['base_completion_basis'] = 'pinned_Hub_rows'
                    report['status'] = report['minimum']['status']
            else:
                report = {'schema_version': LAB_SCHEMA_VERSION, 'models': len(models),
                          'edges': len(edges), 'status': 'VALID'}
        print(canonical_json(report))
        return 1 if report.get('status') == 'INCOMPLETE' else 0
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
