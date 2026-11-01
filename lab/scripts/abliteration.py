"""Plan eight HUMAN abliteration runs, or exercise safe synthetic dispatch."""
import argparse
import json
from pathlib import Path

from lab.abliteration import classic_edit, run_abliteration, validate_recipe
from lab.artifacts import content_hash, write_checkpoint
from lab.registry import expand_inventory
from lab.schema import LAB_SCHEMA_VERSION, canonical_json


def _tiny(config, root, resume=False):
    import torch
    from safetensors.torch import save_file
    torch.set_num_threads(1)
    torch.manual_seed(20261028)
    state = {'writer.weight': torch.randn(4, 6), 'writer.bias': torch.randn(4)}
    direction = torch.tensor([1., 0., 0., 0.])
    writers = {name: dict(layer=0, residual_axis=0) for name in state}
    fixture = dict(type='synthetic-residual-writer-v1', seed=20261028)
    base = dict(edge_type='abliteration', parents=['tiny-base'], seed=20261028,
        parent_identities={'tiny-base': 'lab/tiny-base@local:' + content_hash(fixture)}, inputs={'fixture': fixture}, requested_config=config)
    classic = dict(base, model_id='tiny-classic', method='classic')
    def project(out):
        from lab.artifacts import atomic_json
        edited = classic_edit(state, {0: direction}, writers)
        for tensor in edited.values():
            assert torch.norm(direction @ tensor) < 1e-6
        save_file(edited, str(out / 'model.safetensors'))
        atomic_json(out / 'abliteration.json', dict(method='classic', synthetic=True, writers=writers))
    write_checkpoint(classic['model_id'], root, classic, project, resume=resume)
    heretic = dict(base, model_id='tiny-heretic-dispatch', method='heretic',
                   heretic_config={'n_trials': 1, 'orthogonalize_direction': True})
    def dispatch(recipe, inputs, out):
        # Dispatch/provenance fixture only: these are unedited synthetic tensors.
        save_file(state, str(out / 'model.safetensors'))
        return dict(config=recipe['heretic_config'], runner_kind='injected_test', synthetic=True,
                    per_layer_settings={'0': {'direction_indices': [0]}},
                    optimization_trace=[{'trial': 0, 'status': 'synthetic_dispatch_only'}])
    inputs = dict(sample_ids=[0, 1], source_training_ids=[4, 5], heldout_ids=[2, 3], validation_ids=[2], test_ids=[3],
                  direction_labels=[0, 1], heretic_runner=dispatch)
    write_checkpoint(heretic['model_id'], root, heretic,
                     lambda out: run_abliteration(heretic, inputs, out), resume=resume)
    return dict(status='COMPLETE_SYNTHETIC', real_heretic_output=False,
                model_ids=[classic['model_id'], heretic['model_id']])


def main(argv=None):
    parser = argparse.ArgumentParser(description='Classic/Heretic abliteration (eight real outputs: HUMAN launch)')
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--lab-root', required=True, type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--tiny', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if config.get('schema_version', LAB_SCHEMA_VERSION) != LAB_SCHEMA_VERSION:
        raise ValueError('unsupported abliteration schema')
    recipes = [validate_recipe({**row, **config}) for row in expand_inventory({})['abliteration']]
    if args.dry_run:
        print(canonical_json(dict(recipes=recipes, launch='HUMAN',
            inputs_state=config.get('inputs_state', 'unresolved'),
            heretic_status='blocked_installed_loader_lacks_safe_local_enforcement')))
        return 0
    if args.tiny:
        print(canonical_json(_tiny(config, args.lab_root, args.resume)))
        return 0
    prepared = config.get('prepared_inputs')
    if config.get('inputs_state') != 'frozen' or not prepared:
        raise ValueError('real abliteration batch blocked: verified frozen direction data and disjoint splits are pending')
    # Validate availability for the entire requested batch before starting outputs.
    for row in recipes:
        if row['model_id'] not in prepared:
            raise ValueError(row['model_id'] + ': missing frozen pinned inputs')
        if row['method'] == 'heretic':
            from lab.abliteration import installed_heretic_runner
            installed_heretic_runner(row, prepared[row['model_id']], args.lab_root)
    for row in recipes:
        portable = prepared[row['model_id']]
        pin = portable['model_pin']
        recipe = dict(row, inputs=portable,
                      parent_identities={row['parents'][0]: pin['repo_id'] + '@' + pin['revision']})
        inputs = dict(portable, root=args.lab_root)
        write_checkpoint(recipe['model_id'], args.lab_root, recipe,
            lambda out, r=recipe, i=inputs: run_abliteration(r, i, out), resume=args.resume)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
