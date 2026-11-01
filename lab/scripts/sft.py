"""Plan full SFT batches, or execute a two-step offline synthetic fixture."""
import argparse
import json
from pathlib import Path

from lab.artifacts import content_hash, write_checkpoint
from lab.registry import expand_inventory
from lab.schema import LAB_SCHEMA_VERSION, canonical_json, contained_path
from lab.training import train_sft, train_dpo, validate_full_recipe


def _tiny(config, edge, root):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, AutoModelForCausalLM
    torch.set_num_threads(1)
    torch.manual_seed(20261028)
    parent = 'tiny-base' if edge == 'sft' else 'tiny-sft'
    if edge == 'sft':
        model = LlamaForCausalLM(LlamaConfig(hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                vocab_size=64, max_position_embeddings=32, tie_word_embeddings=True))
        fixture = dict(type='synthetic-offline-llama', seed=20261028, config=model.config.to_dict())
    else:
        path = contained_path(root, parent, parent)
        if not (path / 'lineage.json').is_file():
            raise ValueError('tiny DPO requires the completed tiny SFT fixture first')
        lineage = json.loads((path / 'lineage.json').read_text())
        provenance = json.loads((path / 'recipe.json').read_text())
        if (lineage != dict(name='lab/tiny-sft', bases=['lab/tiny-base'], relation='sft')
                or provenance['config_hash'] != content_hash(provenance['recipe'])):
            raise ValueError('tiny DPO requires valid SFT provenance')
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
                use_safetensors=True, trust_remote_code=False, dtype=torch.float32)
        fixture = dict(type='synthetic-offline-sft', recipe=provenance['recipe'])
    recipe = validate_full_recipe({**config, 'model_id': 'tiny-' + edge, 'edge_type': edge,
        'parents': [parent], 'parent_identities': {parent: 'lab/' + parent + '@local:' + content_hash(fixture)},
        'seed': 20261028, 'steps': 2, 'effective_batch': 4, 'sequence_length': 8,
        'device': 'cpu', 'torch_dtype': 'float32', 'inputs': {'fixture': 'eight-sequential-token-rows-v1'}}, edge)
    tokens = torch.arange(64).reshape(8, 8)
    inputs = dict(model=model, token_ids=tokens, sample_ids=list(range(8)), heldout_ids=[8])
    if edge == 'dpo':
        rejected = tokens.clone()
        rejected[:, 4:] = rejected[:, 4:].roll(1, dims=1)
        inputs.update(parent_relation='sft', rejected_token_ids=rejected,
            chosen_mask=torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]] * 8),
            rejected_mask=torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]] * 8))
    return recipe, inputs


def _main(argv, edge):
    parser = argparse.ArgumentParser(description='Full ' + edge.upper() + ' (real batches: HUMAN launch)')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--lab-root', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--tiny', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if config.get('schema_version', LAB_SCHEMA_VERSION) != LAB_SCHEMA_VERSION:
        raise ValueError('unsupported full tuning schema')
    recipes = [validate_full_recipe({**row, **config}, edge) for row in expand_inventory({})[edge]]
    if args.dry_run:
        print(canonical_json(dict(recipes=recipes, inputs_state=config.get('inputs_state', 'unresolved'), launch='HUMAN')))
        return 0
    trainer = train_sft if edge == 'sft' else train_dpo
    if args.tiny:
        recipe, inputs = _tiny(config, edge, args.lab_root)
        path = write_checkpoint(recipe['model_id'], args.lab_root, recipe,
                                lambda stage: trainer(recipe, inputs, stage), resume=args.resume)
        from transformers import AutoModelForCausalLM
        AutoModelForCausalLM.from_pretrained(path, local_files_only=True, use_safetensors=True,
                                           trust_remote_code=False, dtype='bfloat16')
        print(canonical_json(dict(model_id=recipe['model_id'], steps=2, status='COMPLETE_SYNTHETIC')))
        return 0
    prepared = config.get('prepared_inputs')
    if not prepared or config.get('inputs_state') != 'frozen':
        raise ValueError('real ' + edge.upper() + ' batch blocked: verified dataset license and frozen splits/inputs are pending')
    # Validate every requested input before starting the human-launched batch.
    for recipe in recipes:
        if recipe['model_id'] not in prepared:
            raise ValueError(recipe['model_id'] + ': missing frozen pinned inputs')
    for recipe in recipes:
        portable = prepared[recipe['model_id']]
        if edge == 'sft':
            pin = portable['model_pin']
            identity = pin['repo_id'] + '@' + pin['revision']
        else:
            if not portable.get('local_parent'):
                raise ValueError('DPO requires its matching SFT parent manifest')
            identity = portable['local_parent']['model_key']
        recipe = {**recipe, 'parent_identities': {recipe['parents'][0]: identity}, 'inputs': portable}
        inputs = {**portable, 'root': args.lab_root}
        write_checkpoint(recipe['model_id'], args.lab_root, recipe,
            lambda stage, r=recipe, i=inputs: trainer(r, i, stage), resume=args.resume)
    return 0


def main(argv=None):
    return _main(argv, 'sft')


if __name__ == '__main__':
    raise SystemExit(main())
