"""Plan LoRA batches, or execute an explicit offline tiny fixture."""
import argparse
import json
from pathlib import Path

from lab.artifacts import content_hash, write_checkpoint
from lab.registry import expand_inventory
from lab.schema import LAB_SCHEMA_VERSION, canonical_json
from lab.training import validate_recipe


def _tiny(config):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.set_num_threads(1)
    torch.manual_seed(20261028)
    model = LlamaForCausalLM(LlamaConfig(hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            vocab_size=64, max_position_embeddings=32, tie_word_embeddings=True))
    # Identity is the fixture's explicit content description, never a real base pin.
    fixture = dict(type='synthetic-offline-llama', seed=20261028, config=model.config.to_dict())
    recipe = {**config, 'model_id': 'tiny-lora', 'edge_type': 'lora', 'parents': ['tiny-base'],
              'parent_identities': {'tiny-base': 'lab/tiny-base@local:' + content_hash(fixture)},
              'seed': 20261028, 'rank': 8, 'alpha': 8, 'steps': 2,
              'effective_batch': 4, 'sequence_length': 8, 'torch_dtype': 'float32',
              'inputs': {'fixture': 'eight-sequential-token-rows-v1'},
              'secondary_weights': ['adapter/adapter_model.safetensors']}
    return validate_recipe(recipe), {'model': model, 'token_ids': torch.arange(64).reshape(8, 8)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--lab-root', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--tiny', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if config.get('schema_version', LAB_SCHEMA_VERSION) != LAB_SCHEMA_VERSION:
        raise ValueError('unsupported LoRA schema')
    recipes = [validate_recipe({**row, **config}) for row in expand_inventory({})['lora']]
    if args.dry_run:
        print(canonical_json({'recipes': recipes, 'inputs_state': config.get('inputs_state', 'unresolved'),
                              'launch': 'HUMAN'}))
        return 0
    from lab.training import train_lora
    if args.tiny:
        recipe, inputs = _tiny(config)
        path = write_checkpoint(recipe['model_id'], args.lab_root, recipe,
                                lambda stage: train_lora(recipe, inputs, stage), resume=args.resume)
        # Local reload validates the public checkpoint rather than only its writer.
        from transformers import AutoModelForCausalLM
        AutoModelForCausalLM.from_pretrained(path, local_files_only=True, use_safetensors=True,
                                            trust_remote_code=False, dtype='bfloat16')
        print(canonical_json({'model_id': recipe['model_id'], 'steps': 2, 'status': 'COMPLETE_SYNTHETIC'}))
        return 0
    prepared = config.get('prepared_inputs')
    if not prepared or config.get('inputs_state') != 'frozen':
        raise ValueError('real LoRA batch blocked: verified dataset license and frozen splits/inputs are pending')
    for recipe in recipes:
        key = recipe['anchor'] + ':' + recipe['task']
        if key not in prepared:
            raise ValueError(f'{key}: missing frozen pinned inputs')
    # This path is reserved for the human-launched 24-recipe batch.
    for recipe in recipes:
        portable = prepared[recipe['anchor'] + ':' + recipe['task']]
        pin = portable['model_pin']
        recipe = {**recipe, 'parent_identities': {recipe['parents'][0]: pin['repo_id'] + '@' + pin['revision']},
                  'inputs': portable, 'secondary_weights': ['adapter/adapter_model.safetensors']}
        inputs = {**portable, 'root': args.lab_root}
        write_checkpoint(recipe['model_id'], args.lab_root, recipe,
                         lambda stage, r=recipe, i=inputs: train_lora(r, i, stage), resume=args.resume)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
