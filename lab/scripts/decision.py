"""Plan six decision checkpoints or train offline synthetic classification fixtures."""
import argparse
import json
from pathlib import Path
from lab.artifacts import content_hash, write_checkpoint
from lab.registry import expand_inventory
from lab.schema import LAB_SCHEMA_VERSION, canonical_json
from lab.training import validate_decision_recipe, train_decision, load_decision_checkpoint


def decision_recipes(config):
    if config.get('schema_version') != LAB_SCHEMA_VERSION:
        raise ValueError('unsupported decision schema')
    return [validate_decision_recipe({**config, **row}) for row in expand_inventory({})['decision']]


def tiny_fixture(architecture, embedded=False):
    import torch
    from transformers import (LlamaConfig, LlamaForSequenceClassification,
        ModernBertConfig, ModernBertForSequenceClassification, Qwen3Config, Qwen3ForSequenceClassification)
    torch.set_num_threads(1)
    torch.manual_seed(20261028)
    common = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=4, vocab_size=64, max_position_embeddings=32,
                  num_labels=2, pad_token_id=0)
    if architecture == 'modernbert':
        model = ModernBertForSequenceClassification(ModernBertConfig(**common, local_attention=8,
            global_attn_every_n_layers=2, classifier_pooling='mean', attn_implementation='eager',
            bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2))
    else:
        config_cls, model_cls = (Qwen3Config, Qwen3ForSequenceClassification) if architecture == 'qwen3' else (LlamaConfig, LlamaForSequenceClassification)
        model = model_cls(config_cls(**common, num_key_value_heads=2, tie_word_embeddings=True))
    fixture = dict(type='synthetic-offline-classifier', architecture=architecture, seed=20261028, config=json.loads(json.dumps(model.config.to_dict())))
    recipe = dict(model_id='tiny-decision-' + architecture, edge_type='lora' if embedded else 'sft',
        parents=['tiny-base'], parent_identities={'tiny-base': 'lab/tiny-base@local:' + content_hash(fixture)},
        seed=20261028, steps=2, effective_batch=4, sequence_length=8, learning_rate=2e-4 if embedded else 2e-5,
        label_names=['negative', 'positive'], pad_token_id=0, embedded_adapter=embedded,
        rank=8, alpha=8, target_modules=['q_proj', 'v_proj'], torch_dtype='float32', device='cpu',
        inputs={'fixture': fixture})
    tokens = (torch.arange(64).reshape(8, 8) % 63) + 1
    tokens[:, -2:] = 0
    return validate_decision_recipe(recipe), dict(model=model, token_ids=tokens,
        attention_mask=(tokens != 0).long(), labels=torch.arange(8) % 2,
        sample_ids=list(range(8)), heldout_ids=[8, 9])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--lab-root', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--tiny', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    recipes = decision_recipes(config)
    if args.dry_run:
        print(canonical_json(dict(recipes=recipes, inputs_state=config.get('inputs_state'))))
        return 0
    if args.tiny:
        for architecture, embedded in [('llama', False), ('modernbert', False), ('qwen3', True)]:
            recipe, inputs = tiny_fixture(architecture, embedded)
            path = write_checkpoint(recipe['model_id'], args.lab_root, recipe,
                lambda stage: train_decision(recipe, inputs, stage), resume=args.resume)
            load_decision_checkpoint(path)
        print(canonical_json(dict(status='COMPLETE_SYNTHETIC', fixtures=3, steps=2)))
        return 0
    prepared = config.get('prepared_inputs')
    if config.get('inputs_state') != 'frozen' or not prepared:
        raise ValueError('real decision batch blocked: verified dataset license and frozen splits/inputs are pending')
    if any(row['model_id'] not in prepared for row in recipes):
        raise ValueError('missing frozen decision inputs')
    for recipe in recipes:
        portable = prepared[recipe['model_id']]
        pin = portable['model_pin']
        recipe = {**recipe, 'inputs': portable,
            'parent_identities': {recipe['parents'][0]: pin['repo_id'] + '@' + pin['revision']}}
        inputs = {**portable, 'root': args.lab_root}
        write_checkpoint(recipe['model_id'], args.lab_root, recipe,
            lambda stage, r=recipe, i=inputs: train_decision(r, i, stage), resume=args.resume)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
