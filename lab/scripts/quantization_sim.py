"""Plan twelve recipes or execute a local random tied-weight fixture offline."""
import argparse
import hashlib
import json
from pathlib import Path

from lab.artifacts import content_hash, write_checkpoint
from lab.quantization import checked_source, run_quantization
from lab.registry import expand_inventory
from lab.schema import canonical_json


def _tiny(root):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.set_num_threads(1)
    torch.manual_seed(20261028)
    name = 'tiny-quantization-parent'
    # A conspicuous fixture identity only; this is never an extractor identity.
    identity = 'lab/'+name+'@local:'+'0'*64
    source_recipe = dict(model_id=name, edge_type='sft', parents=['tiny-root'],
        parent_identities={'tiny-root':'lab/tiny-root@local:'+'0'*64},
        fixture='random-two-layer-tied-llama', seed=20261028)
    def writer(stage):
        model = LlamaForCausalLM(LlamaConfig(hidden_size=8, intermediate_size=16,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            vocab_size=16, tie_word_embeddings=True))
        model.save_pretrained(stage, safe_serialization=True)
    path = write_checkpoint(name, root, source_recipe, writer, resume=True)
    weights = {'model.safetensors':hashlib.sha256((path/'model.safetensors').read_bytes()).hexdigest()}
    pin = dict(path=name, model_key=identity,
        config_sha256=content_hash(json.loads((path/'config.json').read_text())),
        config_file_sha256=hashlib.sha256((path/'config.json').read_bytes()).hexdigest(), weight_sha256=weights,
        manifest=dict(repo_id='lab/'+name, revision='local:'+'0'*64,
            files=[dict(filename=k, sha256=v) for k,v in weights.items()]))
    return [dict(model_id='tiny-quantization-s'+str(bits), edge_type='quantization_sim', parents=[name],
        parent_identities={name:identity}, inputs={name:pin}, bits=bits, seed=20261028,
        identity_source='synthetic_fixture_not_extractor') for bits in [8,4,2]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--lab-root', required=True, type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--tiny', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if config.get('schema_version') != '1':
        raise ValueError('unsupported quantization schema')
    recipes = expand_inventory({})['quantization_sim']
    if args.dry_run:
        print(canonical_json(dict(recipes=recipes, inputs_state=config['inputs_state'])))
        return 0
    if args.tiny:
        recipes = _tiny(args.lab_root)
    else:
        if config.get('inputs_state') != 'frozen':
            raise ValueError('real quantization blocked: frozen local parent inputs are required')
        prepared = config.get('prepared_inputs', {})
        recipes = [{**r, 'inputs':{p:prepared[p] for p in r['parents']},
                    'parent_identities':{p:prepared[p]['model_key'] for p in r['parents']}} for r in recipes]
        for recipe in recipes:
            checked_source(recipe, {'root':args.lab_root})
    completed = []
    for recipe in recipes:
        target = write_checkpoint(recipe['model_id'], args.lab_root, recipe,
            lambda stage, row=recipe: run_quantization(row, {'root':args.lab_root}, stage), resume=args.resume)
        if args.tiny:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(target, local_files_only=True,
                trust_remote_code=False, use_safetensors=True)
            if model.model.embed_tokens.weight.data_ptr() != model.lm_head.weight.data_ptr():
                raise ValueError('tiny reload lost tied embeddings')
        completed.append(recipe['model_id'])
    print(canonical_json(dict(completed_ids=completed, status='COMPLETE_SYNTHETIC' if args.tiny else 'COMPLETE')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
