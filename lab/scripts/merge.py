"""Plan 30 portable recipes or run four offline synthetic merge checks."""
import argparse
import hashlib
import json
from pathlib import Path

from lab.artifacts import atomic_json, content_hash, write_checkpoint
from lab.merges import _checked_inputs, prepare_recipe, run_merge
from lab.registry import expand_inventory
from lab.schema import canonical_json, contained_path


def _tiny(root):
    import torch
    from safetensors.torch import save_file
    torch.set_num_threads(1)
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(20261028)
    model = LlamaForCausalLM(LlamaConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, vocab_size=16, max_position_embeddings=32))
    model.config._name_or_path = ''
    model.config.architectures = ['LlamaForCausalLM']
    cfg = model.config.to_dict()
    generator = torch.Generator().manual_seed(20261028)
    names = ['tiny-merge-p1','tiny-merge-p2','tiny-merge-p3','tiny-merge-p4','tiny-merge-base']
    inputs, identities = {}, {}
    for name in names:
        state = {key:(tensor + 0.01*torch.randn(tensor.shape,generator=generator)).contiguous()
                 for key,tensor in model.state_dict().items()}
        # These identities label synthetic fixtures only; real local parents must
        # supply their extractor model_key and pinned file hashes.
        identity = 'lab/'+name+'@local:'+content_hash({'fixture':name,'seed':20261028})
        source = dict(model_id=name, edge_type='lora', parents=['tiny-merge-root'],
                      parent_identities={'tiny-merge-root':'lab/tiny-merge-root@local:'+'0'*64},
                      seed=20261028, fixture='two-layer-random-llama-v1')
        def writer(stage, s=state, c=cfg):
            save_file(s, str(stage/'model.safetensors'))
            atomic_json(stage/'config.json', c)
        path = write_checkpoint(name, root, source, writer, resume=True)
        inputs[name] = {'path':name, 'model_key':identity, 'config_sha256':content_hash(cfg),
            'config_file_sha256':hashlib.sha256((path/'config.json').read_bytes()).hexdigest(),
            'weight_sha256':{'model.safetensors':hashlib.sha256((path/'model.safetensors').read_bytes()).hexdigest()}}
        identities[name] = identity
    recipes = []
    for method in ['linear','slerp','ties','dare_ties']:
        count = {'linear':2,'slerp':2,'ties':4,'dare_ties':3}[method]
        selected = names[:count]+[names[-1]]
        row = {'model_id':'tiny-merge-'+method, 'edge_type':'merge', 'anchor':names[-1],
            'method':method, 'parents':names[:count], 'reference':names[-1], 'seed':20261028,
            'synthetic_fixture':True, 'execution_backend':'mergekit_yaml', 'identity_source':'synthetic_fixture_not_extractor',
            'inputs':{n:inputs[n] for n in selected}, 'parent_identities':{n:identities[n] for n in selected}}
        if method == 'slerp':
            row['t'] = [0.2,0.8]
        else:
            row['configured_weights'] = [1/count]*count
        if method in {'ties','dare_ties'}:
            row.update(density=0.5 if method == 'ties' else 0.8, sparsification_seed=20261028)
        recipes.append(prepare_recipe(row))
    return recipes


def main(argv=None):
    parser = argparse.ArgumentParser(description='Mergekit lab recipes (complete real batch: HUMAN launch)')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--lab-root', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--tiny', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--model-id')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if config.get('schema_version') != '1':
        raise ValueError('unsupported merge schema')
    recipes = [prepare_recipe(row) for row in expand_inventory({})['merge']]
    if args.model_id:
        recipes = [r for r in recipes if r['model_id'] == args.model_id]
        if not recipes:
            raise ValueError('unknown merge model ID')
    if args.dry_run:
        print(canonical_json({'recipes':recipes,'launch':'HUMAN','inputs_state':config['inputs_state']}))
        return 0
    if args.tiny:
        recipes = _tiny(args.lab_root)
    else:
        if config.get('inputs_state') != 'frozen':
            raise ValueError('real merges blocked: trained parents and extractor manifests must be frozen')
        prepared = config.get('prepared_inputs', {})
        frozen = []
        for recipe in recipes:
            if recipe['model_id'] not in prepared:
                raise ValueError(recipe['model_id']+': missing frozen parent inputs')
            inputs = prepared[recipe['model_id']]
            frozen.append(prepare_recipe({**recipe, 'inputs':inputs,
                'parent_identities':{name:pin['model_key'] for name,pin in inputs.items()}}))
        recipes = frozen
        # Preflight all selected parents before writing the first output.
        for recipe in recipes:
            _checked_inputs(recipe, args.lab_root)
    completed = []
    for recipe in recipes:
        write_checkpoint(recipe['model_id'], args.lab_root, recipe,
            lambda stage, r=recipe: run_merge(r, args.lab_root, stage), resume=args.resume)
        completed.append(recipe['model_id'])
    print(canonical_json({'completed_ids':completed,'status':'COMPLETE_SYNTHETIC' if args.tiny else 'COMPLETE'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
