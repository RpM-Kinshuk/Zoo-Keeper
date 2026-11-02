"""Offline frozen training launch preparation and fail-closed human preflight.

Preparation and preflight never train. Only the explicit --execute flag,
reserved for a human launch, calls the existing training writers.
"""
import argparse
import hashlib
import json
from pathlib import Path

from .artifacts import content_hash
from .pins import verify_pin
from .schema import contained_path

PARTITIONS = ('training', 'direction', 'validation', 'scoring')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_files(root, hashes):
    for relative, expected in hashes.items():
        path = contained_path(root, relative, 'frozen evidence')
        if path.is_symlink() or not path.is_file():
            raise ValueError(relative + ': missing regular frozen file')
        if file_hash(path) != expected:
            raise ValueError(relative + ': hash mismatch')


def check_license(root, pin):
    check_files(root,{pin['license_path']:pin['license_sha256']})
    if pin.get('license_evidence_type') == 'pinned_readme_declaration':
        import yaml
        text=contained_path(root,pin['license_path'],'license').read_text()
        parts=text.split('---',2)
        if len(parts)!=3 or parts[0].strip():raise ValueError('missing pinned license declaration')
        declaration=yaml.safe_load(parts[1]).get('license')
        if isinstance(declaration,list) and len(declaration)==1:declaration=declaration[0]
        if declaration!=pin['license']:raise ValueError('pinned license declaration differs from claimed license')


def split_ids(selections):
    if set(selections) != set(PARTITIONS):
        raise ValueError('all four frozen partitions are required')
    seen = set()
    for name in PARTITIONS:
        ids = selections[name]
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError('partition IDs overlap or contain duplicates')
        seen.update(ids)
    return selections


def causal_row(tokenizer, messages, length):
    # Explicit template avoids silently changing an upstream chat template.
    text = ''.join(m['role'] + ': ' + m['content'] + '\n' for m in messages)
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[:length] if len(ids) >= length else None


def preference_row(tokenizer, prompt, chosen, rejected, length):
    prefix = tokenizer.encode(prompt, add_special_tokens=False)[:length // 2]
    if not prefix:
        return None
    completions = [tokenizer.encode(text, add_special_tokens=False) for text in (chosen, rejected)]
    remaining = length - len(prefix)
    if any(len(ids) < remaining for ids in completions):
        return None
    mask = [0] * len(prefix) + [1] * remaining
    return prefix + completions[0][:remaining], prefix + completions[1][:remaining], mask, mask[:]


def preflight(config, root, *, load_model=False):
    """Validate frozen evidence, source bytes, tensor contracts; never optimize."""
    import torch
    from safetensors.torch import load_file
    from .training import validate_recipe, validate_full_recipe, validate_decision_recipe
    root = Path(root)
    check_files(root, config['evidence_hashes'])
    recipe, inputs, edge = config['recipe'], config['inputs'], config['trainer']
    if edge == 'lora':
        recipe = validate_recipe(recipe)
    elif edge == 'decision':
        recipe = validate_decision_recipe(recipe)
    else:
        recipe = validate_full_recipe(recipe, edge)
    verify_pin(inputs['model_pin'])
    verify_pin(inputs['data_pin'])
    for name in ('model_pin', 'data_pin'):
        pin = inputs[name]
        check_license(root,pin)
    partitions = split_ids(config['selections'])
    if not partitions['training'] or inputs['sample_ids'] != partitions['training']:
        raise ValueError('training IDs differ from frozen selections')
    heldout = sum((partitions[name] for name in PARTITIONS[1:]), [])
    if inputs['heldout_ids'] != heldout:
        raise ValueError('heldout IDs differ from frozen selections')
    if content_hash(inputs['sample_ids']) != inputs['data_pin']['selected_index_sha256']:
        raise ValueError('selected index hash mismatch')
    model_path = contained_path(root, inputs['model_path'], 'model')
    for path in model_path.rglob('*'):
        if path.is_symlink():
            raise ValueError('model symlinks are forbidden')
    pin = inputs.get('local_parent', inputs['model_pin'])
    if {p.relative_to(model_path).as_posix() for p in model_path.rglob('*.safetensors')} != set(pin['weight_sha256']):
        raise ValueError('model weight file set differs from pinned provenance')
    if edge == 'dpo' and not inputs.get('local_parent'):
        raise ValueError('DPO blocked until matching completed SFT parent is resolved')
    check_files(model_path, {'config.json': pin['config_file_sha256'], **pin['weight_sha256']})
    if inputs.get('local_parent'):
        local=dict(pin);key=local.pop('model_key')
        if key!='lab/'+recipe['parents'][0]+'@local:'+content_hash(local):raise ValueError('local parent content identity mismatch')
        check_files(model_path,{name+'.json':pin[name+'_sha256'] for name in ('recipe','lineage','training')})
    if content_hash(json.loads((model_path/'config.json').read_text())) != pin['config_sha256']:
        raise ValueError('model canonical config hash mismatch')
    check_files(root, {inputs['tokens_path']: inputs['tokens_sha256']})
    data = load_file(str(contained_path(root, inputs['tokens_path'], 'tokens')))
    ids = data['input_ids']
    expected = (len(inputs['sample_ids']), recipe['sequence_length'])
    vocab = json.loads((model_path/'config.json').read_text())['vocab_size']
    if ids.dtype != torch.long or tuple(ids.shape) != expected or ids.min() < 0 or ids.max() >= vocab:
        raise ValueError('invalid frozen token shape/dtype/vocabulary')
    preprocessing = inputs['data_pin']['preprocessing']
    if edge in {'sft', 'lora'}:
        if any(preprocessing.get(k) != v for k,v in dict(format='causal_tokens_v1',loss='all_tokens',padding='none').items()):
            raise ValueError('invalid causal preprocessing')
    elif edge == 'dpo':
        if any(preprocessing.get(k) != v for k,v in dict(format='preference_tokens_v1',loss='completion_only',padding='none').items()):
            raise ValueError('invalid preference preprocessing')
        rejected = data['rejected_input_ids']
        if rejected.dtype != torch.long or rejected.shape != ids.shape or rejected.min()<0 or rejected.max()>=vocab:
            raise ValueError('invalid rejected tokens')
        for row in range(len(ids)):
            starts=[]
            for name in ('chosen_mask','rejected_mask'):
                mask=data[name][row]
                if mask.shape != ids[row].shape or not torch.all((mask==0)|(mask==1)) or mask[0] != 0 or mask.sum() < 1:
                    raise ValueError('invalid completion mask')
                start=int(torch.nonzero(mask)[0]);starts.append(start)
                if not torch.all(mask[start:]==1):raise ValueError('noncontiguous completion mask')
            if starts[0]!=starts[1] or not torch.equal(ids[row,:starts[0]],rejected[row,:starts[1]]):
                raise ValueError('different preference prompts')
    else:
        expected_pre = dict(format='classification_tokens_v1',padding='right',label_names=recipe['label_names'],pad_token_id=recipe['pad_token_id'])
        if preprocessing != expected_pre:raise ValueError('invalid classification preprocessing')
        labels,mask=data['labels'],data['attention_mask']
        if labels.dtype!=torch.long or labels.shape!=(len(ids),) or not torch.all((labels==0)|(labels==1)):
            raise ValueError('invalid classification labels')
        if mask.shape!=ids.shape or not torch.all((mask==0)|(mask==1)) or not torch.all(mask[:,0]==1) or torch.any(mask[:,1:]>mask[:,:-1]) or torch.any(ids[mask==0]!=recipe['pad_token_id']) or torch.any(ids[mask==1]==recipe['pad_token_id']):
            raise ValueError('invalid classification padding/mask')
    if load_model:
        from .training import _frozen_inputs, _full_inputs
        runtime = {**inputs,'root':root}
        if edge in {'sft','dpo'}:
            model,*_ = _full_inputs(recipe,runtime,edge)
        elif edge=='decision':
            from transformers import AutoModelForSequenceClassification
            model,_=_frozen_inputs(recipe,runtime,model_loader=AutoModelForSequenceClassification,
                model_kwargs=dict(num_labels=2,pad_token_id=recipe['pad_token_id']))
        else:
            model,_=_frozen_inputs(recipe,runtime)
        if edge=='lora':
            from .training import TARGETS
            targets=recipe.get('target_modules',TARGETS.get(model.config.model_type))
            modules=dict(model.named_modules())
            if not targets or any(t not in TARGETS.get(model.config.model_type,[]) for t in targets):raise ValueError('unsupported LoRA targets')
            for target in targets:
                matching=[m for n,m in modules.items() if n.endswith('.'+target)]
                if not matching or any(not isinstance(m,torch.nn.Linear) for m in matching):raise ValueError('unsupported LoRA target modules')
        if edge=='decision' and (model.config.model_type not in {'llama','qwen2','qwen3','modernbert'} or sum(p.numel() for p in model.parameters())>1_700_000_000):raise ValueError('unsupported classification architecture/size')
        del model
    return dict(status='READY_FOR_HUMAN',model_id=recipe['model_id'],trainer=edge,
                training_rows=len(ids),optimizer_started=False,model_loaded=load_model)


def resolve_sft_parent(config, root):
    """Bind an existing, completed matching SFT output; never guess its identity."""
    import copy
    from .artifacts import _lineage
    config = copy.deepcopy(config)
    if config['trainer'] != 'dpo':
        raise ValueError('parent resolution is only valid for DPO')
    parent = config['recipe']['parents'][0]
    path = contained_path(root, 'lab/' + parent, parent)
    try:
        provenance=json.loads((path/'recipe.json').read_text())
        lineage=json.loads((path/'lineage.json').read_text())
        training=json.loads((path/'training.json').read_text())
        source=provenance['recipe']
        if (source['edge_type']!='sft' or provenance['model_id']!=parent
                or provenance['config_hash']!=content_hash(source)
                or lineage!=_lineage(parent,source)
                or training['steps_completed']!=source['steps']
                or training['steps_completed']<1 or training['full_parameters'] is not True
                or source['inputs']['model_pin']!=config['inputs']['model_pin']):
            raise ValueError('invalid SFT provenance')
        weights={p.relative_to(path).as_posix():file_hash(p) for p in path.rglob('*.safetensors')}
        if not weights:raise ValueError('missing SFT weights')
    except (OSError,KeyError,TypeError,ValueError) as error:
        raise ValueError('DPO requires its actual completed SFT output with valid provenance') from error
    pin=dict(config_sha256=content_hash(json.loads((path/'config.json').read_text())),
        config_file_sha256=file_hash(path/'config.json'),weight_sha256=weights,
        recipe_sha256=file_hash(path/'recipe.json'),lineage_sha256=file_hash(path/'lineage.json'),
        training_sha256=file_hash(path/'training.json'))
    pin['model_key']='lab/'+parent+'@local:'+content_hash(pin)
    config['inputs'].update(model_path='lab/'+parent,local_parent=pin)
    config['recipe']['parent_identities']={parent:pin['model_key']}
    return config


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--expected-config-sha256')
    parser.add_argument('--load-model',action='store_true')
    parser.add_argument('--resolve-parent-output',type=Path)
    parser.add_argument('--execute',action='store_true',help='HUMAN only: run the frozen training recipe')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    if args.expected_config_sha256 and file_hash(args.config)!=args.expected_config_sha256:
        raise ValueError('frozen launch config hash mismatch')
    config=json.loads(args.config.read_text())
    if args.resolve_parent_output:
        config=resolve_sft_parent(config,args.data_root)
        preflight(config,args.data_root,load_model=True)
        if args.resolve_parent_output.exists():raise FileExistsError('resolved launch config already exists')
        args.resolve_parent_output.write_text(json.dumps(config,sort_keys=True,indent=2)+'\n')
        args.resolve_parent_output.with_suffix('.json.sha256').write_text(file_hash(args.resolve_parent_output)+'  '+args.resolve_parent_output.name+'\n')
        print(json.dumps({'status':'RESOLVED_ACTUAL_SFT_PARENT','config_sha256':file_hash(args.resolve_parent_output),'model_key':config['inputs']['local_parent']['model_key']}))
        return
    result=preflight(config,args.data_root,load_model=args.load_model or args.execute)
    if args.execute:
        from .artifacts import write_checkpoint
        from .training import train_lora,train_sft,train_dpo,train_decision
        trainer={'lora':train_lora,'sft':train_sft,'dpo':train_dpo,'decision':train_decision}[config['trainer']]
        recipe={**config['recipe'],'inputs':config['inputs']}
        runtime={**config['inputs'],'root':args.data_root}
        path=write_checkpoint(recipe['model_id'],args.data_root/'lab',recipe,
            lambda stage:trainer(recipe,runtime,stage),resume=args.resume)
        result.update(status='COMPLETE',output=str(path),optimizer_started=True)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
