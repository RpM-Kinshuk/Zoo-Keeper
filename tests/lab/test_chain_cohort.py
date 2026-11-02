"""Scoped chains must retain completed intermediates and inherited measurements."""
import copy

import pytest

from lab.scripts.chain_cohort import scoped_tables, verify_chain_ready


def rows():
    base=dict(model_id='smol-135',parents=[],kind='source',complete=True,path='lab/inputs/base',
              repo_id='org/base',revision='a'*40,license='apache-2.0')
    outputs=[]
    for seed in (0,1):
        sft=f'smol-135-gpu-v1-sft-s{seed}';dpo=f'smol-135-gpu-v1-dpo-s{seed}'
        for name,parent,edge in [(sft,'smol-135','sft'),(dpo,sft,'dpo')]:
            outputs.append(dict(model_id=name,parents=[parent],kind='derived',complete=True,path='lab/'+name,
                                repo_id=None,revision=None,license='apache-2.0',edge_type=edge))
        for bits in ([4,8] if seed==0 else [4]):
            name=f'chain-v1-smol-135-dpo-s{seed}-q{bits}'
            outputs.append(dict(model_id=name,parents=[dpo],kind='derived',complete=True,path='lab/'+name,
                                repo_id=None,revision=None,license='apache-2.0',edge_type='quantization_sim'))
    return {'sources':[base],'outputs':outputs}


def test_three_scoped_depth_three_paths_never_earn_formal_credit():
    report=verify_chain_ready(rows())
    assert len(report['chain_paths'])==3
    assert {len(path)-1 for path in report['chain_paths']}=={3}
    assert report['formal_chain_credit']==0
    models,edges=scoped_tables(rows())
    assert len(models)==8 and len(edges)==7


def test_missing_or_incomplete_intermediate_rejects_chain_count():
    ready=rows();ready['outputs']=[r for r in ready['outputs'] if r['model_id']!='smol-135-gpu-v1-sft-s0']
    with pytest.raises(ValueError,match='intermediate'):verify_chain_ready(ready)
    ready=copy.deepcopy(rows());ready['outputs'][0]['complete']=False
    with pytest.raises(ValueError,match='intermediate'):verify_chain_ready(ready)
