import json
import numpy as np
import pytest


def test_every_transport_attempt_checks_global_stop_before_reserving_bytes():
    from scripts.real_comparison import guarded_transport
    attempts=[]
    class Budget:
        def transport(self,*args):attempts.append(args);return 'response'
    stopped=False
    def admit():
        if stopped:raise RuntimeError('stopped')
    transport=guarded_transport(admit,Budget())
    assert transport('HEAD','fixture',{})=='response'
    stopped=True
    for method in ['HEAD','GET','GET']:
        with pytest.raises(RuntimeError,match='stopped'):transport(method,'fixture',{})
    assert len(attempts)==1


def fixture():
    rng=np.random.default_rng(20261028)
    roles=['self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight',
           'self_attn.o_proj.weight','mlp.down_proj.weight','mlp.up_proj.weight','mlp.gate_proj.weight']
    record=dict(layers=2,hidden=3,trace_roles=roles)
    tensors={}
    for layer in range(2):
        for role in roles:
            shape=(3,5) if 'down_proj' in role else ((5,3) if 'mlp.' in role else (3,3))
            tensors[f'model.layers.{layer}.{role}']=rng.normal(size=shape)
    return record,tensors


def test_profiles_read_each_complete_matrix_once_and_match_svd_energy():
    from scripts.real_comparison import extract_profile
    from studies.baselines.residual import signature
    record,tensors=fixture();seen=[]
    def fetch(names):
        seen.extend(names);return [tensors[n] for n in names]
    profile=extract_profile(record,fetch)
    assert len(seen)==len(set(seen))==14
    for role,energy in profile['trace'].items():
        for layer,value in enumerate(energy):
            wanted=np.linalg.norm(np.linalg.svd(tensors[f'model.layers.{layer}.{role}'],compute_uv=False))
            np.testing.assert_allclose(value,wanted,rtol=1e-12)
    vec,conc=signature(tensors['model.layers.0.mlp.down_proj.weight'],tensors['model.layers.0.mlp.up_proj.weight'])
    np.testing.assert_array_equal(profile['vectors'][0],vec)
    assert profile['concentrations'][0]==conc


def test_incomplete_or_nonfinite_matrix_aborts_instead_of_dropping_a_role():
    from scripts.real_comparison import extract_profile
    record,tensors=fixture();tensors['model.layers.0.self_attn.q_proj.weight'][0,0]=np.nan
    with pytest.raises(ValueError):extract_profile(record,lambda ns:[tensors[n] for n in ns])
    del tensors['model.layers.0.mlp.down_proj.weight']
    with pytest.raises(KeyError):extract_profile(record,lambda ns:[tensors[n] for n in ns])


def test_tlt_query_first_orientation_and_all_pool_ties_are_preserved():
    from scripts.real_comparison import score_panel
    vectors=np.eye(2)
    def p(c):return dict(trace={'role':[1.,2.]},vectors=vectors,concentrations=np.array([c,c]))
    profiles={'query':p(.1),'relative1':p(.01),'relative2':p(.01)}
    case=dict(query_key='query',pool_settings={'present':['relative1','relative2'],'removed':['relative2']})
    out=score_panel(case,profiles)
    assert [r['score'] for r in out['pairs'] if r['method']=='TLT']==pytest.approx([.1,.1])
    assert out['pools']['present']['TLT']['top']==['relative1','relative2']
    assert out['pools']['removed']['TLT']['top']==['relative2']
    assert out['pools']['removed']['TLT']['source_decision']=='unsupported'


def test_profile_record_roundtrips_without_pickle_or_numpy_json_values(tmp_path):
    from scripts.real_comparison import extract_profile,save_profile,load_profile
    record,tensors=fixture();profile=extract_profile(record,lambda ns:[tensors[n] for n in ns])
    path=tmp_path/'profile.npz';receipt=save_profile(path,profile)
    json.dumps(receipt,allow_nan=False)
    loaded=load_profile(path)
    np.testing.assert_array_equal(profile['vectors'],loaded['vectors'])
    assert profile['trace']==loaded['trace']
