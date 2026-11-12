"""Offline guards for metadata-only scaling preparation."""
import copy
import pytest


def module():
    from scripts import scaling_pools
    return scaling_pools


def manifest():
    return dict(config=dict(model_type='qwen3',hidden_size=1024,num_hidden_layers=28),
                access='granted',is_adapter=False,view_counts={'matrix':1},params_by_dtype={'BF16':64},
                files=[dict(filename='model.safetensors',sha256='a'*64,size=128)],
                tensors=[dict(name='model.layers.0.mlp.down_proj.weight',shape=[4,8]),
                         dict(name='model.embed_tokens.weight',shape=[100,4])])


def test_inventory_identity_ignores_filenames_but_rejects_unknown_hashes():
    m=module();a=manifest();b=copy.deepcopy(a);b['files'][0]['filename']='renamed.safetensors'
    assert m.inventory_identity(a)==m.inventory_identity(b)
    b['files'][0]['sha256']=None
    with pytest.raises(ValueError):m.inventory_identity(b)
    b['files'][0]['sha256']='a'*40
    with pytest.raises(ValueError):m.inventory_identity(b)


def test_config_match_cannot_hide_changed_tensor_geometry_or_quantization():
    m=module();a=manifest();b=copy.deepcopy(a)
    assert m.compatibility_reason(a,b) is None
    b['tensors'][0]['shape']=[4,9]
    assert m.compatibility_reason(b,a)=='tensor_shapes'
    b=copy.deepcopy(a);b['config']['quantization_config']={'quant_method':'gptq'}
    assert m.compatibility_reason(b,a)=='quantization'
    b=copy.deepcopy(a);b['tensors'][1]['shape']=[101,4]
    assert m.compatibility_reason(b,a) is None


def test_saved_completed_identity_is_not_repinned_on_resume(tmp_path):
    m=module();path=tmp_path/'outcome.json'
    calls=[]
    def measure():
        calls.append(1)
        return dict(repo_id='org/model',revision='a'*40,status='metadata_error',error_class='Denied')
    first=m.resume_outcome(path,'org/model',measure)
    second=m.resume_outcome(path,'org/model',lambda:pytest.fail('must not repin completed input'))
    assert first==second and len(calls)==1
    with pytest.raises(ValueError):m.resume_outcome(path,'other/model',measure)


def test_header_reader_rejects_payload_before_transport():
    m=module()
    from evidence.hub import RemoteFile
    calls=[]
    reader=m.header_reader(token='offline-fixture',transport=lambda *args:calls.append(args))
    remote=RemoteFile('org/model','model.safetensors','a'*40,'https://example.com',100,'b'*64,None)
    with pytest.raises(ValueError):reader.read(remote,40,50)
    assert not calls


def test_transport_rate_limit_bubbles_without_retry_and_saved_limit_does_not_recur(tmp_path,monkeypatch):
    monkeypatch.setenv('HF_HUB_OFFLINE','0')
    m=module()
    from evidence.hub import RemoteFile
    calls=[]
    def transport(*args):
        calls.append(1)
        return 429,{},b''
    reader=m.header_reader(token='offline-fixture',transport=transport)
    remote=RemoteFile('org/model','config.json','a'*40,'https://example.com',100,None,None)
    with pytest.raises(m.RateLimited):reader.read(remote,0,10)
    assert len(calls)==1
    path=tmp_path/'saved.json'
    m.resume_outcome(path,'org/model',lambda:dict(repo_id='org/model',status='metadata_error',http_status=429))
    marker=[]
    m.resume_outcome(path,'org/model',lambda:marker.append('new_request'))
    assert not marker


def test_live_tuple_geometry_matches_cached_json_list_geometry():
    m=module();a=manifest();b=copy.deepcopy(a)
    for tensor in b['tensors']:tensor['shape']=tuple(tensor['shape'])
    assert m.compatibility_reason(b,a) is None
