import json
import numpy as np
import torch
from safetensors.torch import save_file


def fixture(tmp_path):
    from modeldna.io.source import open_source
    from modeldna.io.weights import WeightIndex
    from evidence.safetensors_header import header_length,parse_header
    name='model.layers.0.self_attn.q_proj.weight'
    scale_name=name+'_scale_inv'
    codes=torch.linspace(-448,448,256*256).reshape(256,256).to(torch.float8_e4m3fn)
    scales=torch.tensor([[.001,.125],[.25,.5]],dtype=torch.bfloat16)
    save_file({name:codes,scale_name:scales},str(tmp_path/'model.safetensors'))
    config=dict(model_type='qwen3',hidden_size=256,intermediate_size=256,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,torch_dtype='float32',quantization_config=dict(weight_block_size=[128,128]))
    (tmp_path/'config.json').write_text(json.dumps(config))
    with (tmp_path/'model.safetensors').open('rb') as f:
        first=f.read(8);n=header_length(first);entries,_=parse_header(first+f.read(n),'model.safetensors')
    source=open_source(str(tmp_path)); index=WeightIndex(source)
    expected=codes.float()*scales.float().repeat_interleave(128,0).repeat_interleave(128,1)
    return index,entries,config,expected,name


def test_adapter_preserves_native_sample_and_slice_positions(tmp_path):
    from studies.baselines.fp8_input import DecodedFP8Index
    from modeldna.io.weights import _lcg_offsets
    index,entries,config,expected,name=fixture(tmp_path)
    decoded=DecodedFP8Index(index,entries,config)
    np.testing.assert_array_equal(decoded.read_flat_slice(name,127*256+193,1000),expected.flatten()[127*256+193:127*256+1193].numpy())
    offsets=_lcg_offsets(expected.numel(),5,111,20260704)
    wanted=np.concatenate([expected.flatten()[i:i+111].numpy() for i in offsets])
    np.testing.assert_array_equal(decoded.read_sample(name,20260704,5,111),wanted)
    assert not np.array_equal(index.read_sample(name,20260704,5,111),wanted)


def test_adapted_native_fingerprint_equals_genuinely_decoded_dense_input(tmp_path):
    from studies.baselines.fp8_input import decoded_extractor
    from modeldna.fingerprint.extract import FingerprintExtractor
    from modeldna.io.source import open_source
    torch.set_num_threads(1)
    encoded=tmp_path/'encoded';encoded.mkdir()
    index,entries,config,expected,name=fixture(encoded)
    dense=tmp_path/'dense';dense.mkdir();save_file({name:expected},str(dense/'model.safetensors'))
    (dense/'config.json').write_text(json.dumps(config))
    a=decoded_extractor(open_source(str(encoded)),entries,config,mode='fast')
    left=a.run();right=FingerprintExtractor(open_source(str(dense)),mode='fast').run()
    assert left.arch.n_params==right.arch.n_params
    for field in ['sigma_curves','vector_norms','vector_samples','pcs_samples','spectra_sketch','spectra_exact']:
        assert getattr(left,field)==getattr(right,field),field
