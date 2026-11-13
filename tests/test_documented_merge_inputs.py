from pathlib import Path
from types import SimpleNamespace


def test_relative_output_is_pinned_under_data_root(tmp_path,monkeypatch):
    from scripts import documented_merge_inputs as module
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('HF_TOKEN','offline-fixture')
    class API:
        def model_info(self,repo,**kwargs):
            return SimpleNamespace(id=repo,sha='a'*40,siblings=[])
    monkeypatch.setattr(module,'HfApi',lambda **kwargs:API())
    monkeypatch.setattr(module,'build_manifest',lambda repo,pin,*args:dict(repo_id=repo,revision=pin))
    out=Path('relative');out.mkdir()
    module.pin_case([('query','org/model','main')],out)
    import json
    value=json.loads((out/'inputs.json').read_text())
    assert value['records'][0]['revision']=='a'*40
    manifest=tmp_path/value['records'][0]['manifest']
    assert manifest.is_file() and manifest.is_relative_to(tmp_path)
