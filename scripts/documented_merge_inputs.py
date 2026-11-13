"""Pin one metadata-selected small merge; never reads tensor payloads."""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'Zoo-Keeper'),str(ROOT/'Model-Zoo')]
from scripts.scaling_pools import header_reader,PinnedInfo
from evidence.hub import HubReader
from evidence.manifest import build_manifest,write_manifest
from huggingface_hub import HfApi
from lab.artifacts import atomic_json
def pin_case(requests,out):
    out=out.resolve()
    assert not (out/'inputs.json').exists(),'preserve pinned inputs'
    token=next((os.environ[k] for k in ['HF_TOKEN','HF_UPLOAD','HF_UPLOAD_TOKEN'] if os.environ.get(k)),None);assert token
    api=HfApi(token=token);reader=header_reader(token=token);textreader=HubReader(token=token)
    rows=[]
    for role,repo,revision in requests:
     info=api.model_info(repo,revision=revision,files_metadata=True,timeout=15)
     manifest=build_manifest(repo,info.sha,reader,PinnedInfo(info));path=write_manifest(manifest,out/'manifests')
     row=dict(role=role,repo_id=repo,revision=info.sha,manifest=str(path.relative_to(ROOT)),manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),sources=[])
     names={s.rfilename for s in info.siblings}
     for name in ['README.md','config.json','mergekit_config.yml']:
      if name not in names:continue
      remote=textreader.resolve(repo,name,info.sha);assert 0<=remote.size<=1024**2
      raw=textreader.read(remote,0,remote.size)
      target=out/'sources'/role/name;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(raw)
      row['sources'].append(dict(path=str(target.relative_to(ROOT)),sha256=hashlib.sha256(raw).hexdigest(),size=len(raw)))
     rows.append(row)
    atomic_json(out/'inputs.json',dict(records=rows,header_reader_stats=reader.stats,text_reader_stats=textreader.stats,
     source_metadata='runs/scaling-pool-discovery-20261110/source-candidates.json',
     selection='Smallest conventional two-parent base/Instruct recipe-bearing metadata candidate; selection preceded payloads/outcomes.',
     input_pins='Base reuses actual pinned lab identity; query/Instruct snapshots pinned now; historical construction revisions remain unknown unless recipe records them.',weight_payload_reads=0))
    print(json.dumps([dict(role=r['role'],repo_id=r['repo_id'],revision=r['revision']) for r in rows],indent=2))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query',default='vonjack/SmolLM2-360M-Merged')
    parser.add_argument('--out',type=Path,default=ROOT/'runs/documented-merge-20261112')
    parser.add_argument('--family',choices=['smol','qwen'],default='smol')
    args=parser.parse_args();args.out.mkdir(exist_ok=True,parents=True)
    if args.family=='smol':
        parents=[('base','HuggingFaceTB/SmolLM2-360M','f8027fd0eaeea54caa13c31d31b9fdc459c38b49'),('instruct','HuggingFaceTB/SmolLM2-360M-Instruct','main')]
    else:
        pilot=json.loads((ROOT/'runs/scaling-pool-discovery-20261110/pilot-results.json').read_text())
        parents=[]
        for role,repo in [('base','Qwen/Qwen2.5-0.5B'),('instruct','Qwen/Qwen2.5-0.5B-Instruct')]:
            previous=next(r for r in pilot['rows'] if r['repo_id']==repo)
            parents.append((role,repo,previous['revision']))
    pin_case([('query',args.query,'main')]+parents,args.out)
