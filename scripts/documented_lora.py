"""One public-factor compatibility experiment; --score-only reuses saved bands.

Assumptions, controls and stopping rule live in the run's short RESEARCH.md.
Readers/LoRA planning/native comparison are existing implementations.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
ROLES = ('instruct', 'base', 'sft', 'lab_lora')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def check_band(parent, child, a=None, b=None):
    """Necessary conventional-FP32 merge condition, with outward FP64 endpoints."""
    import torch
    import math
    values = [parent, child] + ([] if a is None and b is None else [a, b])
    if any(t is None or t.dtype != torch.float32 or not torch.isfinite(t).all() for t in values):
        return dict(supported=False, reason='nonfinite_or_non_FP32_operands')
    if parent.ndim != 2 or parent.shape != child.shape:
        return dict(supported=False, reason='incompatible_shape')
    c, y = parent.double(), child.double()
    if a is None:
        reference = c
        violations = int((parent.view(torch.int32) != child.view(torch.int32)).sum())
        point = parent
    else:
        if a.ndim != 2 or b.ndim != 2 or a.shape != (16, c.shape[1]) or b.shape != (c.shape[0], 16):
            return dict(supported=False, reason='unsupported_factor_shape_or_rank')
        if any(not (((t == 0) | ((t.abs() >= 2.**-40) & (t.abs() <= 2.**40))).all()) for t in [parent, a, b]):
            return dict(supported=False, reason='unsupported_operand_range')
        n = 18
        gamma = lambda u: n*u/(1-n*u)
        g32, g64 = gamma(Fraction(1, 2**24)), gamma(Fraction(1, 2**53))
        coefficient = math.nextafter(float((g32+g64)/(1-g64)), math.inf)
        reference = c + 2*(b.double() @ a.double())
        magnitude = c.abs() + 2*(b.double().abs() @ a.double().abs())
        error = torch.nextafter(magnitude*coefficient, torch.full_like(c, math.inf))
        lower = torch.nextafter(reference-error, torch.full_like(c, -math.inf))
        upper = torch.nextafter(reference+error, torch.full_like(c, math.inf))
        violations = int(((y < lower) | (y > upper)).sum())
        point = parent + (b @ a)*2
    return dict(supported=True, checked_values=child.numel(), violations=violations,
        exact_mismatches=int((point.contiguous().view(torch.int32) != child.contiguous().view(torch.int32)).sum()),
        before_sse=float(((y-c)**2).sum()), after_sse=float(((y-reference)**2).sum()), query_ss=float((y*y).sum()))


def prepare(out):
    from evidence.fingerprint import _entries
    from evidence.views import plan_views
    from evidence.sketch import row_block_bounds
    from lab.artifacts import atomic_json
    import modeldna
    assert modeldna.__version__ == '0.1.0'
    p = json.loads((out/'inputs.json').read_text())
    assert 'plan' not in p, 'preserve the prospective inputs'
    manifests = {r: json.loads((ROOT/v['manifest']).read_text()) for r, v in p['records'].items()}
    for v in p['records'].values():
        assert sha(ROOT/v['manifest']) == v['sha256']
    am = manifests['adapter']; cfg = am['adapter_config']
    assert cfg['r'] == 16 and cfg['lora_alpha'] == 32 and cfg['bias'] == 'none'
    assert not any(cfg.get(k) for k in ['fan_in_fan_out','use_rslora','use_dora','use_qalora','modules_to_save','alpha_pattern','rank_pattern'])
    av, coverage = plan_views(_entries(am), am['config'], cfg)
    adapted = {v.name: v for v in av if v.geometry == 'lora_product'}
    assert len(adapted) == 168 and len(adapted)*2 == len(am['tensors'])
    qv, _ = plan_views(_entries(manifests['query']), manifests['query']['config'])
    plan=[]
    for view in qv:
        assert view.decode == 'plain' and len(view.sources) == 1
        first, last = row_block_bounds(view.matrix_shape[0])
        v = adapted.get(view.name)
        if v is not None:
            assert v.scale == 2 and v.matrix_shape == view.matrix_shape
        plan.append(dict(name=view.name, first=first, last=last,
                         factors=list(v.sources) if v else []))
        for role, m in manifests.items():
            if role != 'adapter':
                entry = _entries(m)[view.name]
                assert list(entry.shape) == list(view.matrix_shape) and entry.dtype == ('F32' if role == 'query' else 'BF16')
    assert len(plan) == 169 and sum(bool(v['factors']) for v in plan) == 168
    p['plan'] = plan
    p['native'] = {r: dict(path=path, sha256=sha(ROOT/path)) for r, path in {
        'base':'runs/documented-slerp-20261112/native-base.json',
        'instruct':'runs/documented-slerp-20261112/native-instruct.json',
        'sft':'runs/track-c-k2-modeldna-20261108/q25-05-sft-s0.json',
        'lab_lora':'runs/track-c-k2-modeldna-20261108/q25-05-lora-dialogue-r8.json'}.items()}
    from modeldna.fingerprint.extract import Fingerprint
    for role, value in p['native'].items():
        fp=Fingerprint.from_dict(json.loads((ROOT/value['path']).read_text()))
        assert fp.seed == 20260704 and fp.mode == 'fast'
        if role in ['base','instruct']:assert fp.model_id==p['records'][role]['key'] and fp.revision==p['records'][role]['key'].rsplit('@',1)[1]
        else:assert fp.model_id==Path(p['records'][role]['local_dir']).name and fp.revision=='local-pinned'
    ledger_path=ROOT/'runs/track-c-k2-modeldna-20261108/inputs.json'
    ledger=json.loads(ledger_path.read_text())['run_inputs']
    for role in ['sft','lab_lora']:
        for f in manifests[role]['files']:
            assert ledger[p['records'][role]['local_dir']+'/'+f['filename']]==f['sha256']
    installed=ROOT/'runs/track-c-k2-remaining-20261108/modeldna/installed-package.json'
    package=Path(modeldna.__file__).parent.parent
    for name, expected in json.loads(installed.read_text())['source_file_sha256'].items():
        assert sha(package/name)==expected, name
    sources=[Path(__file__),ROOT/'Zoo-Keeper/tests/test_documented_lora.py',installed,ledger_path]
    sources += [ROOT/'Model-Zoo/evidence'/name for name in ['fingerprint.py','views.py','hub.py','safetensors_header.py','sketch.py']]
    sources += [ROOT/'Zoo-Keeper/studies/baselines/native_source.py',ROOT/'Zoo-Keeper/scripts/task_choice_download.py']
    p['source_sha256']={str(f.relative_to(ROOT)):sha(f) for f in sources}
    p['protocol_sha256']=hashlib.sha256((out/'RESEARCH.md').read_text().split('## Outcome')[0].encode()).hexdigest()
    p['budgets']=dict(seconds=6600,response_reservation=2_000_000_000,weight_identities=6,gpu_devices=[])
    atomic_json(out/'inputs.json',p)
    print('prepared169bands/168public-factor edits; no payloads',flush=True)


def score(out, inputs):
    import torch
    from safetensors import safe_open
    from modeldna.fingerprint.extract import Fingerprint
    from modeldna.compare import compare_fingerprints
    from modeldna.verdict import judge
    totals={r:dict(supported_bands=0,checked_values=0,violations=0,exact_mismatches=0,before_sse=0.,after_sse=0.,query_ss=0.) for r in ROLES}
    bands=[]
    with safe_open(out/'bands.safetensors',framework='pt',device='cpu') as raw:
        for i, spec in enumerate(inputs['plan']):
            prefix=f'{i:03d}.'; y=raw.get_tensor(prefix+'query')
            factors=[raw.get_tensor(prefix+k) for k in ['a','b']] if spec['factors'] else [None,None]
            row=dict(name=spec['name'],candidates={})
            for role in ROLES:
                result=check_band(raw.get_tensor(prefix+role).float(),y,*factors)
                row['candidates'][role]=result
                if result['supported']:
                    totals[role]['supported_bands']+=1
                    for key in totals[role]:
                        if key!='supported_bands':totals[role][key]+=result[key]
            bands.append(row)
    compatible=[r for r,t in totals.items() if t['supported_bands']==len(bands) and t['violations']==0]
    fps={r:Fingerprint.from_dict(json.loads((ROOT/v['path']).read_text())) for r,v in inputs['native'].items()}
    query=Fingerprint.from_dict(json.loads((out/'native-query.json').read_text()))
    pairs={r:compare_fingerprints(query,fp) for r,fp in fps.items()}
    pools={}
    for name, roles in [('present',ROLES),('instruct_removed',ROLES[1:])]:
        verdict=judge([(pairs[r],'Qwen2.5-0.5B') for r in roles],suspect_dtype=query.arch.torch_dtype,
                      candidate_dtypes={fps[r].model_id:fps[r].arch.torch_dtype for r in roles})
        pools[name]=dict(compatible_candidates=[r for r in roles if r in compatible],native=verdict.to_dict())
    return dict(totals=totals,pools=pools,bands=bands)


def run(out):
    import os,signal,socket
    from concurrent.futures import ThreadPoolExecutor
    import torch
    from safetensors.torch import save_file
    from evidence.hub import HubReader
    from evidence.fingerprint import _Sources,_entries
    from evidence.views import plan_views
    from lab.artifacts import atomic_json
    from studies.baselines.native_source import PinnedSource
    from scripts.task_choice_download import ResponseBudget,AcquisitionStopped
    from modeldna.fingerprint.extract import extract_fingerprint,Fingerprint
    started=time.monotonic(); p=json.loads((out/'inputs.json').read_text())
    assert Path(os.environ['ZOO_DATA']).resolve()==ROOT, 'Set ZOO_DATA to this workspace before acquisition'
    assert not (out/'results.json').exists() and not (out/'bands.safetensors').exists()
    def stop(*_):raise AcquisitionStopped('110min experiment deadline')
    seconds=min(6600,int(p.get('execution_resume',{}).get('deadline_unix',time.time()+6600)-time.time()))
    assert seconds>0,'original experiment deadline passed'
    signal.signal(signal.SIGALRM,stop);signal.alarm(seconds)
    for name, expected in p['source_sha256'].items():assert sha(ROOT/name)==expected,name
    assert hashlib.sha256((out/'RESEARCH.md').read_text().split('## Outcome')[0].encode()).hexdigest()==p['protocol_sha256']
    for value in list(p['records'].values())+list(p['native'].values()):
        assert sha(ROOT/value.get('manifest',value.get('path')))==value['sha256']
    token=next(os.environ[k] for k in ['HF_TOKEN','HF_UPLOAD','HF_UPLOAD_TOKEN'] if os.environ.get(k))
    budget=ResponseBudget(used=p.get('execution_resume',{}).get('response_reservation_bytes',0),limit=p['budgets']['response_reservation']); network_bodies=[0]
    def transport(method,url,headers):
        status,response_headers,body=budget.transport(method,url,headers)
        with budget.lock:network_bodies[0]+=len(body)
        return status,response_headers,body
    reader=HubReader(token=token,transport=transport)
    atomic_json(out/'runtime.json',dict(hostname=socket.gethostname(),owned_pid=os.getpid(),gpu_devices=[],stop='kill -TERM owned_pid',status='RUNNING'))
    hash_bytes=0; receipts={}
    for role, record in p['records'].items():
        if 'local_dir' not in record:continue
        manifest=json.loads((ROOT/record['manifest']).read_text())
        for f in manifest['files']:
            path=ROOT/record['local_dir']/f['filename'];assert sha(path)==f['sha256'];hash_bytes+=path.stat().st_size
            receipts[str(path.relative_to(ROOT))]=f['sha256']
    print('verified cached weight files',hash_bytes,flush=True)
    adapter=next(f for f in p['files'] if f['filename'].startswith('adapter/'))
    dest=out/'payload'/adapter['filename'];dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():assert sha(dest)==adapter['sha256']
    else:
        remote=reader.resolve(p['repo_id'],adapter['filename'],p['revision']);assert remote.size==adapter['size'] and remote.sha256==adapter['sha256']
        payload=reader.read(remote,0,remote.size);assert hashlib.sha256(payload).hexdigest()==adapter['sha256'];dest.write_bytes(payload);del payload
    raw={}
    with ThreadPoolExecutor(4) as pool:
        sources={r:_Sources(json.loads((ROOT/v['manifest']).read_text()),reader,pool) for r,v in p['records'].items()}
        sources['adapter'].source=str((out/'payload').resolve())
        for r,v in p['records'].items():
            if 'local_dir' in v:sources[r].source=str((ROOT/v['local_dir']).resolve())
        views={r:{v.name:v for v in plan_views(s.entries,s.manifest.get('config',{}),s.manifest.get('adapter_config'))[0]} for r,s in sources.items()}
        for i,spec in enumerate(p['plan']):
            prefix=f'{i:03d}.';bounds=(spec['first'],spec['last'])
            for role in ('query',)+ROLES:
                _,stored,_=sources[role].block(views[role][spec['name']],bounds)
                raw[prefix+role]=stored.contiguous()
            if spec['factors']:
                factors=sources['adapter'].full(views['adapter'][spec['name']]);a,b=[factors[n] for n in spec['factors']]
                raw[prefix+'a']=a;raw[prefix+'b']=b[bounds[0]:bounds[1]].contiguous()
            if i%24==0:print('acquired bands',i+1,'of',len(p['plan']),flush=True)
        save_file(raw,out/'bands.safetensors');del raw
        print('saved raw bands; native query next',flush=True)
        src=PinnedSource(p['repo_id'],p['revision'],['model.safetensors','config.json','tokenizer.json','tokenizer_config.json'],reader,pool,
                         deadline=started+seconds,byte_cap=512*1024**2)
        fp=extract_fingerprint(src,mode='fast',model_id=p['repo_id']+'@'+p['revision'],revision=p['revision'])
        atomic_json(out/'native-query.json',fp.to_dict())
        source_bytes={r:s.bytes_read for r,s in sources.items()}
    evidence=score(out,p)
    evidence.update(scope='Sampled public-adapter-informed compatibility under the fixed conventional FP32 class; not calibrated historical attribution. Native comparison receives no adapter factors.',
        reader_stats=reader.stats,response_reservation_bytes=budget.reserved,
        network_response_body_bytes=network_bodies[0],
        local_weight_sha_scan_bytes=hash_bytes,source_band_bytes=source_bytes,native_source_bytes=src.bytes_read,
        local_weight_sha256=receipts,input_sha256=sha(out/'inputs.json'),
        raw_sha256={n:sha(out/n) for n in ['bands.safetensors','native-query.json']},adapter_sha256=adapter['sha256'])
    if 'execution_resume' in p:
        prior=p['execution_resume'];evidence['prior_setup']=prior
        evidence['network_response_body_bounds']=[network_bodies[0]+v for v in prior['network_body_bounds']]
        evidence['original_clock_span_seconds']=time.time()-prior['clock_anchor_unix']
        evidence['local_weight_sha_scan_bytes']+=prior['local_sha_scan_bytes']
    evidence['elapsed_seconds']=time.monotonic()-started
    atomic_json(out/'results.json',evidence);signal.alarm(0)
    atomic_json(out/'runtime.json',dict(hostname=socket.gethostname(),owned_pid=os.getpid(),gpu_devices=[],status='COMPLETE'))
    print('complete',evidence['elapsed_seconds'],{k:v['compatible_candidates'] for k,v in evidence['pools'].items()},flush=True)


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    group=parser.add_mutually_exclusive_group();group.add_argument('--prepare',action='store_true');group.add_argument('--score-only',action='store_true')
    args=parser.parse_args();out=args.out.resolve()
    if args.prepare:prepare(out)
    elif args.score_only:
        p=json.loads((out/'inputs.json').read_text())
        expected=json.loads((out/'results.json').read_text())
        assert sha(out/'inputs.json')==expected['input_sha256']
        for name, digest in expected['raw_sha256'].items():assert sha(out/name)==digest,name
        for value in p['native'].values():assert sha(ROOT/value['path'])==value['sha256']
        for name, digest in p['source_sha256'].items():assert sha(ROOT/name)==digest,name
        result=score(out,p)
        assert all(result[k]==expected[k] for k in result)
        print(json.dumps(dict(reproduced=True,pools={k:v['compatible_candidates'] for k,v in result['pools'].items()})))
    else:run(out)
