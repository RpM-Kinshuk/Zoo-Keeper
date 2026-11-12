"""Verify completed fresh cost evidence against the predeclared two inputs."""
import hashlib
import json
import platform
from pathlib import Path
import socket
import sys
from importlib.metadata import version

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'Model-Zoo')]
from evidence.fingerprint import load_fingerprint
r=ROOT/'runs/scaling-cost-probe-20261111'
f=json.loads((r/'freeze.json').read_text());x=json.loads((r/'row_blocks/summary.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert x['counts']=={'complete':2}
for rel,digest in f['input_sha256'].items():assert sha(ROOT/rel)==digest,rel
for row in f['selected']:
 actual=next(a for a in x['models'] if a['repo_id']==row['repo_id'] and a['revision']==row['revision'])
 assert actual['bytes_read']==row['plan']['planned_requested_payload_bytes']>0
 assert actual['range_reads']==row['plan']['nviews']
paths=sorted((r/'row_blocks').glob('*.safetensors'));assert len(paths)==2
artifacts=[]
for p in paths:
 fp=load_fingerprint(p)
 assert fp.meta['sketch_k']==64 and fp.meta['sketch_seed']==20261028 and fp.meta['row_block']==256
 artifacts.append(dict(path=str(p.relative_to(ROOT)),size=p.stat().st_size,sha256=sha(p),meta=fp.meta))
cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown')
result=dict(passed=True,hostname=socket.gethostname(),cpu=cpu,python=platform.python_version(),packages={k:version(k) for k in ['torch','numpy','safetensors']},
 elapsed_extraction_seconds=sum(a['elapsed_seconds'] for a in x['models']),payload_bytes=sum(a['bytes_read'] for a in x['models']),range_reads=sum(a['range_reads'] for a in x['models']),
 observed_input_freeze_to_summary_seconds=(r/'row_blocks/summary.json').stat().st_mtime-(r/'freeze.json').stat().st_mtime,
 invocation_timer='File timestamp interval includes startup; not an instrumented end-to-end query timer.',artifacts=artifacts,
 policy=f['policy'],metadata_concurrent=False,gpu_devices=[],verifier_sha256=sha(Path(__file__)),scope='Two fresh F32 row-block extraction examples, not accuracy or full-query latency.')
(r/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='artifacts'},indent=2))
