"""K4 fresh selected-weight extraction, plain Smol360 aligned views."""
import copy,csv,hashlib,json,os,time
from itertools import combinations
from pathlib import Path
import networkx as nx
import numpy as np
from safetensors import safe_open
from evidence.fingerprint import load_fingerprint
from evidence.sketch import two_sided_sketch
from lineage.lab_report import common_distances
from lab.artifacts import atomic_json
root=Path(os.environ['ZOO_DATA']);repo=Path(__file__).resolve().parents[1]
out=root/'runs/track-b-k4-extraction-repeat-20261107';out.mkdir(parents=True,exist_ok=True)
assert not (out/'results.json').exists()
p=repo/'docs/steering/K4-extraction-protocol.md'
atomic_json(out/'protocol.json',dict(text=p.read_text(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
start=time.monotonic();hashes={}
def bind(p):hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
p=root/'runs/track-b-e6-expanded-20261105/pre-score-freeze.json';bind(p);frozen=json.loads(p.read_text());names=frozen['groups']['smol-360']
old=[];first=[];second=[]
for name in names:
 p=root/frozen['records'][name]['paths']['row_blocks_k64'];bind(p);fp=load_fingerprint(p)
 source=root/'lab/inputs/cohort-v1-smol-360-f8027fd0eaee' if name=='smol-360' else root/'lab'/name
 p=source/'model.safetensors';bind(p)
 copies=[]
 for repeat in range(2):
  sketches=[]
  with safe_open(str(p),framework='pt',device='cpu') as handle:
   for i,key in enumerate(fp.names):
    assert fp.geometry[i]=='matrix'
    assert tuple(handle.get_slice(key).get_shape())==tuple(fp.shape[i])
    a,b=map(int,fp.rows[i]);block=handle.get_slice(key)[a:b,:].float()
    sketches.append(two_sided_sketch(block,64))
  f=copy.copy(fp);f.sketch=np.stack(sketches);f.meta=dict(fp.meta,sketch_k=64);copies.append(f)
 old.append(fp);first.append(copies[0]);second.append(copies[1]);print(name,'twice extracted',flush=True)
 assert time.monotonic()-start<6600
original,views=common_distances(old,k=64);a,_=common_distances(first,k=64);b,_=common_distances(second,k=64)
e_repeat=float(np.max(np.abs(a-b)));e_backend=max(float(np.max(np.abs(original-a))),float(np.max(np.abs(original-b))));e=max(e_repeat,e_backend)
tol=4*e+64*np.finfo(float).eps*max(1.,float(original.max()))
def gaps(d):
 out=[]
 for a,b,c,e in combinations(range(len(d)),4):
  s=sorted([d[a,b]+d[c,e],d[a,c]+d[b,e],d[a,e]+d[b,c]])
  out.append(s[2]-s[1])
 return np.asarray(out)
g=nx.Graph();g.add_nodes_from(names)
with (repo/'lab/edges.csv').open() as f:
 for edge in csv.DictReader(f):
  if edge['role']=='parent' and edge['parent'] in names and edge['child'] in names:
   i,j=names.index(edge['parent']),names.index(edge['child']);g.add_edge(edge['parent'],edge['child'],weight=float(original[i,j]))
assert nx.is_tree(g)
exact=np.asarray([[nx.shortest_path_length(g,a,b,weight='weight') for b in names] for a in names])
noise=np.triu(np.random.default_rng(20261028).uniform(-e,e,exact.shape),1);noise+=noise.T
noisy=np.maximum(0,exact+noise);np.fill_diagonal(noisy,0)
real,control=gaps(original),gaps(noisy)
result=dict(models=len(names),views=len(views),same_backend_distance_noise=e_repeat,original_vs_cpu_distance_sensitivity=e_backend,tolerance_abs=tol,real_gap_median=float(np.median(real)),real_gap_p95=float(np.quantile(real,.95)),real_gap_max=float(real.max()),real_violation_fraction=float(np.mean(real>tol)),exact_tree_violation_fraction=float(np.mean(control>tol)),control_pass=bool(np.all(control<=tol)),quartets=len(real),input_sha256=hashes,driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),truth_table_sha256=hashlib.sha256((repo/'lab/edges.csv').read_bytes()).hexdigest(),scope='Smol360 partial-read only; backend sensitivity not sampling uncertainty')
atomic_json(out/'results.json',result);print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'}),flush=True)
