"""Bounded K4 numerical repeat; see docs/steering/K4-protocol.md."""
import hashlib
import json
import os
from itertools import combinations
from pathlib import Path
import networkx as nx
import numpy as np
from evidence.fingerprint import load_fingerprint
from lineage.lab_report import common_distances
from lab.artifacts import atomic_json

root = Path(os.environ['ZOO_DATA'])
old = root / 'runs/track-b-e6-expanded-20261105'
out = root / 'runs/track-b-k4-numerical-repeat-20261107'
out.mkdir(parents=True, exist_ok=True)
protocol = Path(__file__).resolve().parents[1] / 'docs/steering/K4-protocol.md'
assert not (out / 'results.json').exists(), 'new output only'
freeze = json.loads((old / 'pre-score-freeze.json').read_text())
bound = {}
def bind(path):
    bound[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
bind(old / 'pre-score-freeze.json')
atomic_json(out / 'protocol.json', {'text': protocol.read_text(), 'sha256': hashlib.sha256(protocol.read_bytes()).hexdigest()})
rng = np.random.default_rng(20261028)
def gaps(d):
    return np.asarray([sorted([d[a,b]+d[c,e], d[a,c]+d[b,e], d[a,e]+d[b,c]])[2] - sorted([d[a,b]+d[c,e], d[a,c]+d[b,e], d[a,e]+d[b,c]])[1] for a,b,c,e in combinations(range(len(d)),4)])
rows = []
for anchor,names in freeze['groups'].items():
    for mode,k in [('row_blocks_k64',64), ('full_weights_k128',128)]:
        path = old / (anchor+'-'+mode+'-distances.npz')
        bind(path)
        saved = np.load(path, allow_pickle=False)
        assert list(saved['labels']) == names
        fps = []
        for name in names:
            path = root / freeze['records'][name]['paths'][mode]
            bind(path)
            fps.append(load_fingerprint(path))
        repeated, views = common_distances(fps,k=k)
        original = saved['distance']
        error = float(np.max(np.abs(repeated-original)))
        tol = 4*error + 64*np.finfo(float).eps*max(1.,float(original.max()))
        graph = nx.Graph()
        graph.add_nodes_from(names)
        import csv
        repo = Path(__file__).resolve().parents[1]
        with (repo/'lab/edges.csv').open() as stream:
            for edge in csv.DictReader(stream):
                if edge['role']=='parent' and edge['parent'] in names and edge['child'] in names:
                    i,j=names.index(edge['parent']),names.index(edge['child'])
                    graph.add_edge(edge['parent'],edge['child'],weight=float(original[i,j]))
        assert nx.is_tree(graph)
        exact = np.asarray([[nx.shortest_path_length(graph,a,b,weight='weight') for b in names] for a in names])
        noise = rng.uniform(-error,error,exact.shape)
        noise = np.triu(noise,1); noise += noise.T
        noisy = np.maximum(0,exact+noise); np.fill_diagonal(noisy,0)
        real,control = gaps(original),gaps(noisy)
        rows.append(dict(anchor=anchor,mode=mode,noise_max_abs=error,tolerance_abs=tol,quartets=len(real),
                         gap_median=float(np.median(real)),gap_p95=float(np.quantile(real,.95)),gap_max=float(real.max()),
                         real_fraction=float(np.mean(real>tol)),control_fraction=float(np.mean(control>tol)),
                         control_pass=bool(np.all(control<=tol))))
atomic_json(out/'results.json',dict(rows=rows,input_sha256=bound,scope='numerical repeat only; extraction noise unmeasured',K4_complete=False))
lines=['# K4 numerical repeat only','','Extraction/measurement repeats remain required. No ancestry claim or pivot decision.','','| Family/mode | numerical repeat max | gap median / p95 / max | real violations | exact-tree violations |','|---|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['anchor']}/{r['mode']} | {r['noise_max_abs']:.6g} | {r['gap_median']:.6g} / {r['gap_p95']:.6g} / {r['gap_max']:.6g} | {r['real_fraction']:.6f} | {r['control_fraction']:.6f} |")
lines += ['', 'All rules and noise definitions were saved before execution in protocol.json. Input hashes in results.json. Repeat means fresh fingerprint loads and distance recomputation, not re-extraction.']
tmp=out/'REPORT.md.tmp';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(out/'REPORT.md')
print(json.dumps(rows))
