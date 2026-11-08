"""K3 diagnostic on stored aligned sketches; frozen production fit is untouched."""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import yaml

ROOT = Path(os.environ['ZOO_DATA']).resolve()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lineage.upstream import load_records
from lineage.records import sketch_block, view_diagnostics
from lineage.merge import _layer

OUT = ROOT / 'runs/track-b-k3-deltas-20261107'
OUT.mkdir(exist_ok=True, parents=True)
def write(name, value):
    p = OUT / name
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)

def main():
    if not (OUT/'protocol.json').exists():
        raise RuntimeError('Write protocol before running')
    hashes = {str((OUT/'protocol.json').relative_to(ROOT)): hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
              'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def read(p):
        hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text())
    results = []
    for lane, recipes in [('smol360', 'recipes'), ('qwen', 'recipes-v2-frozen')]:
        run = ROOT / f'runs/track-b-formal-{lane}-merges-v1'
        index = run/'evidence/record-index.json'
        entries = read(index)
        for e in entries:
            for field in ('manifest', 'fingerprint'):
                p = ROOT/e[field]
                hashes[e[field]] = hashlib.sha256(p.read_bytes()).hexdigest()
        records = load_records(index, ROOT/'runs', data_root=ROOT)
        def mid(r):
            return r.key.split('@')[0].removeprefix('lab/')
        byid = {mid(r): r for r in records}
        for rp in sorted((run/recipes).glob('*.json')):
            recipe = read(rp)
            name = recipe['model_id']
            child = byid[name]
            parents = [byid[p] for p in recipe['parents']]
            # Native references have Hub identities: resolve by recipe identity map.
            basekey = recipe['parent_identities'][recipe['reference']]
            base = next(r for r in records if r.key == basekey)
            published = ROOT/'lab'/name/'recipe.json'
            pub = read(published)
            pub_recipe = pub.get('recipe', pub)
            config = yaml.safe_load(recipe['mergekit_yaml'])
            assert yaml.safe_load(pub_recipe['mergekit_yaml']) == config
            aligned = [dict(view_diagnostics(child, r, k=64)[0]) for r in parents+[base]]
            common = sorted(set.intersection(*(set(a) for a in aligned)))
            views = []
            for i in common:
                b = sketch_block(base, aligned[-1][i],64).ravel()
                y = sketch_block(child,i,64).ravel()-b
                x = np.column_stack([sketch_block(p,a[i],64).ravel()-b for p,a in zip(parents,aligned)])
                # SLERP affine base term absorbs coefficients whose sum differs from one.
                if recipe['method']=='slerp':
                    x = np.column_stack((x,b))
                views.append((child.names[i],x,y))
            def fit(selected):
                x = np.concatenate([v[1] for v in selected]); y = np.concatenate([v[2] for v in selected])
                c,_,rank,s = np.linalg.lstsq(x,y,rcond=None)
                return {'coefficients':c.tolist(),'rank':int(rank),'condition':float(s[0]/s[-1]),'residual':float(np.linalg.norm(y-x@c)/np.linalg.norm(y)),'update_relative_to_base':float(np.linalg.norm(y)/np.linalg.norm(np.concatenate([sketch_block(base,aligned[-1][i],64).ravel() for i in common])))}
            result = {'model_id':name,'method':recipe['method'],'fit':fit(views),'n_shared':len(common),'n_requested':len(child.names),'published_config_matches':True}
            if recipe['method']=='linear':
                truth = [float(m['parameters']['weight']) for m in config['models']]
                truth = np.asarray(truth)/sum(truth)
                result['target'] = truth.tolist()
                result['coefficient_error'] = float(np.max(np.abs(np.array(result['fit']['coefficients'])-truth)))
            elif recipe['method']=='slerp':
                t = config['parameters']['t']
                if isinstance(t,list):
                    layers = sorted(set(_layer(v[0])[1] for v in views if _layer(v[0])))
                    perlayer = []
                    for layer in layers:
                        f = fit([v for v in views if _layer(v[0]) and _layer(v[0])[1]==layer])
                        fraction = layer/max(layers)
                        expected_t = float(np.interp(fraction,np.linspace(0,1,len(t)),t))
                        error = max(abs(f['coefficients'][0]-(1-expected_t)),abs(f['coefficients'][1]-expected_t))
                        perlayer.append(dict(layer=layer,expected_t=expected_t,error=error,**f))
                    result['layers']=perlayer
                    result['coefficient_error']=max(p['error'] for p in perlayer)
                else:
                    result['target']=[1-t,t]
                    result['coefficient_error']=float(np.max(np.abs(np.array(result['fit']['coefficients'][:2])-[1-t,t])))
            if 'coefficient_error' in result:
                result['pass'] = result['coefficient_error'] <= .05
            results.append(result)
            print(name, result.get('coefficient_error'), flush=True)
    write('inputs.json', hashes)
    write('results.json',results)
    summary = {}
    for method in sorted({r['method'] for r in results}):
        group = [r for r in results if r['method'] == method]
        summary[method] = dict(n=len(group), coefficient_semantics=(
            'linear_weights' if method == 'linear' else
            'small_angle_interpolation_approximation' if method == 'slerp' else 'effective_projection_only'))
        if method in ('linear', 'slerp'):
            summary[method].update(max_error=max(r['coefficient_error'] for r in group),
                                   passes=sum(r['pass'] for r in group))
    write('summary.json',summary)
    print(json.dumps(summary))
if __name__=='__main__': main()
