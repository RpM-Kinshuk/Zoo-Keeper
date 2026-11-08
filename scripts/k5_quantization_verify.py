"""Independent full-key coverage and original-delivery pin check for K5."""
import hashlib,json,os
from pathlib import Path
from safetensors import safe_open
from lab.artifacts import atomic_json
root=Path(os.environ['ZOO_DATA']);out=root/'runs/track-b-k5-quantization-20261108'
v=json.loads((out/'results.json').read_text())
source=root/'runs/track-a-compatibility-20261105/full-weight-artifact-map137.json'
index=json.loads(source.read_text());mapping={r['model_id']:r for r in index['rows']}
def keys(path):
 result={}
 for p in sorted(path.glob('*.safetensors')):
  with safe_open(str(p),framework='pt',device='cpu') as f:
   for name in f.keys():
    assert name not in result
    result[name]=tuple(f.get_slice(name).get_shape())
 return result
rows=[];proof={str(source.relative_to(root)):hashlib.sha256(source.read_bytes()).hexdigest()}
for row in v['rows']:
 mid=row['model_id']
 if not row['positive']:continue
 child=root/'lab'/mid;recipe=json.loads((child/'recipe.json').read_text());recipe=recipe.get('recipe',recipe);parent=recipe['parents'][0]
 assert keys(child)==keys(root/'lab'/recipe['inputs'][parent]['path']),mid
 delivery=root/mapping[mid]['manifest'];digest=hashlib.sha256(delivery.read_bytes()).hexdigest();assert digest==mapping[mid]['manifest_sha256']
 proof[str(delivery.relative_to(root))]=digest
 for file in json.loads(delivery.read_text())['files']:
  if file['filename'].endswith('.safetensors'):
   assert v['input_sha256'][f"lab/{mid}/"+file['filename']]==file['sha256'],mid
 rows.append(dict(model_id=mid,parent_child_tensor_names_shapes_equal=True,original_child_weight_pins_match=True))
assert len(rows)==12
assert v['summary']['not_measured']==0
atomic_json(out/'verification.json',dict(status='PASS',rows=rows,input_sha256=proof,results_sha256=hashlib.sha256((out/'results.json').read_bytes()).hexdigest()))
print('PASS: all 12 child/parent complete name+shape sets and original child delivery weight pins')
