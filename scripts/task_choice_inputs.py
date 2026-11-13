"""Pinned text/tokenizer preparation only; never downloads model weights."""
import hashlib,json,os,sys,time
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[2];O=R/'runs/task-choice-20261112'
sys.path[:0]=[str(R/'Zoo-Keeper'),str(R/'Model-Zoo'),str(R/'Model-Zoo/esd_experiment/src')]
from lab.artifacts import atomic_json
from evidence.hub import HubReader
from huggingface_hub import HfApi
from scripts.task_choice import split_rows

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def prompt(task,row):
 if task=='amazon_polarity':
  return 'Classify the sentiment of this product review.\nA: negative\nB: positive\nReply with only A or B.\n\nReview:\n'+row['text']
 return 'Answer the question using the passage.\nA: no\nB: yes\nReply with only A or B.\n\n'+row['text']

def main():
 import pandas as pd
 from transformers import AutoTokenizer
 assert not (O/'freeze.json').exists()
 token=next((os.environ[k] for k in ['HF_TOKEN','HF_UPLOAD','HF_UPLOAD_TOKEN'] if os.environ.get(k)),None);assert token
 api=HfApi(token=token);reader=HubReader(token=token)
 inspection=json.loads((O/'input-inspection.json').read_text());models=[r for r in inspection['rows'] if r['kind']=='model']
 def fetch(repo,revision,name,path,expected=None,limit=128*1024**2):
  if path.exists():
   if expected:assert digest(path)==expected
   return
  remote=reader.resolve(repo,name,revision);assert 0<remote.size<=limit
  path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial')
  with tmp.open('wb') as f:
   for start in range(0,remote.size,8*1024**2):f.write(reader.read(remote,start,min(start+8*1024**2,remote.size)))
  if expected:assert digest(tmp)==expected
  tmp.replace(path)
 for m in models:
  assert not m['gated'] and m['license'] in ['apache-2.0','mit']
  folder=O/'inputs'/m['repo_id'].replace('/','--');m['local_path']=str(folder.relative_to(R))
  for f in m['files']:
   n=f['name']
   if '/' not in n and (n.endswith(('.json','.txt','.model')) or n in ['LICENSE','README.md']):
    assert f['size']<16*1024**2
    fetch(m['repo_id'],m['revision'],n,folder/n,f.get('sha256'),16*1024**2)
  tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
  assert tok.chat_template
 mapi={m['repo_id']:AutoTokenizer.from_pretrained(R/m['local_path'],local_files_only=True,trust_remote_code=False) for m in models}
 datasets=[]
 amazon=api.dataset_info('fancyzhx/amazon_polarity',revision='9d9c45c18f8c3cf1b23a3c27917b60cbf28f3289',files_metadata=True,timeout=20)
 boolq=next(r for r in inspection['rows'] if r['repo_id']=='google/boolq')
 af=[dict(name=f.rfilename,size=f.size,sha256=getattr(f.lfs,'sha256',None)) for f in amazon.siblings]
 atomic_json(O/'amazon-metadata.json',dict(repo_id=amazon.id,revision=amazon.sha,license=getattr(amazon.card_data,'license',None),files=af))
 specs=[('amazon_polarity',amazon.id,amazon.sha,'test',af,getattr(amazon.card_data,'license',None)),('boolq',boolq['repo_id'],boolq['revision'],'validation',boolq['files'],boolq['license'])]
 for task,repo,rev,split,files,license in specs:
  parts=[f for f in files if Path(f['name']).name.startswith(split+'-') and f['name'].endswith('.parquet')];assert parts
  frames=[];paths=[]
  for f in parts:
   p=O/'datasets'/task/Path(f['name']).name;fetch('datasets/'+repo,rev,f['name'],p,f['sha256']);frames.append(pd.read_parquet(p));paths.append(str(p.relative_to(R)))
  card=O/'datasets'/task/'README.md';fetch('datasets/'+repo,rev,'README.md',card,limit=2*1024**2)
  frame=pd.concat(frames,ignore_index=True);rows=[];length_excluded=0;token_excluded=0
  for i,row in frame.iterrows():
   if task=='amazon_polarity':
    text=str(row['title'])+'\n'+str(row['content']);group=text;label=int(row['label'])
   else:
    text='Passage: '+str(row['passage'])+'\nQuestion: '+str(row['question']);group=str(row['passage']);label=int(row['answer'])
   if len(text)>2400:length_excluded+=1;continue
   item=dict(id=f'{repo}@{rev}:{split}:{i}',text=text,group=group,label=label)
   rows.append(item)
  selected=split_rows(rows)
  # Verify every selected prompt fits all three tokenizers; never truncate or drop per model.
  for part in selected.values():
   for item in part:
    assert all(len(t.apply_chat_template([dict(role='user',content=prompt(task,item))],tokenize=True,add_generation_prompt=True))<=2048 for t in mapi.values()), 'common prompt limit exceeded; stop preparation'
  target=O/'datasets'/task/'selected.json';atomic_json(target,selected)
  datasets.append(dict(task=task,repo_id=repo,revision=rev,source_split=split,license=license,source_paths=paths,selected_path=str(target.relative_to(R)),source_rows=len(frame),character_length_excluded=length_excluded,token_length_excluded=token_excluded,eligible_rows=len(rows),distinct_groups=len({' '.join(x['group'].casefold().split()) for x in rows})))
  print(json.dumps(dict(task=task,eligible=len(rows),selection='32probe+256validation',no_model_outputs=True)),flush=True)
 atomic_json(O/'prepared.json',dict(prepared_utc=datetime.now(timezone.utc).isoformat(),models=models,datasets=datasets,reader_stats=reader.stats))
if __name__=='__main__':main()
