"""Build scoped tables and a frozen descriptive report from completed real evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import pandas as pd

from lab.artifacts import atomic_json, content_hash
from lab.cohort_evidence import sha, verify_ready_weights
from lab.registry import validate_tables
from lab.schema import MODEL_COLUMNS, EDGE_COLUMNS, canonical_json
from lineage.cli import main as lineage_main
from lineage.engine import configuration, digest


def run_report(data_root, run_root, report_version='v1', protocol_path=None):
    root, run = Path(data_root).resolve(), Path(run_root).resolve()
    if not run.is_relative_to(root):
        raise ValueError('report must remain inside data root')
    ready = json.loads((run/'ready.json').read_text())
    protocol_path = Path(protocol_path) if protocol_path else run/'fallback-pre-score-protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if sha(run/'ready.json') != protocol['ready_sha256']:
        raise ValueError('ready inventory changed after pre-score protocol freeze')
    progress = json.loads((run/'evidence/progress.json').read_text())
    measured = {r['model_id']:r for r in progress['rows']}
    original = {r['model_id']:r for r in ready['sources']+ready['outputs']}
    independent = protocol.get('independent_root_controls') is True
    if independent:
        registry=json.loads((Path(__file__).parents[1]/'pins.json').read_text())
        for source in ready['sources']:
            origin=registry[source['original_model_id']]
            if (origin.get('reference_type')!='polypythias' or origin['repo_id']!=source['repo_id']
                    or origin.get('seed')!=source.get('seed')):
                raise ValueError('independent-root claim lacks certified published seed identity')
    if set(original) != set(measured):
        raise ValueError('first report requires every frozen scoped model measured; no silent drops')
    for model_id, item in measured.items():
        verify_ready_weights(original[model_id], root)
        if item['ready_row_sha256'] != content_hash(original[model_id]):
            raise ValueError('measured model differs from frozen ready input')
        for field in ('manifest','fingerprint'):
            if sha(root/item[field]) != item[field+'_sha256']:
                raise ValueError('evidence hash changed')
    rows, edge_rows = [], []
    for model_id, row in original.items():
        source = bool(row.get('repo_id'))
        manifest = json.loads((root/measured[model_id]['manifest']).read_text())
        key = measured[model_id]['key']
        rows.append(dict(model_id=model_id, kind='base' if source else 'derived',
            path=key if source else row['path'].removeprefix('lab/'),
            base_family=manifest.get('config',{}).get('model_type','unknown'), license=row['license']))
        if not source:
            checkpoint = root/row['path']
            recipe = json.loads((checkpoint/'recipe.json').read_text())['recipe']
            lineage = json.loads((checkpoint/'lineage.json').read_text())
            expected_bases = [measured[parent]['key'] for parent in row['parents']]
            if sorted(lineage['bases']) != sorted(expected_bases):
                raise ValueError('scoped edge parent disagrees with exact checkpoint provenance')
            for parent in row['parents']:
                edge_rows.append(dict(child=model_id,parent=parent,edge_type=row['edge_type'],
                    role='parent',coefficient='',params_json=canonical_json(dict(
                        bits=recipe.get('bits'),method=recipe.get('method'),
                        recipe_sha256=sha(checkpoint/'recipe.json'),scope='small_real_cohort'))))
    models = pd.DataFrame(rows,columns=MODEL_COLUMNS).sort_values('model_id').reset_index(drop=True)
    edges = pd.DataFrame(edge_rows,columns=EDGE_COLUMNS).sort_values(['child','parent']).reset_index(drop=True)
    validate_tables(models,edges)
    scoped = run/('scoped' if report_version=='v1' else 'scoped-'+report_version)
    scoped.mkdir(exist_ok=True)
    for name, frame in (('models',models),('edges',edges)):
        path = scoped/(name+'.csv')
        content = frame.to_csv(index=False,lineterminator='\n')
        if path.exists() and path.read_text()!=content:
            raise ValueError('existing scoped tables differ')
        path.write_text(content)
    method = configuration({})
    method_path = run/'inference-method.json'
    atomic_json(method_path,method)
    code_paths = [Path(__file__), Path(__file__).parents[2]/'lineage/engine.py',
                  Path(__file__).parents[2]/'lineage/upstream.py',Path(__file__).parents[2]/'lab/score.py']
    frozen = dict(method=method,method_hash=digest(method),ready_sha256=sha(run/'ready.json'),
                  pre_score_protocol_sha256=sha(protocol_path),
                  source_sha256={p.name:sha(p) for p in code_paths},
                  evidence_hashes={k:{'manifest':v['manifest_sha256'],'fingerprint':v['fingerprint_sha256']}
                                   for k,v in measured.items()},threshold=protocol['development_threshold'])
    freeze_path = run/('method-freeze.json' if report_version=='v1' else 'method-freeze-'+report_version+'.json')
    if freeze_path.exists() and json.loads(freeze_path.read_text()) != frozen:
        raise ValueError('method freeze differs')
    atomic_json(freeze_path,frozen)
    prediction_path = run/'inference/edges.parquet'
    begin = time.monotonic()
    lineage_main(['infer','--manifests',str(run/'evidence/manifests'),
                  '--fingerprints',str(run/'evidence/t1'),'--data-root',str(root),
                  '--config',str(method_path),'--out',str(prediction_path),'--resume'])
    inference_seconds = time.monotonic()-begin
    predictions = pd.read_parquet(prediction_path)
    parents = {name:set() for name in original}
    for edge in edge_rows:
        parents[edge['child']].add(edge['parent'])
    def ancestors(name):
        found=set();pending=list(parents[name])
        while pending:
            parent=pending.pop()
            if parent not in found:
                found.add(parent);pending.extend(parents[parent])
        return found
    keys = {name:item['key'] for name,item in measured.items()}
    reports = {}
    for partition in ('development','test'):
        ids = sorted(name for name,row in original.items()
                     if protocol['root_split'][row['root_id']] == partition)
        universe_rows, truth_rows, cost_rows = [], [], []
        for query in ids:
            query_key=keys[query]
            for candidate in ids:
                if query==candidate:
                    continue
                # Within-root siblings are legitimate directed controls. There
                # are no fabricated independent-root or cross-architecture negatives.
                same_root=original[query]['root_id']==original[candidate]['root_id']
                if not same_root and not independent:
                    raise ValueError('fallback cohort expected one whole root per partition')
                labels={'parent':candidate in parents[query], 'ancestor':candidate in ancestors(query),
                        'same_root':same_root}
                # Conservatively use verified-Hub tier for the whole comparison
                # when certified published root independence supplies negatives;
                # constructed positives remain individually backed by exact edges.
                tier='verified_hub' if independent else 'lab_exact'
                universe_rows.append(dict(query_key=query_key,candidate_key=keys[candidate],eligible=True,
                    truth_parent=labels['parent'],truth_ancestor=labels['ancestor'],truth_same_root=same_root,
                    truth_tier=tier))
                for estimand,label in labels.items():
                    truth_rows.append(dict(query_key=query_key,candidate_key=keys[candidate],estimand=estimand,
                        label=int(label),truth_tier=tier,root_group=original[query]['root_id'],
                        split=partition,duplicate_group=original[query]['root_id']))
            item=measured[query]
            own=item['tensor_bytes_by_source'].get(query_key,0)
            cost_rows.append(dict(query_key=query_key,bytes_read=item['tensor_bytes_read'],
                seconds=item['elapsed_seconds'],query_bytes=own,
                verification_bytes=item['tensor_bytes_read']-own,reference_bytes=None,
                reference_amortization_queries=len(ids),status='incomplete',
                reason='measured extraction tensor bytes; full source-verification/cached-inference IO not measured or attributed, so no complete cold-reference bytes/answer claim'))
        directory=scoped/partition;directory.mkdir(exist_ok=True)
        frames={'predictions':predictions[predictions.child_key.isin([keys[k] for k in ids]) &
                                         predictions.parent_key.isin([keys[k] for k in ids])].copy(),
                'truth':pd.DataFrame(truth_rows),'universe':pd.DataFrame(universe_rows),'costs':pd.DataFrame(cost_rows)}
        inputs={}
        for name,frame in frames.items():
            path=directory/(name+'.parquet');frame.to_parquet(path,index=False);inputs[name]=str(path)
        inputs.update(models=str(scoped/'models.csv'),edges=str(scoped/'edges.csv'),
                      manifests=str(run/'evidence/manifests'))
        cfg=dict(schema_version='1',deployment_threshold=protocol['development_threshold'],
            deployment_threshold_status='predeclared before scores; demonstration only',seed=20261031,
            pool_sizes=[2,4,8,100,500,1000,5000,10000,20000],baseline_results={},
            method_hash=frozen['method_hash'],scope='real_scoped_'+partition,
            controls='constructed siblings and certified published independent seeds' if independent
                     else 'exact constructed siblings; independent same-architecture controls unavailable')
        config_path=directory/'evaluation.json';atomic_json(config_path,cfg)
        input_path=directory/'inputs.json';atomic_json(input_path,inputs)
        evaluation=run/('evaluation' if report_version=='v1' else 'evaluation-'+report_version)/partition
        if not evaluation.exists():
            lineage_main(['freeze','--config',str(config_path),'--inputs',str(input_path),'--out',str(evaluation)])
        argv=['evaluate','--config',str(config_path),'--out',str(evaluation)]
        for name,path in inputs.items():
            argv.extend(['--'+name,path])
        lineage_main(argv)
        reports[partition]=json.loads((evaluation/'metrics.json').read_text())
    result=dict(schema_version='1',status='REAL_SCOPED_DESCRIPTIVE',models=len(models),edges=len(edges),
        generated_derivatives=len(ready['outputs']),prediction_rows=len(predictions),
        inference_seconds=inference_seconds,method_hash=frozen['method_hash'],
        selected_tensor_bytes=sum(r['tensor_bytes_read'] for r in measured.values()),
        acquisition_cost_scope='selected tensor reads including matched parent blocks; full prep/inference artifact IO remains incomplete',
        independent_controls='VERIFIED_PUBLISHED_SEED_CONTROLS' if independent else 'BLOCKED_SAFE_FORMAT_OR_NETWORK',
        gate_status='NOT_EVALUABLE',
        calibrated_claims=int(predictions.calibrated.sum()),
        partitions={part:{e:report['by_estimand'][e] for e in ('parent','ancestor','same_root')}
                    for part,report in reports.items()},
        training_bundle='runs/track-b-training-launches-v1',release_staging='deferred')
    report_json=run/('first-real-report.json' if report_version=='v1' else 'first-real-report-'+report_version+'.json')
    report_md=run/('REPORT.md' if report_version=='v1' else 'REPORT-'+report_version+'.md')
    atomic_json(report_json,result)
    lines=['# First real scoped lab report','',
        f"{len(models)} real models, {len(edges)} exact construction edges; {len(predictions)} detected prediction rows.",
        'Development: Pythia seeds1/2. Held-out test: seeds3/4; every root and its derivatives stay in one partition.'
        if independent else 'Development: SmolLM2-135M root and five derivatives. Held-out test: OLMo root and three derivatives.',
        'Threshold 0.75 was declared before any cohort score; no calibration or test-threshold tuning.',
        '', '| Partition | Estimand | AUROC | Query coverage | Frozen TPR | False matches/query |',
        '| --- | --- | --- | --- | --- | --- |']
    for partition,report in reports.items():
        for estimand,m in report['by_estimand'].items():
            at=m['at_frozen_threshold']
            lines.append(f"| {partition} | {estimand} | {m['auroc']} | {m['query_coverage']} | {at['tpr']} | {at['false_matches_per_query']} |")
    lines.extend(['','Sibling controls have exact non-parent/non-ancestor labels. Independent-root negatives use pinned published PolyPythias seed evidence; comparisons are conservatively reported at verified-Hub tier.'
        if independent else 'Sibling controls have exact non-parent/non-ancestor labels. No comparable independently pretrained controls were safely available: same-root AUROC is undefined.',
        'The unfitted default method abstains on direction; zero false parent claims must be read with its TPR and coverage, not as successful detection.',
        'T2, a 5k reference pool and matched baselines are absent. G0/P1 remain NOT_EVALUABLE.',
        'Cost artifacts retain actual selected tensor reads and parent-read breakdowns; complete cold-reference/preparation/inference IO is not claimed.',
        'See evaluation/development and evaluation/test for frozen protocols, input hashes, metrics, coverage and pool curves.',
        'The failed MLP-permutation audit is excluded, not counted. Quantization is simulated; perturbation makes no preservation claim.',
        'Training launch bundles are separate and have not been executed. Release staging is deferred.'])
    report_md.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':result['status'],'models':len(models),'edges':len(edges),
                      'predictions':len(predictions),'report':str(report_md)}),flush=True)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--report-version',default='v1',choices=['v1','v2','v3'])
    parser.add_argument('--protocol',type=Path)
    args=parser.parse_args(argv)
    run_report(args.data_root,args.run_root,args.report_version,args.protocol)


if __name__=='__main__':
    main()
