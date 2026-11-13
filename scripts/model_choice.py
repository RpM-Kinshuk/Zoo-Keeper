"""Fixed saved-score coverage and retrospective leave-benchmark-out choices."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'Zoo-Keeper'), str(ROOT/'Model-Zoo')]
OUT = ROOT/'runs/model-choice-20261112'
BENCHMARKS = ['BBH Raw','GPQA Raw','IFEval Raw','MATH Lvl 5 Raw','MMLU-PRO Raw','MUSR Raw']
NATIVE_METRICS = dict(zip(BENCHMARKS, ['acc_norm,none','acc_norm,none','strict_acc,none',
                                     'exact_match,none','acc,none','acc_norm,none']))
BUDGETS = [('reported_le_2B',2.), ('reported_le_8B',8.), ('unrestricted',None)]


def evaluate_scenario(scores, metadata, target, budget):
    """Select without the target column; use its values only for final scoring."""
    if not scores.index.is_unique or not metadata.index.is_unique:
        raise ValueError('unique candidate identities required')
    if set(scores.index) != set(metadata.index) or target not in scores or len(scores.columns)<2:
        raise ValueError('aligned candidate rows and non-target benchmarks required')
    if not np.isfinite(scores.to_numpy(dtype=float)).all() or not np.isfinite(metadata[['reported_params_b','historical_downloads']].to_numpy(dtype=float)).all():
        raise ValueError('finite complete candidate evidence required')
    keys = sorted(k for k in metadata.index if budget is None or metadata.loc[k,'reported_params_b']<=budget)
    result = dict(target=target, budget_reported_params_b=budget, candidates=keys, n_candidates=len(keys))
    if len(keys)<2:
        return dict(result, status='insufficient_candidates', policies={}, uniform=None)
    known = scores.loc[keys].drop(columns=[target])
    # The selectors receive only metadata or non-target columns.
    decision_values = dict(largest=metadata.loc[keys,'reported_params_b'],
        downloads=metadata.loc[keys,'historical_downloads'], other_scores=known.mean(axis=1))
    selections = {}
    for policy, values in decision_values.items():
        best = float(values.max()); tied = sorted(k for k in keys if float(values.loc[k])==best)
        selections[policy] = dict(selected=tied[0], selection_value=best, selection_ties=tied, n_selection_ties=len(tied))
    y = scores.loc[keys,target]
    maximum = float(y.max()); oracle = sorted(k for k in keys if float(y.loc[k])==maximum)
    for row in selections.values():
        observed=float(y.loc[row['selected']])
        row.update(target_value=observed, regret=maximum-observed, target_best_hit=row['selected'] in oracle)
    return dict(result, status='scored', available_benchmarks=list(known.columns), oracle_value=maximum,
        oracle_keys=oracle, n_oracle_ties=len(oracle), policies=selections,
        uniform=dict(expected_target_value=float(y.mean()), expected_regret=maximum-float(y.mean()),
                     target_best_probability=len(oracle)/len(keys)))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(record, revision):
    if record is None or record.get('status')!='ok' or record.get('resolved_revision')!=revision:
        return None
    rows = record.get('weights', [])
    if not rows or any(r['size']<=0 or re.fullmatch(r'[0-9a-f]{64}',r.get('sha256','')) is None for r in rows):
        return None
    return sorted((r['path'],r['size'],r['sha256']) for r in rows)


def prepare():
    from lab.artifacts import atomic_json
    assert not (OUT/'freeze.json').exists(), 'preserve frozen inputs'
    models_path=ROOT/'runs/track-a-expanded-r6-20261105/zoo/models.parquet'
    scores_path=models_path.with_name('scores.parquet')
    raw_path=ROOT/'Model-Research/data/sampled_leaderboard.csv'
    audit=ROOT/'runs/track-a-k10-20261107'
    for name,h in json.loads((audit/'input-freeze.json').read_text()).items():
        assert sha(ROOT/name)==h
    scores=pd.read_parquet(scores_path);models=pd.read_parquet(models_path).set_index('model_key')
    raw=pd.read_csv(raw_path,dtype=str,keep_default_na=False)
    saved=json.loads((audit/'metadata.json').read_text())
    pairs=json.loads((audit/'pairs.json').read_text())
    runs=[]
    for (key,repo,source_revision,harness), group in scores.groupby(['model_key','repo_id','source_revision','harness'],dropna=False):
        precision=harness.split(':',1)[1]; measured_revision=key.rsplit('@',1)[1]
        matched=raw.loc[raw.fullname.str.casefold().eq(repo.casefold()) & raw['Model sha'].eq(source_revision) & raw.Precision.eq(precision)]
        assert len(matched)==1 and sorted(group.benchmark)==sorted(BENCHMARKS)
        idx=matched.index[0]; row=matched.iloc[0]
        assert set(group.metric)=={'raw_accuracy'} and set(group.source)=={'open_llm_leaderboard_v2'}
        for value in group.itertuples():
            assert float(row[value.benchmark])==value.value and 0<=value.value<=1
        left=inventory(saved.get(key),measured_revision)
        right=inventory(saved.get(repo+'@'+source_revision),source_revision) if source_revision else None
        joined=left is not None and right is not None and left==right
        old=next(r for r in pairs if r['repo_id']==repo and r['measured_revision']==measured_revision and r['score_revision']==source_revision)
        assert joined==(old['status']=='weight_identity_verified')
        reported=float(row['#Params (B)']) if row['#Params (B)'] else None
        downloads=float(row['downloads']) if row['downloads'] else None
        chat={'true':True,'false':False}.get(row['Chat Template'].casefold())
        reasons=[]
        if precision!='bfloat16':reasons.append('other_precision')
        if chat is not True:reasons.append('chat_template_not_true')
        if not joined:reasons.append('weight_identity_unknown')
        if reported is None or not np.isfinite(reported) or reported<=0:reasons.append('reported_size_unknown')
        if downloads is None or not np.isfinite(downloads) or downloads<0:reasons.append('download_count_unknown')
        weight_group=hashlib.sha256(json.dumps(left).encode()).hexdigest() if left else None
        runs.append(dict(model_key=key,repo_id=repo,source_revision=source_revision,precision=precision,harness=harness,
            reported_chat_template=chat,eval_name=row['eval_name'],source_csv_row=int(idx)+2,
            reported_params_b=reported,historical_downloads=downloads,pipeline_tag=row['pipeline_tag'] or None,
            leaderboard_license=row['Hub License'],catalog_license=models.loc[key,'license'],
            submission_date=row['Submission Date'],reported_generation=row['Generation'],reported_type=row['Type'],
            weight_identity_status='verified' if joined else 'unknown', same_commit=source_revision==measured_revision,
            weight_inventory_group=weight_group,primary_eligible=not reasons,exclusion_reasons=reasons,
            source_scores={b:float(row[b]) for b in BENCHMARKS},published_metric_context=NATIVE_METRICS,
            protocol_status='Published suite/display context; exact historical per-run code/task/template configuration unknown.'))
    primary=[r for r in runs if r['primary_eligible']]
    assert len({r['model_key'] for r in primary})==len(primary)
    aliases={g:sorted({r['model_key'] for r in primary if r['weight_inventory_group']==g}) for g in {r['weight_inventory_group'] for r in primary}}
    aliases={g:k for g,k in aliases.items() if len(k)>1}
    coverage=dict(catalog_models=len(models),scored_model_keys=int(scores.model_key.nunique()),score_rows=len(scores),
        evaluation_runs=len(runs),csv_rows_matched=len(runs),raw_values_matched=len(scores),
        primary_models=len(primary),primary_runs=[r['eval_name'] for r in primary],primary_weight_alias_groups=aliases,
        existing_catalog_known_params=int(models.loc[scores.model_key.unique(),'params'].notna().sum()),
        existing_catalog_known_downloads=int(models.loc[scores.model_key.unique(),'downloads'].notna().sum()),
        recovered_reported_size_models=len({r['model_key'] for r in runs if r['reported_params_b'] is not None}),
        recovered_download_models=len({r['model_key'] for r in runs if r['historical_downloads'] is not None}),
        recorded_pipeline_models=len({r['model_key'] for r in runs if r['pipeline_tag']}),
        weight_joined_score_rows=sum(6 for r in runs if r['weight_identity_status']=='verified'),
        same_commit_score_rows=int(scores.revision_verified.sum()))
    atomic_json(OUT/'candidate-evidence.json',dict(runs=runs,coverage=coverage))
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['source_scores','published_metric_context']} for r in runs]).to_csv(OUT/'candidate-evidence.csv',index=False)
    paths=[Path(__file__),OUT/'PROTOCOL.md',models_path,scores_path,raw_path,
        audit/'input-freeze.json',audit/'protocol.md',audit/'metadata.json',audit/'pairs.json',
        OUT/'candidate-evidence.json',OUT/'candidate-evidence.csv',ROOT/'Zoo-Keeper/tests/test_model_choice.py']
    paths += list((OUT/'references').glob('*'))
    atomic_json(OUT/'freeze.json',dict(source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths if p.is_file()},
        candidate_keys=sorted(r['model_key'] for r in primary),benchmarks=BENCHMARKS,budgets=BUDGETS,
        coverage=coverage,duplicate_policy='Retain evaluated model keys; flag exact weight inventories; no independence claim.',
        scope='Retrospective other-task selector baseline on a score-covered convenience panel; no weights, new task runs, C6 or general model-choice benefit.'))
    print(json.dumps(coverage,indent=2),flush=True)


def run():
    from lab.artifacts import atomic_json
    assert (OUT/'prospective-review.md').exists()
    assert not (OUT/'results.json').exists(), 'preserve completed diagnostic'
    started=time.monotonic();freeze=json.loads((OUT/'freeze.json').read_text())
    for name,h in freeze['source_sha256'].items():assert sha(ROOT/name)==h,name
    records=json.loads((OUT/'candidate-evidence.json').read_text())['runs']
    selected={r['model_key']:r for r in records if r['primary_eligible']}
    keys=freeze['candidate_keys'];assert set(selected)==set(keys)
    scores=pd.DataFrame({k:selected[k]['source_scores'] for k in keys}).T
    metadata=pd.DataFrame({k:{f:selected[k][f] for f in ['reported_params_b','historical_downloads']} for k in keys}).T
    cases=[]
    for name,cap in freeze['budgets']:
        for target in freeze['benchmarks']:
            case=evaluate_scenario(scores,metadata,target,cap);case['scenario']=name;cases.append(case)
    summaries=[]
    for scenario,_ in freeze['budgets']:
        group=[r for r in cases if r['scenario']==scenario and r['status']=='scored']
        for policy in ['largest','downloads','other_scores','uniform']:
            regrets=[r['uniform']['expected_regret'] if policy=='uniform' else r['policies'][policy]['regret'] for r in group]
            hits=[r['uniform']['target_best_probability'] if policy=='uniform' else float(r['policies'][policy]['target_best_hit']) for r in group]
            summaries.append(dict(scenario=scenario,policy=policy,n_scored_targets=len(group),
                mean_regret_pp=float(np.mean(regrets)*100) if regrets else None,
                worst_regret_pp=float(max(regrets)*100) if regrets else None,
                target_best_hits_or_expected=float(sum(hits)) if hits else None))
    atomic_json(OUT/'results.json',dict(cases=cases,summaries=summaries,coverage=freeze['coverage'],
        elapsed_seconds=time.monotonic()-started,freeze_sha256=sha(OUT/'freeze.json'),
        runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
        model_weight_reads=0,new_evaluation_jobs=0,scope=freeze['scope']))
    print(json.dumps(summaries,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    (prepare if args.prepare else run)()


if __name__=='__main__':main()
