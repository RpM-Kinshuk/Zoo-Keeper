import hashlib
import json

import pandas as pd

from studies.heritability.historical import audit_predictions, discover_runs


def make_run(tmp_path):
    run = tmp_path / 'research_runs' / 'example'
    run.mkdir(parents=True)
    key = 'org/a@' + 'a'*40
    rows = []
    for method in ['weight', 'no_weight']:
        rows.append({'model_key':key, 'benchmark':'v2', 'task':'math', 'harness':'h1',
                     'precision':'bf16', 'quantization':'none', 'method':method,
                     'fold':'0', 'actual':.4, 'prediction':.5, 'split_group':'root-a',
                     'duplicate_group':'dup-a', 'root_group':'root-a'})
    pd.DataFrame(rows).to_csv(run/'predictions.csv', index=False)
    (run/'protocol.json').write_text(json.dumps({'seed':17}))
    pd.DataFrame([{'fold':'0','split_group':'root-b','partition':'train'},
                  {'fold':'0','split_group':'root-a','partition':'test'}]).to_csv(run/'splits.csv',index=False)
    manifest = {'artifacts':{name:{'sha256':hashlib.sha256((run/name).read_bytes()).hexdigest()}
                             for name in ['predictions.csv','protocol.json','splits.csv']}}
    (run/'artifact_manifest.json').write_text(json.dumps(manifest))
    scores = pd.DataFrame([dict(rows[0], accuracy=.4, revision_verified=True)])
    scores = pd.concat([scores, scores.assign(precision='fp16',accuracy=.7)], ignore_index=True)
    mapping = {'prediction_file':'predictions.csv', 'prediction_scale':'raw_accuracy',
               'prediction_semantics':'binary_accuracy', 'setting':'root_holdout',
               'split_file':'splits.csv','protocol_file':'protocol.json',
               'manifest_file':'artifact_manifest.json','baseline_method':'no_weight'}
    return run, scores, mapping


def test_audited_raw_predictions_join_exact_deployment(tmp_path):
    run, scores, mapping = make_run(tmp_path)
    predictions, exclusions, audit = audit_predictions(run, scores, mapping)
    assert len(predictions) == 2
    assert predictions.revision_verified.all()
    assert predictions.provenance_status.eq('eligible').all()
    assert predictions.actual.eq(.4).all()
    assert audit['prediction_to_score_join']['numerator'] == 2
    assert audit['prediction_to_score_join']['denominator'] == 2
    assert exclusions.empty


def test_normalized_predictions_are_not_reinterpreted_as_raw(tmp_path):
    run, scores, mapping = make_run(tmp_path)
    mapping['prediction_scale'] = 'normalized_percentage_points'
    predictions, excluded, audit = audit_predictions(run, scores, mapping)
    assert predictions.empty
    assert 'normalized_target_not_raw' in set(excluded.reason)
    assert audit['input_predictions'] == 2


def test_missing_membership_and_crossing_roots_are_reported(tmp_path):
    run, scores, mapping = make_run(tmp_path)
    mapping.pop('split_file')
    predictions, excluded, _ = audit_predictions(run, scores, mapping)
    assert 'split_unverifiable' in set(excluded.reason)
    assert not predictions.provenance_status.eq('eligible').any()
    # Rehash the altered split artifact so this checks leakage rather than corruption.
    mapping['split_file'] = 'splits.csv'
    p = run/'splits.csv'
    pd.DataFrame([{'fold':'0','split_group':'root-a','partition':'train'},
                  {'fold':'0','split_group':'root-a','partition':'test'}]).to_csv(p,index=False)
    manifest=json.loads((run/'artifact_manifest.json').read_text())
    manifest['artifacts']['splits.csv']['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    (run/'artifact_manifest.json').write_text(json.dumps(manifest))
    _, excluded, _ = audit_predictions(run,scores,mapping)
    assert 'historical_leakage' in set(excluded.reason)


def test_zero_bytes_missing_runs_and_corrupted_provenance(tmp_path):
    missing = discover_runs([tmp_path/'absent'])
    assert missing.iloc[0].status == 'not_found'
    run, scores, mapping = make_run(tmp_path)
    (run/'predictions.csv').write_bytes(b'')
    assert 'missing_zero_bytes' in set(discover_runs([run]).status)
    predictions, excluded, _ = audit_predictions(run,scores,mapping)
    assert predictions.empty
    assert 'missing_zero_bytes' in set(excluded.reason)


def test_mismatched_baseline_support_and_missing_revision_are_not_eligible(tmp_path):
    run,scores,mapping=make_run(tmp_path)
    table=pd.read_csv(run/'predictions.csv')
    table=table[table.method.eq('weight')]
    table.to_csv(run/'predictions.csv',index=False)
    manifest=json.loads((run/'artifact_manifest.json').read_text())
    manifest['artifacts']['predictions.csv']['sha256']=hashlib.sha256((run/'predictions.csv').read_bytes()).hexdigest()
    (run/'artifact_manifest.json').write_text(json.dumps(manifest))
    predictions, excluded, _=audit_predictions(run,scores,mapping)
    assert 'baseline_support_mismatch' in set(excluded.reason)
    assert predictions.provenance_status.ne('eligible').all()


def test_discovery_is_bounded_and_portable(tmp_path):
    run,_,_=make_run(tmp_path)
    candidates=discover_runs([tmp_path])
    assert candidates.path.str.contains('predictions.csv').any()
    assert not candidates.path.str.startswith('/').any()
    assert candidates.loc[candidates.status.eq('ok'),'sha256'].str.len().eq(64).all()


def rehash(run, *names):
    manifest = json.loads((run/'artifact_manifest.json').read_text())
    for name in names:
        manifest['artifacts'][name] = {'sha256': hashlib.sha256((run/name).read_bytes()).hexdigest()}
    (run/'artifact_manifest.json').write_text(json.dumps(manifest))


def test_changed_protocol_and_missing_input_hash_block_primary_outputs(tmp_path):
    run, scores, mapping = make_run(tmp_path)
    (run/'protocol.json').write_text(json.dumps({'seed': 18}))
    predictions, excluded, audit = audit_predictions(run, scores, mapping)
    assert predictions.provenance_status.ne('eligible').all()
    assert 'artifact_hash_mismatch' in set(excluded.reason)
    assert audit['protocol_verified'] is False
    rehash(run, 'protocol.json')
    (run/'input.csv').write_text('model\na\n')
    (run/'protocol.json').write_text(json.dumps({'seed': 17, 'inputs': {'files': [
        {'path': 'input.csv', 'sha256': '0'*64}]}}))
    rehash(run, 'protocol.json')
    predictions, excluded, audit = audit_predictions(run, scores, mapping)
    assert predictions.provenance_status.ne('eligible').all()
    assert 'artifact_hash_mismatch' in set(excluded.reason)


def test_unverified_checkpoint_and_actual_mismatch_are_descriptive(tmp_path):
    run, scores, mapping = make_run(tmp_path)
    scores['revision_verified'] = False
    predictions, excluded, audit = audit_predictions(run, scores, mapping)
    assert not predictions.revision_verified.any()
    assert predictions.setting.eq('historical_unverified').all()
    assert 'deployment_unverified' in set(excluded.reason)
    assert audit['prediction_to_score_join']['numerator'] == 0
    scores['revision_verified'] = True
    scores.loc[scores.precision.eq('bf16'), 'accuracy'] = .41
    predictions, excluded, _ = audit_predictions(run, scores, mapping)
    assert 'actual_score_mismatch' in set(excluded.reason)
    assert predictions.provenance_status.ne('eligible').all()


def test_duplicate_leakage_and_baseline_fold_identity(tmp_path):
    run, scores, mapping = make_run(tmp_path)
    pd.DataFrame([{'fold': '0', 'split_group': 'root-b', 'duplicate_group': 'dup-a', 'partition': 'train'},
                  {'fold': '0', 'split_group': 'root-a', 'duplicate_group': 'dup-a', 'partition': 'test'}]).to_csv(run/'splits.csv', index=False)
    rehash(run, 'splits.csv')
    predictions, excluded, _ = audit_predictions(run, scores, mapping)
    assert 'historical_leakage' in set(excluded.reason)
    assert predictions.provenance_status.ne('eligible').all()
    table = pd.read_csv(run/'predictions.csv')
    table.loc[table.method.eq('no_weight'), 'fold'] = 1
    table.to_csv(run/'predictions.csv', index=False)
    rehash(run, 'predictions.csv')
    _, excluded, _ = audit_predictions(run, scores, mapping)
    assert 'baseline_support_mismatch' in set(excluded.reason)


def test_schema_mapping_and_out_of_range_predictions(tmp_path):
    run, scores, mapping = make_run(tmp_path)
    table = pd.read_csv(run/'predictions.csv').rename(columns={'method': 'feature_block', 'fold': 'outer_fold'})
    table['prediction'] = 1.5
    table.to_csv(run/'predictions.csv', index=False)
    rehash(run, 'predictions.csv')
    mapping['columns'] = {'method': 'feature_block', 'fold': 'outer_fold'}
    predictions, excluded, audit = audit_predictions(run, scores, mapping)
    assert len(predictions) == 2
    assert predictions.prediction.eq(1.5).all()
    assert predictions.provenance_status.eq('eligible').all()
    assert audit['out_of_range_predictions'] == 2
    assert excluded.empty


def test_discovery_ignores_unrelated_files_secrets_and_symlinks(tmp_path):
    run, _, _ = make_run(tmp_path)
    (run/'checkpoint.bin').write_bytes(b'weights')
    (run/'unrelated.csv').write_text('nothing\n')
    (run/'.env.hidden').mkdir()
    (run/'.env.hidden'/'predictions.csv').write_text('secret\n')
    (run/'linked_predictions.csv').symlink_to(run/'predictions.csv')
    discovered = discover_runs([tmp_path])
    assert not discovered.path.str.contains(r'checkpoint|unrelated|\.env|linked').any()
    assert len(discovered) == 4


def test_baseline_alone_cannot_claim_a_paired_comparison(tmp_path):
    run, scores, mapping = make_run(tmp_path)
    table = pd.read_csv(run/'predictions.csv')
    table = table[table.method.eq('no_weight')]
    table.to_csv(run/'predictions.csv', index=False)
    rehash(run, 'predictions.csv')
    predictions, excluded, _ = audit_predictions(run, scores, mapping)
    assert 'baseline_support_mismatch' in set(excluded.reason)
    assert predictions.provenance_status.ne('eligible').all()


def test_case_variant_checkpoints_cannot_cross_train_and_test(tmp_path):
    run, scores, mapping = make_run(tmp_path)
    key = 'org/a@' + 'a'*40
    pd.DataFrame([
        {'fold': '0', 'split_group': 'root-b', 'partition': 'train', 'model_key': 'Org/A@' + 'A'*40},
        {'fold': '0', 'split_group': 'root-a', 'partition': 'test', 'model_key': key},
    ]).to_csv(run/'splits.csv', index=False)
    rehash(run, 'splits.csv')
    predictions, excluded, _ = audit_predictions(run, scores, mapping)
    assert 'historical_leakage' in set(excluded.reason)
    assert predictions.provenance_status.ne('eligible').all()


def test_held_out_membership_accepts_repo_case_and_preserves_original_ids(tmp_path):
    run, scores, mapping = make_run(tmp_path)
    pd.DataFrame([
        {'fold': '0', 'split_group': 'root-b', 'partition': 'train', 'model_key': 'org/b@' + 'b'*40},
        {'fold': '0', 'split_group': 'root-a', 'partition': 'test', 'model_key': 'Org/A@' + 'A'*40},
    ]).to_csv(run/'splits.csv', index=False)
    rehash(run, 'splits.csv')
    predictions, excluded, _ = audit_predictions(run, scores, mapping)
    assert predictions.provenance_status.eq('eligible').all()
    assert predictions.model_key.eq('org/a@' + 'a'*40).all()
    assert excluded.empty


def test_excluded_baseline_provenance_cannot_support_eligible_weight_rows(tmp_path):
    run, scores, mapping = make_run(tmp_path)
    pd.DataFrame([
        {'fold': '0', 'split_group': 'root-b', 'partition': 'train', 'duplicate_group': 'dup-b'},
        {'fold': '0', 'split_group': 'root-a', 'partition': 'test', 'duplicate_group': 'dup-a'},
    ]).to_csv(run/'splits.csv', index=False)
    table = pd.read_csv(run/'predictions.csv')
    table.loc[table.method.eq('no_weight'), 'duplicate_group'] = 'wrong-duplicate'
    table.to_csv(run/'predictions.csv', index=False)
    rehash(run, 'splits.csv', 'predictions.csv')
    predictions, excluded, _ = audit_predictions(run, scores, mapping)
    assert 'split_unverifiable' in set(excluded.reason)
    assert 'baseline_support_mismatch' in set(excluded.reason)
    assert predictions.provenance_status.ne('eligible').all()


def test_baseline_and_weight_must_have_identical_settings(tmp_path):
    for baseline_setting in ['within_family', 'historical_unverified']:
        run, scores, mapping = make_run(tmp_path/baseline_setting)
        table = pd.read_csv(run/'predictions.csv')
        table['setting'] = 'root_holdout'
        table.loc[table.method.eq('no_weight'), 'setting'] = baseline_setting
        table.to_csv(run/'predictions.csv', index=False)
        rehash(run, 'predictions.csv')
        predictions, excluded, _ = audit_predictions(run, scores, mapping)
        assert 'baseline_support_mismatch' in set(excluded.reason)
        assert predictions.provenance_status.ne('eligible').all()


def test_third_method_mismatch_preserves_only_complete_baseline_comparisons(tmp_path):
    run, scores, mapping = make_run(tmp_path)
    table = pd.read_csv(run/'predictions.csv')
    third = table[table.method.eq('weight')].assign(method='weight_incomplete')
    # Same unique rows, different multiplicity: a set comparison is insufficient.
    table = pd.concat([table, third, third], ignore_index=True)
    table.to_csv(run/'predictions.csv', index=False)
    rehash(run, 'predictions.csv')
    predictions, excluded, _ = audit_predictions(run, scores, mapping)
    eligible = predictions[predictions.provenance_status.eq('eligible')]
    assert set(eligible.method) == {'weight', 'no_weight'}
    assert len(eligible) == 2
    assert predictions.loc[predictions.method.eq('weight_incomplete'), 'provenance_status'].ne('eligible').all()
    assert 'baseline_support_mismatch' in set(excluded.reason)
