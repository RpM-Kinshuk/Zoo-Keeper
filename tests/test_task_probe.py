import math
import pytest


def test_progress_can_record_transformers_loading_info_sets(tmp_path):
    import json
    from scripts.task_probe import loading_record,write
    # Transformers 5.9 returns sets here, even when loading succeeds.
    loading=dict(missing_keys=set(),unexpected_keys=set(),mismatched_keys=[],error_msgs=[])
    write(tmp_path/'worker.json',dict(status='LOADED',loading_info=loading_record(loading)))
    assert json.loads((tmp_path/'worker.json').read_text())['loading_info']['missing_keys']==[]


def test_balanced_splits_keep_normalized_groups_separate():
    from scripts.task_probe import balanced_split
    rows=[dict(id=f'{label}:{i}',text=f'class {label} sentence {i}',label=label)
          for label in range(6) for i in range(40)]
    rows += [dict(id='duplicate',text=' CLASS 0 SENTENCE 0 ',label=0)]
    split,stats=balanced_split(rows,6)
    assert [len(split[k]) for k in ['probe','validation']]==[32,128]
    assert not ({r['group'] for r in split['probe']} & {r['group'] for r in split['validation']})
    for part in split.values():
        counts=[sum(r['label']==label for r in part) for label in range(6)]
        assert max(counts)-min(counts)<=1
    assert stats['duplicate_rows']==1


def test_rotation_aggregation_uses_semantic_labels_and_mean_log_probability():
    from scripts.task_probe import aggregate_rotations
    # Opposite mappings; arithmetic probability averaging would choose class0.
    assert aggregate_rotations([[math.log(.9),math.log(.099)],
                                [math.log(.4),math.log(.01)]],[0,1])==1
    assert aggregate_rotations([[0.,0.],[0.,0.]],[0,1])==0


def test_probe_selection_requires_complete_cohort_and_discloses_tag_fallback():
    from scripts.task_probe import choose
    meta={'small':dict(params_b=.6,downloads=20,task_match=False),
          'large':dict(params_b=3.8,downloads=10,task_match=False)}
    result=choose({'small':[True]*32,'large':[True]*32},meta)
    assert result['choices']==dict(probe='small',largest='large',downloads='small',task_labels='small')
    assert result['task_label_status']=='no_matches_downloads_fallback'
    with pytest.raises(ValueError):choose({'small':[True]*32},meta)
    with pytest.raises(ValueError):choose({'small':[True]*31,'large':[True]*32},meta)


def test_literal_continuation_must_extend_exact_prompt_by_one_distinct_token():
    from scripts.task_probe import label_tokens
    class Tokenizer:
        def encode(self,text,**kwargs):
            if text=='prefix':return [1,2]
            if text=='prefix A':return [1,2,3]
            return [1,9,4]  # A boundary merge changes the prefix.
    with pytest.raises(ValueError):label_tokens(Tokenizer(),'prefix',2)
