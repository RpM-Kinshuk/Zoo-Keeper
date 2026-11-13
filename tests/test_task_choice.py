import pytest


def test_exact_letter_parser_counts_verbose_and_empty_outputs_wrong():
    from scripts.task_choice import parse_label
    assert parse_label(' B\n', 4) == 1
    assert parse_label('B.', 4) is None
    assert parse_label('Answer: B', 4) is None
    assert parse_label('', 4) is None
    assert parse_label('C', 2) is None
    assert parse_label('A', 2, hit_limit=True) is None


def test_probe_tie_uses_cost_and_never_requires_validation():
    from scripts.task_choice import choose
    meta = {'big': {'params_b': 3., 'downloads': 20., 'historical_mean': .9},
            'small': {'params_b': 1., 'downloads': 30., 'historical_mean': .8}}
    probe = {'big': [True, False], 'small': [False, True]}
    choices = choose(probe, meta)
    assert choices == {'probe': 'small', 'largest': 'big', 'downloads': 'small', 'historical_mean': 'big'}
    with pytest.raises(ValueError, match='aligned'):
        choose({'big': [True], 'small': [False, True]}, meta)


def test_split_keeps_duplicate_text_groups_together_and_is_order_invariant():
    from scripts.task_choice import split_rows
    rows = [{'id': str(i), 'text': 'same' if i < 2 else str(i), 'label': i % 2} for i in range(20)]
    a = split_rows(rows, 3, 8)
    b = split_rows(rows[::-1], 3, 8)
    assert a == b
    assert len(a['probe']) == 3 and len(a['validation']) == 8
    assert not {r['text'] for r in a['probe']} & {r['text'] for r in a['validation']}
    assert len({r['text'] for rs in a.values() for r in rs}) == 11


def test_paired_report_uses_discordances_and_finite_panel_regret():
    from scripts.task_choice import score_choices
    correct = {'a': [True, True, False, False], 'b': [True, False, True, True]}
    result = score_choices(correct, {'probe': 'a', 'largest': 'b'})
    assert result['policies']['probe']['accuracy'] == .5
    assert result['policies']['probe']['finite_panel_regret'] == .25
    assert result['probe_vs_largest']['gain'] == -.25
    assert result['probe_vs_largest']['probe_only_correct'] == 1
    assert result['probe_vs_largest']['largest_only_correct'] == 2
    assert result['uniform_expected_accuracy'] == .625


def test_fresh_cache_receipt_rejects_inventory_and_file_changes(tmp_path):
    from scripts.task_choice import validate_cached_file
    p=tmp_path/'weights';p.write_bytes(b'valid')
    s=p.stat();receipt=dict(absolute_path=str(p.resolve()),dev=s.st_dev,ino=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,expected_sha256='a'*64,observed_sha256='a'*64)
    validate_cached_file(p,'a'*64,receipt)
    with pytest.raises(ValueError,match='inventory'):
        validate_cached_file(p,'b'*64,receipt)
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):
        validate_cached_file(p,'a'*64,receipt)
