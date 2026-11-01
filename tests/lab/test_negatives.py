"""Offline contracts for authoritative seed rosters and sibling truth."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from lab.negatives import resolve_independents
from lab.pins import verify_pin
from lab.registry import expand_inventory, materialize_tables


SIZES = [14, 31, 70, 160, 410]
SOURCE = {'url': 'https://publisher.example/paper.pdf', 'locator': 'Table 5; section 2',
          'statement': 'Distinct independently initialized pretraining runs for seeds 1 through 9.'}


def config():
    return {'schema_version': '1', 'roster_source': SOURCE,
            'roster': [{'size_m': size, 'seed': seed,
                        'repo_id': f'publisher/run-{size}-{seed}',
                        'url': f'https://huggingface.co/publisher/run-{size}-{seed}'}
                       for size in SIZES for seed in range(1, 10)],
            'siblings': [{'model_id': f'posttraining-sibling-slot-{i}', 'status': 'unresolved',
                          'unresolved_reason': 'Published branches not yet verified'} for i in range(2)]}


class Hub:
    """Only selected model metadata is available; enumeration cannot succeed."""
    def __init__(self, cfg):
        self.rows = {r['repo_id']: r for r in cfg['roster'] +
                     [r for r in cfg['siblings'] if r.get('repo_id')]}

    def model_info(self, repo_id, *, revision):
        expected = self.rows[repo_id].get('revision', 'main')
        assert revision == expected
        return {'sha': expected if len(expected) == 40 else hashlib.sha1(repo_id.encode()).hexdigest(), 'gated': False,
                'card_data': {'license': 'apache-2.0'},
                'siblings': [{'rfilename': n} for n in ['README.md', 'config.json', 'tokenizer.json']]}

    def download(self, repo_id, filename, *, revision, repo_type):
        expected = self.rows[repo_id].get('revision', 'main')
        assert revision == (expected if len(expected) == 40 else hashlib.sha1(repo_id.encode()).hexdigest())
        assert repo_type == 'model'
        return {'README.md': '---\nlicense: apache-2.0\n---\n',
                'config.json': json.dumps({'model_type': 'gpt_neox',
                                          'hidden_size': self.rows[repo_id].get('size_m', 160)}),
                'tokenizer.json': '{}'}[filename]


def resolve(cfg):
    hub = Hub(cfg)
    return resolve_independents(cfg, api=hub, download_text=hub.download)


def test_complete_roster_pins_slots_and_retains_independence_evidence():
    cfg = config()
    rows = resolve(cfg)
    assert len(rows) == 47
    pinned = [r for r in rows if r['status'] == 'pinned']
    assert len(pinned) == 45
    assert len({(r['size_m'], r['seed']) for r in pinned}) == 45
    assert rows == resolve({**cfg, 'roster': list(reversed(cfg['roster']))})
    for row in pinned:
        verify_pin(row)
        assert row['independence_evidence']['source'] == SOURCE
        assert len(row['independence_evidence']['roster_sha256']) == 64
        assert row['license_evidence_type'] == 'pinned_readme_declaration'
        assert row['base_family'] == f"pythia-{row['size_m']}m"
        assert not row.get('parents') and not row.get('edges')
        labels = row['evaluation_labels']
        assert len(labels) == 8
        assert all(l['eligible'] and l['truth_same_root'] is False and
                   l['truth_ancestor'] is False and l['truth_parent'] is False for l in labels)
        assert all(l['candidate_key'].startswith(f"poly-size-slot-{SIZES.index(row['size_m'])}-") for l in labels)
    assert rows[0]['model_id'] == 'poly-size-slot-0-seed-slot-0'
    assert rows[0]['seed'] == 1


@pytest.mark.parametrize('change', ['duplicate', 'missing', 'repository', 'url', 'source', 'seed0'])
def test_bad_rosters_fail_before_api(change):
    cfg = config()
    if change == 'duplicate': cfg['roster'][-1] = copy.deepcopy(cfg['roster'][0])
    if change == 'missing': cfg['roster'].pop()
    if change == 'repository': cfg['roster'][1]['repo_id'] = cfg['roster'][0]['repo_id']
    if change == 'url': cfg['roster'][0]['url'] = 'https://huggingface.co/other/model'
    if change == 'source': cfg['roster_source'] = {}
    if change == 'seed0': cfg['roster'][0]['seed'] = 0
    with pytest.raises(ValueError):
        resolve_independents(cfg, api=None, download_text=None)


def test_reference_pins_never_add_construction_edges():
    cfg = config()
    refs = [r for r in resolve(cfg) if r['status'] == 'pinned']
    inventory = expand_inventory({})
    reference_rows = [r for r in inventory['references'] if r['reference_type'] == 'polypythias']
    inventory = {key: [] for key in inventory}
    inventory['references'] = reference_rows
    models, edges = materialize_tables(inventory, {r['model_id']: r for r in refs})
    assert len(models) == 45
    assert set(models.kind) == {'independent'}
    assert edges.empty


def sibling_config():
    cfg = config()
    evidence = {'source': {'url': 'https://publisher.example/branches', 'locator': 'Lineage',
                           'statement': 'Both branches independently tuned from the same root.'},
                'common_root': 'publisher/root@' + 'a' * 40, 'nonancestry': True}
    cfg['siblings'] = [{'model_id': f'posttraining-sibling-slot-{i}', 'repo_id': f'publisher/branch-{i}',
                        'base_family': 'example', 'lineage_evidence': evidence} for i in range(2)]
    return cfg


def test_siblings_are_same_root_positives_and_parent_negatives_without_edges():
    rows = resolve(sibling_config())
    siblings = [r for r in rows if r['reference_type'] == 'sibling']
    assert len(siblings) == 2
    for row in siblings:
        verify_pin(row)
        assert not row.get('parents') and not row.get('edges')
        assert len(row['evaluation_labels']) == 1
        label = row['evaluation_labels'][0]
        assert label['truth_same_root'] is True
        assert label['truth_parent'] is False and label['truth_ancestor'] is False
        assert label['eligible'] is True


@pytest.mark.parametrize('change', ['nonancestry', 'root', 'architecture'])
def test_sibling_evidence_cannot_turn_ancestors_or_other_architectures_into_negatives(change):
    cfg = sibling_config()
    if change == 'nonancestry': cfg['siblings'][0]['lineage_evidence']['nonancestry'] = False
    if change == 'root': cfg['siblings'][0]['lineage_evidence']['common_root'] = 'publisher/root@main'
    if change == 'architecture': cfg['siblings'][1]['size_m'] = 31
    with pytest.raises(ValueError): resolve(cfg)


def test_checked_in_published_roster_is_complete_and_deterministic():
    cfg = json.loads((Path(__file__).parents[2] / 'lab/configs/negatives.json').read_text())
    rows = resolve(cfg)
    assert len([r for r in rows if r['status'] == 'pinned']) == 47
    assert not [r for r in rows if r['status'] == 'unresolved']
    assert len([r for r in rows if r['reference_type'] == 'sibling']) == 2
    assert {r['size_m'] for r in cfg['roster']} == {14, 31, 70, 160, 410}
    assert cfg['roster'][0]['repo_id'] == 'EleutherAI/pythia-14m-seed1'
    assert cfg['roster'][-1]['repo_id'] == 'EleutherAI/pythia-410m-seed9'


def test_classifier_dropout_metadata_does_not_change_causal_architecture():
    cfg = config()
    hub = Hub(cfg)

    def download(repo_id, filename, **kwargs):
        text = hub.download(repo_id, filename, **kwargs)
        if filename == 'config.json':
            model = json.loads(text)
            model['classifier_dropout'] = 0.1 if repo_id.endswith('-1') else None
            return json.dumps(model)
        return text

    rows = resolve_independents(cfg, api=hub, download_text=download)
    pinned = [r for r in rows if r['status'] == 'pinned']
    assert len(pinned) == 45
    assert all(len(row['evaluation_labels']) == 8 for row in pinned)
    first, second = pinned[:2]
    assert first['architecture_sha256'] == second['architecture_sha256']
    assert first['config_sha256'] != second['config_sha256']
    assert first['config_file_sha256'] != second['config_file_sha256']


def test_hidden_dimension_change_is_not_compatible_architecture():
    cfg = config()
    hub = Hub(cfg)

    def download(repo_id, filename, **kwargs):
        text = hub.download(repo_id, filename, **kwargs)
        if filename == 'config.json' and repo_id == 'publisher/run-14-1':
            model = json.loads(text)
            model['hidden_size'] = 999
            return json.dumps(model)
        return text

    with pytest.raises(ValueError, match='inconsistent architectures'):
        resolve_independents(cfg, api=hub, download_text=download)


def test_sibling_runtime_config_does_not_change_weight_architecture():
    cfg = sibling_config()
    hub = Hub(cfg)

    def download(repo_id, filename, **kwargs):
        text = hub.download(repo_id, filename, **kwargs)
        if filename == 'config.json' and repo_id.startswith('publisher/branch-'):
            model = json.loads(text)
            model['use_cache'] = repo_id.endswith('-1')
            if repo_id.endswith('-1'):
                model['pad_token_id'] = 2
                model['attention_dropout'] = 0.0
            return json.dumps(model)
        return text

    rows = resolve_independents(cfg, api=hub, download_text=download)
    a, b = [r for r in rows if r['reference_type'] == 'sibling']
    assert a['architecture_sha256'] == b['architecture_sha256']
    assert a['config_file_sha256'] != b['config_file_sha256']
    assert a['evaluation_labels'][0]['truth_same_root'] is True


def test_explicit_revisions_are_used_without_resolving_main_again():
    cfg = sibling_config()
    for row in cfg['roster'] + cfg['siblings']:
        row['revision'] = hashlib.sha1(row['repo_id'].encode()).hexdigest()
    rows = resolve(cfg)
    assert len([r for r in rows if r['status'] == 'pinned']) == 47


def test_checked_in_tables_cover_the_complete_planned_inventory():
    import pandas as pd
    from lab.registry import validate_tables
    root = Path(__file__).parents[2] / 'lab'
    pins = json.loads((root / 'pins.json').read_text())
    models = pd.read_csv(root / 'models.csv', keep_default_na=False)
    edges = pd.read_csv(root / 'edges.csv', keep_default_na=False)
    inventory = expand_inventory({})
    expected_models, expected_edges = materialize_tables(inventory, pins)
    validate_tables(models, edges)
    assert len(models) == 166 and len(edges) == 187
    assert len(pins) == 54
    assert set(models.model_id) == {r['model_id'] for group in inventory.values() for r in group}
    assert (root / 'models.csv').read_text() == expected_models.to_csv(index=False, lineterminator='\n')
    assert (root / 'edges.csv').read_text() == expected_edges.to_csv(index=False, lineterminator='\n')
    pairs = json.loads((root / 'reference_pairs.json').read_text())
    assert sum(p['label'] == 'independent' for p in pairs['pairs']) == 180
    assert sum(p['label'] == 'sibling' for p in pairs['pairs']) == 1
    sibling_ids = set(pairs['sibling_model_ids'])
    assert not any(sibling_ids & set(p['model_ids']) for p in pairs['pairs'] if p['label'] == 'independent')
