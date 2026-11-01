"""Metadata-only references; certified labels remain outside the construction DAG."""
import hashlib
import json
import re

from .pins import resolve_pins


_SIZES = (14, 31, 70, 160, 410)
_HUB_ROOT = re.compile(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[0-9a-f]{40}\Z')


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _source(source):
    if (not isinstance(source, dict) or
            not all(isinstance(source.get(k), str) and source[k].strip()
                    for k in ('url', 'locator', 'statement')) or
            not source['url'].startswith('https://')):
        raise ValueError('reference requires authoritative source URL, locator, and statement')


def _label(query, candidate, *, same_root):
    return {'query_key': query, 'candidate_key': candidate, 'eligible': True,
            'truth_parent': False, 'truth_ancestor': False,
            'truth_same_root': same_root, 'truth_tier': 'verified_hub'}


def resolve_independents(config: dict, *, api, download_text) -> list[dict]:
    """Pin an explicit published roster using selected Hub metadata only.

    The config transcribes the cited authoritative roster: repository names are
    never inferred or enumerated. The caller supplies a Hub API and the same
    pinned-text callback used by :func:`lab.pins.resolve_pins`. Unverified sibling
    slots remain unresolved records, never negative truth. Returned labels use
    model IDs; exporters must join them to their pinned candidate/query keys.
    """
    if config.get('schema_version') != '1':
        raise ValueError('unsupported negatives schema version')
    source = config.get('roster_source')
    _source(source)
    roster = config.get('roster', [])
    expected = {(size, seed) for size in _SIZES for seed in range(1, 10)}
    pairs, repositories = set(), set()
    entries, metadata = [], {}
    # Validate the entire selection before issuing even one metadata request.
    for row in roster:
        size, seed, repo = row.get('size_m'), row.get('seed'), row.get('repo_id')
        if type(size) is not int or type(seed) is not int or (size, seed) not in expected:
            raise ValueError('roster requires published sizes and seeds 1 through 9')
        if (size, seed) in pairs or repo in repositories:
            raise ValueError('duplicate roster size/seed or repository')
        if not isinstance(repo, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo):
            raise ValueError('invalid roster repository')
        if row.get('url') != 'https://huggingface.co/' + repo:
            raise ValueError('roster selected URL does not match repository')
        pairs.add((size, seed))
        repositories.add(repo)
        size_slot, seed_slot = _SIZES.index(size), seed - 1
        label = f'poly-size-slot-{size_slot}-seed-slot-{seed_slot}'
        entries.append({'model_id': label, 'repo_id': repo, 'base_family': f'pythia-{size}m',
                        'revision': row.get('revision', 'main')})
        metadata[label] = {'reference_type': 'polypythias', 'size_m': size, 'seed': seed,
                           'size_slot': size_slot, 'seed_slot': seed_slot,
                           'selected_url': row['url']}
    if pairs != expected or len(roster) != 45:
        raise ValueError('incomplete roster: expected five sizes by nine seeds')
    roster_hash = _hash(sorted(roster, key=lambda r: (r['size_m'], r['seed'])))
    siblings = config.get('siblings', [])
    sibling_ids = {'posttraining-sibling-slot-0', 'posttraining-sibling-slot-1'}
    if len(siblings) != 2 or {r.get('model_id') for r in siblings} != sibling_ids:
        raise ValueError('expected two distinct sibling slots')
    unresolved = []
    selected_siblings = []
    for row in sorted(siblings, key=lambda r: r['model_id']):
        label = row['model_id']
        if row.get('status') == 'unresolved':
            if not row.get('unresolved_reason') or row.get('repo_id'):
                raise ValueError('unresolved sibling requires reason and no selected repository')
            unresolved.append({'model_id': label, 'reference_type': 'sibling',
                               'status': 'unresolved', 'unresolved_reason': row['unresolved_reason']})
            continue
        evidence = row.get('lineage_evidence', {})
        _source(evidence.get('source'))
        if evidence.get('nonancestry') is not True or not _HUB_ROOT.fullmatch(evidence.get('common_root', '')):
            raise ValueError('siblings require pinned common root and documented nonancestry')
        repo = row.get('repo_id')
        if (not isinstance(repo, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo)
                or repo in repositories or not row.get('base_family')):
            raise ValueError('sibling requires distinct repository and architecture family')
        repositories.add(repo)
        entries.append({'model_id': label, 'repo_id': repo, 'base_family': row['base_family'],
                        'revision': row.get('revision', 'main')})
        metadata[label] = {'reference_type': 'sibling', 'lineage_evidence': evidence}
        selected_siblings.append(row)
    if selected_siblings and (len(selected_siblings) != 2 or
            selected_siblings[0]['lineage_evidence'] != selected_siblings[1]['lineage_evidence']):
        raise ValueError('both siblings must document the same root and pair evidence')

    architectures = {}
    def read(repo_id, filename, **kwargs):
        text = download_text(repo_id, filename, **kwargs)
        if filename == 'config.json':
            model = json.loads(text)
            # Exact configurations are a conservative architecture compatibility
            # check. Provenance, runtime flags, and dropout do not
            # change the causal-model tensor topology. Raw config hashes remain
            # untouched in the pin evidence.
            architecture = {k: v for k, v in model.items() if k not in
                            {'_name_or_path', 'transformers_version', 'torch_dtype',
                             'classifier_dropout', 'attention_dropout',
                             'pad_token_id', 'use_cache'}}
            if not architecture.get('model_type'):
                raise ValueError('reference configuration requires model_type')
            architectures[repo_id] = _hash(architecture)
        return text

    pins = resolve_pins({'entries': sorted(entries, key=lambda r: r['model_id'])},
                        api=api, download_text=read)
    for label, pin in pins.items():
        pin.update(metadata[label])
        pin['evaluation_labels'] = []
        if pin['reference_type'] == 'polypythias':
            pin['independence_evidence'] = {'source': source, 'roster_sha256': roster_hash,
                                           'relation': 'independently_initialized_pretraining'}
        if pin['status'] == 'pinned':
            pin['architecture_sha256'] = architectures[pin['repo_id']]
    # Independent roots are certified by the training-run source. Only equal
    # architectures are eligible for weight-comparison negatives.
    for size in _SIZES:
        group = [p for p in pins.values() if p.get('size_m') == size and p['status'] == 'pinned']
        if len({p['architecture_sha256'] for p in group}) > 1:
            raise ValueError('published size group has inconsistent architectures')
        for pin in group:
            pin['evaluation_labels'] = [_label(pin['model_id'], other['model_id'], same_root=False)
                                        for other in group if other is not pin]
    if selected_siblings:
        group = [pins[r['model_id']] for r in selected_siblings]
        if all(p['status'] == 'pinned' for p in group):
            if group[0]['architecture_sha256'] != group[1]['architecture_sha256']:
                raise ValueError('sibling references require the same architecture')
            for pin, other in (group, list(reversed(group))):
                pin['evaluation_labels'] = [_label(pin['model_id'], other['model_id'], same_root=True)]
    return sorted([*pins.values(), *unresolved], key=lambda r: r['model_id'])
