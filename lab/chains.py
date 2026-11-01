"""Check the public minimum using verified completion IDs, without launching jobs."""
from collections import defaultdict

from .registry import expand_inventory, validate_tables


def chain_paths(models, edges) -> list[list[str]]:
    validate_tables(models, edges)
    children = defaultdict(set)
    incoming = set()
    for edge in edges.itertuples():
        if edge.role == 'parent':
            children[edge.parent].add(edge.child)
            incoming.add(edge.child)
    paths = []

    def walk(path):
        if not children[path[-1]]:
            if len(path) > 1:
                paths.append(path)
            return
        for child in sorted(children[path[-1]]):
            walk(path + [child])

    for root in sorted(set(models.model_id) - incoming):
        walk([root])
    return sorted(paths)


def verify_minimum(models, edges, completed_ids: set[str]) -> dict:
    """A planned row is not completion; callers supply IDs verified on disk.

    The blind contribution is deliberately never read or inferred here. It
    remains pending C's attestation even when every public minimum is met.
    """
    validate_tables(models, edges)
    inventory = expand_inventory({})
    present = set(models.model_id)
    complete = set(completed_ids) & present
    limits = {'lora': 24, 'sft': 8, 'dpo': 8, 'merge': 30,
              'abliteration': 8, 'quantization_sim': 12,
              'laundering': 16, 'decision': 6}
    checks = {}
    planned = set()
    for group, minimum in limits.items():
        ids = {row['model_id'] for row in inventory[group]}
        planned.update(ids)
        count = len(ids & complete)
        checks[group] = {'count': count, 'minimum': minimum,
                         'verdict': 'PASS' if count >= minimum else 'FAIL'}
    decoder_bases = [row for row in inventory['bases'] if row['model_id'] != 'modernbert']
    planned.update(row['model_id'] for row in inventory['bases'])
    n_bases = sum(row['model_id'] in complete for row in decoder_bases)
    families = {row['base_family'] for row in decoder_bases if row['model_id'] in complete}
    checks['bases'] = {'count': n_bases, 'minimum': 6, 'verdict': 'PASS' if n_bases >= 6 else 'FAIL'}
    checks['base_families'] = {'count': len(families), 'minimum': 3,
                             'verdict': 'PASS' if len(families) >= 3 else 'FAIL'}
    references = {row['model_id'] for row in inventory['references']}
    planned.update(references)
    n_references = len(references & complete)
    checks['references'] = {'count': n_references, 'minimum': 47,
                            'verdict': 'PASS' if n_references >= 47 else 'FAIL'}
    parent_edges = set(edges.loc[edges.role == 'parent', ['parent', 'child']].itertuples(index=False, name=None))
    paths = []
    for anchor in ('q25-05', 'q3-06', 'smol-360'):
        path = [anchor, anchor + '-sft-s0', anchor + '-dpo-s0', anchor + '-quantization-s8']
        if set(path) <= complete and all(pair in parent_edges for pair in zip(path, path[1:])):
            paths.append(path)
    checks['chains'] = {'count': len(paths), 'minimum': 3,
                        'verdict': 'PASS' if len(paths) == 3 else 'FAIL'}
    public_ok = planned <= complete and all(check['verdict'] == 'PASS' for check in checks.values())
    checks['laundering_combined'] = {'minimum': 20, 'public_count': checks['laundering']['count'],
                                     'verdict': 'PENDING_C_ATTESTATION'}
    return {'status': 'INCOMPLETE', 'public_verdict': 'PASS' if public_ok else 'FAIL',
            'checks': checks, 'chain_paths': paths, 'missing_ids': sorted(planned - complete),
            'unknown_completion_ids': sorted(set(completed_ids) - present)}
