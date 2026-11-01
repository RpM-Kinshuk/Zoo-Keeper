"""Deterministic planned inventory, validated DAGs, and paired table snapshots."""
import heapq
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from .artifacts import atomic_json, content_hash
from .schema import (MODEL_COLUMNS, EDGE_COLUMNS, EDGE_TYPES, HUB_PATH,
                     LAB_SCHEMA_VERSION, LOCAL_KEY, canonical_json, contained_path,
                     validate_id, validate_relative_path)


def _blank(value):
    return value is None or (isinstance(value, str) and value == '')


def topological_order(models: pd.DataFrame, edges: pd.DataFrame) -> list[str]:
    children = {model_id: set() for model_id in models.model_id}
    indegree = {model_id: 0 for model_id in models.model_id}
    for edge in edges.itertuples(index=False):
        if edge.parent not in children or edge.child not in children:
            raise ValueError(f'{edge.child}: absent parent or child {edge.parent}')
        if edge.child not in children[edge.parent]:
            children[edge.parent].add(edge.child)
            indegree[edge.child] += 1
    ready = [model_id for model_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        parent = heapq.heappop(ready)
        order.append(parent)
        for child in sorted(children[parent]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(order) != len(children):
        raise ValueError('cycle involving ' + ','.join(sorted(k for k, v in indegree.items() if v)))
    return order


def validate_tables(models: pd.DataFrame, edges: pd.DataFrame) -> None:
    if list(models.columns) != MODEL_COLUMNS:
        raise ValueError('models: expected columns ' + ','.join(MODEL_COLUMNS))
    if list(edges.columns) != EDGE_COLUMNS:
        raise ValueError('edges: expected columns ' + ','.join(EDGE_COLUMNS))
    duplicate = models.loc[models.model_id.duplicated(), 'model_id'].tolist()
    if duplicate:
        raise ValueError('duplicate model IDs: ' + ','.join(map(str, duplicate)))
    kinds = {}
    for row in models.itertuples(index=False):
        validate_id(row.model_id)
        if row.kind not in {'base', 'derived', 'independent'}:
            raise ValueError(f'{row.model_id}: unknown kind {row.kind}')
        kinds[row.model_id] = row.kind
        if not isinstance(row.path, str):
            raise ValueError(f'{row.model_id}: missing path')
        if '@' in row.path:
            if not HUB_PATH.fullmatch(row.path) or row.kind == 'derived':
                raise ValueError(f'{row.model_id}: invalid pinned Hub path')
        else:
            validate_relative_path(row.path, row.model_id)
        for key in ['base_family', 'license']:
            if not isinstance(getattr(row, key), str) or not getattr(row, key).strip():
                raise ValueError(f'{row.model_id}: missing {key}')
    seen = set()
    relations = {}
    for row in edges.itertuples(index=False):
        if row.child not in kinds:
            raise ValueError(f'{row.child}: absent child')
        if row.parent not in kinds:
            raise ValueError(f'{row.child}: absent parent {row.parent}')
        if row.child == row.parent:
            raise ValueError(f'{row.child}: self-edge')
        if kinds[row.child] == 'independent':
            raise ValueError(f'{row.child}: independent has incoming truth edge')
        key = (row.child, row.parent, row.role)
        if key in seen:
            raise ValueError(f'{row.child}: duplicate edge from {row.parent}')
        seen.add(key)
        if row.edge_type not in EDGE_TYPES:
            raise ValueError(f'{row.child}: unknown edge type {row.edge_type}')
        if row.role not in {'parent', 'base'} or (row.role == 'base' and row.edge_type != 'merge'):
            raise ValueError(f'{row.child}: invalid role {row.role}')
        if row.child in relations and relations[row.child] != row.edge_type:
            raise ValueError(f'{row.child}: inconsistent edge types')
        relations[row.child] = row.edge_type
        if not _blank(row.coefficient):
            try:
                finite = math.isfinite(float(row.coefficient))
            except (TypeError, ValueError):
                finite = False
            if not finite or row.edge_type != 'merge' or row.role == 'base':
                raise ValueError(f'{row.child}: invalid coefficient')
        try:
            params = json.loads(row.params_json)
            canonical_json(params)
            if not isinstance(params, dict):
                raise ValueError('expected object')
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{row.child}: invalid params_json') from exc
    topological_order(models, edges)
    incoming = set(edges.child)
    for model_id, kind in kinds.items():
        if kind == 'derived' and model_id not in incoming:
            raise ValueError(f'{model_id}: derived model has no parents')
        if kind == 'base' and model_id in incoming:
            raise ValueError(f'{model_id}: base has incoming truth edge')


def expand_inventory(config: dict) -> dict[str, list[dict]]:
    default = json.loads((Path(__file__).parent / 'configs/inventory.json').read_text())
    settings = {**default, **config}
    if settings['schema_version'] != LAB_SCHEMA_VERSION or settings['seed'] != 20261028:
        raise ValueError('inventory: unsupported schema or seed')
    groups = ['lora', 'sft', 'dpo', 'merge', 'abliteration', 'quantization_sim', 'laundering', 'decision']
    result = {key: [] for key in ['bases', *groups, 'references']}
    for base in settings['bases']:
        result['bases'].append({**base, 'kind': 'base', 'status': 'unresolved',
                                'unresolved_reason': 'requires verified base pin and license evidence'})
    anchors = settings['anchors']
    merge_anchors = settings['merge_anchors']
    by_id = {r['model_id']: r for r in result['bases']}
    if len(by_id) != 7 or len(anchors) != 4 or merge_anchors != anchors[:3]:
        raise ValueError('inventory: expected seven bases, four anchors, first three merge anchors')
    seed_index = 0

    def add(group, anchor, suffix, parents=None, edge_type=None, **parameters):
        nonlocal seed_index
        row = {'model_id': anchor + '-' + suffix, 'kind': 'derived', 'anchor': anchor,
               'base_family': by_id[anchor]['base_family'], 'path': anchor + '-' + suffix,
               'parents': parents if parents is not None else [anchor],
               'edge_type': edge_type or group, 'seed': settings['seed'] + seed_index,
               **parameters}
        seed_index += 1
        result[group].append(row)
        return row['model_id']

    for anchor in anchors:
        for task in ['instruction', 'dialogue']:
            for rank in [8, 64, 256]:
                add('lora', anchor, f'lora-{task}-r{rank}', task=task, rank=rank, alpha=rank,
                    rslora=False, steps=128, effective_batch=16, sequence_length=512,
                    learning_rate=2e-4, optimizer='adamw', scheduler='constant', dropout=0)
    for anchor in anchors:
        for index in range(2):
            add('sft', anchor, f'sft-s{index}', seed_index=index,
                training_seed=20261028 + index, task='instruction', steps=256,
                effective_batch=16, sequence_length=512, learning_rate=2e-5,
                max_base_parameters=1700000000)
    for sft in result['sft']:
        add('dpo', sft['anchor'], f'dpo-s{sft["seed_index"]}', parents=[sft['model_id']],
            seed_index=sft['seed_index'], reference_model=sft['model_id'], task='preference',
            steps=128, beta=0.1, effective_batch=8, sequence_length=512,
            learning_rate=5e-6, max_base_parameters=1700000000)
    for anchor in merge_anchors:
        parents = [r['model_id'] for r in sorted(result['lora'], key=lambda r: (r['task'], r['rank']))
                   if r['anchor'] == anchor][:4]
        for weight in [0.25, 0.5, 0.75]:
            add('merge', anchor, 'merge-linear-' + str(weight).replace('.', ''),
                parents=parents[:2], method='linear', reference=anchor,
                coefficients={parents[0]: weight, parents[1]: 1 - weight},
                configured_weights=[weight, 1 - weight])
        for t in [0.25, 0.5, [0.2, 0.8]]:
            add('merge', anchor, 'merge-slerp-' + ('layerwise' if isinstance(t, list) else str(t).replace('.', '')),
                parents=parents[:2], method='slerp', reference=anchor, t=t,
                interpolation_schedule='linear_by_layer' if isinstance(t, list) else 'constant')
        for method, sizes, density in [('ties', [2, 4], 0.5), ('dare_ties', [3, 4], 0.8)]:
            for count in sizes:
                add('merge', anchor, f'merge-{method}-{count}', parents=parents[:count],
                    method=method, reference=anchor, density=density,
                    configured_weights=[1 / count] * count,
                    sparsification_seed=settings['seed'] + seed_index)
    for anchor in anchors:
        for method in ['classic', 'heretic']:
            add('abliteration', anchor, 'abliteration-' + method, method=method)
    for anchor in anchors:
        for bits in [8, 4, 2]:
            parent = anchor + '-dpo-s0' if anchor in merge_anchors else anchor
            add('quantization_sim', anchor, f'quantization-s{bits}', parents=[parent], bits=bits,
                symmetric=True, per_row=True, signed=True, output_dtype='bfloat16')
    for anchor in anchors:
        for method in ['mlp_permutation', 'rescaling', 'value_rotation', 'targeted_perturbation']:
            params = {'relative_frobenius_norm': 1e-3, 'target_matrices': 4} if method == 'targeted_perturbation' else {}
            add('laundering', anchor, 'laundering-' + method, method=method, **params)
    for anchor, index in [('q25-05', 0), ('q25-15', 0), ('q3-06', 0),
                          ('smol-360', 0), ('modernbert', 0), ('modernbert', 1)]:
        embedded = anchor == 'q3-06'
        params = {'rank': 8, 'alpha': 8} if embedded else {}
        add('decision', anchor, f'decision-s{index}', edge_type='lora' if embedded else 'sft',
            seed_index=index, labels=2, steps=32, effective_batch=16, sequence_length=128,
            learning_rate=2e-4 if embedded else 2e-5, embedded_adapter=embedded, **params)
    slots = settings['reference_slots']
    if slots != {'published_size_slots': 5, 'published_seed_slots': 9, 'sibling_slots': 2}:
        raise ValueError('inventory: expected 45 PolyPythias and two sibling metadata slots')
    for size in range(5):
        for seed in range(9):
            result['references'].append({
                'model_id': f'poly-size-slot-{size}-seed-slot-{seed}', 'kind': 'independent',
                'reference_type': 'polypythias', 'size_slot': size, 'seed_slot': seed,
                'status': 'unresolved', 'unresolved_reason': 'requires authoritative roster, pin, and license evidence',
            })
    for index in range(2):
        result['references'].append({
            'model_id': f'posttraining-sibling-slot-{index}', 'kind': 'independent',
            'reference_type': 'sibling', 'sibling_slot': index, 'status': 'unresolved',
            'unresolved_reason': 'requires published branches, common-root and nonancestry evidence, pin, and license',
        })
    for key in result:
        result[key].sort(key=lambda row: row['model_id'])
    return result


def materialize_tables(inventory: dict, pins: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    external = inventory['bases'] + inventory['references']
    unresolved = [r['model_id'] for r in external if r['model_id'] not in pins]
    if unresolved:
        raise ValueError('unresolved inventory pins: ' + ','.join(sorted(unresolved)))
    models, edges = [], []
    licenses = {}
    for row in external:
        model_id = row['model_id']
        pin = pins[model_id]
        path = f'{pin.get("repo_id", "")}@{pin.get("revision", "")}'
        if not HUB_PATH.fullmatch(path) or pin.get('status', 'pinned') != 'pinned':
            raise ValueError(f'{model_id}: unresolved or invalid full commit pin')
        if row.get('repo_id') and pin['repo_id'] != row['repo_id']:
            raise ValueError(f'{model_id}: pin repository does not match inventory')
        license_id = pin.get('license', '')
        family = pin.get('base_family', row.get('base_family', ''))
        licenses[model_id] = license_id
        models.append([model_id, row['kind'], path, family, license_id])
    for group in ['lora', 'sft', 'dpo', 'merge', 'abliteration', 'quantization_sim', 'laundering', 'decision']:
        for row in inventory[group]:
            model_id = row['model_id']
            models.append([model_id, 'derived', row['path'], row['base_family'], licenses[row['anchor']]])
            parameters = dict(row)
            if row.get('method') in {'ties', 'dare_ties'}:
                parameters['coefficient_semantics'] = 'configured_task_weight'
            params = canonical_json(parameters)
            for index, parent in enumerate(row['parents']):
                coefficient = row.get('coefficients', {}).get(parent, '')
                if row.get('method') in {'ties', 'dare_ties'}:
                    coefficient = row['configured_weights'][index]
                edges.append([model_id, parent, row['edge_type'], 'parent', coefficient, params])
            if row.get('reference'):
                edges.append([model_id, row['reference'], row['edge_type'], 'base', '', params])
    model_frame = pd.DataFrame(models, columns=MODEL_COLUMNS).sort_values('model_id').reset_index(drop=True)
    edge_frame = pd.DataFrame(edges, columns=EDGE_COLUMNS).sort_values(['child', 'parent', 'role']).reset_index(drop=True)
    validate_tables(model_frame, edge_frame)
    return model_frame, edge_frame


def write_tables(models: pd.DataFrame, edges: pd.DataFrame, output_root: Path) -> Path:
    validate_tables(models, edges)
    models = models.sort_values('model_id').reset_index(drop=True)
    edges = edges.sort_values(['child', 'parent', 'role']).reset_index(drop=True)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    versions = contained_path(root, 'inventories', 'tables')
    versions.mkdir(exist_ok=True)
    model_bytes = models.to_csv(index=False, lineterminator='\n').encode()
    edge_bytes = edges.to_csv(index=False, lineterminator='\n').encode()
    version = content_hash({'models': model_bytes.decode(), 'edges': edge_bytes.decode()})
    destination = contained_path(versions, version, 'tables')
    staging = Path(tempfile.mkdtemp(prefix='.inventory-', dir=versions))
    try:
        (staging / 'models.csv').write_bytes(model_bytes)
        (staging / 'edges.csv').write_bytes(edge_bytes)
        if destination.exists():
            if (destination.is_symlink() or (destination / 'models.csv').read_bytes() != model_bytes
                    or (destination / 'edges.csv').read_bytes() != edge_bytes):
                raise ValueError('tables: existing inventory version differs')
        else:
            os.rename(staging, destination)
        atomic_json(root / 'current.json', {'schema_version': LAB_SCHEMA_VERSION,
                                           'inventory': f'inventories/{version}'})
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def read_tables(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(output_root)
    pointer = json.loads((root / 'current.json').read_text())
    if pointer.get('schema_version') != LAB_SCHEMA_VERSION:
        raise ValueError('tables: unsupported schema version')
    directory = contained_path(root, pointer['inventory'], 'tables')
    models = pd.read_csv(contained_path(directory, 'models.csv', 'tables'), keep_default_na=False)
    edges = pd.read_csv(contained_path(directory, 'edges.csv', 'tables'), keep_default_na=False)
    validate_tables(models, edges)
    return models, edges


def verify_tables(models: pd.DataFrame, edges: pd.DataFrame, lab_root: Path) -> dict:
    validate_tables(models, edges)
    completed, missing, incomplete = [], [], []
    paths_by_id = dict(zip(models.model_id, models.path))
    for row in models.sort_values('model_id').itertuples(index=False):
        if HUB_PATH.fullmatch(row.path):
            continue
        target = contained_path(lab_root, row.path, row.model_id)
        if not target.exists():
            missing.append(row.model_id)
            continue
        for path in target.rglob('*'):
            contained_path(target, path.relative_to(target).as_posix(), row.model_id)
            if path.is_symlink():
                raise ValueError(f'{row.model_id}: symlink in checkpoint')
        try:
            metadata = json.loads((target / 'recipe.json').read_text())
            lineage = json.loads((target / 'lineage.json').read_text())
            from .artifacts import _lineage
            recipe = metadata['recipe']
            if (metadata['schema_version'] != LAB_SCHEMA_VERSION or metadata['model_id'] != row.model_id
                    or metadata['config_hash'] != content_hash(recipe)
                    or metadata['input_hash'] != content_hash({'parents': recipe['parent_identities'], 'inputs': recipe.get('inputs', {})})
                    or lineage != _lineage(row.model_id, recipe)
                    or not any(p.is_file() for p in target.rglob('*.safetensors'))):
                raise ValueError('incomplete or mismatched metadata')
            expected = edges.loc[edges.child == row.model_id]
            contributors = set(recipe['parents'])
            references = {recipe['reference']} if recipe.get('reference') else set()
            if (contributors != set(expected.loc[expected.role == 'parent', 'parent'])
                    or references != set(expected.loc[expected.role == 'base', 'parent'])
                    or set(expected.edge_type) != {recipe['edge_type']}):
                raise ValueError('checkpoint parents or roles differ from table')
            for parent in contributors | references:
                registered = paths_by_id[parent]
                identity = recipe['parent_identities'][parent]
                if HUB_PATH.fullmatch(registered):
                    if identity != registered:
                        raise ValueError(f'parent {parent} pin differs from registry')
                else:
                    local = LOCAL_KEY.fullmatch(identity)
                    if not local or local.group(1) != parent:
                        raise ValueError(f'parent {parent} local identity differs from registry')
        except (KeyError, TypeError, ValueError, OSError) as exc:
            incomplete.append({'model_id': row.model_id, 'reason': str(exc)})
            continue
        completed.append(row.model_id)
    return {'schema_version': LAB_SCHEMA_VERSION, 'completed_ids': completed,
            'missing_ids': missing, 'incomplete': incomplete,
            'status': 'PASS' if not missing and not incomplete else 'INCOMPLETE'}
