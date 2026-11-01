import pandas as pd
import pytest


@pytest.fixture
def tables():
    models = pd.DataFrame([
        ['b', 'base', 'org/base@' + 'a' * 40, 'tiny', 'apache-2.0'],
        ['c', 'derived', 'c', 'tiny', 'apache-2.0'],
        ['i', 'independent', 'org/independent@' + 'b' * 40, 'tiny', 'apache-2.0'],
    ], columns=['model_id', 'kind', 'path', 'base_family', 'license'])
    edges = pd.DataFrame([['c', 'b', 'sft', 'parent', '', '{}']],
                         columns=['child', 'parent', 'edge_type', 'role', 'coefficient', 'params_json'])
    return models, edges
