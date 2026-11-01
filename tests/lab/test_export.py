import csv

import pandas as pd
import pytest

from lab.export import export_models
from lab.schema import MODEL_COLUMNS


def test_export_hub_and_local_models(tmp_path):
    root = tmp_path / 'lab'
    (root / 'child').mkdir(parents=True)
    models = pd.DataFrame([
        ['base', 'base', 'org/base@'+'a'*40, 'family', 'mit'],
        ['child', 'derived', 'child', 'family', 'mit'],
    ], columns=MODEL_COLUMNS)
    output = tmp_path / 'invocation.csv'
    export_models(models, root, output)
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [{'repo_id': 'org/base', 'revision': 'a'*40},
                    {'repo_id': str(root / 'child'), 'revision': ''}]


@pytest.mark.parametrize('path', ['../escape', '/absolute', 'org/base@main'])
def test_export_refuses_escape_or_missing_pin(tmp_path, path):
    models = pd.DataFrame([['m', 'derived', path, 'family', 'mit']], columns=MODEL_COLUMNS)
    with pytest.raises(ValueError):
        export_models(models, tmp_path / 'lab', tmp_path / 'models.csv')


def test_export_refuses_missing_checkpoint(tmp_path):
    models = pd.DataFrame([['m', 'derived', 'm', 'family', 'mit']], columns=MODEL_COLUMNS)
    with pytest.raises(ValueError, match='missing'):
        export_models(models, tmp_path / 'lab', tmp_path / 'models.csv')
