"""Offline query fixtures shared with Model-Zoo's deterministic universe."""

import json

import pandas as pd
import pytest

from zoo.build import build_zoo
from zoo.schema import SCORES_COLUMNS
from zoo.testing import make_universe, SHA1, SHA2, SHA4, SHA10


@pytest.fixture(scope="module")
def universe(tmp_path_factory):
    return make_universe(tmp_path_factory.mktemp("query-universe"))


@pytest.fixture(scope="module")
def built_zoo(universe):
    manifests = [json.loads(path.read_text()) for path in sorted(universe.manifests.glob("*.manifest.json"))]
    frame = universe.root / "frame.csv"
    pd.DataFrame([dict(repo_id=m["repo_id"], revision=m["revision"], modality="text",
                       downloads=1000 - i, params=1000000 + i)
                  for i, m in enumerate(manifests)]).to_csv(frame, index=False)
    rows = []
    for repo, pin, skill, value, verified in [
        ("org/child", SHA2, "math", .6, True),
        ("org/child", SHA2, "math", .8, True),
        ("org/child", SHA2, "math", 0., False),
        ("org/child", SHA2, "writing", 1., True),
        ("org/child", SHA10, "math", .1, True),
        ("org/parent", SHA1, "math", .9, False),
        ("org/sibling", SHA4, "math", .95, True),
    ]:
        rows.append(dict(model_key=f"{repo}@{pin}", repo_id=repo, benchmark=f"bench-{len(rows)}",
                         skill=skill, value=value, metric="accuracy", source="synthetic",
                         source_revision=pin if verified else "f" * 40, revision_verified=verified,
                         harness="fixture", n_items=10))
    return build_zoo(universe.manifests, [universe.fingerprints], universe.root / "zoo",
                     universe.root, frame=frame, score_tables=[pd.DataFrame(rows, columns=SCORES_COLUMNS)])


@pytest.fixture(scope="module")
def z(built_zoo, universe):
    from zoo_keeper import Zoo

    with Zoo(built_zoo, data_root=universe.root) as opened:
        yield opened
