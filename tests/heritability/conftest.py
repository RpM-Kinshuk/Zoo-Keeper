import hashlib
import json

import pandas as pd
import pytest


SHA = "a" * 40
OTHER_SHA = "b" * 40


@pytest.fixture
def registry(tmp_path):
    evidence = tmp_path / "denominator.json"
    evidence.write_text(json.dumps({"task": "math", "n_items": 100}))
    proof = {"path": "denominator.json", "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()}
    binary = dict(column="Math Raw", benchmark="v2", task="math", harness="h2", scale="raw_accuracy", kind="binary", n_items=100, denominator_evidence=proof)
    composite = dict(column="Composite Raw", benchmark="v2", task="composite", harness="h2", scale="raw_accuracy", kind="composite")
    columns = dict(record_id="id", repo_id="Model", revision="Model sha", precision="Precision", quantization="Quantization")
    return {"evidence_root": tmp_path, "schemas": {
        "v2": {"columns": dict(columns), "tasks": [binary, composite]},
        "v1": {"columns": dict(columns), "tasks": [dict(binary, benchmark="v1", harness="h1")]},
    }}


@pytest.fixture
def pins():
    return pd.DataFrame([dict(repo_id="ORG/Model", revision=SHA, precision="bf16", quantization="none", harness="h2", evidence_source="manifests/input.json")])


@pytest.fixture
def score_sources(tmp_path):
    base = dict(Model="org/model", **{"Model sha": SHA, "Precision": "bf16", "Quantization": "none", "Math Raw": 0.2, "Math": 0.0, "Composite Raw": 0.4})
    rows = [dict(base, id="a"), dict(base, id="fp16", Precision="fp16"), dict(base, id="recorded", **{"Model sha": OTHER_SHA}), dict(base, id="missing_precision", Precision=""), dict(base, id="missing_revision", **{"Model sha": ""}), dict(base, id="invalid", **{"Math Raw": 1.2}), dict(base, id="conflict", Model="org/conflict")]
    frames = [pd.DataFrame(rows), pd.DataFrame([dict(base, id="a"), dict(base, id="conflict", Model="org/conflict", **{"Math Raw": 0.3})]), pd.DataFrame([dict(base, id="v1")])]
    sources = []
    for i, frame in enumerate(frames):
        path = tmp_path / f"source{i}.csv"
        frame.to_csv(path, index=False)
        sources.append(dict(path=path, name=f"source{i}", schema="v1" if i == 2 else "v2"))
    return sources
