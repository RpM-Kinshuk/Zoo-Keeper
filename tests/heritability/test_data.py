import copy
import json

import pandas as pd
import pytest

from studies.heritability.data import assemble_scores, inventory_inputs
from studies.heritability.protocol import freeze_protocol


def test_identity_raw_and_variance(score_sources, pins, registry):
    scores, excluded, audit = assemble_scores(score_sources, pins, registry)
    a = scores.query("record_id == 'a'").iloc[0]
    assert a.accuracy == 0.2
    assert a.measurement_variance == pytest.approx(0.0016)
    assert a.revision_verified
    assert not scores.query("revision_status == 'recorded'").revision_verified.any()
    assert set(scores.precision) >= {'bf16', 'fp16'}
    assert 'conflicting_score' in set(excluded.reason)
    assert 'variance_unavailable' in set(excluded.reason)
    assert audit['verified_joins'] <= audit['eligible_joins']


def test_zero_bytes_and_protocol(tmp_path):
    p = tmp_path / 'empty.csv'; p.write_bytes(b'')
    inv = inventory_inputs([p], data_root=tmp_path)
    assert inv['files'][0]['status'] == 'missing_zero_bytes'
    out = tmp_path / 'frozen.json'
    assert freeze_protocol({'seed': 20261028}, inv, out) == freeze_protocol({'seed': 20261028}, inv, out)
    with pytest.raises(ValueError, match='frozen'):
        freeze_protocol({'seed': 1}, inv, out)
    assert json.loads(out.read_text())["config"]["seed"] == 20261028


def test_identity_boundaries_and_provenance(score_sources, pins, registry):
    scores, excluded, audit = assemble_scores(score_sources, pins, registry)
    math = scores.query("task == 'math'")
    assert not (math.record_id == "invalid").any()
    assert "invalid_observation" in set(excluded.reason)
    assert set(math.benchmark) == {"v1", "v2"}
    assert not math.query("record_id == 'missing_precision'").revision_verified.any()
    assert not math.query("record_id == 'missing_revision'").model_key.any()
    assert set(math.query("repo_id == 'org/model'").revision) >= {"a" * 40, "b" * 40}
    a = math.query("record_id == 'a'").iloc[0]
    assert len(json.loads(a.source_provenance)) == 2
    assert scores.record_id.is_unique
    assert not math.query("repo_id == 'org/conflict'").shape[0]
    no_pins, _, _ = assemble_scores(score_sources, pd.DataFrame(), registry)
    assert not no_pins.revision_verified.any()
    assert audit["score_to_model_join"]["denominator_definition"]


@pytest.mark.parametrize("n, reason", [(None, "variance_unavailable"), (0, "invalid_n_items")])
def test_denominator_not_invented(score_sources, pins, registry, n, reason):
    changed = copy.deepcopy(registry)
    changed["schemas"]["v2"]["tasks"][0]["n_items"] = n
    scores, excluded, _ = assemble_scores(score_sources, pins, changed)
    a = scores.query("record_id == 'a'").iloc[0]
    assert pd.isna(a.n_items)
    assert pd.isna(a.measurement_variance)
    assert reason in set(excluded.reason)


def test_evidence_and_determinism(score_sources, pins, registry):
    changed = copy.deepcopy(registry)
    changed["schemas"]["v2"]["tasks"][0]["denominator_evidence"]["sha256"] = "0" * 64
    scores, _, _ = assemble_scores(score_sources, pins, changed)
    assert pd.isna(scores.query("record_id == 'a'").iloc[0].measurement_variance)
    for schema in registry["schemas"].values():
        del schema["columns"]["record_id"]
    first, _, _ = assemble_scores(score_sources, pins, registry)
    second, _, _ = assemble_scores(list(reversed(score_sources)), pins, registry)
    pd.testing.assert_frame_equal(first, second)


def test_inventory_is_relative_explicit_and_safe(tmp_path):
    p = tmp_path / "score.csv"; p.write_text("x\n1\n")
    inv = inventory_inputs([p, tmp_path / "absent.csv"], data_root=tmp_path)
    assert {row["path"] for row in inv["files"]} == {"score.csv", "absent.csv"}
    assert {row["status"] for row in inv["files"]} == {"ok", "missing"}
    with pytest.raises(ValueError, match="env"):
        inventory_inputs([tmp_path / ".env"], data_root=tmp_path)
    with pytest.raises(ValueError, match="root"):
        inventory_inputs([tmp_path.parent / "outside.csv"], data_root=tmp_path)


def test_self_attested_or_wrong_deployment_pin_never_verifies(score_sources, pins, registry):
    for evidence, precision, harness in [("source0", "bf16", "h2"), ("manifest.json", "fp16", "h2"), ("manifest.json", "bf16", "h1")]:
        altered = pins.copy()
        altered["evidence_source"] = evidence
        altered["precision"] = precision
        altered["harness"] = harness
        scores, _, _ = assemble_scores(score_sources[:1], altered, registry)
        assert not scores.query("record_id == 'a'").revision_verified.any()


def test_explicit_v1_mapping_and_unavailable_harness_export_copies(tmp_path, pins, registry):
    mapping = copy.deepcopy(registry["schemas"]["v1"])
    mapping["columns"]["eval_name"] = "eval_name"
    mapping["tasks"][0]["harness"] = ""
    row = {"id": "v1_explicit", "Model": "org/model", "Model sha": "a" * 40, "Precision": "bf16", "Quantization": "none", "Math Raw": 0.2, "eval_name": "export_model_bf16"}
    first = tmp_path / "first.csv"; second = tmp_path / "second.csv"
    pd.DataFrame([row]).to_csv(first, index=False)
    pd.DataFrame([dict(row, normalized_distractor=0.0)]).to_csv(second, index=False)
    sources = [dict(path=first, name="first", mapping=mapping), dict(path=second, name="second", mapping=mapping)]
    scores, _, _ = assemble_scores(sources, pins, registry)
    assert len(scores) == 1
    assert scores.iloc[0].benchmark == "v1"
    assert scores.iloc[0].harness == ""
    assert scores.iloc[0].revision_status == "unverified"
    assert not scores.revision_verified.any()
    assert len(json.loads(scores.iloc[0].source_provenance)) == 2


def test_unknown_scale_missing_inputs_and_raw_endpoints(tmp_path, pins, registry):
    mapping = copy.deepcopy(registry["schemas"]["v2"])
    mapping["tasks"] = mapping["tasks"][:1]
    rows = [{"id": f"endpoint{i}", "Model": f"org/model{i}", "Model sha": "a" * 40, "Precision": "bf16", "Quantization": "none", "Math Raw": p} for i, p in enumerate([0.0, 1.0])]
    path = tmp_path / "endpoints.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    sources = [dict(path=path, name="endpoints", mapping=mapping)]
    scores, _, _ = assemble_scores(sources, pins, registry)
    assert list(scores.accuracy) == [0.0, 1.0]
    assert list(scores.measurement_variance) == [0.0, 0.0]
    mapping["tasks"][0]["scale"] = "normalized"
    scores, excluded, _ = assemble_scores(sources, pins, registry)
    assert scores.empty
    assert set(excluded.reason) == {"unsupported_scale"}
    empty = tmp_path / "empty.csv"; empty.write_bytes(b"")
    scores, excluded, _ = assemble_scores([dict(path=empty, name="empty", mapping=mapping)], pins, registry)
    assert scores.empty
    assert set(excluded.reason) == {"missing_zero_bytes"}
