"""Query contracts against offline, real fingerprint artifacts."""

import pytest

from zoo.testing import SHA1, SHA2, SHA3, SHA4, SHA5, SHA7, SHA8, SHA10, SHA11, SHA12
from zoo_keeper import AmbiguousModel, NotComparable, UnknownModel, Zoo


def test_find_filters_and_literal_strings(z):
    df = z.find(text="child")
    assert set(df.model_key) >= {"org/child@" + SHA2, "org/child-vocab@" + SHA3}
    assert "org/child@" + SHA2 in set(z.find(license="mit").model_key)
    assert set(z.find(license=["mit"]).license) == {"mit"}
    assert len(z.find(license='it\'s "odd"')) == 1
    assert len(z.find(text="'; DROP TABLE models; --")) == 0 and len(z.find()) > 0

def test_resolve_ambiguous_and_unknown(z):
    with pytest.raises(AmbiguousModel) as err:
        z.resolve("org/child")
    assert SHA2 in str(err.value) and SHA10 in str(err.value)
    with pytest.raises(UnknownModel):
        z.resolve("org/missing")

def test_info_separates_measured_and_predicted(z):
    card = z.info("org/child@" + SHA2)
    assert card["predictions"] == [] and all("revision_verified" in s for s in card["measured_scores"])
    assert card["coverage"]["status"] in {"measured", "partial"}
    assert z.info("org/t0only@" + SHA8)["coverage"]["status"] == "t0_only"

def test_similar_and_not_comparable(z):
    df = z.similar("org/child@" + SHA2, k=5)
    assert df.iloc[0].model_key in {"org/parent@" + SHA1, "org/sibling@" + SHA4, "org/child-vocab@" + SHA3,
                                    "org/decider@" + SHA11}
    assert "org/dup@" + SHA5 not in set(df.model_key)               # own duplicate group excluded
    assert "org/dup@" + SHA5 in z.info("org/child@" + SHA2)["duplicates"]
    with pytest.raises(NotComparable) as err:
        z.similar("org/t0only@" + SHA8)
    assert err.value.reason == "t0_only"
    with pytest.raises(NotComparable):
        z.similar("org/other-arch@" + SHA7)

def test_family_license_chain_and_diff(z):
    fam = z.family("org/child@" + SHA2)
    assert fam["license_chain"][:2] == [("org/child@" + SHA2, "mit"), ("org/parent@" + SHA1, "apache-2.0")]
    assert fam["license_differs"] is True
    d = z.diff("org/child@" + SHA2, "org/parent@" + SHA1)
    assert d["relatedness"]["verdict"] == "uncalibrated" and d["summary"]["counts"]["changed"] > 0
    assert z.check("org/child@" + SHA2)["verdict"] == "uncalibrated"

def test_decision_models_are_queryable(z):
    assert set(z.find(system_one=True).model_key) == {"org/decider@" + SHA11}
    card = z.info("org/decider@" + SHA11)
    assert card["decision"]["decision_type"] == "binary" and card["decision"]["labels"] == ["no", "yes"]
    d = z.diff("org/decider@" + SHA11, "org/parent@" + SHA1)
    assert [h["name"] for h in d["summary"]["heads"]] == ["score.weight"] and d["summary"]["counts"]["added"] == 1
    assert "org/parent-GGUF@" + SHA12 in set(z.find(architecture="llama").model_key)


@pytest.fixture
def snapshot(built_zoo, tmp_path):
    """Make editable table snapshots without touching the shared generation."""
    import shutil
    import pyarrow as pa
    import pyarrow.parquet as pq

    def make(edits=None):
        path = tmp_path / f"snapshot-{len(list(tmp_path.iterdir()))}"
        shutil.copytree(built_zoo.resolve(), path)
        for name, edit in (edits or {}).items():
            file = path / f"{name}.parquet"
            table = pq.read_table(file)
            rows = table.to_pylist()
            edit(rows)
            pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), file)
        return path

    return make


def test_find_all_metadata_filters_and_order(z):
    child = "org/child@" + SHA2
    row = z.find(text="ORG/CHILD", license=["mit"], modality="text", access="granted",
                 weight_format="safetensors", gated=False, params_min=1e6, params_max=2e6,
                 decision=False, system_one=False, architecture="llama", limit=1)
    assert len(row) == 1 and row.iloc[0].model_key == child
    assert list(row.columns) == ["model_key", "repo_id", "modality", "params", "license",
                                 "downloads", "coverage_status", "dup_group"]
    assert z.find(license=[]).empty and z.find(limit=0).empty
    assert z.find(modality="text'; DROP TABLE models; --").empty
    assert z.find(architecture="llama' OR true --").empty
    assert z.find(text="%").empty  # Substring, not SQL LIKE wildcard.
    assert list(z.find().model_key) == list(z.find().sort_values(
        ["downloads", "model_key"], ascending=[False, True]).model_key)
    with pytest.raises(ValueError):
        z.find(limit=-1)


def test_measured_score_means_use_only_relevant_verified_revision(z):
    child, sibling, parent = "org/child@" + SHA2, "org/sibling@" + SHA4, "org/parent@" + SHA1
    assert set(z.find(skill="math", min_score=.69).model_key) == {child, sibling}
    assert set(z.find(skill="math", min_score=.71).model_key) == {sibling}
    assert set(z.find(skill="math", min_score=.69, verified_only=False).model_key) == {parent, sibling}
    assert set(z.find(skill="writing").model_key) == {child}
    assert z.find(skill="math' OR true --", min_score=0).empty
    scores = z.info(child)["measured_scores"]
    assert len(scores) == 4 and any(not row["revision_verified"] for row in scores)
    assert len(z.info("org/child@" + SHA10)["measured_scores"]) == 1


def test_metadata_only_and_environment_data_root(built_zoo, universe, monkeypatch):
    monkeypatch.delenv("ZOO_DATA", raising=False)
    with Zoo(built_zoo) as metadata:
        assert len(metadata.find()) == 12
        coverage = metadata.info("org/child@" + SHA2)["coverage"]
        assert coverage["skipped_files"] is None and "unavailable" in coverage["skipped_files_note"]
        with pytest.raises(ValueError, match="data_root|ZOO_DATA"):
            metadata.diff("org/child@" + SHA2, "org/parent@" + SHA1)
    monkeypatch.setenv("ZOO_DATA", str(universe.root))
    with Zoo(built_zoo) as from_env:
        assert from_env.diff("org/child@" + SHA2, "org/parent@" + SHA1)["summary"]["counts"]["changed"] > 0


def test_info_coverage_and_decision_names(z):
    import json

    card = z.info("org/decider@" + SHA11)
    assert card["identity"]["revision"] == SHA11
    assert card["decision"]["head_views"] == ["score.weight"]
    assert card["duplicates_note"] == "identical on every measured block"
    assert card["predictions_note"] == "predictions arrive in zoo v1"
    coverage = card["coverage"]
    assert coverage["skipped_tensor_reasons"]["not_a_matrix"] == 1
    assert coverage["skipped_files"] == []
    assert z.info("org/gated")["coverage"]["status"] == "no_access"
    json.dumps(card, allow_nan=False)


def test_similar_collapses_other_groups_by_downloads(snapshot, universe):
    def downloads(rows):
        for row in rows:
            if row["model_key"] == "org/dup@" + SHA5:
                row["downloads"] = 1e6
    with Zoo(snapshot({"models": downloads}), data_root=universe.root) as opened:
        collapsed = opened.similar("org/parent@" + SHA1, k=100)
        assert "org/dup@" + SHA5 in set(collapsed.model_key)
        assert "org/child@" + SHA2 not in set(collapsed.model_key)
        assert collapsed.dup_group.is_unique
        expanded = opened.similar("org/parent@" + SHA1, k=100, collapse_duplicates=False)
        assert {"org/dup@" + SHA5, "org/child@" + SHA2} <= set(expanded.model_key)
        assert "org/other-arch@" + SHA7 not in set(expanded.model_key)
        assert opened.similar("org/parent@" + SHA1, k=0).empty
        with pytest.raises(ValueError):
            opened.similar("org/parent@" + SHA1, k=-1)


def test_similar_t2_only_reason_and_missing_common_views(snapshot, universe):
    def t2_only(rows):
        child = next(row for row in rows if row["model_key"] == "org/child@" + SHA2)
        child.update(fingerprint_t1="", fingerprint_t2="fingerprints/only.t2.fingerprint.safetensors")
    with Zoo(snapshot({"models": t2_only}), data_root=universe.root) as opened:
        with pytest.raises(NotComparable) as err:
            opened.similar("org/child@" + SHA2)
        assert err.value.reason == "no_t1_fingerprint"
    path = snapshot()
    import shutil
    shutil.rmtree(path / "retrieval")
    with Zoo(path) as opened:
        with pytest.raises(NotComparable) as err:
            opened.similar("org/child@" + SHA2)
        assert err.value.reason == "no_compatible_views"


def test_family_tiers_directions_cycles_and_depth(snapshot):
    child, parent, sibling = "org/child@" + SHA2, "org/parent@" + SHA1, "org/sibling@" + SHA4
    def edges(rows):
        template = dict(rows[0])
        rows.clear()
        for left, right, estimand, tier in [
            (child, parent, "parent", "declared"),
            (child, sibling, "parent", "detected"),
            (child, parent, "ancestor", "verified"),
            (parent, child, "parent", "declared"),
            (sibling, child, "same_root", "verified"),
        ]:
            rows.append(dict(template, child_key=left, parent_key=right, estimand=estimand,
                             evidence_tier=tier, evidence=f"{tier}-{estimand}"))
    with Zoo(snapshot({"lineage": edges})) as opened:
        family = opened.family(child, depth=20)
        assert family["ancestors"] == [parent]
        assert family["descendants"] == [parent]
        assert family["same_root"] == [sibling]
        assert family["license_chain"] == [(child, "mit"), (parent, "apache-2.0")]
        assert opened.family(child, depth=0)["ancestors"] == []
        assert opened.family(child, depth=0)["license_chain"] == [(child, "mit")]
        assert opened.family(sibling)["descendants"] == []
        with pytest.raises(ValueError):
            opened.family(child, depth=-1)
        lineage = opened.info(child)["lineage"]
        assert {row["key"] for row in lineage["same_root"]} == {sibling}
        assert {row["evidence_tier"] for row in lineage["parents"]} == {"declared", "detected", "verified"}


def test_check_missing_and_ambiguous_declared_base(z, snapshot, universe):
    assert z.check("org/unrelated") == {"verdict": "base_not_in_zoo"}
    assert z.check("org/parent") == {"verdict": "base_not_in_zoo"}
    def ambiguous(rows):
        next(row for row in rows if row["repo_id"] == "org/sibling")["declared_bases"] = ["org/child"]
    with Zoo(snapshot({"models": ambiguous}), data_root=universe.root) as opened:
        with pytest.raises(AmbiguousModel):
            opened.check("org/sibling")


@pytest.mark.parametrize("bad_path", ["../outside.t1.fingerprint.safetensors", "/tmp/outside.t1.fingerprint.safetensors"])
def test_diff_rejects_paths_outside_data_root(snapshot, universe, bad_path):
    def escape(rows):
        next(row for row in rows if row["model_key"] == "org/child@" + SHA2)["fingerprint_t1"] = bad_path
    with Zoo(snapshot({"models": escape}), data_root=universe.root) as opened:
        with pytest.raises(ValueError, match="relative|escapes|outside"):
            opened.diff("org/child@" + SHA2, "org/parent@" + SHA1)


def test_diff_rejects_wrong_loaded_identity_and_no_common_tier(snapshot, universe):
    def wrong_identity(rows):
        parent = next(row for row in rows if row["model_key"] == "org/parent@" + SHA1)
        next(row for row in rows if row["model_key"] == "org/child@" + SHA2)["fingerprint_t1"] = parent["fingerprint_t1"]
    with Zoo(snapshot({"models": wrong_identity}), data_root=universe.root) as opened:
        with pytest.raises(ValueError, match="identity"):
            opened.diff("org/child@" + SHA2, "org/parent@" + SHA1)
    def t2_only(rows):
        next(row for row in rows if row["model_key"] == "org/child@" + SHA2).update(
            fingerprint_t1="", fingerprint_t2="unused.t2.fingerprint.safetensors")
    with Zoo(snapshot({"models": t2_only}), data_root=universe.root) as opened:
        with pytest.raises(NotComparable) as err:
            opened.diff("org/child@" + SHA2, "org/parent@" + SHA1)
        assert err.value.reason == "no_common_tier"


def test_reader_keeps_resolved_generation(snapshot, tmp_path):
    first = snapshot()
    def renamed(rows):
        for row in rows:
            row["license"] = "new-generation"
    second = snapshot({"models": renamed})
    pointer = tmp_path / "published"
    pointer.symlink_to(first.name, target_is_directory=True)
    with Zoo(pointer) as old:
        replacement = tmp_path / "replacement"
        replacement.symlink_to(second.name, target_is_directory=True)
        replacement.replace(pointer)
        assert old.info("org/child@" + SHA2)["metadata"]["license"] == "mit"
        assert "org/dup@" + SHA5 in old.info("org/child@" + SHA2)["duplicates"]
        with Zoo(pointer) as new:
            assert new.info("org/child@" + SHA2)["metadata"]["license"] == "new-generation"


def test_manifest_skipped_files_sha_and_containment(snapshot, universe, tmp_path):
    import hashlib
    import json

    root = tmp_path / "data"
    root.mkdir()
    manifest = root / "child.manifest.json"
    payload = dict(repo_id="org/child", revision=SHA2,
                   skipped_files=[dict(path="extra.bin", reason="unsupported")])
    manifest.write_text(json.dumps(payload))
    path = snapshot()
    metadata = json.loads((path / "zoo.json").read_text())
    metadata["inputs"] = [dict(kind="file", path=manifest.name,
                               sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())]
    (path / "zoo.json").write_text(json.dumps(metadata))
    with Zoo(path, data_root=root) as opened:
        assert opened.info("org/child@" + SHA2)["coverage"]["skipped_files"] == payload["skipped_files"]
        manifest.write_text(json.dumps(dict(payload, skipped_files=[])))
        assert opened.info("org/child@" + SHA2)["coverage"]["skipped_files"] is None
    metadata["inputs"][0]["path"] = "../outside.manifest.json"
    (path / "zoo.json").write_text(json.dumps(metadata))
    with Zoo(path, data_root=root) as opened:
        assert opened.info("org/child@" + SHA2)["coverage"]["skipped_files"] is None


@pytest.fixture(scope="module")
def tiered_zoo(tmp_path_factory):
    """Two tiny local checkpoints measured at both tiers, never Hub identities."""
    import json
    from types import SimpleNamespace
    import torch
    from safetensors.torch import save_file
    from evidence.fingerprint import fingerprint_model
    from evidence.hub import HubReader
    from evidence.manifest import build_manifest, write_manifest
    from zoo.build import build_zoo

    root = tmp_path_factory.mktemp("query-tiers")
    generator = torch.Generator().manual_seed(720)
    weight = torch.randn(300, 16, generator=generator)
    keys, fingerprints = [], []
    for name, matrix in [("base", weight), ("child", weight + .01)]:
        checkpoint = root / name
        checkpoint.mkdir()
        save_file({"linear.weight": matrix}, checkpoint / "model.safetensors")
        manifest = build_manifest(str(checkpoint), "main", HubReader(), api=None)
        key = manifest["repo_id"] + "@" + manifest["revision"]
        if keys:
            manifest["declared_bases"] = [keys[0]]
        keys.append(key)
        write_manifest(manifest, root / "manifests")
        fingerprints.append(fingerprint_model(manifest, HubReader(), "t2", root / "fingerprints",
                                               also_t1_blocks=(256,), read_threads=1))
    path = build_zoo(root / "manifests", [root / "fingerprints"], root / "zoo", root)
    return SimpleNamespace(root=root, path=path, keys=keys, fingerprints=fingerprints)


def test_diff_prefers_t1_then_falls_back_to_real_t2_and_forwards_calibration(tiered_zoo, tmp_path):
    import json
    import shutil
    import pyarrow as pa
    import pyarrow.parquet as pq
    from evidence.sketch import ROW_BLOCK, SKETCH_SEED

    base, child = tiered_zoo.keys
    calibration = tmp_path / "null.json"
    null = dict(statistic="median_cosine", sketch_k=32, row_block=ROW_BLOCK, sketch_seed=SKETCH_SEED,
                n_pairs=99, tier="t1", null_quantiles=[-.1, .1], related_threshold=.8)
    calibration.write_text(json.dumps(null))
    with Zoo(tiered_zoo.path, data_root=tiered_zoo.root) as opened:
        assert opened.diff(child, base, null=calibration)["relatedness"]["verdict"] == "related"
        assert opened.check(child, null=calibration)["verdict"] == "consistent"
    fallback = tmp_path / "fallback"
    shutil.copytree(tiered_zoo.path.resolve(), fallback)
    table = pq.read_table(fallback / "models.parquet")
    rows = table.to_pylist()
    next(row for row in rows if row["model_key"] == child)["fingerprint_t1"] = ""
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), fallback / "models.parquet")
    null["tier"] = "t2"
    calibration.write_text(json.dumps(null))
    with Zoo(fallback, data_root=tiered_zoo.root) as opened:
        result = opened.diff(child, base, null=calibration)
        assert result["relatedness"]["verdict"] == "related"
        assert result["a"]["revision"] == child.rsplit("@", 1)[1]


def test_diff_rejects_fingerprint_symlink_escape(snapshot, universe, tmp_path):
    import shutil

    target = next(universe.fingerprints.glob("*.t1.fingerprint.safetensors"))
    root = tmp_path / "data"
    root.mkdir()
    outside = tmp_path / "external.t1.fingerprint.safetensors"
    shutil.copyfile(target, outside)
    (root / "escape.t1.fingerprint.safetensors").symlink_to(outside)
    def escape(rows):
        next(row for row in rows if row["model_key"] == "org/child@" + SHA2)["fingerprint_t1"] = "escape.t1.fingerprint.safetensors"
    with Zoo(snapshot({"models": escape}), data_root=root) as opened:
        with pytest.raises(ValueError, match="escapes"):
            opened.diff("org/child@" + SHA2, "org/parent@" + SHA1)


def test_diff_rejects_embedded_t2_in_t1_slot_before_outside_hdf_open(tiered_zoo, tmp_path, monkeypatch):
    import shutil
    from pathlib import Path
    import pyarrow as pa
    import pyarrow.parquet as pq
    import evidence.fingerprint as fingerprint_module

    root = tmp_path / "data"
    root.mkdir()
    artifact = root / "misplaced.t2.fingerprint.safetensors"
    source = tiered_zoo.fingerprints[1]
    shutil.copyfile(source, artifact)
    companion = artifact.with_name("misplaced.t2.spectra.h5")
    outside = tmp_path / "outside.t2.spectra.h5"
    shutil.copyfile(source.with_name(source.name.removesuffix(".t2.fingerprint.safetensors")
                                   + ".t2.spectra.h5"), outside)
    companion.symlink_to(outside)
    path = tmp_path / "zoo"
    shutil.copytree(tiered_zoo.path.resolve(), path)
    table = pq.read_table(path / "models.parquet")
    rows = table.to_pylist()
    child = tiered_zoo.keys[1]
    next(row for row in rows if row["model_key"] == child)["fingerprint_t1"] = artifact.name
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path / "models.parquet")
    outside_opens = []
    original_file = fingerprint_module.h5py.File

    def observe_hdf_open(file, *args, **kwargs):
        if Path(file).resolve() == outside:
            outside_opens.append(Path(file))
        return original_file(file, *args, **kwargs)

    monkeypatch.setattr(fingerprint_module.h5py, "File", observe_hdf_open)
    with Zoo(path, data_root=root) as opened:
        with pytest.raises(ValueError, match="tier"):
            opened.diff(child, tiered_zoo.keys[0])
    assert outside_opens == []
