import hashlib
import json
from pathlib import Path

import pandas as pd


def catalog(tmp_path, specs):
    models, inputs = [], []
    for index, changes in enumerate(specs):
        repo = f"owner/model-{index}"
        revision = str(index) * 40
        manifest = dict(repo_id=repo, revision=revision,
                        files=[dict(filename="model.safetensors", size=100)])
        manifest.update(changes.pop("manifest", {}))
        path = tmp_path / f"{index}.manifest.json"
        path.write_text(json.dumps(manifest))
        inputs.append(dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        row = dict(model_key=repo + "@" + revision, repo_id=repo, revision=revision,
                   is_adapter=False, weight_format="safetensors", access="granted",
                   gated=False, license="apache-2.0", downloads=20-index, params=None)
        row.update(changes)
        models.append(row)
    return pd.DataFrame(models), dict(inputs=inputs)


def test_all_shards_count_missing_is_unknown_and_boundary_is_inclusive(tmp_path):
    from scripts.find_weight_budget import weight_inventory, select_budget
    models, metadata = catalog(tmp_path, [
        dict(manifest=dict(files=[dict(filename="a.safetensors", size=60),
                                  dict(filename="b.safetensors", size=40)])),
        dict(manifest=dict(files=[dict(filename="a.safetensors", size=None)])),
        dict(manifest=dict(files=[])),
    ])
    inventory = weight_inventory(models, metadata, tmp_path)
    assert inventory.iloc[0].selected_weight_bytes == 100
    assert inventory.iloc[1:].selected_weight_bytes.isna().all()
    assert inventory.iloc[1:].inventory_status.tolist() == ["unknown_size", "empty_inventory"]
    assert select_budget(models, inventory, 100, 10).model_key.tolist() == [models.iloc[0].model_key]
    assert select_budget(models, inventory, 99, 10).empty


def test_budget_filter_precedes_limit_and_preserves_find_order(tmp_path):
    from scripts.find_weight_budget import weight_inventory, select_budget, find_with_budget
    models, metadata = catalog(tmp_path, [
        dict(manifest=dict(files=[dict(filename="model.safetensors", size=101)])), {}, {},
    ])
    inventory = weight_inventory(models, metadata, tmp_path)
    result = select_budget(models, inventory, 100, 1)
    assert result.model_key.tolist() == [models.iloc[1].model_key]
    assert result.iloc[0].params is None  # Missing nominal params never become guessed bytes.
    class FakeZoo:
        def find(self, **kwargs):
            assert kwargs == dict(license=["apache-2.0", "mit"], access="granted", gated=False,
                                  weight_format="safetensors", limit=len(models))
            return models.drop(columns=["revision", "is_adapter"])
    assert find_with_budget(FakeZoo(), models, inventory, 100, 1).model_key.tolist() == result.model_key.tolist()


def test_corrupt_wrong_identity_duplicate_names_and_unsafe_paths_are_unknown(tmp_path):
    from scripts.find_weight_budget import weight_inventory
    models, metadata = catalog(tmp_path, [
        {}, dict(manifest=dict(revision="f" * 40)),
        dict(manifest=dict(files=[dict(filename="same.safetensors", size=1)] * 2)),
        {},
    ])
    p = tmp_path / "0.manifest.json"
    p.write_text(p.read_text().replace('100', '101'))
    metadata["inputs"][3]["path"] = "../outside.manifest.json"
    result = weight_inventory(models, metadata, tmp_path)
    assert result.selected_weight_bytes.isna().all()
    assert result.inventory_status.tolist() == [
        "manifest_hash_mismatch", "missing_manifest", "duplicate_filename", "missing_manifest"]


def test_public_shortlist_excludes_adapters_local_pins_and_ambiguous_file_sets(tmp_path):
    from scripts.find_weight_budget import weight_inventory, select_budget
    models, metadata = catalog(tmp_path, [{}, dict(is_adapter=True), {},
        dict(manifest=dict(files=[dict(filename="other.bin", size=10)]))])
    # Exact manifest identity remains bound; a local pin is not a public download.
    models.loc[2, "revision"] = "local:" + "c" * 64
    models.loc[2, "model_key"] = models.loc[2, "repo_id"] + "@" + models.loc[2, "revision"]
    p = tmp_path / "2.manifest.json"
    m = json.loads(p.read_text()); m["revision"] = models.loc[2, "revision"]
    p.write_text(json.dumps(m)); metadata["inputs"][2]["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    inventory = weight_inventory(models, metadata, tmp_path)
    result = select_budget(models, inventory, 100, 10)
    assert result.model_key.tolist() == [models.iloc[0].model_key]
    assert inventory.iloc[3].inventory_status == "complete"  # Metadata-only bin accounting is allowed.
    assert inventory.iloc[3].all_safetensors == False


def test_budget_inventory_reads_only_manifests_and_conflicts_remain_unknown(tmp_path, monkeypatch):
    from scripts.find_weight_budget import weight_inventory
    models, metadata = catalog(tmp_path, [{}])
    original = Path.read_bytes
    def guarded(path):
        assert path.name.endswith(".manifest.json")
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    extra = tmp_path / "extra.manifest.json"
    extra.write_text((tmp_path / "0.manifest.json").read_text().replace('100', '200'))
    metadata["inputs"].append(dict(path=extra.name, sha256=hashlib.sha256(extra.read_bytes()).hexdigest()))
    row = weight_inventory(models, metadata, tmp_path).iloc[0]
    assert row.inventory_status == "conflicting_manifests"
    assert pd.isna(row.selected_weight_bytes)
