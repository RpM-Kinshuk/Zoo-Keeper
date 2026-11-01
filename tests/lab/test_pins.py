from types import SimpleNamespace
import json
from pathlib import Path

import pytest

from lab.pins import resolve_pins, verify_pin


SHA = "a" * 40


class FakeAPI:
    def __init__(self, license="apache-2.0", denied=False):
        self.calls = []
        self.license = license
        self.denied = denied

    def model_info(self, repo_id, *, revision, files_metadata=False):
        self.calls.append((repo_id, revision))
        if self.denied:
            error = PermissionError("no_access hf_secret_marker")
            raise error
        return SimpleNamespace(
            sha=revision if len(revision) == 40 else SHA, gated=False, card_data={"license": self.license},
            siblings=[SimpleNamespace(rfilename=name) for name in
                      ["LICENSE", "config.json", "tokenizer_config.json"]],
        )

    dataset_info = model_info


def config(repo="org/model"):
    return {"entries": [{"model_id": "base", "repo_id": repo,
                         "revision": "main", "kind": "model"}]}


def download_text(repo_id, filename, *, revision, repo_type):
    assert len(revision) == 40 and int(revision, 16) >= 0
    return {"LICENSE": "Apache License\nVersion 2.0",
            "config.json": '{"model_type":"llama"}',
            "tokenizer_config.json": "{}"}[filename]


def test_full_pins_with_license_and_configuration_hashes():
    api = FakeAPI()
    out = resolve_pins(config(), api=api, download_text=download_text)
    pin = out["base"]
    assert pin["revision"] == SHA
    assert pin["license"] == "apache-2.0"
    assert pin["status"] == "pinned"
    assert len(pin["license_sha256"]) == 64
    assert len(pin["config_sha256"]) == 64
    assert len(pin["tokenizer_sha256"]) == 64
    assert api.calls == [("org/model", "main")]
    verify_pin(pin)


@pytest.mark.parametrize("license", [None, "unknown", ["apache-2.0", "mit"]])
def test_missing_or_ambiguous_license_fails(license):
    with pytest.raises(ValueError, match="license"):
        resolve_pins(config(), api=FakeAPI(license), download_text=download_text)


def test_forbidden_base_rejected_before_request():
    api = FakeAPI()
    with pytest.raises(ValueError, match="Qwen2.5-3B"):
        resolve_pins(config("Qwen/Qwen2.5-3B"), api=api,
                     download_text=download_text)
    assert api.calls == []


def test_denied_repository_is_recorded_without_secret_exception_text():
    api = FakeAPI(denied=True)
    out = resolve_pins(config(), api=api, download_text=download_text)
    assert out["base"]["status"] == "no_access"
    assert len(api.calls) == 1
    assert "hf_secret_marker" not in str(out)


def test_missing_license_file_requires_review():
    api = FakeAPI()
    original = api.model_info

    def no_license(*args, **kwargs):
        info = original(*args, **kwargs)
        info.siblings = info.siblings[1:]
        return info

    api.model_info = no_license
    with pytest.raises(ValueError, match="license evidence"):
        resolve_pins(config(), api=api, download_text=download_text)


def test_revision_and_path_validation():
    pin = resolve_pins(config(), api=FakeAPI(), download_text=download_text)["base"]
    for changed in ({"revision": "main"}, {"license_path": "/private/LICENSE"},
                    {"license_path": "../LICENSE"}, {"license_sha256": "bad"}):
        with pytest.raises(ValueError):
            verify_pin({**pin, **changed})


def test_seven_base_configuration_uses_full_pins():
    bases = json.loads((Path(__file__).parents[2] / "lab/configs/bases.json").read_text())
    api = FakeAPI()
    out = resolve_pins(bases, api=api, download_text=download_text)
    assert len(out) == 7
    assert all(out[entry["model_id"]]["revision"] == entry["revision"] for entry in bases["entries"])
    assert len(api.calls) == 7


def test_canonical_configuration_hash_retains_raw_file_hash():
    def changed_text(repo, name, **kwargs):
        return '{ "model_type": "llama" }' if name == "config.json" else download_text(repo, name, **kwargs)

    a = resolve_pins(config(), api=FakeAPI(), download_text=download_text)["base"]
    b = resolve_pins(config(), api=FakeAPI(), download_text=changed_text)["base"]
    assert a["config_sha256"] == b["config_sha256"]
    assert a["config_file_sha256"] != b["config_file_sha256"]


def test_dataset_pin_requires_frozen_valid_fields():
    model_pin = resolve_pins(config(), api=FakeAPI(), download_text=download_text)["base"]
    for fields in ({}, {"split": "", "selected_index_sha256": "a"*64, "preprocessing": {"seed": 1}},
                   {"split": "train", "selected_index_sha256": "invalid", "preprocessing": {"seed": 1}},
                   {"split": "train", "selected_index_sha256": "a"*64, "preprocessing": None}):
        with pytest.raises(ValueError, match="dataset"):
            verify_pin({**model_pin, "kind": "dataset", **fields})


def test_nested_third_party_license_is_not_project_evidence():
    api = FakeAPI()
    original = api.model_info

    def nested(*args, **kwargs):
        info = original(*args, **kwargs)
        info.siblings[0].rfilename = "third_party/LICENSE"
        return info

    api.model_info = nested
    with pytest.raises(ValueError, match="license evidence"):
        resolve_pins(config(), api=api, download_text=download_text)


def test_pinned_readme_declaration_is_accepted_but_must_match():
    api = FakeAPI()
    original = api.model_info

    def readme(*args, **kwargs):
        info = original(*args, **kwargs)
        info.siblings[0].rfilename = "README.md"
        return info

    api.model_info = readme

    def text(repo, name, **kwargs):
        return "---\nlicense: apache-2.0\n---\nModel" if name == "README.md" else download_text(repo, name, **kwargs)

    pin = resolve_pins(config(), api=api, download_text=text)["base"]
    assert pin["license_evidence_type"] == "pinned_readme_declaration"
    assert pin["license_path"].endswith("README.md")
    with pytest.raises(ValueError, match="license"):
        resolve_pins(config(), api=api, download_text=lambda *a, **k: "---\nlicense: mit\n---")


def test_pin_config_saves_evidence_under_data_root(tmp_path, monkeypatch):
    import huggingface_hub
    from lab.pins import pin_config

    cache = tmp_path / "cache"
    cache.mkdir()

    def fake_download(repo_id, filename, *, revision, repo_type, token):
        path = cache / filename
        path.write_text(download_text(repo_id, filename, revision=revision, repo_type=repo_type))
        return str(path)

    monkeypatch.setenv("HF_TOKEN", "hf_secret_marker")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kwargs: FakeAPI())
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    root = tmp_path / "data"
    output = root / "lab/inputs/pins.json"
    pins = pin_config(config(), output, data_root=root)
    assert output.exists()
    assert (root / pins["base"]["license_path"]).read_text().startswith("Apache License")
    assert "hf_secret_marker" not in output.read_text()
