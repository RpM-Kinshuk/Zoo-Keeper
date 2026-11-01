"""Pin selected metadata inputs without fetching weights or trusting Hub code."""

import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import tempfile


PERMISSIVE_LICENSES = frozenset({
    "apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "cc0-1.0",
    "cc-by-4.0",
})
_SHA = re.compile(r"[0-9a-f]{40}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def _field(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _relative(path):
    p = PurePosixPath(path)
    return bool(path) and not p.is_absolute() and ".." not in p.parts and "\\" not in path


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_pin(pin: dict) -> None:
    """Reject unresolved identity, licensing, or nonportable evidence paths."""
    label = pin.get("model_id", pin.get("repo_id", "input"))
    if pin.get("status") != "pinned":
        raise ValueError(f"{label}: input is not pinned")
    if not _SHA.fullmatch(pin.get("revision", "")):
        raise ValueError(f"{label}: revision must be a full commit SHA")
    if pin.get("license") not in PERMISSIVE_LICENSES:
        raise ValueError(f"{label}: missing, ambiguous, or unsupported license")
    if not _relative(pin.get("license_path", "")):
        raise ValueError(f"{label}: invalid license evidence path")
    if not _HASH.fullmatch(pin.get("license_sha256", "")):
        raise ValueError(f"{label}: invalid license evidence hash")
    for field in ("config_sha256", "tokenizer_sha256"):
        if pin.get("kind", "model") == "model" and not _HASH.fullmatch(pin.get(field, "")):
            raise ValueError(f"{label}: missing {field}")
    if pin.get("kind") == "dataset":
        split = pin.get("split")
        if not isinstance(split, str) or not split.strip():
            raise ValueError(f"{label}: dataset requires a nonempty split")
        if not _HASH.fullmatch(pin.get("selected_index_sha256", "")):
            raise ValueError(f"{label}: dataset requires a frozen selected index hash")
        if not isinstance(pin.get("preprocessing"), dict) or not pin["preprocessing"]:
            raise ValueError(f"{label}: dataset requires a preprocessing configuration")


def resolve_pins(config: dict, *, api, download_text) -> dict:
    """Resolve each selected branch once, then hash metadata at that commit.

    ``api`` is an authenticated Hub client (or an offline fake). ``download_text``
    accepts repo ID, filename, revision and repo_type; a live callback keeps the
    evidence at the returned relative path. Permission failures are data, never
    copied exception messages. Unknown license declarations require review.
    """
    entries = config.get("entries", [])
    ids = [entry["model_id"] for entry in entries]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate model_id in pin configuration")
    for entry in entries:
        label, repo_id = entry["model_id"], entry["repo_id"]
        if not _ID.fullmatch(label):
            raise ValueError(f"{label}: invalid input ID")
        if repo_id.casefold().startswith("qwen/qwen2.5-3b"):
            raise ValueError("Qwen2.5-3B is excluded from the lab")
        if entry.get("kind", "model") not in {"model", "dataset"}:
            raise ValueError(f"{label}: invalid input kind")

    result = {}
    for entry in entries:
        label, repo_id = entry["model_id"], entry["repo_id"]
        kind = entry.get("kind", "model")
        resolve = api.model_info if kind == "model" else api.dataset_info
        try:
            info = resolve(repo_id, revision=entry.get("revision", "main"))
        except Exception as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            if isinstance(error, PermissionError) or status in {401, 403}:
                result[label] = {"model_id": label, "repo_id": repo_id,
                                 "kind": kind, "status": "no_access"}
                continue
            raise RuntimeError(f"{label}: metadata request failed") from None
        sha = _field(info, "sha", "")
        if not isinstance(sha, str) or not _SHA.fullmatch(sha):
            raise ValueError(f"{label}: metadata did not resolve a full commit SHA")
        requested = entry.get("revision", "main")
        if _SHA.fullmatch(requested) and requested != sha:
            raise ValueError(f"{label}: returned revision differs from pinned request")
        if _field(info, "gated", False) and not entry.get("gated_approved", False):
            raise ValueError(f"{label}: gated input requires the owner's agreement")
        card = _field(info, "card_data") or _field(info, "cardData") or {}
        license = _field(card, "license")
        if not isinstance(license, str) or license.lower() not in PERMISSIVE_LICENSES:
            raise ValueError(f"{label}: missing, ambiguous, or unsupported license")
        license = license.lower()
        if entry.get("license") and entry["license"].lower() != license:
            raise ValueError(f"{label}: license differs from configured expectation")
        files = sorted(_field(sibling, "rfilename", "") for sibling in _field(info, "siblings", []))
        license_files = [name for name in files if _relative(name) and "/" not in name and
                         PurePosixPath(name).name.lower() in
                         {"license", "license.txt", "license.md", "copying"}]

        def read(name):
            try:
                text = download_text(repo_id, name, revision=sha, repo_type=kind)
            except Exception:
                raise RuntimeError(f"{label}: selected metadata file unavailable") from None
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{label}: empty metadata evidence")
            return text

        if license_files:
            evidence = {name: read(name) for name in license_files}
            if len({_sha256(text) for text in evidence.values()}) > 1:
                raise ValueError(f"{label}: conflicting license evidence files; review required")
            license_name = license_files[0]
            license_text = evidence[license_name]
            evidence_type = "standalone_license"
        elif "README.md" in files:
            license_name = "README.md"
            license_text = read(license_name)
            front = re.match(r"\A---\s*\n(.*?)\n---(?:\s|$)", license_text, re.S)
            declaration = re.search(r"^license:\s*['\"]?([a-zA-Z0-9.-]+)['\"]?\s*$",
                                    front[1], re.M) if front else None
            if not declaration or declaration[1].lower() != license:
                raise ValueError(f"{label}: missing or conflicting README license declaration")
            evidence_type = "pinned_readme_declaration"
        else:
            raise ValueError(f"{label}: missing pinned license evidence file; review required")
        pin = {"model_id": label, "repo_id": repo_id, "kind": kind,
               "revision": sha, "license": license, "status": "pinned",
               "license_path": f"lab/inputs/evidence/{label}/{license_name}",
               "license_sha256": _sha256(license_text),
               "license_evidence_type": evidence_type}
        if entry.get("base_family"):
            pin["base_family"] = entry["base_family"]
        if kind == "model":
            weight_hashes = {}
            for sibling in _field(info, "siblings", []):
                name = _field(sibling, "rfilename", "")
                digest = _field(_field(sibling, "lfs", {}), "sha256", "")
                if name.endswith(".safetensors") and _relative(name) and _HASH.fullmatch(digest or ""):
                    weight_hashes[name] = digest
            pin["weight_sha256"] = dict(sorted(weight_hashes.items()))
            if "config.json" not in files:
                raise ValueError(f"{label}: missing model configuration")
            config_text = read("config.json")
            try:
                model_config = json.loads(config_text)
            except ValueError:
                raise ValueError(f"{label}: invalid model configuration") from None
            if not isinstance(model_config, dict):
                raise ValueError(f"{label}: invalid model configuration")
            pin["config_sha256"] = _sha256(json.dumps(model_config, sort_keys=True, separators=(",", ":"), allow_nan=False))
            pin["config_file_sha256"] = _sha256(config_text)
            token_files = [name for name in files if name in
                           {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}]
            if not token_files:
                raise ValueError(f"{label}: missing tokenizer metadata")
            token_hashes = {name: _sha256(read(name)) for name in token_files}
            pin["tokenizer_sha256"] = _sha256(json.dumps(token_hashes, sort_keys=True, separators=(",", ":")))
        else:
            for field in ("split", "selected_index_sha256", "preprocessing"):
                if field not in entry:
                    raise ValueError(f"{label}: dataset requires frozen {field}")
                pin[field] = entry[field]
        verify_pin(pin)
        result[label] = pin
    return result


def metadata_callbacks(data_root: Path, entries: list[dict]):
    """Build authenticated metadata/evidence callbacks without making requests."""
    from types import SimpleNamespace
    from functools import partial
    from .schema import contained_path

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN must be set before metadata pinning")
    from huggingface_hub import HfApi, hf_hub_download

    destinations = {entry["repo_id"]: entry["model_id"] for entry in entries}

    def download(repo_id, filename, *, revision, repo_type):
        # Hub-cache internals and raw cards stay outside both repositories.
        try:
            source = hf_hub_download(repo_id, filename, revision=revision,
                                     repo_type=repo_type, token=token)
            content = Path(source).read_bytes()
        except Exception:
            raise RuntimeError("selected pinned metadata download failed") from None
        text = content.decode("utf-8")
        relative = f"lab/inputs/evidence/{destinations[repo_id]}/{filename}"
        target = contained_path(data_root, relative, destinations[repo_id])
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".evidence-", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return text

    client = HfApi(token=token)
    api = SimpleNamespace(model_info=partial(client.model_info, files_metadata=True),
                          dataset_info=client.dataset_info)
    return api, download


def pin_config(config: dict, output: Path, *, data_root: Path) -> dict:
    """Run authenticated metadata pinning and save immutable-revision evidence."""
    from .artifacts import atomic_json

    output = Path(output).resolve()
    if not output.is_relative_to(Path(data_root).resolve()):
        raise ValueError("pin output must live under ZOO_DATA")
    api, download = metadata_callbacks(data_root, config.get("entries", []))
    pins = resolve_pins(config, api=api, download_text=download)
    atomic_json(output, pins)
    return pins
