"""Assemble explicitly mapped Raw observations without inventing evidence.

Source descriptors contain ``path``, ``name`` (a portable audit label), and a
``schema`` key from registry['schemas'], or an inline ``mapping``. Mappings have
``columns`` for deployment identity and ``tasks`` for individual Raw columns.
Registry evidence paths resolve under ``evidence_root`` (default current dir).
Pinned models are independent manifest records, with repo_id, revision,
harness, precision, quantization and evidence_source; optional task/benchmark
columns constrain their deployment. The caller is responsible for authenticating
the independent manifest; a score export must never masquerade as a manifest.
"""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import pandas as pd

from .protocol import canonical_json


SCORE_COLUMNS = ["record_id", "model_key", "repo_id", "revision", "benchmark", "task", "harness", "precision", "quantization", "accuracy", "n_items", "measurement_variance", "variance_source", "revision_status", "revision_verified", "source_hash", "source_row", "source_provenance", "eval_name"]
EXCLUSION_COLUMNS = ["stage", "record_id", "reason", "detail"]
IDENTITY = ["repo_id", "revision", "benchmark", "task", "harness", "precision", "quantization"]


def _text(value):
    return "" if pd.isna(value) else str(value).strip()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_file(path):
    path = Path(path)
    if any(part.startswith(".env") for part in path.parts):
        raise ValueError("refusing to open .env inputs")
    resolved = path.resolve()
    if any(part.startswith(".env") for part in resolved.parts):
        raise ValueError("refusing to open .env inputs")
    return resolved


def inventory_inputs(paths: list[Path], *, data_root: Path) -> dict:
    """Inventory only supplied files beneath a declared root; never recurse."""
    root = Path(data_root).resolve()
    entries = []
    for item in paths:
        path = _safe_file(item)
        if not path.is_relative_to(root):
            raise ValueError("input lies outside data root")
        record = {"path": path.relative_to(root).as_posix(), "size_bytes": None, "sha256": None}
        if not path.exists():
            record["status"] = "missing"
        elif not path.is_file():
            record["status"] = "not_a_file"
        else:
            record.update(size_bytes=path.stat().st_size, sha256=_sha256(path))
            record["status"] = "ok" if record["size_bytes"] else "missing_zero_bytes"
        entries.append(record)
    return {"files": sorted(entries, key=lambda row: row["path"])}


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def _evidence(proof, root):
    if not isinstance(proof, dict) or not proof.get("path") or not proof.get("sha256"):
        return False
    path = _safe_file(Path(root) / proof["path"])
    root = Path(root).resolve()
    if Path(proof["path"]).is_absolute() or not path.is_relative_to(root):
        return False
    return path.is_file() and path.stat().st_size > 0 and _sha256(path) == proof["sha256"]


def _variance(task, row, accuracy, evidence_root):
    """Return n, variance, source, reason; missing evidence is never filled."""
    if task.get("kind") == "binary":
        raw_n = row.get(task["n_items_column"]) if task.get("n_items_column") else task.get("n_items")
        n = _number(raw_n)
        if raw_n is not None and not pd.isna(raw_n) and (n is None or n <= 0 or not n.is_integer()):
            return None, None, "", "invalid_n_items"
        proof = task.get("denominator_evidence")
        if n is not None and _evidence(proof, evidence_root):
            return int(n), accuracy * (1 - accuracy) / n, f"binomial:{proof['path']}@{proof['sha256']}", None
    elif task.get("measurement_variance_column") or "measurement_variance" in task:
        raw_v = row.get(task["measurement_variance_column"]) if task.get("measurement_variance_column") else task["measurement_variance"]
        variance = _number(raw_v)
        proof = task.get("variance_evidence")
        if variance is not None and variance >= 0 and _evidence(proof, evidence_root):
            return None, variance, f"documented:{proof['path']}@{proof['sha256']}", None
    return None, None, "", "variance_unavailable"


def _model_key(repo, revision):
    if repo and re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        return f"{repo}@{revision.lower()}"
    if repo.startswith("lab/") and re.fullmatch(r"local:[0-9a-fA-F]{64}", revision):
        return f"{repo}@{revision.lower()}"
    return ""


def _verified(record, pins, source):
    if not record["model_key"] or not all(record[key] for key in ("precision", "quantization", "harness")):
        return False
    for pin in pins:
        evidence = _text(pin.get("evidence_source", ""))
        if not evidence or evidence in {source["name"], str(source["path"]), record["source_hash"]}:
            continue
        if _text(pin.get("repo_id", "")).casefold() != record["repo_id"].casefold():
            continue
        if _text(pin.get("revision", "")).lower() != record["revision"].lower():
            continue
        if any(_text(pin.get(key, "")) != record[key] for key in ("harness", "precision", "quantization")):
            continue
        if any(_text(pin.get(key, "")) and _text(pin[key]) != record[key] for key in ("benchmark", "task")):
            continue
        if _text(pin.get("model_key", "")) and _text(pin["model_key"]).casefold() != record["model_key"].casefold():
            continue
        return True
    return False


def assemble_scores(sources: list[dict], pinned_models: pd.DataFrame, task_registry: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return descriptive scores, explicit exclusions, and join/variance coverage.

    Exclusions include analysis-specific ineligibility: an unsupported variance
    does not discard its descriptive observation. Conflicts and invalid observed
    accuracies are removed entirely. source_row is the 1-based data row (no header).
    source_provenance retains every collapsed source hash and row as sorted JSON.
    """
    exclusions, records = [], []
    input_counts = Counter()
    pins = pinned_models.to_dict("records")
    evidence_root = task_registry.get("evidence_root", Path.cwd())

    def exclude(record_id, reason, detail):
        exclusions.append(dict(stage="scores", record_id=record_id, reason=reason, detail=detail))

    for source in sources:
        path = _safe_file(source["path"])
        name = source.get("name", path.name)
        if Path(name).is_absolute():
            raise ValueError("source audit names must be relative")
        source = dict(source, name=name)
        mapping = source.get("mapping", task_registry.get("schemas", {}).get(source.get("schema")))
        if not path.is_file() or path.stat().st_size == 0:
            exclude(name, "missing_zero_bytes" if path.is_file() else "missing", name)
            continue
        if not mapping:
            exclude(name, "missing_schema_mapping", name)
            continue
        digest = _sha256(path)
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        columns = mapping.get("columns", {})
        for index, source_row in enumerate(frame.to_dict("records"), 1):
            for task_index, task in enumerate(mapping.get("tasks", [])):
                input_counts[name] += 1
                alias = _text(source_row.get(columns.get("record_id"), ""))
                # One fixture/source ID may describe a wide row with many tasks.
                record_id = alias if task_index == 0 else f"{alias}:{task.get('task', task_index)}" if alias else ""
                fallback_id = f"{name}:{index}:{task.get('task', task_index)}"
                label = record_id or fallback_id
                if task.get("scale") != "raw_accuracy":
                    exclude(label, "unsupported_scale", "only explicitly mapped raw_accuracy is accepted")
                    continue
                if task.get("column") not in source_row:
                    exclude(label, "missing_score_column", str(task.get("column")))
                    continue
                accuracy = _number(source_row[task["column"]])
                if accuracy is None or not 0 <= accuracy <= 1:
                    exclude(label, "invalid_observation", "Raw accuracy must be finite and in [0, 1]")
                    continue
                record = {key: _text(source_row.get(column, "")) for key, column in columns.items() if key != "record_id"}
                for key in IDENTITY:
                    record.setdefault(key, "")
                record.setdefault("eval_name", "")
                for key in ("benchmark", "task", "harness", "precision", "quantization"):
                    if key in task:
                        record[key] = str(task[key])
                record["revision"] = record["revision"].lower()
                if not record["benchmark"] or not record["task"]:
                    exclude(label, "missing_task_identity", "benchmark and task must be explicit")
                    continue
                record.update(record_id=record_id, accuracy=accuracy, source_hash=digest, source_row=index)
                record["model_key"] = _model_key(record["repo_id"], record["revision"])
                record["revision_verified"] = _verified(record, pins, source)
                deployment_recorded = record["model_key"] and all(record[key] for key in ("harness", "precision", "quantization"))
                record["revision_status"] = "verified" if record["revision_verified"] else "recorded" if deployment_recorded else "unverified"
                n, variance, variance_source, reason = _variance(task, source_row, accuracy, evidence_root)
                record.update(n_items=n, measurement_variance=variance, variance_source=variance_source)
                record["_reasons"] = []
                if reason:
                    record["_reasons"].append(reason)
                if not record["revision_verified"]:
                    record["_reasons"].append("deployment_unverified")
                if not record["model_key"]:
                    record["_reasons"].append("unpinned_model")
                record["_source"] = name
                identity = [record[key].casefold() if key == "repo_id" else record[key] for key in IDENTITY]
                # eval_name identifies copies of the same leaderboard export;
                # absent that evidence, incomplete deployments remain separate.
                if not all(identity):
                    identity.extend(["export", record["eval_name"]] if record["eval_name"] else [digest, index])
                record["_identity"] = canonical_json(identity).decode().strip()
                records.append(record)

    grouped = {}
    for record in records:
        grouped.setdefault(record["_identity"], []).append(record)
    rows = []
    used_ids = set()
    for identity, copies in sorted(grouped.items()):
        copies.sort(key=lambda row: (row["source_hash"], row["source_row"], row["_source"]))
        aliases = {row["record_id"] for row in copies if row["record_id"]}
        generated = "score_" + hashlib.sha256(identity.encode()).hexdigest()
        record_id = next(iter(aliases)) if len(aliases) == 1 else generated
        if record_id in used_ids:
            record_id = generated
        used_ids.add(record_id)
        if len({row["accuracy"] for row in copies}) > 1:
            exclude(record_id, "conflicting_score", json.dumps([dict(source=row["_source"], source_hash=row["source_hash"], source_row=row["source_row"], accuracy=row["accuracy"]) for row in copies], sort_keys=True))
            continue
        supported = [row for row in copies if row["measurement_variance"] is not None]
        if len({(row["n_items"], row["measurement_variance"]) for row in supported}) > 1:
            exclude(record_id, "conflicting_measurement_evidence", "documented denominator/variance evidence differs at one deployment identity")
            continue
        record = (supported[0] if supported else copies[0]).copy()
        record["record_id"] = record_id
        record["revision_verified"] = any(row["revision_verified"] for row in copies)
        if record["revision_verified"]:
            record["revision_status"] = "verified"
        provenance = [dict(source=row["_source"], source_hash=row["source_hash"], source_row=row["source_row"]) for row in copies]
        record["source_provenance"] = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
        for reason in sorted({reason for row in copies for reason in row["_reasons"]}):
            if reason == "deployment_unverified" and record["revision_verified"]:
                continue
            if reason in {"variance_unavailable", "invalid_n_items"} and supported:
                continue
            exclude(record_id, reason, "descriptive observation retained; excluded from the corresponding verified/measurement-corrected analysis")
        rows.append({key: record[key] for key in SCORE_COLUMNS})
    scores = pd.DataFrame(rows, columns=SCORE_COLUMNS).sort_values("record_id").reset_index(drop=True)
    scores["revision_verified"] = scores["revision_verified"].astype(bool)
    excluded = pd.DataFrame(exclusions, columns=EXCLUSION_COLUMNS).drop_duplicates().sort_values(EXCLUSION_COLUMNS).reset_index(drop=True)
    verified = int(scores.revision_verified.sum())
    eligible = len(scores)
    coverage = {
        "input_observations": sum(input_counts.values()), "input_observations_by_source": dict(sorted(input_counts.items())),
        "retained_observations": eligible, "verified_joins": verified, "eligible_joins": eligible,
        "score_to_model_join": {"numerator": verified, "denominator": eligible, "fraction": verified / eligible if eligible else None, "denominator_definition": "retained unique deployment/task observations, including unpinned and missing deployment evidence"},
        "prediction_to_score_join": {"numerator": 0, "denominator": 0, "fraction": None, "denominator_definition": "not assessed by source assembly; predictions are a later audit"},
        "retained_by_task": dict(sorted(Counter(f"{row['benchmark']}/{row['task']}/{row['harness']}" for row in rows).items())),
        "reasons": dict(sorted(Counter(excluded.reason).items())),
        "variance_available": int(scores.measurement_variance.notna().sum()),
        "variance_unavailable": int(scores.measurement_variance.isna().sum()),
        "v1_status": "supplied_explicitly" if any(row["benchmark"] == "v1" for row in rows) else "unavailable_no_explicit_source",
    }
    return scores, excluded, coverage
