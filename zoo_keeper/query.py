"""Queries over one immutable zoo generation and its measured evidence."""

from collections import Counter, deque
import hashlib
import json
import os
from pathlib import Path

import duckdb
import pandas as pd
from safetensors import safe_open

from evidence.compare import check_verdict, diff_fingerprints
from evidence.fingerprint import load_fingerprint
from zoo import ZOO_VERSION
from zoo.retrieval import nearest
from zoo.schema import model_key


class AmbiguousModel(ValueError):
    """A bare repository matches several pinned revisions."""


class UnknownModel(KeyError):
    """The requested identity is absent from the zoo."""


class NotComparable(ValueError):
    """Measured evidence cannot support the requested comparison."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


_TABLES = ("models", "views", "coverage", "lineage", "scores", "predictions")
_FIND_COLUMNS = ("model_key", "repo_id", "modality", "params", "license", "downloads",
                 "coverage_status", "dup_group")
_SIMILAR_COLUMNS = ("model_key", "cosine", "dup_group", "license")
_TIERS = {"verified": 0, "detected": 1, "declared": 2}


def _nonnegative(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


class Zoo:
    """Open a published generation once; optional data_root locates evidence.

    A reader keeps its resolved generation even when a builder replaces the
    publication symlink. Metadata and retrieval do not require original files.
    """

    def __init__(self, path: Path, data_root: Path | None = None):
        self.path = Path(path).resolve(strict=True)
        root = data_root if data_root is not None else os.environ.get("ZOO_DATA")
        self.data_root = Path(root).resolve() if root else None
        self.metadata = json.loads((self.path / "zoo.json").read_text())
        if self.metadata.get("zoo_version") != ZOO_VERSION:
            raise ValueError("Unsupported zoo version")
        self._connection = duckdb.connect(":memory:")
        try:
            for name in _TABLES:
                # Names are fixed; paths go through DuckDB's relation API.
                self._connection.read_parquet(str(self.path / f"{name}.parquet")).create_view(name)
        except BaseException:
            self.close()
            raise

    def close(self):
        self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _records(self, sql, values=()):
        cursor = self._connection.execute(sql, values)
        names = [column[0] for column in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]

    def resolve(self, model: str) -> str:
        exact = self._records("SELECT model_key FROM models WHERE model_key = ?", [model])
        if exact:
            return exact[0]["model_key"]
        matches = self._records("SELECT model_key FROM models WHERE repo_id = ? ORDER BY model_key", [model])
        if not matches:
            raise UnknownModel(model)
        keys = [row["model_key"] for row in matches]
        if len(keys) > 1:
            raise AmbiguousModel(f"Several revisions match {model}: {', '.join(keys)}")
        return keys[0]

    def _model(self, model):
        key = self.resolve(model)
        return self._records("SELECT * FROM models WHERE model_key = ?", [key])[0]

    def find(self, modality=None, license=None, params_min=None, params_max=None,
             weight_format=None, access=None, gated=None, text=None, skill=None,
             min_score=None, verified_only=True, decision=None, system_one=None,
             architecture=None, limit=50) -> pd.DataFrame:
        """Filter metadata and means of relevant measured rows at pinned revisions."""
        _nonnegative(limit, "limit")
        clauses, values = [], []
        for clause, value in [
            ("m.modality = ?", modality), ("m.params >= ?", params_min),
            ("m.params <= ?", params_max), ("m.weight_format = ?", weight_format),
            ("m.access = ?", access), ("m.gated = ?", gated),
            ("m.is_decision = ?", decision), ("m.system_one = ?", system_one),
            ("m.architecture = ?", architecture),
        ]:
            if value is not None:
                clauses.append(clause)
                values.append(value)
        if license is not None:
            clauses.append("list_contains(?, m.license)" if isinstance(license, list) else "m.license = ?")
            values.append(license)
        if text is not None:
            clauses.append("contains(lower(m.repo_id), lower(?))")
            values.append(text)
        if skill is not None or min_score is not None:
            score_clauses = ["s.model_key = m.model_key"]
            if skill is not None:
                score_clauses.append("s.skill = ?")
                values.append(skill)
            if verified_only:
                score_clauses.append("s.revision_verified = ?")
                values.append(True)
            having = ""
            if min_score is not None:
                having = " HAVING avg(s.value) >= ?"
                values.append(min_score)
            clauses.append("EXISTS (SELECT 1 FROM scores s WHERE " + " AND ".join(score_clauses)
                           + " GROUP BY s.model_key" + having + ")")
        sql = "SELECT " + ", ".join("m." + column for column in _FIND_COLUMNS) + " FROM models m"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY m.downloads DESC NULLS LAST, m.model_key LIMIT ?"
        return self._connection.execute(sql, [*values, limit]).df()

    def _lineage(self, key):
        edges = self._records(
            "SELECT * FROM lineage WHERE child_key = ? OR parent_key = ? "
            "ORDER BY child_key, parent_key, estimand, evidence_tier, evidence", [key, key])
        result = dict(parents=[], children=[], same_root=[])
        for edge in edges:
            if edge["estimand"] == "same_root":
                relation = "same_root"
            elif edge["estimand"] in ("parent", "ancestor"):
                relation = "parents" if edge["child_key"] == key else "children"
            else:
                continue
            other = edge["parent_key"] if edge["child_key"] == key else edge["child_key"]
            result[relation].append(dict(key=other, **{field: value for field, value in edge.items()
                                                      if field not in ("child_key", "parent_key")}))
        return result

    def _artifact(self, relative):
        if self.data_root is None:
            raise ValueError("Evidence files require data_root or ZOO_DATA")
        relative = Path(relative)
        if relative.is_absolute():
            raise ValueError("Artifact path must be relative to data_root")
        path = (self.data_root / relative).resolve()
        if not path.is_relative_to(self.data_root):
            raise ValueError("Artifact path escapes data_root")
        return path

    def _skipped_files(self, key):
        # v0 models.parquet has no skipped_files field. Recover it only from
        # exact, contained manifest inputs whose build-time hashes still match.
        if self.data_root is not None:
            for entry in self.metadata.get("inputs", []):
                if entry.get("kind") != "file" or not str(entry.get("path", "")).endswith(".manifest.json"):
                    continue
                try:
                    path = self._artifact(entry["path"])
                    payload = path.read_bytes()
                    if hashlib.sha256(payload).hexdigest() != entry.get("sha256"):
                        continue
                    manifest = json.loads(payload)
                    if model_key(manifest["repo_id"], manifest["revision"]) == key and "skipped_files" in manifest:
                        return manifest["skipped_files"], None
                except (OSError, ValueError, KeyError, TypeError):
                    continue
        return None, "skipped files unavailable: matching original manifest and data_root required"

    def info(self, model) -> dict:
        row = self._model(model)
        key = row["model_key"]
        identity_fields = ("model_key", "repo_id", "revision")
        decision_fields = ("is_decision", "system_one", "decision_type", "num_labels", "labels")
        fingerprint_fields = ("fingerprint_t1", "fingerprint_t2", "spectra_t2")
        excluded = {*identity_fields, *decision_fields, *fingerprint_fields, "coverage_status", "skipped_tensors"}
        decision = {field: row[field] for field in decision_fields}
        labels = decision["labels"] or {}
        decision["labels"] = [labels[label] for label in sorted(labels, key=lambda label: (
            0, int(label)) if str(label).isdigit() else (1, str(label)))]
        decision["head_views"] = [view["name"] for view in self._records(
            "SELECT DISTINCT name FROM views WHERE model_key = ? AND geometry = ? ORDER BY name", [key, "head"])]
        skipped = row["skipped_tensors"] or []
        files, note = self._skipped_files(key)
        duplicates = self._records("SELECT model_key FROM models WHERE dup_group = ? AND model_key <> ? ORDER BY model_key",
                                   [row["dup_group"], key])
        return dict(
            identity={field: row[field] for field in identity_fields},
            metadata={field: value for field, value in row.items() if field not in excluded},
            coverage=dict(status=row["coverage_status"], skipped_tensors=skipped,
                          skipped_tensor_reasons=dict(sorted(Counter(item["reason"] for item in skipped).items())),
                          skipped_files=files, skipped_files_note=note),
            measured_scores=self._records("SELECT * FROM scores WHERE model_key = ? ORDER BY benchmark, source, source_revision, harness", [key]),
            predictions=self._records("SELECT * FROM predictions WHERE model_key = ? ORDER BY skill", [key]),
            predictions_note="predictions arrive in zoo v1", lineage=self._lineage(key),
            duplicates=[item["model_key"] for item in duplicates],
            duplicates_note="identical on every measured block", decision=decision,
            fingerprints={field: row[field] for field in fingerprint_fields})

    def similar(self, model, k=10, collapse_duplicates=True) -> pd.DataFrame:
        _nonnegative(k, "k")
        row = self._model(model)
        if not row["fingerprint_t1"]:
            raise NotComparable("no_t1_fingerprint" if row["fingerprint_t2"] else row["coverage_status"])
        candidates = self._records(
            "SELECT model_key, dup_group, license, downloads FROM models "
            "WHERE arch_signature = ? AND fingerprint_t1 <> ?", [row["arch_signature"], ""])
        if len(candidates) < 2:
            raise NotComparable("no_same_architecture_models")
        if k == 0:
            return pd.DataFrame(columns=_SIMILAR_COLUMNS)
        metadata = self.path / "retrieval" / f"{row['arch_signature']}.json"
        if not metadata.exists():
            raise NotComparable("no_compatible_views")
        matches = nearest(self.path, row["model_key"], row["arch_signature"], len(candidates))
        if not matches:
            raise NotComparable("no_compatible_views")
        known = {candidate["model_key"]: candidate for candidate in candidates}
        output = []
        for key, cosine in matches:
            if key not in known:
                raise ValueError("Retrieval model disagrees with architecture table")
            other = known[key]
            if other["dup_group"] != row["dup_group"]:
                output.append(dict(model_key=key, cosine=cosine, dup_group=other["dup_group"],
                                   license=other["license"], downloads=other["downloads"]))
        if collapse_duplicates:
            representatives = {}
            # Stable ties use model_key; missing download counts sort last.
            for item in sorted(output, key=lambda item: (
                    -(item["downloads"] if item["downloads"] is not None else float("-inf")), item["model_key"])):
                representatives.setdefault(item["dup_group"], item)
            output = list(representatives.values())
        output.sort(key=lambda item: (-item["cosine"], item["model_key"]))
        return pd.DataFrame(output[:k], columns=_SIMILAR_COLUMNS)

    def family(self, model, depth=2) -> dict:
        _nonnegative(depth, "depth")
        key = self.resolve(model)
        edges = self._records("SELECT * FROM lineage WHERE estimand IN (?, ?) "
                              "ORDER BY child_key, parent_key, evidence", ["parent", "ancestor"])
        strongest = {}
        for edge in edges:
            child, rank = edge["child_key"], _TIERS.get(edge["evidence_tier"], 3)
            strongest[child] = min(strongest.get(child, 3), rank)
        parents, children = {}, {}
        for edge in edges:
            child, parent = edge["child_key"], edge["parent_key"]
            if _TIERS.get(edge["evidence_tier"], 3) == strongest[child]:
                parents.setdefault(child, set()).add(parent)
                children.setdefault(parent, set()).add(child)

        def traverse(graph):
            visited, result, queue = {key}, [], deque([(key, 0)])
            while queue:
                current, distance = queue.popleft()
                if distance >= depth:
                    continue
                for neighbor in sorted(graph.get(current, ())):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        result.append(neighbor)
                        queue.append((neighbor, distance + 1))
            return result

        licenses = {row["model_key"]: row["license"] for row in self._records("SELECT model_key, license FROM models")}
        chain, seen, current = [], set(), key
        for _ in range(depth + 1):
            chain.append((current, licenses.get(current)))
            seen.add(current)
            choices = sorted(parents.get(current, set()) - seen)
            if not choices:
                break
            current = choices[0]
        roots = sorted({edge["key"] for edge in self._lineage(key)["same_root"] if edge["key"] != key})
        return dict(ancestors=traverse(parents), descendants=traverse(children), same_root=roots,
                    license_chain=chain, license_differs=len({license for _, license in chain if license is not None}) > 1)

    def _fingerprint(self, row, tier):
        path = self._artifact(row[f"fingerprint_{tier}"])
        # The evidence loader chooses companion access from embedded metadata,
        # so reject a mislabeled artifact before it can open any companion.
        with safe_open(path, framework="numpy") as handle:
            metadata = handle.metadata()
        if model_key(metadata["repo_id"], metadata["revision"]) != row["model_key"]:
            raise ValueError(f"Fingerprint identity does not match model: {row['model_key']}")
        if metadata.get("tier") != tier:
            raise ValueError(f"Fingerprint tier does not match {tier}")
        if tier == "t2":
            # load_fingerprint also opens this companion file. Check containment
            # before handing it to the evidence loader (including symlinks).
            companion = path.with_name(path.name.removesuffix(".t2.fingerprint.safetensors") + ".t2.spectra.h5")
            if not companion.resolve().is_relative_to(self.data_root):
                raise ValueError("T2 spectra path escapes data_root")
        fingerprint = load_fingerprint(path)
        if model_key(fingerprint.meta["repo_id"], fingerprint.meta["revision"]) != row["model_key"]:
            raise ValueError(f"Fingerprint identity does not match model: {row['model_key']}")
        if fingerprint.meta.get("tier") != tier:
            raise ValueError(f"Fingerprint tier does not match {tier}")
        return fingerprint

    def diff(self, a, b, null: Path | None = None) -> dict:
        """Return measured evidence, exposing summary.heads as {name: ...} records.

        The query interface wraps upstream head-name strings without altering
        any computed values or the upstream fingerprint/diff formats.
        """
        left, right = self._model(a), self._model(b)
        tier = next((tier for tier in ("t1", "t2")
                     if left[f"fingerprint_{tier}"] and right[f"fingerprint_{tier}"]), None)
        if tier is None:
            raise NotComparable("no_common_tier")
        calibration = json.loads(Path(null).read_text()) if null is not None else None
        result = diff_fingerprints(self._fingerprint(left, tier), self._fingerprint(right, tier), null=calibration)
        result["summary"]["heads"] = [dict(name=name) for name in result["summary"]["heads"]]
        return result

    def check(self, model, null: Path | None = None) -> dict:
        row = self._model(model)
        for declared in row["declared_bases"] or []:
            try:
                base = self.resolve(declared)
            except UnknownModel:
                continue
            result = self.diff(row["model_key"], base, null=null)
            result["verdict"] = check_verdict(result)
            return result
        return {"verdict": "base_not_in_zoo"}
