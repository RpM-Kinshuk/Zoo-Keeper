"""Find recorded accessible Apache/MIT non-adapters by selected weight-file bytes.

This offline adapter reuses Zoo.find; it changes no database or score schema.
The budget covers the manifest's selected files, not RAM/VRAM, tokenizer/config
files, adapter bases, or the smallest alternative quantization in a repository.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pandas as pd


def weight_inventory(models, metadata, data_root):
    """Join snapshot-hashed manifests to exact keys; unknown is never zero."""
    root = Path(data_root).resolve()
    records, issues = {}, []
    for item in metadata["inputs"]:
        name = item.get("path", "")
        if not name.endswith(".manifest.json"):
            continue
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path.name == ".env":
            issues.append(dict(path=name, reason="unsafe_manifest_path"))
            continue
        try:
            raw = path.read_bytes()
            manifest = json.loads(raw)
            key = manifest["repo_id"] + "@" + manifest["revision"]
        except (OSError, ValueError, KeyError, TypeError):
            issues.append(dict(path=name, reason="unreadable_manifest"))
            continue
        digest = hashlib.sha256(raw).hexdigest()
        row = dict(model_key=key, selected_weight_bytes=None, inventory_status="complete",
                   selected_file_count=0, all_safetensors=False,
                   manifest_path=name, manifest_sha256=digest)
        if digest != item.get("sha256"):
            row["inventory_status"] = "manifest_hash_mismatch"
        else:
            files = manifest.get("files")
            if not isinstance(files, list) or not files:
                row["inventory_status"] = "empty_inventory"
            elif any(not isinstance(f, dict) or not isinstance(f.get("filename"), str)
                     for f in files):
                row["inventory_status"] = "invalid_file_inventory"
            else:
                names = [f["filename"] for f in files]
                row.update(selected_file_count=len(files),
                           all_safetensors=all(n.endswith(".safetensors") for n in names))
                if len(set(names)) != len(names):
                    row["inventory_status"] = "duplicate_filename"
                elif any(not n.endswith((".safetensors", ".gguf", ".bin", ".pt", ".pth", ".ckpt", ".onnx"))
                         for n in names):
                    row["inventory_status"] = "unrecognized_weight_file"
                elif any(type(f.get("size")) is not int or f["size"] <= 0 for f in files):
                    row["inventory_status"] = "unknown_size"
                else:
                    row["selected_weight_bytes"] = sum(f["size"] for f in files)
        if key in records and records[key] != row:
            row.update(inventory_status="conflicting_manifests", selected_weight_bytes=None)
        records[key] = row
    result = []
    for key in models.model_key:
        result.append(records.get(key, dict(model_key=key, selected_weight_bytes=None,
            inventory_status="missing_manifest", selected_file_count=0,
            all_safetensors=False, manifest_path=None, manifest_sha256=None)))
    frame = pd.DataFrame(result)
    frame["selected_weight_bytes"] = pd.array(frame.selected_weight_bytes, dtype="Int64")
    frame.attrs["manifest_issues"] = issues
    return frame


def select_budget(found, inventory, max_weight_bytes, limit):
    """Apply file budget before top-k, keeping the existing find ordering."""
    for value, name in [(max_weight_bytes, "max_weight_bytes"), (limit, "limit")]:
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    joined = found.merge(inventory, on="model_key", how="left", sort=False, validate="one_to_one")
    eligible = (joined.inventory_status.eq("complete") & joined.all_safetensors.eq(True)
                & joined.selected_weight_bytes.le(max_weight_bytes)
                & joined.is_adapter.eq(False)
                & joined.revision.str.fullmatch(r"[0-9a-fA-F]{40}")
                & ~joined.repo_id.str.startswith("lab/"))
    return joined.loc[eligible.fillna(False)].head(limit).copy()


def find_with_budget(zoo, models, inventory, max_weight_bytes, limit=10):
    found = zoo.find(license=["apache-2.0", "mit"], access="granted", gated=False,
                     weight_format="safetensors", limit=len(models))
    found = found.merge(models[["model_key", "revision", "is_adapter"]],
                        on="model_key", sort=False, validate="one_to_one")
    return select_budget(found, inventory, max_weight_bytes, limit)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zoo", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--max-weight-bytes", required=True, type=int)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)
    from zoo_keeper import Zoo
    try:
        with Zoo(args.zoo, data_root=args.data_root) as zoo:
            models = pd.read_parquet(zoo.path / "models.parquet")
            inventory = weight_inventory(models, zoo.metadata, args.data_root)
            found = find_with_budget(zoo, models, inventory, args.max_weight_bytes, args.limit)
        print(json.dumps(dict(
            budget_scope="Selected safetensors files only; not full package, RAM/VRAM or skill. Recorded access, not a live availability check.",
            max_weight_bytes=args.max_weight_bytes, catalog_entries=len(models),
            inventory_status=dict(Counter(inventory.inventory_status)),
            manifest_issues=inventory.attrs["manifest_issues"],
            matches=json.loads(found.to_json(orient="records"))), indent=2, allow_nan=False))
    except (OSError, ValueError) as error:
        parser.exit(2, f"budget query: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
