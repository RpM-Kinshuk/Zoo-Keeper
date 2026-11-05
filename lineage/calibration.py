"""Calibration of the frozen median-cosine statistic; no data acquisition."""
from copy import deepcopy
import hashlib
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import genpareto

from evidence.compatibility import canonical_mode

METHOD_VERSION = "lineage-calibration-v1"
STATISTIC = dict(statistic="median_cosine", sketch_k=64, tier="row_blocks", row_block=256)
STRATA = {"same_root": ("independent",), "ancestor": ("independent", "sibling"),
          "parent": ("independent", "sibling", "non_immediate_ancestor")}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _frame_hash(frame):
    rows = frame.to_json(orient="records", double_precision=15)
    return _hash(sorted(json.loads(rows), key=lambda row: json.dumps(row, sort_keys=True)))


def exact_shortlist_permutation(observed: float, null_scores: np.ndarray) -> dict:
    """Rank test conditional on exchangeability of the observed candidate label.

    Calling this mathematical primitive asserts that assumption. It does not
    certify a retrieval or sampling design; use apply_null to preserve validity.
    """
    null = np.asarray(null_scores, dtype=np.float64)
    if null.ndim != 1 or not np.isfinite(null).all() or not math.isfinite(observed):
        raise ValueError("finite observed score and one-dimensional null required")
    return dict(p_value=float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1)),
                n_permutations=len(null) + 1, validity="exchangeable",
                assumption="observed and null candidate labels are conditionally exchangeable",
                tie_rule="greater_or_equal")


def query_control(p_values: np.ndarray, *, alpha: float = 0.05, pool_size: int,
                  validity: str = "unvalidated") -> np.ndarray:
    """e-BH, counting unretrieved candidates as zero e-values in the full pool."""
    if validity != "exchangeable":
        raise ValueError("query control requires validated exchangeable superuniform p-values")
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p <= 0) | (p > 1)):
        raise ValueError("p-values must be finite and in (0,1]")
    if not 0 < alpha < 1 or isinstance(pool_size, bool) or not isinstance(pool_size, (int, np.integer)) or pool_size < len(p) or pool_size < 1:
        raise ValueError("alpha in (0,1) and full positive integer pool_size required")
    e = .5 / np.sqrt(p)
    order = np.argsort(-e, kind="stable")
    passing = e[order] >= pool_size / (alpha * np.arange(1, len(p) + 1))
    result = np.zeros(len(p), dtype=bool)
    if passing.any():
        result[order[:np.flatnonzero(passing)[-1] + 1]] = True
    return result


def _gpd(samples, clusters, quantile):
    values = np.asarray(samples, dtype=np.float64)
    threshold = float(np.quantile(values, quantile))
    indices = np.flatnonzero(values > threshold)
    result = dict(status="insufficient_tail", validity="approximate_gpd", threshold=threshold,
                  n_exceedances=len(indices), empirical_floor=1 / (len(values) + 1),
                  heldout_diagnostics=None, bootstrap_intervals=None)
    if len(indices) < 50:
        return result
    # Frozen deterministic split; no held-out diagnostic participates in fitting.
    training_count = max(50, int(.8 * len(indices)))
    train_indices, held_indices = indices[:training_count], indices[training_count:]
    excess = values[train_indices] - threshold
    try:
        shape, loc, scale = genpareto.fit(excess, floc=0)
        if not np.isfinite([shape, scale]).all() or scale <= 0 or np.any(1 + shape * excess / scale <= 0):
            raise ValueError("invalid GPD support")
        result.update(status="fitted_approximate", shape=float(shape), loc=float(loc), scale=float(scale))
        held = values[held_indices] - threshold
        if len(held):
            probabilities = (np.arange(len(held)) + .5) / len(held)
            theoretical = genpareto.ppf(probabilities, shape, loc=0, scale=scale)
            result["heldout_diagnostics"] = dict(n=len(held),
                qq_rmse=float(np.sqrt(np.mean((np.sort(held) - theoretical)**2))),
                coverage_90=float(np.mean(held <= genpareto.ppf(.9, shape, loc=0, scale=scale))))
        else:
            result["heldout_diagnostics"] = dict(status="insufficient_heldout_tail", n=0)
        # Bootstrap root/seed blocks rather than treating dependent views as draws.
        blocks = sorted(set(clusters))
        block_samples = {block: values[np.asarray(clusters) == block] for block in blocks}
        rng = np.random.default_rng(20261028)
        fits = []
        for _ in range(20):
            boot = np.concatenate([block_samples[blocks[i]] for i in rng.integers(len(blocks), size=len(blocks))])
            boot = boot[boot > threshold] - threshold
            if len(boot) < 50:
                continue
            s, _, t = genpareto.fit(boot, floc=0)
            if np.isfinite([s, t]).all() and t > 0 and np.all(1 + s * boot / t > 0):
                fits.append([s, t])
        result["bootstrap_intervals"] = dict(block="query_root_seed", repetitions=20,
            successful=len(fits), intervals=np.quantile(fits, [.025,.975], axis=0).tolist() if fits else None)
    except (ValueError, FloatingPointError):
        result["status"] = "invalid_fit"
    return result


def fit_null(scores: pd.DataFrame, *, estimand: str, settings: dict,
             tail_quantile: float = 0.9) -> dict:
    """Fit certified negatives separately, retaining provenance and exclusions.

    Extra certification/root/architecture fields are deliberately mandatory:
    negative_kind or different declared families alone never certifies truth.
    exchangeability_design is an explicit caller attestation, not inferred from n.
    """
    if estimand not in STRATA or not 0 < tail_quantile < 1:
        raise ValueError("supported estimand and tail quantile in (0,1) required")
    if any((canonical_mode(settings.get(k)) if k == "tier" else settings.get(k)) != v
           for k, v in STATISTIC.items()):
        raise ValueError("calibration requires median_cosine at k64/row_blocks/row256")
    source_hashes = settings.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes or any(
        not isinstance(value, str) or len(value) != 64 or
        any(c not in "0123456789abcdef" for c in value) for value in source_hashes.values()
    ):
        raise ValueError("immutable SHA256 source_hashes required")
    required = {"query_key", "candidate_key", "query_family", "candidate_family", "negative_kind",
                "score", "split", "certified", "truth_known", "query_arch", "candidate_arch",
                "query_root", "candidate_root", "seed"}
    if not required.issubset(scores.columns):
        raise ValueError(f"missing certification/provenance columns: {sorted(required - set(scores.columns))}")
    frame = scores.copy(deep=True)
    excluded = {}
    rules = [("not_calibration", frame.split != "calibration"),
             ("unknown_truth", ~frame.truth_known.eq(True)),
             ("uncertified", ~frame.certified.eq(True)),
             ("architecture_mismatch", frame.query_arch.isna() | frame.candidate_arch.isna() | (frame.query_arch != frame.candidate_arch)),
             ("wrong_stratum", ~frame.negative_kind.isin(STRATA[estimand])),
             ("family_mismatch", frame.query_family.isna() | (frame.query_family != frame.candidate_family)),
             ("missing_root_seed", frame[["query_root", "candidate_root", "seed"]].isna().any(axis=1)),
             ("invalid_score", ~np.isfinite(pd.to_numeric(frame.score, errors="coerce")))]
    keep = np.ones(len(frame), dtype=bool)
    for reason, mask in rules:
        mask = np.asarray(mask)
        excluded[reason] = int(np.sum(keep & mask))
        keep &= ~mask
    frame = frame.loc[keep].copy()
    if (frame.groupby("query_family").query_arch.nunique() > 1).any():
        raise ValueError("multiple architecture signatures in one family require separate calibration artifacts")
    keys = ["query_key", "candidate_key", "negative_kind"]
    # Repeated views must not manufacture extra pair observations.
    for _, group in frame.groupby(keys, sort=True):
        if group[list(required)].drop_duplicates().shape[0] > 1:
            raise ValueError("conflicting duplicate pair records")
    excluded["duplicate_pair"] = int(frame.duplicated(keys).sum())
    frame = frame.drop_duplicates(keys).sort_values(keys)
    # Connect every shared root/seed, including a candidate reused by queries.
    # Pair counts and views do not increase the number of independent blocks.
    links = {}

    def component(node):
        links.setdefault(node, node)
        while links[node] != node:
            links[node] = links[links[node]]
            node = links[node]
        return node

    for row in frame.itertuples():
        left = component((str(row.query_root), str(row.seed)))
        right = component((str(row.candidate_root), str(row.seed)))
        links[max(left, right)] = min(left, right)
    frame["cluster"] = [json.dumps(component((str(r), str(seed))))
                        for r,seed in zip(frame.query_root, frame.seed)]
    strata = {}
    for (family, kind), group in frame.groupby(["query_family", "negative_kind"], sort=True):
        # A fixed representative per root/seed avoids a fictitious pair-count floor.
        representatives = group.drop_duplicates("cluster")
        samples = representatives.score.astype(float).tolist()
        strata.setdefault(family, {})[kind] = dict(samples=samples, n_pairs=len(group),
            n_clusters=len(representatives), clusters=representatives.cluster.tolist(),
            gpd=_gpd(samples, representatives.cluster.tolist(), tail_quantile))
    protocol = dict(settings=deepcopy(settings), estimand=estimand, tail_quantile=tail_quantile,
                    method_version=METHOD_VERSION, representative_rule="first sorted pair per query-root/seed")
    artifact = dict(**protocol, source_hash=_frame_hash(scores), protocol_hash=_hash(protocol),
        source_hashes=deepcopy(settings["source_hashes"]), strata=strata, required_strata=list(STRATA[estimand]),
        n_pairs=len(frame), excluded=excluded,
        architectures=frame.groupby("query_family").query_arch.first().to_dict(),
        root_groups=sorted(set(frame.query_root) | set(frame.candidate_root)),
        seed_groups=sorted(map(str, frame.seed.unique())),
        cluster_counts=dict(query_root_seed=frame.cluster.nunique()),
        validity="exchangeable" if settings.get("exchangeability_design") else "unvalidated",
        assumptions=["conditional exchangeability under recorded caller-supplied design",
                     "superuniform marginal p-values; e-BH permits arbitrary candidate dependence"],
        uncertainty_unit="connected_query_candidate_root_seed")
    return dict(artifact, artifact_hash=_hash(artifact))


def apply_null(observed: float, calibration: dict, *, family: str, settings: dict) -> dict:
    """Conservative composite p-value, with explicit mismatch/transfer abstention."""
    if calibration.get("artifact_hash") != _hash({k:v for k,v in calibration.items() if k != "artifact_hash"}):
        raise ValueError("calibration artifact hash mismatch")
    abstain = dict(status="uncalibrated", p_value=None, validity="unvalidated", guarantee="none")
    if any((canonical_mode(settings.get(k)) != canonical_mode(value) if k == "tier"
            else settings.get(k) != value) for k, value in calibration["settings"].items()):
        return dict(abstain, reason="settings_mismatch")
    if family not in calibration["strata"]:
        return dict(abstain, reason="unseen_family", guarantee="transfer_unvalidated")
    if settings.get("architecture") != calibration["architectures"].get(family):
        return dict(abstain, reason="architecture_mismatch")
    available = calibration["strata"][family]
    if any(kind not in available for kind in calibration["required_strata"]):
        return dict(abstain, reason="missing_required_stratum", available_strata=sorted(available))
    if calibration["validity"] != "exchangeable":
        return dict(abstain, reason="exchangeability_unvalidated")
    ps = {kind: exact_shortlist_permutation(observed, np.asarray(available[kind]["samples"]))["p_value"]
          for kind in calibration["required_strata"]}
    return dict(status="calibrated", p_value=max(ps.values()), stratum_p_values=ps,
                validity="exchangeable", guarantee="conditional_exchangeability", available_strata=sorted(available))
