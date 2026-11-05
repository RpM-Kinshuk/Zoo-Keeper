"""Frozen development scorer and family-conditional whole-set conformal sets."""
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from .calibration import STRATA, _frame_hash, _hash

METHOD_VERSION = "lineage-conformal-v1"
UNKNOWN = "not_in_reference"


def _probabilities(probabilities):
    if any(not np.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
        raise ValueError("probabilities must be finite in [0,1]")
    if UNKNOWN not in probabilities:
        raise ValueError("not_in_reference probability required")


def whole_set_nonconformity(probabilities: dict[str, float], true_labels) -> float:
    """Max nonconformity covers all true labels, including multiple parents."""
    _probabilities(probabilities)
    labels = list(true_labels) or [UNKNOWN]
    if any(label not in probabilities for label in labels):
        raise ValueError("true labels missing from fixed candidate universe")
    return float(max(1 - probabilities[label] for label in labels))


def fit_development_scorer(development: pd.DataFrame, *, estimand: str,
                           candidate_universe_hash: str) -> dict:
    """Fit average true-label NLL only on explicit development rows.

    Rows hold score dictionaries of the frozen median cosine and true-label
    lists. Empty lists denote real held-out roots outside the reference pool.
    The caller supplies truth and the frozen candidate-universe hash.
    """
    required = {"split", "scores", "true_labels"}
    if not required.issubset(development.columns) or development.empty or not development.split.eq("development").all():
        raise ValueError("nonempty development-only rows required")
    if estimand not in STRATA or len(candidate_universe_hash) != 64 or any(c not in "0123456789abcdef" for c in candidate_universe_hash):
        raise ValueError("supported estimand and SHA256 candidate universe required")
    examples = []
    for row in development.itertuples():
        labels = sorted(row.scores)
        scores = np.asarray([row.scores[label] for label in labels], dtype=np.float64)
        if UNKNOWN in labels or not np.isfinite(scores).all():
            raise ValueError("finite candidate median cosine scores required")
        labels.append(UNKNOWN)
        truth = list(row.true_labels) or [UNKNOWN]
        if any(label not in labels for label in truth):
            raise ValueError("true labels absent from fixed candidate universe")
        examples.append((scores, [labels.index(label) for label in truth]))

    def objective(parameters):
        temperature = np.exp(parameters[0])
        losses = []
        for scores, truth in examples:
            logits = np.r_[scores / temperature, parameters[1]]
            losses.append(logsumexp(logits) - np.mean(logits[truth]))
        return float(np.mean(losses))

    fit = minimize(objective, [0., 0.], method="L-BFGS-B", bounds=[(-10, 10), (-30, 30)])
    if not fit.success or not np.isfinite(fit.fun):
        raise ValueError("development scorer optimization failed")
    model = dict(method_version=METHOD_VERSION, estimand=estimand,
                 candidate_universe_hash=candidate_universe_hash,
                 temperature=float(np.exp(fit.x[0])), not_in_reference_intercept=float(fit.x[1]),
                 development_hash=_frame_hash(development), frozen=True,
                 statistic=dict(statistic="median_cosine", sketch_k=64, tier="row_blocks", row_block=256),
                 objective="mean query average true-label negative log probability",
                 development_nll=float(fit.fun))
    return dict(model, protocol_hash=_hash(model))


def candidate_probabilities(scores: dict[str, float], *, model: dict) -> dict[str, float]:
    payload = {key:value for key,value in model.items() if key != "protocol_hash"}
    if not model.get("frozen") or _hash(payload) != model.get("protocol_hash"):
        raise ValueError("development scorer hash mismatch or not frozen")
    if UNKNOWN in scores or not np.isfinite(list(scores.values())).all():
        raise ValueError("finite candidate cosine scores required")
    labels = sorted(scores) + [UNKNOWN]
    temperature = model["temperature"]
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("positive finite temperature required")
    logits = np.r_[[scores[key] / temperature for key in labels[:-1]], model["not_in_reference_intercept"]]
    probabilities = np.exp(logits - logsumexp(logits))
    return dict(zip(labels, map(float, probabilities)))


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _artifact_hash(artifact):
    # The n+1 quantile can be +infinity: give nonfinite floats a stable tagged
    # representation for hashing without relaxing strict JSON hashing elsewhere.
    def canonical(value):
        if isinstance(value, float) and not math.isfinite(value):
            return {"nonfinite_float": str(value)}
        if isinstance(value, dict):
            return {key: canonical(item) for key,item in value.items()}
        if isinstance(value, (list, tuple)):
            return [canonical(item) for item in value]
        return value

    return _hash(canonical({key:value for key,value in artifact.items() if key != "artifact_hash"}))


def fit_conformal(calibration: pd.DataFrame, *, alpha: float = 0.1) -> dict:
    """Mondrian order statistic ceil((n+1)(1-alpha)), infinity beyond n.

    Rows must be query-level maxima from a frozen development scorer; labels
    may be lists. DataFrame.attrs must contain SHA256 scorer_hash and
    candidate_universe_hash for the frozen model and reference construction.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    if not all(_is_sha256(calibration.attrs.get(field)) for field in ("scorer_hash", "candidate_universe_hash")):
        raise ValueError("frozen scorer_hash and candidate_universe_hash SHA256 hashes required")
    if not {"family", "true_label", "nonconformity"}.issubset(calibration.columns):
        raise ValueError("family, true_label, nonconformity required")
    if "split" in calibration and not calibration.split.eq("calibration").all():
        raise ValueError("conformal fitting requires calibration-only rows")
    if calibration.family.isna().any() or calibration.true_label.isna().any():
        raise ValueError("family and true label cannot be missing")
    scores = pd.to_numeric(calibration.nonconformity, errors="coerce")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("finite nonconformities in [0,1] required")
    families = {}
    for family, group in calibration.groupby("family", sort=True):
        values = np.sort(group.nonconformity.to_numpy(dtype=np.float64))
        rank = math.ceil((len(values) + 1) * (1 - alpha))
        families[family] = dict(n=len(values), rank=rank,
            threshold=float(values[rank-1]) if rank <= len(values) else float("inf"))
    protocol = dict(alpha=alpha, method_version=METHOD_VERSION,
                    coverage_scope="whole_true_set_within_family",
                    nonconformity="max_1_minus_probability_over_whole_true_set",
                    assumptions="exchangeable calibration and query within each supported family; frozen scorer and candidate universe",
                    scorer_hash=calibration.attrs.get("scorer_hash"),
                    candidate_universe_hash=calibration.attrs.get("candidate_universe_hash"))
    artifact = dict(protocol, families=families, source_hash=_frame_hash(calibration),
                protocol_hash=_hash(protocol),
                root_groups=sorted(map(str, calibration.query_root.unique())) if "query_root" in calibration else [],
                seed_groups=sorted(map(str, calibration.seed.unique())) if "seed" in calibration else [],
                reporting_metrics=["whole_true_set_inclusion", "marginal_candidate_coverage"])
    return dict(artifact, artifact_hash=_artifact_hash(artifact))


def prediction_set(candidate_probabilities: dict[str, float], *, family: str,
                   calibration: dict, scorer_hash: str | None = None,
                   candidate_universe_hash: str | None = None) -> dict:
    """Apply only to the frozen scorer/universe; otherwise return a full set.

    The optional keywords preserve the original interface. Omitting them
    abstains for supported families; no provenance is inferred from scores.
    """
    if calibration.get("artifact_hash") != _artifact_hash(calibration):
        raise ValueError("conformal calibration artifact hash mismatch")
    probabilities = dict(candidate_probabilities)
    # Unsupported families always return the full augmented universe.
    if family not in calibration["families"]:
        if any(not np.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
            raise ValueError("probabilities must be finite in [0,1]")
        return dict(labels=sorted(set(probabilities) | {UNKNOWN}), coverage_scope="none",
                    status="uncalibrated_family", guarantee="transfer_unvalidated")
    _probabilities(probabilities)
    if (not _is_sha256(scorer_hash) or not _is_sha256(candidate_universe_hash)
        or scorer_hash != calibration["scorer_hash"]
        or candidate_universe_hash != calibration["candidate_universe_hash"]):
        return dict(labels=sorted(probabilities), coverage_scope="none",
                    status="uncalibrated_provenance", guarantee="none",
                    reason="missing_or_mismatched_scorer_or_candidate_universe_hash")
    threshold = calibration["families"][family]["threshold"]
    return dict(labels=sorted(label for label,p in probabilities.items() if 1-p <= threshold),
                coverage_scope=calibration["coverage_scope"], status="calibrated",
                guarantee="conditional_within_family_exchangeability")
