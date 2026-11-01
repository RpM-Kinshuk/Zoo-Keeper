from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lineage.calibration import (apply_null, exact_shortlist_permutation, fit_null,
                                 query_control)


SETTINGS = dict(statistic="median_cosine", sketch_k=64, tier="t1", row_block=256,
                source_hashes={"upstream_x9": "a" * 64},
                exchangeability_design="independent query-root/seed blocks, fixed candidate labels")


def null_frame(n=12):
    return pd.DataFrame([dict(query_key=f"q{i}", candidate_key=f"c{i}",
        query_family="f", candidate_family="f", query_arch="cfg:a", candidate_arch="cfg:a",
        negative_kind=kind, certified=True, truth_known=True, score=i / (n + 1),
        split="calibration", query_root=f"qr{i}", candidate_root=f"cr{i}" if kind == "independent" else f"qr{i}", seed=i)
        for i in range(n) for kind in ("independent", "sibling", "non_immediate_ancestor")])


@pytest.fixture
def calibration_fixture():
    artifact = fit_null(null_frame(), estimand="parent", settings=SETTINGS)
    return SimpleNamespace(artifact=artifact, apply=lambda **kwargs: apply_null(
        .95, artifact, family=kwargs.pop("family", "f"), settings={**SETTINGS, "architecture":"cfg:a", **kwargs}))


def test_exact_permutation_uses_ties_and_finite_floor():
    out = exact_shortlist_permutation(0.9, np.asarray([0.1,0.9,0.2]))
    assert out["p_value"] == 0.5
    assert exact_shortlist_permutation(1.0, np.arange(9)/10)["p_value"] == 0.1
    assert out["validity"] == "exchangeable"
    # Exhaust all four candidate-label assignments: ranks are conservative.
    values = np.asarray([.1, .9, .2, .9])
    ps = [exact_shortlist_permutation(x, np.delete(values, i))["p_value"]
          for i, x in enumerate(values)]
    for alpha in (.25, .5, .75, 1):
        assert np.mean(np.asarray(ps) <= alpha) <= alpha


def test_full_pool_multiplicity_is_not_shortlist_size():
    p = np.asarray([1e-8, 0.8])
    assert query_control(p, alpha=0.05, pool_size=2, validity="exchangeable").tolist() == [True,False]
    assert query_control(p, alpha=0.05, pool_size=5000, validity="exchangeable").tolist() == [False,False]


def test_settings_and_family_transfer_do_not_fake_calibration(calibration_fixture):
    assert calibration_fixture.apply(tier="t2")["status"] == "uncalibrated"
    assert calibration_fixture.apply(row_block=128)["status"] == "uncalibrated"
    assert calibration_fixture.apply(sketch_k=32)["status"] == "uncalibrated"
    assert calibration_fixture.apply(family="unseen")["guarantee"] == "transfer_unvalidated"


@pytest.mark.parametrize("validity", ["unvalidated", "approximate_gpd", "unknown"])
def test_unvalidated_or_gpd_p_values_rejected(validity):
    with pytest.raises(ValueError):
        query_control(np.asarray([1e-8]), pool_size=1, validity=validity)


def test_sibling_not_same_root_negative():
    out = fit_null(null_frame(), estimand="same_root", settings=SETTINGS)
    assert out["n_pairs"] == 12
    assert set(out["strata"]["f"]) == {"independent"}


def test_gpd_thin_tail_falls_back(calibration_fixture):
    for stratum in calibration_fixture.artifact["strata"]["f"].values():
        assert stratum["gpd"]["status"] == "insufficient_tail"


def test_stouffer_never_used():
    frame = null_frame()
    out = fit_null(pd.concat([frame, frame.assign(view="duplicate")]), estimand="parent", settings=SETTINGS)
    assert out["n_pairs"] == len(frame)
    assert out["cluster_counts"]["query_root_seed"] == 12


def test_unknown_truth_excluded():
    frame = null_frame()
    frame.loc[0, "truth_known"] = False
    frame.loc[1, "certified"] = False
    frame.loc[2, "split"] = "test"
    frame.loc[3, "candidate_arch"] = "other"
    out = fit_null(frame, estimand="parent", settings=SETTINGS)
    assert out["n_pairs"] == len(frame) - 4
    assert sum(out["excluded"].values()) == 4


def test_retrieval_uses_full_pool_denominator():
    assert not query_control(np.asarray([1e-8]), pool_size=5000, validity="exchangeable").any()


def test_missing_stratum_and_exchangeability_abstain():
    frame = null_frame()
    out = fit_null(frame[frame.negative_kind == "independent"], estimand="parent", settings=SETTINGS)
    assert apply_null(.99, out, family="f", settings=SETTINGS)["status"] == "uncalibrated"
    settings = {k:v for k,v in SETTINGS.items() if k != "exchangeability_design"}
    out = fit_null(frame, estimand="parent", settings=settings)
    assert apply_null(.99, out, family="f", settings=settings)["validity"] == "unvalidated"


def test_hashes_are_frozen_and_conflicting_pairs_rejected(calibration_fixture):
    out = calibration_fixture.artifact
    assert len(out["protocol_hash"]) == len(out["source_hash"]) == 64
    assert out["source_hashes"] == SETTINGS["source_hashes"]
    frame = null_frame()
    with pytest.raises(ValueError, match="conflicting"):
        fit_null(pd.concat([frame, frame.iloc[:1].assign(score=.99)]), estimand="parent", settings=SETTINGS)


@pytest.mark.parametrize("p", [[0], [-1], [np.nan], [np.inf], [1.1]])
def test_invalid_p_values_rejected(p):
    with pytest.raises(ValueError):
        query_control(np.asarray(p), pool_size=1, validity="exchangeable")


def test_gpd_diagnostics_are_approximate():
    frame = null_frame(600)
    frame["score"] = np.tile(np.linspace(0, .99, 600), 3)  # deterministic, not real calibration
    out = fit_null(frame, estimand="parent", settings=SETTINGS)
    gpd = out["strata"]["f"]["independent"]["gpd"]
    assert gpd["n_exceedances"] >= 50
    assert gpd["validity"] == "approximate_gpd"
    assert "heldout_diagnostics" in gpd
    assert "bootstrap_intervals" in gpd


def test_shared_candidate_roots_are_one_uncertainty_block():
    frame = null_frame()
    frame["candidate_root"] = "shared-root"
    frame["seed"] = 7
    out = fit_null(frame, estimand="parent", settings=SETTINGS)
    assert out["cluster_counts"]["query_root_seed"] == 1
    assert out["strata"]["f"]["independent"]["n_clusters"] == 1


def test_artifact_mutation_is_detected(calibration_fixture):
    calibration_fixture.artifact["strata"]["f"]["independent"]["samples"][0] = -.8
    with pytest.raises(ValueError, match="hash"):
        calibration_fixture.apply()


def test_architectures_cannot_be_pooled_in_one_family_stratum():
    frame = null_frame()
    frame.loc[:2, ["query_arch", "candidate_arch"]] = "cfg:other"
    with pytest.raises(ValueError, match="architecture"):
        fit_null(frame, estimand="parent", settings=SETTINGS)


def test_source_hashes_must_be_content_hashes():
    with pytest.raises(ValueError, match="source_hashes"):
        fit_null(null_frame(), estimand="parent", settings={**SETTINGS, "source_hashes":{"upstream_x9":"mutable-path"}})


@pytest.mark.parametrize("architecture", [None, "cfg:b"])
def test_null_application_requires_observed_architecture(calibration_fixture, architecture):
    settings = dict(SETTINGS)
    if architecture is not None:
        settings["architecture"] = architecture
    out = apply_null(.95, calibration_fixture.artifact, family="f", settings=settings)
    assert out["status"] == "uncalibrated"
    assert out["p_value"] is None
    assert out["validity"] == "unvalidated"
    assert out["reason"] == "architecture_mismatch"


def test_null_records_architecture_and_matching_application(calibration_fixture):
    assert calibration_fixture.artifact["architectures"] == {"f":"cfg:a"}
    assert calibration_fixture.apply()["status"] == "calibrated"
