import numpy as np
import pandas as pd
import pytest

from lineage.conformal import (fit_conformal, prediction_set, fit_development_scorer,
                               candidate_probabilities, whole_set_nonconformity)


PROVENANCE = dict(scorer_hash="a"*64, candidate_universe_hash="b"*64)


def with_provenance(frame):
    frame.attrs.update(PROVENANCE)
    return frame


@pytest.fixture
def conformal_fixture():
    frame = pd.DataFrame(dict(family=["f"]*10, true_label=["parent"]*10,
        nonconformity=[.05,.1,.15,.2,.25,.3,.35,.4,.45,.5]))
    return fit_conformal(with_provenance(frame), alpha=.1)


def test_mondrian_finite_sample_quantile(conformal_fixture):
    out = prediction_set({"a": .5, "b": .49, "not_in_reference": .5}, family="f", calibration=conformal_fixture, **PROVENANCE)
    assert out["labels"] == ["a", "not_in_reference"]
    assert conformal_fixture["families"]["f"]["threshold"] == .5
    assert out["coverage_scope"] == "whole_true_set_within_family"


def test_conformal_unseen_family_abstains(conformal_fixture):
    out = prediction_set({"a": .99}, family="new", calibration=conformal_fixture, **PROVENANCE)
    assert out["labels"] == ["a", "not_in_reference"]
    assert out["status"] == "uncalibrated_family"
    assert out["guarantee"] == "transfer_unvalidated"


def test_quantile_above_n_is_infinity():
    calibration = fit_conformal(with_provenance(pd.DataFrame([dict(family="f", true_label="a", nonconformity=.1)])), alpha=.1)
    assert np.isinf(calibration["families"]["f"]["threshold"])
    assert prediction_set({"a":0, "not_in_reference":1}, family="f", calibration=calibration, **PROVENANCE)["labels"] == ["a", "not_in_reference"]


def test_multi_parent_whole_set_nonconformity():
    assert whole_set_nonconformity({"a":.8, "b":.2, "not_in_reference":.1}, ["a", "b"]) == .8
    assert whole_set_nonconformity({"not_in_reference":.3}, []) == .7
    with pytest.raises(ValueError):
        whole_set_nonconformity({"a":.8, "not_in_reference":.2}, ["missing"])


def test_scorer_is_fitted_only_on_development_and_frozen():
    development = pd.DataFrame([dict(split="development", scores={"a":.8, "b":.2}, true_labels=["a"]),
        dict(split="development", scores={"a":.2, "b":.3}, true_labels=[])])
    model = fit_development_scorer(development, estimand="parent", candidate_universe_hash="b"*64)
    assert model["temperature"] > 0
    assert len(model["protocol_hash"]) == 64
    probabilities = candidate_probabilities({"a":.8, "b":.2}, model=model)
    assert set(probabilities) == {"a", "b", "not_in_reference"}
    assert sum(probabilities.values()) == pytest.approx(1)
    with pytest.raises(ValueError):
        fit_development_scorer(development.assign(split="test"), estimand="parent", candidate_universe_hash="b"*64)


def test_conformal_metadata_and_invalid_inputs(conformal_fixture):
    assert len(conformal_fixture["source_hash"]) == 64
    with pytest.raises(ValueError):
        prediction_set({"a":np.nan}, family="f", calibration=conformal_fixture, **PROVENANCE)
    with pytest.raises(ValueError):
        prediction_set({"a":.8}, family="f", calibration=conformal_fixture, **PROVENANCE)


@pytest.mark.parametrize("field,value", [("threshold", .99), ("n", 1000), ("rank", 1)])
def test_conformal_fitted_content_hash_detects_mutation(conformal_fixture, field, value):
    conformal_fixture["families"]["f"][field] = value
    with pytest.raises(ValueError, match="hash"):
        prediction_set({"a":.2, "not_in_reference":.8}, family="f", calibration=conformal_fixture, **PROVENANCE)


@pytest.mark.parametrize("attrs", [{}, {"scorer_hash":"a"*64},
    {"scorer_hash":"mutable-path", "candidate_universe_hash":"b"*64}])
def test_conformal_requires_frozen_fit_provenance(attrs):
    frame = pd.DataFrame([dict(family="f", true_label="a", nonconformity=.1)])
    frame.attrs.update(attrs)
    with pytest.raises(ValueError, match="hash"):
        fit_conformal(frame)


@pytest.mark.parametrize("provenance", [{}, {"scorer_hash":"c"*64, "candidate_universe_hash":"b"*64},
    {"scorer_hash":"a"*64, "candidate_universe_hash":"c"*64}])
def test_missing_or_mismatched_application_provenance_abstains(conformal_fixture, provenance):
    out = prediction_set({"a":.1, "not_in_reference":.9}, family="f", calibration=conformal_fixture, **provenance)
    assert out["status"] == "uncalibrated_provenance"
    assert out["labels"] == ["a", "not_in_reference"]
    assert out["guarantee"] == "none"
    assert out["coverage_scope"] == "none"
