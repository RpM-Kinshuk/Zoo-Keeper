"""Command-line contracts over the offline query universe."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from evidence.sketch import ROW_BLOCK, SKETCH_SEED
from zoo.testing import SHA1, SHA2, SHA3, SHA8, SHA11
from zoo_keeper import Zoo
from zoo_keeper.cli import main


def test_cli_find(built_zoo, capsys):
    assert main(["find", "--zoo", str(built_zoo), "--text", "child", "--json"]) == 0
    assert any(r["model_key"].startswith("org/child") for r in json.loads(capsys.readouterr().out))


def test_cli_error_exit_codes(built_zoo, capsys):
    assert main(["info", "--zoo", str(built_zoo), "org/child"]) == 2           # ambiguous
    assert "org/child@" in capsys.readouterr().err
    assert main(["similar", "--zoo", str(built_zoo), "org/t0only@" + SHA8]) == 2


@pytest.mark.parametrize("options, expected", [
    (["--modality", "text", "--license", "mit", "apache-2.0", "--params_min", "1000000",
      "--params_max", "2000000", "--format", "safetensors", "--access", "granted",
      "--architecture", "llama", "--text", "child", "--limit", "1"], "org/child-vocab@" + SHA3),
    (["--decision", "--system_one"], "org/decider@" + SHA11),
    (["--text", "org/parent", "--skill", "math", "--min_score", ".8", "--include_unverified"],
     "org/parent@" + SHA1),
])
def test_find_options_select_models(built_zoo, capsys, options, expected):
    assert main(["find", "--zoo", str(built_zoo), "--json", *options]) == 0
    assert [row["model_key"] for row in json.loads(capsys.readouterr().out)] == [expected]


def test_filters_remain_literal_and_verified_by_default(built_zoo, capsys):
    assert main(["find", "--zoo", str(built_zoo), "--json", "--text", "'; DROP TABLE models; --"]) == 0
    assert json.loads(capsys.readouterr().out) == []
    assert main(["find", "--zoo", str(built_zoo), "--json", "--text", "org/parent",
                 "--skill", "math", "--min_score", ".8"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_defaults_use_zoo_data_for_zoo_and_evidence(built_zoo, universe, monkeypatch, capsys):
    builds = universe.root / "builds"
    builds.mkdir(exist_ok=True)
    (builds / "current").symlink_to(built_zoo, target_is_directory=True)
    monkeypatch.setenv("ZOO_DATA", str(universe.root))
    assert main(["diff", "org/child@" + SHA2, "org/parent@" + SHA1, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["relatedness"]["verdict"] == "uncalibrated"


def test_similar_and_family_options(built_zoo, capsys):
    common = ["--zoo", str(built_zoo), "--json"]
    model = "org/parent@" + SHA1
    assert main(["similar", model, *common, "-k", "0"]) == 0
    assert json.loads(capsys.readouterr().out) == []
    assert main(["similar", model, *common, "-k", "20"]) == 0
    collapsed = json.loads(capsys.readouterr().out)
    assert main(["similar", model, *common, "-k", "20", "--keep_duplicates"]) == 0
    assert len(json.loads(capsys.readouterr().out)) > len(collapsed)
    assert main(["family", model, *common, "--depth", "0"]) == 0
    family = json.loads(capsys.readouterr().out)
    assert family["ancestors"] == family["descendants"] == []
    assert family["license_chain"] == [[model, "apache-2.0"]]


def test_calibration_and_data_root_are_forwarded(built_zoo, universe, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ZOO_DATA", raising=False)
    calibration = tmp_path / "null.json"
    calibration.write_text(json.dumps(dict(
        statistic="median_cosine", sketch_k=32, row_block=ROW_BLOCK, sketch_seed=SKETCH_SEED,
        n_pairs=99, tier="t1", null_quantiles=[-.1, .1], related_threshold=.8)))
    common = ["--zoo", str(built_zoo), "--data_root", str(universe.root),
              "--null", str(calibration), "--json"]
    assert main(["diff", "org/child@" + SHA2, "org/parent@" + SHA1, *common]) == 0
    assert json.loads(capsys.readouterr().out)["relatedness"]["verdict"] == "related"
    assert main(["check", "org/child@" + SHA2, *common]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "consistent"


def test_check_inconsistent_returns_one(built_zoo, capsys, monkeypatch):
    monkeypatch.setattr(Zoo, "check", lambda self, model, null=None: {"verdict": "inconsistent"})
    assert main(["check", "--zoo", str(built_zoo), "org/parent", "--json"]) == 1
    output = capsys.readouterr()
    assert json.loads(output.out) == {"verdict": "inconsistent"}
    assert output.err == ""


def test_json_normalizes_missing_dataframe_values(built_zoo, capsys, monkeypatch):
    monkeypatch.setattr(Zoo, "find", lambda self, **kwargs: pd.DataFrame([
        {"model_key": "missing", "params": float("nan"), "license": pd.NA}]))
    assert main(["find", "--zoo", str(built_zoo), "--json"]) == 0
    output = capsys.readouterr()
    def reject_constant(value):
        pytest.fail(f"Nonstandard JSON constant: {value}")
    assert json.loads(output.out, parse_constant=reject_constant) == [
        {"model_key": "missing", "params": None, "license": None}]
    assert output.err == ""


@pytest.mark.parametrize("command", [
    ["info", "org/missing", "--json"],
    ["info", "org/child", "--json"],
    ["similar", "org/t0only@" + SHA8, "--json"],
    ["find", "--limit", "-1", "--json"],
    ["family", "org/parent", "--depth", "-1"],
    ["diff", "org/child@" + SHA2, "org/parent@" + SHA1, "--null", "/missing/null.json"],
])
def test_user_errors_have_only_stderr(built_zoo, universe, monkeypatch, capsys, command):
    monkeypatch.setenv("ZOO_DATA", str(universe.root))
    assert main([*command, "--zoo", str(built_zoo)]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err and "Traceback" not in output.err


def test_missing_zoo_is_clean_error(tmp_path, capsys):
    assert main(["find", "--zoo", str(tmp_path / "missing"), "--json"]) == 2
    output = capsys.readouterr()
    assert output.out == "" and output.err


def test_programming_errors_propagate(built_zoo, monkeypatch):
    def broken(self, **kwargs):
        raise RuntimeError("implementation defect")
    monkeypatch.setattr(Zoo, "find", broken)
    with pytest.raises(RuntimeError, match="implementation defect"):
        main(["find", "--zoo", str(built_zoo)])


def test_readable_output_and_module_entrypoint(built_zoo, capsys):
    assert main(["find", "--zoo", str(built_zoo), "--text", "child", "--limit", "1"]) == 0
    assert "model_key" in capsys.readouterr().out
    result = subprocess.run([sys.executable, "-m", "zoo_keeper.cli", "info", "org/parent",
                             "--zoo", str(built_zoo), "--json"],
                            cwd=Path(__file__).resolve().parents[1],
                            env=os.environ.copy(), capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["identity"]["model_key"] == "org/parent@" + SHA1
    assert result.stderr == ""
