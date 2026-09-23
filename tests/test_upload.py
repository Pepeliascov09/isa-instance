"""Fast tests of the upload validation (isaspace.ui.upload), of the engine
subprocess launcher (isaspace.ui.runner) and of the New instance space block.

Usage, from the root, in the .venv-isa:
    .venv-isa/bin/python -m pytest tests/test_upload.py
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from isaspace.ui import runner
from isaspace.ui.upload import (
    GOOD_MIN, MEASURED_TIMES, TIMING_ALGOS, degenerate_algorithms, estimated_time, good_fraction,
    mean_ranking, prelim_rows, validate_annotations, validate_feature_info, validate_metadata,
)

ROOT = Path(__file__).resolve().parents[1]
IS_DIR = ROOT / "resultados" / "is"
DATASETS = ["iris", "diabetes", "blood-transfusion-service-center", "hill-valley"]


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _valid(n=30, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"instances": [f"i{k}" for k in range(n)]})
    for f in ("a", "b", "c"):
        df[f"feature_{f}"] = rng.normal(size=n)
    for a in ("x", "y"):
        df[f"algo_{a}"] = rng.uniform(size=n)
    df["group"] = rng.choice(["p", "q"], size=n)
    return df


# ------------------------------------------------------------------ metadata
def test_valid_metadata():
    v = validate_metadata(_csv(_valid()))
    assert v.ok and not v.errors and not v.warnings
    assert (v.n, v.features, v.algos, v.annotations) == (
        30, ["feature_a", "feature_b", "feature_c"], ["algo_x", "algo_y"], ["group"])


def test_without_instances_column():
    v = validate_metadata(_csv(_valid().rename(columns={"instances": "id"})))
    assert not v.ok
    assert any("Missing the **instances** column" in e for e in v.errors)


def test_repeated_label():
    df = _valid()
    df.loc[5, "instances"] = df.loc[3, "instances"]
    df.loc[9, "instances"] = df.loc[3, "instances"]
    v = validate_metadata(_csv(df))
    assert not v.ok
    [error] = [e for e in v.errors if "Repeated labels" in e]
    assert "'i3'" in error and "(1 distinct)" in error


def test_empty_label():
    df = _valid()
    df.loc[4, "instances"] = ""
    v = validate_metadata(_csv(df))
    assert not v.ok
    assert any("without a label" in e and "6" in e for e in v.errors)   # line 6 of the file


def test_non_numeric_column():
    df = _valid()
    df[["feature_b", "algo_y"]] = df[["feature_b", "algo_y"]].astype(object)
    df.loc[2, "feature_b"] = "abc"
    df.loc[7, "algo_y"] = "unknown"          # not a pandas NA marker such as "n/a"
    v = validate_metadata(_csv(df))
    assert not v.ok
    assert any("**feature_b** has non-numeric values: 'abc'" in e for e in v.errors)
    assert any("**algo_y** has non-numeric values: 'unknown'" in e for e in v.errors)


def test_only_two_features():
    v = validate_metadata(_csv(_valid().drop(columns="feature_c")))
    assert not v.ok
    assert any("At least 3 **feature_" in e and "has 2" in e for e in v.errors)


def test_only_one_algorithm():
    v = validate_metadata(_csv(_valid().drop(columns="algo_y")))
    assert not v.ok
    assert any("At least 2 **algo_" in e and "has 1" in e for e in v.errors)


def test_nan_reported_per_column_without_blocking():
    df = _valid()
    df.loc[[1, 2, 3], "feature_a"] = np.nan
    df.loc[[4], "algo_x"] = np.nan
    v = validate_metadata(_csv(df))
    assert v.ok
    assert v.nan == {"feature_a": 3, "algo_x": 1}
    assert any("feature_a: 3" in w and "algo_x: 1" in w for w in v.warnings)


def test_algorithm_all_nan_blocks():
    df = _valid()
    df["algo_y"] = np.nan
    v = validate_metadata(_csv(df))
    assert not v.ok and any("Algorithms without any value" in e for e in v.errors)


def test_empty_and_binary_file():
    assert validate_metadata(b"").errors == ["The file is empty."]
    assert "UTF-8" in validate_metadata(b"\xff\xfe\x00instances").errors[0]


def test_columns_with_the_same_name():
    v = validate_metadata(b"instances,feature_a,feature_a,feature_b,algo_x,algo_y\n1,1,1,1,1,1\n")
    assert any("same name" in e for e in v.errors)


@pytest.mark.parametrize("dataset", DATASETS)
def test_project_datasets_metadata_is_valid(dataset):
    v = validate_metadata((IS_DIR / dataset / "metadata.csv").read_bytes())
    assert v.ok, v.errors


# ----------------------------------------------------------- auxiliary files
def test_invalid_and_valid_annotations():
    v = validate_metadata(_csv(_valid()))
    _, errors = validate_annotations(b"{not json", v)
    assert errors and "not valid JSON" in errors[0]
    _, errors = validate_annotations(json.dumps({"group": "text"}).encode(), v)
    assert errors and "is not one of" in errors[0]
    _, errors = validate_annotations(json.dumps({"feature_a": "numeric"}).encode(), v)
    assert errors and "is not an annotation column" in errors[0]
    types, errors = validate_annotations(json.dumps({"group": "categorical"}).encode(), v)
    assert (types, errors) == ({"group": "categorical"}, [])


def test_feature_info_without_columns():
    v = validate_metadata(_csv(_valid()))
    assert validate_feature_info(b"feature,familia\na,x\n", v) == [
        "feature_info.csv is missing the columns family."]
    assert validate_feature_info(b"feature,family\na,x\n", v) == []


# ------------------------------------------------------------- PRELIM rule
@pytest.mark.parametrize("higher", [True, False])
@pytest.mark.parametrize("absolute", [True, False])
def test_good_fraction_equals_instancespace(higher, absolute):
    """Same rule as instancespace's compute_binary_performance, with NaN,
    zeros and ties for the best."""
    prelim = pytest.importorskip("instancespace.stages.prelim")
    from instancespace.data.options import GeneralOptions, PerformanceOptions

    rng = np.random.default_rng(1)
    y = rng.uniform(0, 1, size=(300, 4))
    y[rng.random(y.shape) < 0.05] = np.nan
    y[rng.random(y.shape) < 0.05] = 0.0
    y[:10, 1] = y[:10, 0]
    eps = 0.3 if absolute else 0.15
    df = pd.DataFrame(y, columns=[f"algo_{k}" for k in "abcd"])
    expected = prelim.compute_binary_performance(
        y, PerformanceOptions(max_perf=higher, abs_perf=absolute, epsilon=eps,
                              beta_threshold=0.55),
        GeneralOptions.default()).y_bin.mean(axis=0)
    np.testing.assert_array_equal(good_fraction(df, higher, absolute, eps).to_numpy(), expected)


@pytest.mark.parametrize("dataset", DATASETS)
def test_good_fraction_equals_the_written_algorithm_bin(dataset):
    folder = IS_DIR / dataset
    v = validate_metadata((folder / "metadata.csv").read_bytes())
    perf = json.loads((folder / "run_options.json").read_text())["perf"]
    fr = good_fraction(prelim_rows(v), perf["max_perf"], perf["abs_perf"], perf["epsilon"])
    written = pd.read_csv(folder / "algorithm_bin.csv", index_col=0).mean()
    written.index = [c.removeprefix("algo_") for c in written.index]
    pd.testing.assert_series_equal(fr, written.reindex(fr.index), check_names=False)


def test_degenerate_threshold_does_not_detect_the_direction():
    """With an absolute threshold, flipping the direction turns f into 1 - f
    (except for ties at epsilon): the <5%/>95% check fires in both directions
    on iris, and in neither on diabetes. It detects a degenerate epsilon, not a
    wrong direction."""
    for name, fires in (("iris", True), ("diabetes", False)):
        v = validate_metadata((IS_DIR / name / "metadata.csv").read_bytes())
        rows = prelim_rows(v)
        right = good_fraction(rows, True, True, 0.5)
        flipped = good_fraction(rows, False, True, 0.5)
        at_eps = (rows == 0.5).mean().to_numpy()          # good in both directions
        np.testing.assert_allclose(right + flipped, 1.0 + at_eps)
        assert bool(degenerate_algorithms(right)) is fires
        assert set(degenerate_algorithms(flipped)) == set(degenerate_algorithms(right))
    v = validate_metadata((IS_DIR / "iris" / "metadata.csv").read_bytes())
    assert (good_fraction(prelim_rows(v), False, True, 0.5) < GOOD_MIN).any()


# ---------------------------------------------------------------- ranking
def test_mean_ranking_flips_with_the_direction():
    v = validate_metadata((IS_DIR / "iris" / "metadata.csv").read_bytes())
    higher = mean_ranking(prelim_rows(v), True)
    lower = mean_ranking(prelim_rows(v), False)
    assert list(lower.index) == list(higher.index)[::-1]
    assert higher.is_monotonic_decreasing and lower.is_monotonic_increasing
    means = prelim_rows(v).mean()
    assert higher.index[0] == means.idxmax().removeprefix("algo_")


def test_mean_ranking_with_ties_is_an_exact_reverse():
    df = pd.DataFrame({"algo_a": [1.0, 1.0], "algo_b": [0.5, 0.5], "algo_c": [0.5, 0.5]})
    assert list(mean_ranking(df, True).index) == ["a", "b", "c"]
    assert list(mean_ranking(df, False).index) == ["c", "b", "a"]


# ------------------------------------------------------------------- time
def test_estimated_time_reproduces_the_measurements_and_grows():
    for n, (total, _) in MEASURED_TIMES.items():
        assert estimated_time(n, TIMING_ALGOS) == pytest.approx(total, rel=0.1)
    assert estimated_time(1000, 10) > estimated_time(1000, 6) > estimated_time(500, 6)
    assert estimated_time(0, 6) is None


# ----------------------------------------------------------------- runner
def test_safe_name_and_unique_folder(tmp_path):
    assert runner.safe_name("my metadata (v2).csv") == "my_metadata_v2_csv"
    assert runner.safe_name("///") == "metadata"
    now = datetime(2026, 9, 22, 12, 0, 0)
    a = runner.new_run_dir("x", tmp_path, now)
    b = runner.new_run_dir("x", tmp_path, now)
    assert a.name == "x_20260922-120000" and b.name == "x_20260922-120000_2"
    assert (a / runner.INPUT_DIR).is_dir()


def test_subprocess_error_returns_a_message_without_traceback(tmp_path):
    """Metadata instancespace refuses: the error comes in one line, the log stays on disk."""
    path = runner.new_run_dir("broken", tmp_path)
    meta = runner.write_inputs(
        path, b"instances,feature_a,feature_b,algo_x,algo_y\n1,1,2,0.1,0.2\n2,3,4,0.5,0.6\n")
    stages = []
    run = runner.launch(meta, path, {}, on_stage=lambda _r, s: stages.append(s))
    end = time.time() + 120
    while not run.finished and time.time() < end:
        time.sleep(0.2)
    assert run.finished and run.ok is False
    assert "three features" in run.error and "\n" not in run.error
    assert "Traceback" in run.log.read_text()        # the traceback only goes to the log
    assert not (path / "run_info.json").exists()


# --------------------------------------------------------------- UI block
def test_new_space_block_only_enables_after_the_choice(tmp_path):
    from isaspace.ui.new_space import UPSIDE_DOWN, NewSpacePanel

    block = NewSpacePanel(tmp_path, 260, on_done=lambda _p: None)
    assert block.w_run.disabled
    block.w_meta.filename = "my metadata.csv"
    block.w_meta.value = (IS_DIR / "diabetes" / "metadata.csv").read_bytes()
    assert block.w_name.value == "my_metadata"
    assert block.w_run.disabled and "the performance direction" in block.missing.object
    assert block.ranking.object == ""                       # no direction yet: no ranking
    block.w_direction.value = "max"
    assert UPSIDE_DOWN in block.ranking.object and "Best under this direction" in block.ranking.object
    block.w_threshold.value = "abs"
    assert block.w_run.disabled and "ε" in block.missing.object
    block.w_eps.value = 0.5
    assert not block.w_run.disabled and block.missing.object == ""
    assert "Preview" in block.preview.object and not block.preview_warning.objects   # 70-78%
    block.w_k.value, block.w_usesim.value = 4, True
    assert block.options() == {"perf": {"max_perf": True, "abs_perf": True, "epsilon": 0.5},
                               "sifted": {"k": 4}, "trace": {"use_sim": True}}
    block.w_ann.value = b'{"class": "text"}'                    # an invalid type blocks
    assert block.w_run.disabled
    assert "is not one of" in block.validation_msgs.objects[0].object
    block.w_ann.value = b'{"class": "categorical", "row_original": "identifier"}'
    assert not block.w_run.disabled
    assert any("Estimated time" in o.object for o in block.time.objects)
