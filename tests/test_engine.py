"""Validation of the auxiliary files by the engine (without running the pipeline).

run_instancespace reads and validates annotations.json, degenerate_report.csv
and feature_info.csv before the build; an error there must show up in seconds,
not after PYTHIA. Needs the .venv-isa (imports instancespace).
"""

import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

engine = pytest.importorskip("isaspace.engine")

METADATA = (
    "instances,label,weight,feature_a,feature_b,feature_c,algo_x,algo_y\n"
    + "\n".join(f"i{k},{'ab'[k % 2]},{k},{k * 0.1},{(k * 7) % 5},{k % 3},{0.1 * (k % 9)},{0.9 - 0.1 * (k % 9)}"
                for k in range(30))
    + "\n"
)


@pytest.fixture
def inputs(tmp_path):
    folder = tmp_path / "input"
    folder.mkdir()
    (folder / "metadata.csv").write_text(METADATA)
    return folder


def test_without_auxiliary_files(inputs):
    present, types = engine._read_auxiliary(inputs / "metadata.csv")
    assert present == {} and types == {}


def test_valid_auxiliary_files(inputs):
    (inputs / "annotations.json").write_text(json.dumps({"label": "categorical", "weight": "identifier"}))
    (inputs / "degenerate_report.csv").write_text("feature,raw_variance,iqr,reason\nd,0.0,0.0,constant\n")
    (inputs / "feature_info.csv").write_text("feature,family\na,f1\nb,f1\nc,f2\nd,f2\n")
    present, types = engine._read_auxiliary(inputs / "metadata.csv")
    assert sorted(present) == ["annotations.json", "degenerate_report.csv", "feature_info.csv"]
    assert types == {"label": "categorical", "weight": "identifier"}


@pytest.mark.parametrize("content, error", [
    ({"label": "numeric"}, "non-numeric"),
    ({"weight": "text"}, "is not one of"),
    ({"feature_a": "numeric"}, "is not an annotation column"),
    ({"missing": "numeric"}, "is not an annotation column"),
    (["label"], "object"),
])
def test_invalid_annotations_json_fails_before_running(inputs, tmp_path, content, error):
    (inputs / "annotations.json").write_text(json.dumps(content))
    t0 = time.perf_counter()
    with pytest.raises(ValueError, match=error):
        engine.run_instancespace(inputs / "metadata.csv", tmp_path / "out")
    assert time.perf_counter() - t0 < 5
    assert not (tmp_path / "out").exists()


def test_integer_with_fractional_values_is_an_error(inputs):
    (inputs / "metadata.csv").write_text(METADATA.replace(",3,0.30", ",3.5,0.30"))
    (inputs / "annotations.json").write_text(json.dumps({"weight": "integer"}))
    with pytest.raises(ValueError, match="non-integer"):
        engine._read_auxiliary(inputs / "metadata.csv")


def test_report_without_required_columns_is_an_error(inputs):
    (inputs / "degenerate_report.csv").write_text("feature,reason\nd,constant\n")
    with pytest.raises(ValueError, match="missing columns"):
        engine._read_auxiliary(inputs / "metadata.csv")


def test_folder_from_another_tool_is_refused(inputs, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "coordinates.csv").write_text("Row,z_1,z_2\n1,0,0\n")
    with pytest.raises(ValueError, match="did not come from"):
        engine.run_instancespace(inputs / "metadata.csv", other)
