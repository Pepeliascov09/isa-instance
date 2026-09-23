"""Standard orientation (pyispace's adjust_rotation convention) of the engine.

- Unit tests of isaspace.engine.orientation on synthetic data.
- The convention on the four examples of resultados/is (written with the
  orientation on): the centroid of the instances where most algorithms are
  bad is at 135 degrees, the top left.
- Invariance (marked slow): the engine runs again on the four examples with
  orient=False, and everything that is not geometry is compared with the
  rotated outputs: footprint areas, densities and purities, footprint
  membership of every instance, PYTHIA recommendations and metrics, PILOT r2.

Needs the .venv-isa (imports instancespace).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

engine = pytest.importorskip("isaspace.engine")
from isaspace.ui import runner  # noqa: E402
from isaspace.ui.loader_is import load_is_output  # noqa: E402

IS_DIR = ROOT / "resultados" / "is"
DATASETS = ["iris", "diabetes", "blood-transfusion-service-center", "hill-valley"]
GEOMETRIC = {"coordinates.csv", "coordinates_trace.csv", "bounds.csv", "bounds_prunned.csv",
             "projection_matrix.csv"}


def _majority_bad(r):
    good = r.instances[[f"algo_{a}_bin" for a in r.algos]].to_numpy()
    return (len(r.algos) - good.sum(axis=1)) >= len(r.algos) / 2


def _angle(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


# --------------------------------------------------------------------------- #
# unit tests
# --------------------------------------------------------------------------- #
def _synthetic(seed=0, n=300, n_algos=6):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2))
    z -= z.mean(axis=0)
    good = np.ones((n, n_algos), dtype=bool)
    hard = z[:, 0] + 0.5 * z[:, 1] > 1
    good[hard, :4] = False
    return z, good, hard


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bad_centroid_goes_to_135_degrees_with_a_proper_rotation(seed):
    z, good, hard = _synthetic(seed)
    rot, rec = engine.orientation(z, good)
    assert rec["applied"] and rec["n_instances_bad"] == int(hard.sum())
    assert np.allclose(rot @ rot.T, np.eye(2)) and np.isclose(np.linalg.det(rot), 1.0)
    assert _angle(rot @ z[hard].mean(axis=0)) == pytest.approx(135.0, abs=1e-9)
    assert _angle(rec["centroid_bad_after"]) == pytest.approx(135.0, abs=1e-9)
    zr = engine.rotate(z, rot)
    np.testing.assert_allclose(zr, z @ rot.T, atol=1e-14)
    # distances between points are kept: a rotation, not a reflection or a scaling
    np.testing.assert_allclose(np.linalg.norm(zr[1:] - zr[:-1], axis=1),
                               np.linalg.norm(z[1:] - z[:-1], axis=1), atol=1e-12)
    # applied again, the rotation is the identity, and the R2 does not change
    rot2, rec2 = engine.orientation(zr, good)
    np.testing.assert_allclose(rot2, np.eye(2), atol=1e-12)
    assert rec2["gradient_r2"] == pytest.approx(rec["gradient_r2"], abs=1e-12)


def test_a_tie_counts_as_bad_like_scipy_mode_in_pyispace():
    z = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    good = np.array([[1, 1, 1, 0, 0, 0],      # 3 x 3: mode 0 -> bad
                     [1, 1, 1, 1, 0, 0],
                     [1, 1, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1, 1]], dtype=bool)
    rot, rec = engine.orientation(z, good)
    assert rec["n_instances_bad"] == 1
    assert _angle(rot @ z[0]) == pytest.approx(135.0, abs=1e-9)


def test_no_rotation_without_bad_instances_or_with_all_bad():
    z = np.random.default_rng(3).normal(size=(50, 2))
    for good, reason in ((np.ones((50, 4), bool), "no instance"),
                         (np.zeros((50, 4), bool), "every instance")):
        rot, rec = engine.orientation(z, good)
        assert rot is None and not rec["applied"] and reason in rec["reason_not_applied"]


def test_weak_gradient_gets_a_warning():
    rng = np.random.default_rng(4)
    z = rng.normal(size=(400, 2))
    good = rng.random((400, 6)) > 0.5            # difficulty unrelated to z
    _, rec = engine.orientation(z, good)
    assert rec["weak_gradient"] and rec["gradient_r2"] < engine.WEAK_GRADIENT_R2
    assert "weak difficulty gradient" in rec["warning"]


def test_runner_passes_no_orient():
    assert "--no-orient" not in runner.command("m.csv", "out", {})
    assert runner.command("m.csv", "out", {}, orient=False)[-1] == "--no-orient"


# --------------------------------------------------------------------------- #
# the convention on the four examples
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dataset", DATASETS)
def test_hard_region_is_at_the_top_left(dataset):
    r = load_is_output(IS_DIR / dataset)
    o = r.orientation
    assert o["enabled"] and o["applied"]
    rot = np.array(o["matrix"])
    assert np.isclose(np.linalg.det(rot), 1.0) and np.allclose(rot @ rot.T, np.eye(2))
    bad = _majority_bad(r)
    assert int(bad.sum()) == o["n_instances_bad"]
    centroid = r.instances.loc[bad, ["z_1", "z_2"]].to_numpy().mean(axis=0)
    assert _angle(centroid) == pytest.approx(135.0, abs=1e-9)
    assert centroid[0] < 0 < centroid[1]                  # top left of z_1 x z_2
    assert o["weak_gradient"] is (dataset == "hill-valley")
    assert (o["warning"] is not None) is (dataset == "hill-valley")


# --------------------------------------------------------------------------- #
# invariance: with and without the rotation
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def unrotated(tmp_path_factory):
    out = {}
    for name in DATASETS:
        folder = tmp_path_factory.mktemp(f"unrotated_{name}")
        engine.run_instancespace(ROOT / "resultados" / "isa" / name / "metadata.csv", folder,
                                 orient=False)
        out[name] = folder
    return out


@pytest.mark.slow
@pytest.mark.parametrize("dataset", DATASETS)
def test_non_geometric_outputs_are_byte_identical(unrotated, dataset):
    rotated, plain = IS_DIR / dataset, unrotated[dataset]
    names = sorted(p.name for p in plain.glob("*.csv"))
    assert names == sorted(p.name for p in rotated.glob("*.csv"))
    for name in names:
        if name in GEOMETRIC or name.startswith("footprint_") and name != "footprint_performance.csv":
            continue
        assert (plain / name).read_bytes() == (rotated / name).read_bytes(), name


@pytest.mark.slow
@pytest.mark.parametrize("dataset", DATASETS)
def test_footprints_pythia_and_pilot_are_invariant(unrotated, dataset):
    rot_r, plain_r = load_is_output(IS_DIR / dataset), load_is_output(unrotated[dataset])
    o = rot_r.orientation
    rot = np.array(o["matrix"])
    assert plain_r.orientation["enabled"] is False and plain_r.orientation["applied"] is False
    assert plain_r.orientation["gradient_r2"] == pytest.approx(o["gradient_r2"], abs=1e-12)

    # geometry: every point is the unrotated one rotated
    np.testing.assert_allclose(rot_r.instances[["z_1", "z_2"]].to_numpy(),
                               engine.rotate(plain_r.instances[["z_1", "z_2"]].to_numpy(), rot),
                               rtol=0, atol=1e-12)
    # footprints: areas, densities, purities, parts and membership
    pairs = [(k, rot_r.footprints[k], plain_r.footprints[k]) for k in rot_r.footprints]
    pairs += [("space", rot_r.footprint_space, plain_r.footprint_space),
              ("hard", rot_r.footprint_hard, plain_r.footprint_hard)]
    for key, a, b in pairs:
        assert a.status == b.status and len(a.polygons) == len(b.polygons), key
        assert a.area == pytest.approx(b.area, rel=1e-9, abs=1e-12), key
        for field in ("normalized_area", "normalized_density", "purity"):
            assert getattr(a, field) == getattr(b, field) or (
                np.isnan(getattr(a, field)) and np.isnan(getattr(b, field))), (key, field)
        assert rot_r.instances_in_footprint(a) == plain_r.instances_in_footprint(b), key
    pd.testing.assert_frame_equal(rot_r.footprint_performance, plain_r.footprint_performance)
    assert rot_r.bounds.area == pytest.approx(plain_r.bounds.area, rel=1e-9)
    # PYTHIA: recommendations, probabilities and metrics
    pd.testing.assert_frame_equal(rot_r.pythia_selection, plain_r.pythia_selection)
    pd.testing.assert_frame_equal(rot_r.pythia_proba, plain_r.pythia_proba)
    pd.testing.assert_frame_equal(rot_r.pythia_confusion, plain_r.pythia_confusion)
    pd.testing.assert_frame_equal(rot_r.svm_table, plain_r.svm_table)
    # PILOT r2
    pd.testing.assert_frame_equal(rot_r.pilot_r2, plain_r.pilot_r2)
    # projection matrix: the rotated one (4 decimals) still gives the rotated z
    x = pd.read_csv(IS_DIR / dataset / "feature_process.csv", index_col=0)
    a = rot_r.projection_matrix[x.columns].to_numpy()
    np.testing.assert_allclose(x.to_numpy() @ a.T, rot_r.instances[["z_1", "z_2"]].to_numpy(),
                               atol=5e-4)
    np.testing.assert_allclose(a, rot @ plain_r.projection_matrix[x.columns].to_numpy(), atol=2e-4)


def test_run_info_records_the_rotation():
    for name in DATASETS:
        o = json.loads((IS_DIR / name / "run_info.json").read_text())["orientation"]
        for key in ("angle_deg", "matrix", "determinant", "gradient_r2", "centroid_bad_after",
                    "rotated_files", "convention", "bad_instance_rule"):
            assert key in o, (name, key)
        assert "projection_matrix.csv" in o["rotated_files"]
        assert "coordinates.csv" in o["rotated_files"]
