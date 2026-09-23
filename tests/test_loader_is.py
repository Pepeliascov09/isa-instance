"""Tests of isaspace.ui.loader_is.

The tests on real data read resultados/is/<name>/ (generate it with
scripts/run_is_all.py in the .venv-isa) and cover the errors the legacy loader
(isaspace/ui/loader.py) would silently make with the instancespace output. The
synthetic tests build a minimal folder in tmp_path for what the four datasets
do not exercise (non-numeric labels, holes, name collisions, source, ties).

Usage, from the root, with the .venv-isa:  .venv-isa/bin/python -m pytest tests/
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaspace.ui import loader_is  # noqa: E402
from isaspace.ui.loader import _split_polygons  # noqa: E402  (legacy loader)
from isaspace.ui.loader_is import (  # noqa: E402
    CATEGORICAL, DECLARED, EMPTY, FORCED, IDENTIFIER, INFERRED, INTEGER, NUMERIC, OK, TIE,
    Polygon, best_ties, load_is_output,
)

IS_DIR = ROOT / "resultados" / "is"
DATASETS = ["iris", "diabetes", "blood-transfusion-service-center", "hill-valley"]
# instances with a tie for the best observed value (checked against the raw
# algorithm_raw.csv with the PRELIM rule when this round started)
TIES = {"iris": 123, "diabetes": 217, "blood-transfusion-service-center": 215, "hill-valley": 54}


def _folder(name):
    path = IS_DIR / name
    if not (path / "run_info.json").is_file():
        pytest.skip(f"{path} not generated (run scripts/run_is_all.py in the .venv-isa)")
    return path


@pytest.fixture(scope="module", params=DATASETS)
def result(request):
    return load_is_output(_folder(request.param))


# --------------------------------------------------------------------------- #
# real data: the silent errors of the legacy loader
# --------------------------------------------------------------------------- #
def test_two_part_footprint_of_iris_becomes_two_polygons():
    path = _folder("iris")
    r = load_is_output(path)
    two_parts = 0
    for (algo, kind), fp in r.footprints.items():
        if fp.file is None:
            continue
        raw = pd.read_csv(path / fp.file)
        parts = sorted(raw["Part"].unique())
        assert len(fp.polygons) == len(parts), (algo, kind)
        expected = [int(((raw["Part"] == p) & (raw["Ring"] == "exterior")).sum()) for p in parts]
        assert [p.n_vertices for p in fp.polygons] == expected, (algo, kind)
        if len(parts) == 2:
            two_parts += 1
            # the legacy loader only splits on NaN rows: it would join the parts in one ring
            legacy = _split_polygons(pd.read_csv(path / fp.file, index_col="Row"))
            assert len(legacy) == 1 and len(legacy[0]) == sum(expected)
    assert two_parts > 0, "iris should have a two-part footprint"


def test_footprint_without_file_is_empty(result):
    files = result.run_info["footprint_files"]
    for (algo, kind), fp in result.footprints.items():
        no_file = files[algo][kind] is None
        assert no_file == (fp.status == EMPTY), (algo, kind, fp.status)
        if no_file:
            assert fp.polygons == [] and fp.status != OK and fp.area == 0
    assert len(result.footprints) == 2 * len(result.algos)


def test_iris_has_footprints_without_file():
    r = load_is_output(_folder("iris"))
    empty = r.empty_footprints
    assert empty, "iris should have empty best footprints"
    for key in empty:
        assert not (r.path / f"footprint_{key[0]}_{key[1]}.csv").exists()


def test_1_and_0_based_portfolios_give_the_same_name_when_they_agree(result):
    p = pd.read_csv(result.path / "portfolio.csv", dtype={"Row": str}).set_index("Row")["Best_Algorithm"]
    s = pd.read_csv(result.path / "portfolio_svm.csv", dtype={"Row": str}).set_index("Row")["Best_Algorithm"]
    inst = result.instances
    agree = (p - 1 == s).to_numpy()
    assert agree.any()
    assert (inst["best_algo"][agree] == inst["best_algo_svm"][agree]).all()
    # reading portfolio_svm with the legacy loader's 1-based rule would be off by one
    naive = [result.algos[i - 1] if 1 <= i <= len(result.algos) else None for i in s[agree]]
    assert all(a != b for a, b in zip(naive, inst["best_algo_svm"][agree]))
    none = (s == -1).to_numpy()
    assert inst["best_algo_svm"][none].isna().all()
    assert inst["best_algo"].notna().all()   # PRELIM always picks one


def test_ties_for_the_best_observed_value(result):
    inst = result.instances
    raw = inst[[f"algo_{a}" for a in result.algos]].to_numpy(dtype=float)
    expected = (raw == raw.max(axis=1, keepdims=True)).sum(axis=1)     # max_perf, no NaN here
    assert (inst["n_tied_best"].to_numpy() == expected).all()
    assert int((inst["n_tied_best"] > 1).sum()) == TIES[result.name]
    tied = inst["n_tied_best"] > 1
    assert (inst.loc[tied, "best_algo_or_tie"] == TIE).all()
    # without a tie it is portfolio.csv's best algorithm; with a tie, portfolio.csv
    # picked one of the tied algorithms at random
    assert (inst.loc[~tied, "best_algo_or_tie"] == inst.loc[~tied, "best_algo"]).all()
    for row in inst[tied].itertuples():
        values = {a: getattr(row, f"algo_{a}") for a in result.algos}
        assert values[row.best_algo] == max(values.values())


def test_annotations_match_the_table_via_row_original(result):
    inst = result.instances
    table = pd.read_csv(ROOT / "resultados" / f"table_{result.name}.csv", index_col=0)
    t = table.loc[inst["row_original"].to_numpy()]
    assert (inst["class"].to_numpy() == t["class"].astype(str).to_numpy()).all()
    np.testing.assert_allclose(inst["ih"].to_numpy(dtype=float), t["ih"].to_numpy(), rtol=1e-12, atol=0)
    assert (inst["n_wrong"].to_numpy() == t["n_wrong"].to_numpy()).all()
    assert result.annotations["class"] == CATEGORICAL
    assert result.annotations["ih"] == NUMERIC
    assert result.annotations["n_wrong"] == INTEGER          # declared in annotations.json


def test_coordinates_rows_equal_metadata(result):
    coords = pd.read_csv(result.path / "coordinates.csv", dtype={"Row": str})
    meta = pd.read_csv(result.path / "metadata.csv", dtype={"instances": str})
    assert len(coords) == len(meta) == result.n
    assert list(coords["Row"]) == list(meta["instances"]) == list(result.instances.index)
    assert result.run_info["n_instances"] == result.run_info["n_instances_input"] == result.n


def test_binary_files_become_bool(result):
    inst = result.instances
    for a in result.algos:
        assert inst[f"algo_{a}_bin"].dtype == bool
        assert inst[f"algo_{a}_svm"].dtype == bool
    assert inst["IsBetaEasy"].dtype == bool
    # algo_*_bin is the perf threshold on algo_* (max_perf, abs_perf, epsilon=0.5)
    eps = result.run_options["perf"]["epsilon"]
    for a in result.algos:
        assert (inst[f"algo_{a}_bin"] == (inst[f"algo_{a}"] >= eps)).all()


def test_sifted_report_is_consistent(result):
    rep = result.sifted_report
    assert set(rep["status"]) <= {"kept", "dropped_correlation", "dropped_redundancy"}
    assert list(rep.loc[rep["status"] == "kept", "feature"]) == result.features
    kept = rep[rep["status"] == "kept"].set_index("feature")["cluster"]
    for row in rep[rep["status"] == "dropped_redundancy"].itertuples():
        assert row.kept_instead in kept.index
        assert kept[row.kept_instead] == row.cluster
    assert result.run_info["warnings"] == []


def test_pythia_proba_out_of_and_in_sample(result):
    sub, hat = result.pythia_proba, result.pythia_proba_hat
    for proba in (sub, hat):
        assert proba.shape == (result.n, len(result.algos))
        assert list(proba.index) == list(result.instances.index)
        assert list(proba.columns) == result.algos
        assert ((proba.to_numpy() >= 0) & (proba.to_numpy() <= 1)).all()
    assert not np.allclose(sub.to_numpy(), hat.to_numpy())
    # algorithm_svm.csv (y_hat) against pr0_hat < 0.5: the run_info count
    y_hat = result.instances[[f"algo_{a}_svm" for a in result.algos]].to_numpy()
    disagree = int(np.sum(y_hat != (hat.to_numpy() < 0.5)))
    assert disagree == result.run_info["pythia"]["y_hat_disagrees_with_pr0_hat"]


def test_coordinates_is_the_pilot_z_and_trace_only_with_jitter(result):
    rob = result.run_info["trace_robustness"]
    z = result.instances[["z_1", "z_2"]]
    raw = pd.read_csv(result.path / "coordinates.csv", dtype={"Row": str}).set_index("Row")
    assert z.equals(raw[["z_1", "z_2"]])
    # pairs of distinct positions closer than the threshold, in the PILOT z
    pos = np.unique(z.to_numpy(), axis=0)
    d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    close = int(np.sum(np.triu(d < rob["threshold"], k=1)))
    assert close == rob["distinct_close_pairs"]
    if not rob["jitter_applied"]:
        assert result.coordinates_trace is None
        assert not (result.path / "coordinates_trace.csv").exists()
        return
    tr = result.coordinates_trace
    moved = set(rob["perturbed_labels"])
    differ = (tr != z).any(axis=1)
    assert set(differ[differ].index) == moved
    assert np.abs((tr - z).to_numpy()).max() <= rob["max_shift"] + 1e-12


def test_pythia_confusion_reproduces_svm_table(result):
    conf = result.pythia_confusion
    assert (conf.sum(axis=1) == result.n).all()
    tab = result.svm_table.loc[result.algos]
    acc = 100 * (conf.tp + conf.tn) / result.n
    prec = 100 * conf.tp / (conf.tp + conf.fp)
    rec = 100 * conf.tp / (conf.tp + conf.fn)
    # svm_table rounds the percentages to one decimal
    np.testing.assert_allclose(acc.round(1), tab["CV_model_accuracy"], atol=0.051)
    np.testing.assert_allclose(prec.round(1), tab["CV_model_precision"], atol=0.051)
    np.testing.assert_allclose(rec.round(1), tab["CV_model_recall"], atol=0.051)
    # tn + fp = bad instances in the observed performance
    bad = (~result.instances[[f"algo_{a}_bin" for a in result.algos]]).sum().to_numpy()
    assert ((conf.tn + conf.fp).to_numpy() == bad).all()


def test_pythia_selection(result):
    sel = result.pythia_selection
    inst = result.instances
    assert list(sel.index) == list(inst.index)
    assert sel["selection0"].equals(inst["best_algo_svm"])
    has = sel["selection0"].notna()
    assert (sel.loc[has, "selection1"] == sel.loc[has, "selection0"]).all()
    assert sel["selection1"].notna().all()
    # without a recommendation, selection1 falls back to the algorithm with most good instances
    default = inst[[f"algo_{a}_bin" for a in result.algos]].mean().idxmax()[len("algo_"):-len("_bin")]
    assert (sel.loc[~has, "selection1"] == default).all()


def test_features_outside_pilot_come_from_the_metadata(result):
    meta = pd.read_csv(result.path / "metadata.csv", dtype={"instances": str}).set_index("instances")
    all_feats = [c[len("feature_"):] for c in meta.columns if c.startswith("feature_")]
    assert result.features_all == all_feats
    assert set(result.features_outside_pilot) == set(all_feats) - set(result.features)
    outside = result.sifted_report.set_index("feature").loc[result.features_outside_pilot, "status"]
    assert (outside != "kept").all()
    for f in all_feats:
        assert (result.instances[f"feature_{f}"].to_numpy() == meta[f"feature_{f}"].to_numpy()).all()


def test_pilot_r2(result):
    r2 = result.pilot_r2
    assert list(r2["variable"]) == result.features + result.algos
    assert list(r2["kind"]) == ["feature"] * len(result.features) + ["algorithm"] * len(result.algos)
    assert ((r2["r2"] >= 0) & (r2["r2"] <= 1)).all()


def test_sifted_correlations_and_silhouette(result):
    c = result.sifted_correlations
    feats = list(result.sifted_report["feature"])
    assert len(c) == len(feats) * len(result.algos)
    assert list(result.sifted_rho.index) == feats
    rho, pval = result.sifted_rho, result.sifted_pval
    rep = result.sifted_report.set_index("feature")
    for f in feats:
        j = rho.loc[f].abs().idxmax()
        assert rho.loc[f, j] == pytest.approx(rep.loc[f, "rho"])
        assert pval.loc[f, j] == pytest.approx(rep.loc[f, "pval"])
    sil = result.sifted_silhouette
    k = result.run_options["sifted"]["k"]
    assert list(sil["k"]) == list(range(3, 3 + len(sil)))
    assert list(sil.loc[sil["used"], "k"]) == [k]
    assert sil["best"].sum() == 1 and sil.loc[sil["best"], "silhouette"].iloc[0] == sil["silhouette"].max()


def test_space_and_hard_footprints(result):
    space, hard = result.footprint_space, result.footprint_hard
    assert space.kind == "space" and space.status == OK and space.area > 0
    for fp in (space, hard):
        assert (fp.file is None) == (fp.status == EMPTY) == (fp.polygons == [])
    # the normalized areas of footprint_performance are area / space area
    for fp in result.footprints.values():
        if fp.normalized_area >= 0.1:
            assert fp.area / fp.normalized_area == pytest.approx(space.area, rel=0.01)
    if hard.status != EMPTY:
        assert hard.normalized_area == pytest.approx(hard.area / space.area, rel=1e-6)


def test_cloister_as_polygon(result):
    for pol in (result.bounds, result.bounds_pruned):
        assert pol is not None and pol.n_vertices >= 3 and pol.area > 0
    # the boundary's bounding box covers the instance cloud
    z = result.instances[["z_1", "z_2"]].to_numpy()
    ext = result.bounds.exterior
    assert (z.min(axis=0) >= ext.min(axis=0) - 1e-9).all()
    assert (z.max(axis=0) <= ext.max(axis=0) + 1e-9).all()


def test_new_loader_refuses_the_pyispace_folder():
    legacy = ROOT / "resultados" / "isa" / "iris"
    if not legacy.is_dir():
        pytest.skip("resultados/isa/iris missing")
    with pytest.raises(FileNotFoundError, match="run_info.json"):
        load_is_output(legacy)


# --------------------------------------------------------------------------- #
# declared types, degenerate_report and the Features tab
# --------------------------------------------------------------------------- #
def test_declared_types_come_from_annotations_json(result):
    declared = json.loads((result.path / "annotations.json").read_text())
    assert declared == {"row_original": IDENTIFIER, "class": CATEGORICAL, "ih": NUMERIC,
                        "n_wrong": INTEGER}
    assert result.run_info["annotation_types"]["declared"] == declared
    for col, kind in declared.items():
        assert result.annotations[col] == kind
        assert result.annotation_origins[col] == DECLARED
    # the IC7 generator declares everything: nothing inferred; row_original is an identifier
    assert result.inferred_annotations == []
    ro = result.instances["row_original"]
    assert pd.api.types.is_integer_dtype(ro) and ro.is_unique       # values as they came
    assert pd.api.types.is_numeric_dtype(result.instances["n_wrong"])
    assert result.instances["class"].map(type).eq(str).all()


def test_forced_type_overrides_declaration_and_heuristic():
    path = _folder("hill-valley")
    r = load_is_output(path, annotation_types={"row_original": CATEGORICAL, "class": NUMERIC})
    assert r.annotations["row_original"] == CATEGORICAL and r.annotation_origins["row_original"] == FORCED
    assert r.annotations["class"] == NUMERIC and r.annotation_origins["class"] == FORCED
    assert pd.api.types.is_numeric_dtype(r.instances["class"])       # "0"/"1" -> 0/1


def _received_features(name):
    table = pd.read_csv(ROOT / "resultados" / f"table_{name}.csv", index_col=0, nrows=1)
    return [c[len("feature_"):] for c in table.columns if c.startswith("feature_")]


def test_degenerate_report_covers_what_did_not_reach_the_engine(result):
    deg = result.degenerate_report
    assert deg is not None and list(deg.columns) == ["feature", "raw_variance", "iqr", "reason"]
    received = _received_features(result.name)
    assert set(deg["feature"]) == set(received) - set(result.features_all)
    assert not set(deg["feature"]) & set(result.features_all)
    assert deg["reason"].str.len().gt(0).all()


def test_degenerate_report_of_iris():
    r = load_is_output(_folder("iris"))
    assert list(r.degenerate_report["feature"]) == ["kDN", "MV", "CB", "N1", "Harmfulness"]


def test_features_table(result):
    t = result.features_table()
    received = _received_features(result.name)
    assert list(t["feature"]) == received                  # feature_info.csv order
    assert set(t["status"]) <= {"kept", "dropped_degenerate", "dropped_correlation", "dropped_redundancy"}
    assert (t["status"] == "dropped_degenerate").sum() == len(result.degenerate_report)
    kept = t[t["status"] == "kept"]
    assert list(kept["feature"]) == result.features
    assert kept["r2_pilot"].notna().all() and t.loc[t["status"] != "kept", "r2_pilot"].isna().all()
    red = t[t["status"] == "dropped_redundancy"]
    assert red["replaced_by"].isin(result.features).all()
    assert t.loc[t["status"] != "dropped_degenerate", "max_abs_rho"].ge(0).all()
    derived = {"CL", "CLD", "DS", "DCP", "TD_U", "TD_P"}
    assert set(t.loc[t["family"] == "model_derived", "feature"]) == derived & set(received)
    assert set(t["family"]) == {"model_derived", "geometric"}


def test_footprint_membership_matches_trace(result):
    """The loader's membership reproduces the elements TRACE counted."""
    for name, fp in (("space", result.footprint_space), ("hard", result.footprint_hard)):
        expected = result.run_info["special_footprints"][name]["elements"]
        got = len(result.instances_in_footprint(fp)) if fp.polygons else 0
        assert got == expected, name


def test_polygon_contains_with_hole_and_boundaries():
    square = Polygon(np.array([[0, 0], [4, 0], [4, 4], [0, 4]], float),
                     [np.array([[1, 1], [2, 1], [2, 2], [1, 2]], float)])
    pts = np.array([[3, 3], [1.5, 1.5], [0, 2], [4, 4], [1, 1.5], [5, 5], [-0.1, 2]])
    assert square.contains(pts).tolist() == [True, False, True, True, True, False, False]


def test_best_ties_follows_prelim_rule():
    y = np.array([[0.9, 0.9, 0.1], [0.2, np.nan, 0.8], [np.nan, np.nan, np.nan], [0.0, 0.0, 0.0]])
    assert best_ties(y, True).tolist() == [2, 1, 0, 3]
    assert best_ties(y, False).tolist() == [1, 1, 0, 3]


# --------------------------------------------------------------------------- #
# synthetic folder
# --------------------------------------------------------------------------- #
def _csv(path, text):
    path.write_text(text.strip() + "\n")


@pytest.fixture
def synthetic(tmp_path):
    """Four instances with text labels, two algorithms, a footprint with a
    hole, one without a file and an annotation called z_1."""
    p = tmp_path / "synth"
    p.mkdir()
    labels = ["alpha", "beta", "gamma", "delta"]
    _csv(p / "coordinates.csv", "Row,z_1,z_2\n" + "\n".join(
        f"{r},{x},{y}" for r, x, y in zip(labels, [0, 4, 4, 0], [0, 0, 4, 4])))
    _csv(p / "algorithm_raw.csv", "Row,A,B\n" + "\n".join(
        f"{r},{a},{b}" for r, a, b in zip(labels, [0.9, 0.2, 0.7, 0.6], [0.1, 0.8, 0.3, 0.4])))
    _csv(p / "algorithm_bin.csv", "Row,A,B\n" + "\n".join(
        f"{r},{a},{b}" for r, a, b in zip(labels, ["True", "False", "True", "True"],
                                          ["False", "True", "False", "False"])))
    _csv(p / "algorithm_svm.csv", "Row,A,B\n" + "\n".join(f"{r},True,False" for r in labels))
    _csv(p / "feature_raw.csv", "Row,f1,f2\n" + "\n".join(f"{r},{i},{2 * i}" for i, r in enumerate(labels)))
    _csv(p / "feature_process.csv", "Row,f1,f2\n" + "\n".join(f"{r},{i / 3},{-i / 3}" for i, r in enumerate(labels)))
    _csv(p / "good_algos.csv", "Row,NumGoodAlgos\n" + "\n".join(f"{r},1" for r in labels))
    _csv(p / "beta_easy.csv", "Row,IsBetaEasy\n" + "\n".join(f"{r},False" for r in labels))
    _csv(p / "portfolio.csv", "Row,Best_Algorithm\n" + "\n".join(
        f"{r},{v}" for r, v in zip(labels, [1, 2, 1, 1])))
    _csv(p / "portfolio_svm.csv", "Row,Best_Algorithm\n" + "\n".join(
        f"{r},{v}" for r, v in zip(labels, [0, 1, -1, 0])))
    _csv(p / "metadata.csv", "instances,Source,group,z_1,weight,feature_f1,feature_f2,algo_A,algo_B\n" + "\n".join(
        f"{r},{s},{g},{k},{w},0,0,0,0" for r, s, g, k, w in
        zip(labels, ["S1", "S1", "S2", "S2"], ["x", "y", "x", ""], [1, 0, 1, 0], [0.5, 1.5, 2.5, 3.5])))
    _csv(p / "footprint_A_good.csv", """
Row,Part,Ring,Vertex,z_1,z_2
1,1,exterior,1,0,0
2,1,exterior,2,4,0
3,1,exterior,3,4,4
4,1,exterior,4,0,4
5,1,hole_1,1,1,1
6,1,hole_1,2,2,1
7,1,hole_1,3,2,2
8,1,hole_1,4,1,2
9,2,exterior,1,10,10
10,2,exterior,2,11,10
11,2,exterior,3,10,11
""")
    _csv(p / "footprint_performance.csv", """
Row,Area_Good_Normalized,Density_Good_Normalized,Purity_Good,Area_Best_Normalized,Density_Best_Normalized,Purity_Best
A,0.9,1.0,0.8,0.0,0.0,0.0
B,0.0,0.0,0.0,0.0,0.0,0.0
""")
    _csv(p / "projection_matrix.csv", "Row,f1,f2\nZ_{1},0.5,0.5\nZ_{2},-0.5,0.5")
    _csv(p / "svm_table.csv", "Row,Avg_Perf_all_instances\nA,0.6\nB,0.4\nOracle,0.9\nSelector,0.6")
    _csv(p / "sifted_report.csv", "feature,status,rho,rho_algo,pval,n_algos_sig,cluster,kept_instead\n"
         "f1,kept,0.5,A,0.01,1,,\nf2,kept,-0.4,B,0.02,1,,")
    _csv(p / "pythia_proba.csv", "Row,A,B,A_hat,B_hat\n" + "\n".join(
        f"{r},0.1,0.9,0.2,0.8" for r in labels))
    _csv(p / "pythia_confusion.csv", "Algorithm,tn,fp,fn,tp\nA,1,0,0,3\nB,3,0,1,0")
    _csv(p / "pythia_selection.csv", "Row,selection0,selection1\n" + "\n".join(
        f"{r},{a},{b}" for r, a, b in zip(labels, ["A", "B", "", "A"], ["A", "B", "A", "A"])))
    _csv(p / "pilot_r2.csv", "variable,kind,r2\nf1,feature,0.9\nf2,feature,0.5\n"
         "A,algorithm,0.3\nB,algorithm,0.2")
    _csv(p / "sifted_correlations.csv", "feature,algorithm,rho,pval\nf1,A,0.5,0.01\n"
         "f1,B,-0.1,0.5\nf2,A,0.2,0.3\nf2,B,-0.4,0.02")
    _csv(p / "sifted_silhouette.csv", "k,silhouette,used,best")
    _csv(p / "footprint_space.csv", "Row,Part,Ring,Vertex,z_1,z_2\n1,1,exterior,1,0,0\n"
         "2,1,exterior,2,4,0\n3,1,exterior,3,4,4\n4,1,exterior,4,0,4")
    (p / "bounds.csv").write_text("Row,z_1,z_2\nbnd_pnt_1,-1,-1\nbnd_pnt_2,5,-1\nbnd_pnt_3,5,5\nbnd_pnt_4,-1,5\n")
    (p / "run_options.json").write_text(json.dumps({"trace": {"purity": 0.55},
                                                    "perf": {"epsilon": 0.5, "max_perf": True}}))
    (p / "run_info.json").write_text(json.dumps({
        "algorithms": ["A", "B"],
        "footprint_files": {"A": {"good": "footprint_A_good.csv", "best": None},
                            "B": {"good": None, "best": None}},
        "trace_robustness": {"jitter_applied": False},
        "special_footprints": {
            "space": {"file": "footprint_space.csv", "area": 16.0, "purity": 1.0},
            "hard": {"file": None, "area": 0.0, "purity": 0.0,
                     "normalized_area": 0.0, "normalized_density": 0.0},
        },
    }))
    return p


def test_synthetic_text_labels_holes_and_empty(synthetic):
    r = load_is_output(synthetic)
    assert list(r.instances.index) == ["alpha", "beta", "gamma", "delta"]
    good = r.footprints[("A", "good")]
    assert good.status == OK and len(good.polygons) == 2
    outer = good.polygons[0]
    assert outer.n_vertices == 4 and len(outer.holes) == 1
    assert outer.area == pytest.approx(16 - 1)            # 4x4 square minus a 1x1 hole
    assert good.area == pytest.approx(15 + 0.5)
    for key in [("A", "best"), ("B", "good"), ("B", "best")]:
        assert r.footprints[key].status == EMPTY
    assert sorted(r.empty_footprints) == [("A", "best"), ("B", "best"), ("B", "good")]


def test_synthetic_portfolios_bool_and_annotations(synthetic):
    r = load_is_output(synthetic)
    inst = r.instances
    assert list(inst["best_algo"]) == ["A", "B", "A", "A"]
    assert list(inst["best_algo_svm"]) == ["A", "B", None, "A"]
    assert list(inst["algo_A_bin"]) == [True, False, True, True]
    assert inst["IsBetaEasy"].dtype == bool
    assert r.source_column == "source" and list(inst["source"]) == ["S1", "S1", "S2", "S2"]
    # an annotation called z_1 does not overwrite the coordinate
    assert r.annotation_renames == {"z_1": "ann_z_1"}
    assert list(inst["z_1"]) == [0, 4, 4, 0]
    assert r.annotations == {"group": CATEGORICAL, "ann_z_1": CATEGORICAL, "weight": NUMERIC}
    assert pd.isna(inst.loc["delta", "group"])
    assert list(inst["ann_z_1"]) == ["1", "0", "1", "0"]
    assert list(r.projection_matrix.index) == ["z_1", "z_2"]
    assert r.bounds.area == pytest.approx(36)
    forced = load_is_output(synthetic, annotation_types={"z_1": NUMERIC})
    assert forced.annotations["ann_z_1"] == NUMERIC


def test_synthetic_tie_replaces_the_random_pick(synthetic):
    f = synthetic / "algorithm_raw.csv"
    f.write_text(f.read_text().replace("delta,0.6,0.4", "delta,0.6,0.6"))
    r = load_is_output(synthetic)
    inst = r.instances
    assert list(inst["n_tied_best"]) == [1, 1, 1, 2]
    assert list(inst["best_algo_or_tie"]) == ["A", "B", "A", TIE]
    assert inst.loc["delta", "best_algo"] == "A"          # portfolio.csv's pick is kept apart


def test_synthetic_portfolio_against_unique_best_is_an_error(synthetic):
    f = synthetic / "portfolio.csv"
    f.write_text(f.read_text().replace("alpha,1", "alpha,2"))
    with pytest.raises(ValueError, match="unique best"):
        load_is_output(synthetic)


def test_synthetic_new_files(synthetic):
    r = load_is_output(synthetic)
    assert list(r.pythia_proba.columns) == ["A", "B"] and list(r.pythia_proba_hat.columns) == ["A", "B"]
    assert r.pythia_proba.loc["alpha", "A"] == 0.1 and r.pythia_proba_hat.loc["alpha", "A"] == 0.2
    assert list(r.pythia_selection["selection0"]) == ["A", "B", None, "A"]
    assert r.pythia_confusion.loc["B", "fn"] == 1
    assert r.sifted_rho.loc["f2", "B"] == -0.4 and r.sifted_pval.loc["f1", "A"] == 0.01
    assert r.sifted_silhouette.empty
    assert r.coordinates_trace is None
    assert r.footprint_space.status == OK and r.footprint_space.area == pytest.approx(16)
    assert r.footprint_hard.status == EMPTY


def test_synthetic_coordinates_trace_without_jitter_is_an_error(synthetic):
    (synthetic / "coordinates_trace.csv").write_text(
        (synthetic / "coordinates.csv").read_text())
    with pytest.raises(ValueError, match="coordinates_trace.csv"):
        load_is_output(synthetic)


def test_synthetic_selection_with_unknown_algorithm_is_an_error(synthetic):
    f = synthetic / "pythia_selection.csv"
    f.write_text(f.read_text().replace("delta,A,A", "delta,Z,A"))
    with pytest.raises(ValueError, match="unknown algorithms"):
        load_is_output(synthetic)


def test_synthetic_without_auxiliary_files(synthetic):
    r = load_is_output(synthetic)
    assert set(r.annotation_origins.values()) == {INFERRED}
    assert r.degenerate_report is None and r.feature_info is None
    t = r.features_table()
    assert list(t["feature"]) == ["f1", "f2"] and "family" not in t.columns


@pytest.mark.parametrize("declaration, error", [
    ({"group": NUMERIC}, "non-numeric"),          # 'x', 'y' are not numbers
    ({"weight": "text"}, "is not one of"),
    ({"feature_f1": NUMERIC}, "is not an annotation column"),
])
def test_synthetic_invalid_declaration_is_an_error(synthetic, declaration, error):
    (synthetic / "annotations.json").write_text(json.dumps(declaration))
    with pytest.raises(ValueError, match=error):
        load_is_output(synthetic)


def test_synthetic_integer_with_fraction_is_an_error(synthetic):
    (synthetic / "annotations.json").write_text(json.dumps({"weight": INTEGER}))
    with pytest.raises(ValueError, match="non-integer"):
        load_is_output(synthetic)


def test_synthetic_identifier(synthetic):
    (synthetic / "annotations.json").write_text(json.dumps({"weight": IDENTIFIER}))
    r = load_is_output(synthetic)
    assert r.annotations["weight"] == IDENTIFIER and r.annotation_origins["weight"] == DECLARED
    assert list(r.instances["weight"]) == [0.5, 1.5, 2.5, 3.5]          # values as they came


def test_synthetic_valid_declaration(synthetic):
    (synthetic / "annotations.json").write_text(json.dumps({"weight": NUMERIC, "z_1": INTEGER}))
    r = load_is_output(synthetic)
    assert r.annotations["weight"] == NUMERIC and r.annotation_origins["weight"] == DECLARED
    assert r.annotations["ann_z_1"] == INTEGER and r.annotation_origins["ann_z_1"] == DECLARED
    assert r.annotation_origins["group"] == INFERRED


def test_synthetic_explicit_errors(synthetic):
    # Row misaligned between files is an error, not a silent join
    f = synthetic / "algorithm_bin.csv"
    f.write_text(f.read_text().replace("delta", "epsilon"))
    with pytest.raises(ValueError, match="algorithm_bin.csv"):
        load_is_output(synthetic)


def test_synthetic_footprint_in_the_legacy_format_is_refused(synthetic):
    (synthetic / "footprint_A_good.csv").write_text("Row,z_1,z_2\n0,0,0\n1,1,0\n2,0,1\n")
    with pytest.raises(ValueError, match="instancespace format"):
        load_is_output(synthetic)


@pytest.mark.parametrize("module", ["loader_is.py", "app.py", "upload.py", "new_space.py", "runner.py"])
def test_ui_does_not_import_the_backend(module):
    source = (Path(loader_is.__file__).parent / module).read_text()
    imports = re.findall(r"^\s*(?:import|from)\s+(\w+)", source, flags=re.MULTILINE)
    assert not set(imports) & {"instancespace", "sklearn", "pyispace", "pyhard"}, imports
