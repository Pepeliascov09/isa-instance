"""ISA engine on top of the ``instancespace`` package (Muñoz et al.).

Runs the full pipeline (PREPROCESSING, PRELIM, SIFTED, PILOT, PYTHIA,
CLOISTER, TRACE) stage by stage on a metadata.csv and writes the folder that
``isaspace.ui.loader_is`` reads:

- everything ``Model.save_to_csv`` writes (coordinates.csv, feature_*.csv,
  algorithm_*.csv, good_algos.csv, beta_easy.csv, portfolio*.csv,
  footprint_<algo>_<good|best>.csv, footprint_performance.csv,
  projection_matrix.csv, bounds*.csv, svm_table.csv);
- ``coordinates.csv`` rewritten with the uncorrected PILOT z and, only when
  jitter is applied, ``coordinates_trace.csv`` with the z TRACE used;
- ``metadata.csv``: byte-for-byte copy of the input metadata (annotations and
  source included) and, if present next to it, ``annotations.json`` (declared
  annotation types, validated before running), ``degenerate_report.csv`` and
  ``feature_info.csv``;
- ``run_options.json``: the effective options, ``dataclasses.asdict`` of
  ``InstanceSpaceOptions`` (``InstanceSpaceOptions.from_dict`` reads it back);
- ``sifted_report.csv``, ``sifted_correlations.csv``, ``sifted_silhouette.csv``
  (``Model.sifted``), ``pilot_r2.csv`` (``Model.pilot.r2``);
- ``pythia_proba.csv`` (pr0_sub and pr0_hat), ``pythia_confusion.csv``
  (cvcmat) and ``pythia_selection.csv`` (selection0 and selection1);
- ``footprint_space.csv`` and ``footprint_hard.csv`` (``Model.trace.space`` and
  ``.hard``), in the same schema as the algorithm footprints;
- ``run_info.json``: versions, timings per stage, number of instances, the
  near-duplicate correction before TRACE, metrics of the space and hard
  footprints, and warnings. It is written last: a folder without it is an
  incomplete run.

The format of every file is in docs/output_format.md.

TRACE robustness: the legacy TRACE alpha shape in instancespace 0.3.0 returns
an empty polygon when the projection has DISTINCT points ~1e-14 apart
(hill-valley: good footprints and space area went to zero); identical points
do no harm, TRACE merges them with np.unique. After PILOT,
``run_instancespace`` counts the pairs closer than ``NEAR_DUPLICATE_THRESHOLD``;
if there are distinct positions that close, it adds to each one normal noise
with standard deviation ``JITTER_SCALE`` and a fixed seed (identical points
get the same shift) and passes that z only to TRACE, with
``run_stage(TraceStage, z=...)``. PYTHIA (which tunes hyperparameters on z and
moves probabilities by up to 0.08 under that perturbation) and CLOISTER (which
does not use z) run without the correction. The runner keeps the override in
the Model, but coordinates.csv is rewritten with the PILOT z; the TRACE z goes
to coordinates_trace.csv. Everything is recorded in
``run_info.json["trace_robustness"]``.

Orientation: PILOT's projection has an arbitrary rotation, so the same kind
of region can land anywhere from one dataset to the next. After the WHOLE
pipeline (so nothing instancespace computes depends on it), the engine rotates
every geometric output with the convention of pyispace's adjust_rotation
(pyispace/pilot.py:97-103, called by train.py:143-150): the centroid of the
instances for which most algorithms are bad goes to 135 degrees, the top left
of the z_1 x z_2 plane. Only a proper rotation (det = +1, no reflection),
about the origin (the PILOT z is centered). Rotated: coordinates.csv,
coordinates_trace.csv, every footprint_*.csv, bounds.csv, bounds_prunned.csv
and projection_matrix.csv. Recorded in ``run_info.json["orientation"]``, with
the R^2 of the number of bad algorithms regressed on (z_1, z_2) as the
strength of the difficulty gradient. ``orient=False`` (CLI ``--no-orient``)
keeps PILOT's orientation.

Command line (used by the UI, isaspace.ui.runner, to run the engine in a
subprocess): see ``main``.

Requires Python 3.12 and the .venv-isa (instancespace 0.3.0); it does not run
in the 3.11 .venv.
"""

import argparse
import copy
import dataclasses
import json
import platform
import shutil
import sys
import time
import traceback
import warnings
from collections import Counter
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.spatial import cKDTree

from instancespace import InstanceSpace
from instancespace.data import metadata as is_metadata
from instancespace.data.options import InstanceSpaceOptions
from instancespace.stages.cloister import CloisterStage
from instancespace.stages.pilot import PilotStage
from instancespace.stages.prelim import PrelimStage
from instancespace.stages.preprocessing import PreprocessingStage
from instancespace.stages.pythia import PythiaStage
from instancespace.stages.sifted import SiftedStage
from instancespace.stages.trace import TraceStage

from isaspace.ui.loader_is import declared_type_errors
from isaspace.ui.runner import PREFIX

STAGES = [
    ("PREPROCESSING", PreprocessingStage),
    ("PRELIM", PrelimStage),
    ("SIFTED", SiftedStage),
    ("PILOT", PilotStage),
    ("PYTHIA", PythiaStage),
    ("CLOISTER", CloisterStage),
    ("TRACE", TraceStage),
]

# Default options, with the field names of the instancespace dataclasses (the
# same as run_options.json). Anything not listed here keeps the library
# default, including all of SIFTED (rho=0.1, pval=0.05, k=6, GA).
DEFAULT_OPTIONS = {
    "perf": {
        # algo_* is a performance where HIGHER is better (accuracy or
        # probability of the true class). With False, "good" becomes
        # performance <= epsilon, i.e. the error.
        "max_perf": True,
        # absolute threshold: the instance is good for the algorithm if algo_* >= epsilon
        "abs_perf": True,
        "epsilon": 0.5,
    },
    "trace": {
        # footprints of the observed performance (y_bin and portfolio.csv), not
        # of the PYTHIA predictions; the library default is True
        "use_sim": False,
    },
}

NEAR_DUPLICATE_THRESHOLD = 1e-6   # projection point pairs closer than this
JITTER_SCALE = 1e-6               # std of the normal noise added to those points
JITTER_SEED = 0

# orientation (pyispace adjust_rotation convention)
TARGET_ANGLE_DEG = 135.0     # centroid of the majority-bad instances: top left
WEAK_GRADIENT_R2 = 0.3       # below this the difficulty gradient is weak (same
                             # threshold as the UI's low PILOT r2 warning)
# files whose z_1, z_2 columns are rotated (metadata.csv is never touched: an
# annotation may be called z_1)
ROTATED_FILES = ("coordinates.csv", "coordinates_trace.csv", "bounds.csv", "bounds_prunned.csv")
ROTATED_PATTERNS = ("footprint_*_good.csv", "footprint_*_best.csv", "footprint_space.csv",
                    "footprint_hard.csv")

STATUS_KEPT = "kept"
STATUS_CORRELATION = "dropped_correlation"
STATUS_REDUNDANCY = "dropped_redundancy"
STATUS_PREPROCESSING = "dropped_preprocessing"   # removed before SIFTED
STATUS_UNDETERMINED = "undetermined"             # inconsistent reconstruction

# fixed files the engine writes; only these (and footprint_*) are deleted when
# a folder is rewritten
SAVE_TO_CSV_FILES = (
    "coordinates.csv", "bounds.csv", "bounds_prunned.csv", "feature_raw.csv",
    "feature_process.csv", "algorithm_raw.csv", "algorithm_process.csv",
    "algorithm_bin.csv", "good_algos.csv", "beta_easy.csv", "portfolio.csv",
    "algorithm_svm.csv", "portfolio_svm.csv", "footprint_performance.csv",
    "projection_matrix.csv", "svm_table.csv",
)
EXTRA_FILES = (
    "metadata.csv", "run_options.json", "coordinates_trace.csv",
    "sifted_report.csv", "sifted_correlations.csv", "sifted_silhouette.csv",
    "pilot_r2.csv", "pythia_proba.csv", "pythia_confusion.csv",
    "pythia_selection.csv", "footprint_space.csv", "footprint_hard.csv",
    "run_info.json",
)
# optional files next to the input metadata, copied when present
AUXILIARY = {
    "annotations.json": None,                                     # validated separately
    "degenerate_report.csv": ("feature", "raw_variance", "iqr", "reason"),
    "feature_info.csv": ("feature", "family"),
}
EXTRA_FILES += tuple(AUXILIARY)
# pythia_proba.csv: column <algo> = pr0_sub (default), <algo>_hat = pr0_hat
HAT_SUFFIX = "_hat"
# SiftedStage.evaluate_cluster tries k = 3 .. (features left after correlation) - 1
SILHOUETTE_K_MIN = 3
FOOTPRINT_PATTERNS = (
    "footprint_*_good.csv", "footprint_*_best.csv", "footprint_*_vertices.csv",
    "footprint_*_tetrahedra.csv", "footprint_*_boundary_faces.csv",
)
GENERATED_BY = "isaspace.engine.run_instancespace"


# --------------------------------------------------------------------------- #
# options and metadata
# --------------------------------------------------------------------------- #
def _key(name):
    """Normalized form of an option key: 'MaxPerf', 'max_perf' -> 'maxperf'."""
    return str(name).casefold().replace("_", "")


def build_options(options=None):
    """InstanceSpaceOptions from DEFAULT_OPTIONS overridden by `options`.

    `options` is a dict per group ({"trace": {"use_sim": True}}), with the
    names of run_options.json or those of the MATLAB options.json (MaxPerf,
    usesim, PI); a user key replaces the equivalent default key. A ready
    InstanceSpaceOptions is returned unchanged.
    """
    if isinstance(options, InstanceSpaceOptions):
        return options
    merged = copy.deepcopy(DEFAULT_OPTIONS)
    for group, values in (options or {}).items():
        base = merged.get(group)
        if isinstance(values, dict) and isinstance(base, dict):
            new = {_key(k) for k in values}
            base = {k: v for k, v in base.items() if _key(k) not in new}
            merged[group] = {**base, **values}
        else:
            merged[group] = copy.deepcopy(values)
    return InstanceSpaceOptions.from_dict(merged)


def _read_metadata(path):
    """from_csv_file returns None on error and only logs; here the error becomes an exception."""
    errors = []
    sink = logger.add(lambda m: errors.append(m.record["message"]), level="ERROR")
    try:
        meta = is_metadata.from_csv_file(path)
    finally:
        logger.remove(sink)
    if meta is None:
        raise ValueError(f"invalid metadata ({path}): " + (" | ".join(errors) or "no detail"))
    return meta


def _read_auxiliary(metadata_path):
    """Validate the auxiliary files next to the metadata; return ({name: path}
    of those present, declared types). An error becomes ValueError before the
    pipeline runs."""
    folder = Path(metadata_path).parent
    present = {n: folder / n for n in AUXILIARY if (folder / n).is_file()}
    for name, required in AUXILIARY.items():
        if name in present and required is not None:
            cols = list(pd.read_csv(present[name], nrows=0).columns)
            missing = [c for c in required if c not in cols]
            if missing:
                raise ValueError(f"{name}: missing columns {missing}")
    types = {}
    if "annotations.json" in present:
        try:
            types = json.loads(present["annotations.json"].read_text())
        except ValueError as exc:
            raise ValueError(f"invalid annotations.json: {exc}") from exc
        errors = declared_type_errors(pd.read_csv(metadata_path), types)
        if errors:
            raise ValueError("annotations.json: " + "; ".join(errors))
    return present, types


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, (tuple, set)):
        return list(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


# --------------------------------------------------------------------------- #
# TRACE robustness
# --------------------------------------------------------------------------- #
def _close_pairs(z, threshold):
    """Pairs (i, j) with Euclidean distance < threshold, and the distances."""
    pairs = cKDTree(z).query_pairs(r=threshold, output_type="ndarray")
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=int), np.empty(0)
    d = np.linalg.norm(z[pairs[:, 0]] - z[pairs[:, 1]], axis=1)
    return pairs[d < threshold], d[d < threshold]


def _min_distance(z):
    d, _ = cKDTree(z).query(z, k=2)
    return float(np.min(d[:, 1]))


def _fix_near_duplicates(z, labels, apply):
    """Count the near-duplicate pairs of z and, if `apply`, separate them.

    IDENTICAL points are not perturbed: TRACE merges them with np.unique before
    the alpha shape and they do not cause the degeneracy. Separating them
    creates exactly distinct positions ~1e-7 apart, and in blood-transfusion
    (822 identical pairs) that zeroed the six good footprints. The jitter is
    for DISTINCT positions closer than the threshold; the identical points of
    a position get the same shift and remain identical.

    Returns (corrected z or None, record for run_info["trace_robustness"]).
    """
    z = np.asarray(z, dtype=float)
    pairs, d = _close_pairs(z, NEAR_DUPLICATE_THRESHOLD)
    positions, inv = np.unique(z, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    position_pairs, _ = _close_pairs(positions, NEAR_DUPLICATE_THRESHOLD)
    record = {
        "threshold": NEAR_DUPLICATE_THRESHOLD,
        "near_duplicate_pairs": int(len(pairs)),
        "identical_pairs": int(np.sum(d == 0)),
        "distinct_close_pairs": int(len(position_pairs)),
        "min_distance_distinct_before": (_min_distance(positions)
                                         if len(positions) > 1 else None),
        "rule": "jitter only on distinct positions closer than the threshold; identical "
                "points are not separated (TRACE merges them with np.unique)",
        "jitter_applied": False,
    }
    if len(position_pairs) == 0:
        return None, record
    if not apply:
        record["reason_no_jitter"] = "fix_near_duplicates=False"
        return None, record

    idx = np.unique(position_pairs.ravel())
    rng = np.random.default_rng(JITTER_SEED)
    shift = np.zeros_like(positions)
    shift[idx] = rng.normal(scale=JITTER_SCALE, size=(idx.size, z.shape[1]))
    z_fixed = z + shift[inv]
    points = np.flatnonzero(np.isin(inv, idx))
    pairs_after, _ = _close_pairs(positions + shift, NEAR_DUPLICATE_THRESHOLD)
    record.update({
        "jitter_applied": True,
        "jitter_scale": JITTER_SCALE,
        "jitter_seed": JITTER_SEED,
        "jitter_distribution": "normal(0, scale) on each coordinate, one draw per position",
        "perturbed_positions": int(idx.size),
        "perturbed_points": int(points.size),
        "perturbed_labels": [str(labels[i]) for i in points],
        "max_shift": float(np.max(np.linalg.norm(shift[idx], axis=1))),
        "min_distance_distinct_after": _min_distance(positions + shift),
        "distinct_close_pairs_after": int(len(pairs_after)),
        "applied_to": "TRACE only, via run_stage(TraceStage, z=...); coordinates.csv keeps "
                      "the PILOT z and the perturbed z goes to coordinates_trace.csv; PYTHIA "
                      "used the PILOT z and CLOISTER does not use z",
    })
    return z_fixed, record


# --------------------------------------------------------------------------- #
# sifted_report
# --------------------------------------------------------------------------- #
def _correlation_survivors(rho, pval, opts_sifted):
    """Reproduces SiftedStage.select_features_by_performance (sifted.py:774-808).

    Keeps the feature most correlated with each algorithm and every feature
    with |rho| >= sifted.rho and p <= sifted.pval for some algorithm.
    """
    filtered = np.abs(rho)
    filtered[np.isnan(rho) | (pval > opts_sifted.pval)] = 0
    ordered = np.sort(filtered, axis=0)[::-1, :]
    row = np.argsort(-filtered, axis=0)
    keep = np.zeros(rho.shape[0], dtype=bool)
    keep[np.unique(row[0, :])] = True
    keep[np.unique(row[ordered >= opts_sifted.rho])] = True
    return np.where(keep)[0]


def _sifted_report(model, feats_pre, feats_input, opts):
    """One row per input feature; returns (DataFrame, warnings)."""
    s = model.sifted
    algos = list(model.data.algo_labels)
    selvars = [int(i) for i in np.asarray(s.selvars).ravel()]
    warns = []
    if [feats_pre[i] for i in selvars] != list(model.data.feat_labels):
        warns.append("sifted_report: selvars does not reproduce Model.data.feat_labels")

    rho = None if s.rho is None else np.asarray(s.rho, dtype=float)
    pval = None if s.pval is None else np.asarray(s.pval, dtype=float)
    if rho is not None and rho.shape[0] == len(feats_pre) and pval is not None:
        survivors = [int(i) for i in _correlation_survivors(rho, pval, opts.sifted)]
    else:
        survivors = list(range(len(feats_pre)))
        if rho is not None:
            warns.append("sifted_report: rho/pval without one row per feature; "
                         "correlation filter not reconstructed")
    if not set(selvars) <= set(survivors):
        warns.append("sifted_report: kept feature outside the correlation survivors")

    cluster_of = {}
    if s.clust is not None:
        clust = np.asarray(s.clust, dtype=bool)
        if clust.shape[0] != len(survivors):
            warns.append(f"sifted_report: clust has {clust.shape[0]} rows and the "
                         f"correlation reconstruction gave {len(survivors)} features")
        else:
            for pos, i in enumerate(survivors):
                cols = np.flatnonzero(clust[pos])
                if cols.size == 1:
                    cluster_of[i] = int(cols[0]) + 1
    kept_in_cluster = {}
    for i in selvars:
        if i in cluster_of:
            kept_in_cluster.setdefault(cluster_of[i], []).append(feats_pre[i])
    for c, fs in kept_in_cluster.items():
        if len(fs) != 1:
            warns.append(f"sifted_report: cluster {c} with {len(fs)} kept features")

    rows = []
    for f in feats_input:
        rec = {"feature": f, "status": None, "rho": np.nan, "rho_algo": None,
               "pval": np.nan, "n_algos_sig": np.nan, "cluster": None, "kept_instead": None}
        if f not in feats_pre:
            rec["status"] = STATUS_PREPROCESSING
            rows.append(rec)
            continue
        i = feats_pre.index(f)
        if rho is not None and rho.shape[0] == len(feats_pre) and np.isfinite(rho[i]).any():
            j = int(np.nanargmax(np.abs(rho[i])))
            rec["rho"], rec["rho_algo"] = float(rho[i, j]), algos[j]
            if pval is not None:
                rec["pval"] = float(pval[i, j])
                sig = (np.abs(rho[i]) >= opts.sifted.rho) & (pval[i] <= opts.sifted.pval)
                rec["n_algos_sig"] = int(np.sum(sig))
        rec["cluster"] = cluster_of.get(i)
        if i in selvars:
            rec["status"] = STATUS_KEPT
        elif i not in survivors:
            rec["status"] = STATUS_CORRELATION
        elif i in cluster_of:
            rec["status"] = STATUS_REDUNDANCY
            rec["kept_instead"] = ";".join(kept_in_cluster.get(cluster_of[i], [])) or None
        else:
            rec["status"] = STATUS_UNDETERMINED
            warns.append(f"sifted_report: reason for dropping {f} not reconstructed")
        rows.append(rec)
    df = pd.DataFrame(rows)
    df["cluster"] = df["cluster"].astype("Int64")
    df["n_algos_sig"] = df["n_algos_sig"].astype("Int64")
    return df, warns


# --------------------------------------------------------------------------- #
# per-instance files
# --------------------------------------------------------------------------- #
def _good_rule(perf):
    """Text of the PRELIM y_bin rule (prelim.py:120-170) for these options."""
    eps = perf.epsilon
    if perf.max_perf:
        return (f"good = algo_* >= {eps}" if perf.abs_perf
                else f"good = 1 - algo_*/best <= {eps}")
    return (f"good = algo_* <= {eps}" if perf.abs_perf
            else f"good = algo_*/best - 1 <= {eps}")


def _per_instance(values, labels, columns):
    """Same layout as instancespace's _write_array_to_csv: index Row = label."""
    return pd.DataFrame(np.asarray(values), columns=columns,
                        index=pd.Index([str(x) for x in labels], name="Row"))


def _write_coordinates(outdir, labels, z_pilot, z_trace):
    """coordinates.csv with the PILOT z; coordinates_trace.csv only if jitter was applied."""
    cols = [f"z_{i}" for i in range(1, z_pilot.shape[1] + 1)]
    _per_instance(z_pilot, labels, cols).to_csv(outdir / "coordinates.csv")
    if z_trace is not None:
        _per_instance(z_trace, labels, cols).to_csv(outdir / "coordinates_trace.csv")


def _write_pythia(outdir, model, algos, labels):
    """pythia_proba.csv, pythia_confusion.csv and pythia_selection.csv.

    Returns (dict for run_info["pythia"], warnings)."""
    p = model.pythia
    sub = np.asarray(p.pr0_sub, dtype=float)
    hat = np.asarray(p.pr0_hat, dtype=float)
    pd.concat([
        _per_instance(sub, labels, algos),
        _per_instance(hat, labels, [f"{a}{HAT_SUFFIX}" for a in algos]),
    ], axis=1).to_csv(outdir / "pythia_proba.csv")

    # training cvcmat (pythia.py:861-868): [tn, fp, fn, tp] of y_bin against
    # y_sub. The evaluation path (pythia.py:515) uses another order, so the
    # order is checked against the Model's own accuracy/precision/recall.
    cm = np.asarray(p.cvcmat, dtype=float)
    conf = pd.DataFrame(cm, columns=["tn", "fp", "fn", "tp"],
                        index=pd.Index(algos, name="Algorithm")).astype(int)
    warns = []
    n = conf.sum(axis=1).replace(0, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        computed = {
            "accuracy": ((conf.tp + conf.tn) / n).to_numpy(),
            "precision": (conf.tp / (conf.tp + conf.fp)).to_numpy(),
            "recall": (conf.tp / (conf.tp + conf.fn)).to_numpy(),
        }
    for name, values in computed.items():
        ref = np.asarray(getattr(p, name), dtype=float)
        ok = np.isclose(values, ref, equal_nan=True) | (np.isnan(values) & (ref == 0))
        if not ok.all():
            warns.append(f"pythia_confusion: {name} recomputed from the matrix does not "
                         f"match Model.pythia.{name}")
    conf.to_csv(outdir / "pythia_confusion.csv")

    def name_of(i):
        return algos[i] if 0 <= int(i) < len(algos) else None

    sel0 = np.asarray(p.selection0).ravel()
    sel1 = np.asarray(p.selection1).ravel()
    pd.DataFrame({"selection0": [name_of(i) for i in sel0], "selection1": [name_of(i) for i in sel1]},
                 index=pd.Index([str(x) for x in labels], name="Row")
                 ).to_csv(outdir / "pythia_selection.csv")

    y_hat = np.asarray(p.y_hat, dtype=bool)
    info = {
        "instance_algorithm_pairs": int(y_hat.size),
        "y_hat_disagrees_with_pr0_hat": int(np.sum(y_hat != (hat < 0.5))),
        "selection0_none": int(np.sum(sel0 < 0)),
        "selection1_differs_from_selection0": int(np.sum(sel0 != sel1)),
    }
    return info, warns


def _write_pilot_r2(outdir, model):
    """R^2 of each column of [x, y] against the reconstruction z B' (pilot.py:478)."""
    feats = [str(f) for f in model.data.feat_labels]
    algos = [str(a) for a in model.data.algo_labels]
    r2 = np.asarray(model.pilot.r2, dtype=float).ravel()
    warns = []
    if len(r2) != len(feats) + len(algos):
        warns.append(f"pilot_r2: {len(r2)} values for {len(feats)} features + {len(algos)} algorithms")
    names = (feats + algos)[:len(r2)]
    kinds = (["feature"] * len(feats) + ["algorithm"] * len(algos))[:len(r2)]
    pd.DataFrame({"variable": names, "kind": kinds, "r2": r2[:len(names)]}).to_csv(
        outdir / "pilot_r2.csv", index=False)
    return warns


def _write_sifted_extras(outdir, model, feats_pre, opts):
    """sifted_correlations.csv (long format) and sifted_silhouette.csv."""
    s = model.sifted
    algos = [str(a) for a in model.data.algo_labels]
    warns = []
    rows = []
    if s.rho is not None:
        rho = np.asarray(s.rho, dtype=float)
        pval = None if s.pval is None else np.asarray(s.pval, dtype=float)
        if rho.shape != (len(feats_pre), len(algos)):
            warns.append(f"sifted_correlations: rho {rho.shape} is not features x algorithms")
        else:
            for i, f in enumerate(feats_pre):
                for j, a in enumerate(algos):
                    rows.append((f, a, rho[i, j], np.nan if pval is None else pval[i, j]))
    pd.DataFrame(rows, columns=["feature", "algorithm", "rho", "pval"]).to_csv(
        outdir / "sifted_correlations.csv", index=False)

    sil = pd.DataFrame(columns=["k", "silhouette", "used", "best"])
    if s.silhouette_scores:
        v = np.asarray(s.silhouette_scores, dtype=float)
        ks = np.arange(SILHOUETTE_K_MIN, SILHOUETTE_K_MIN + len(v))
        sil = pd.DataFrame({"k": ks, "silhouette": v, "used": ks == opts.sifted.k,
                            "best": np.arange(len(v)) == int(np.nanargmax(v))})
        if s.clust is not None and len(v) != np.asarray(s.clust).shape[0] - SILHOUETTE_K_MIN:
            warns.append("sifted_silhouette: number of k tried does not match clust")
        if opts.sifted.k not in ks:
            warns.append(f"sifted_silhouette: k used ({opts.sifted.k}) outside the k tried")
    sil.to_csv(outdir / "sifted_silhouette.csv", index=False)
    return warns


def _write_special_footprint(outdir, name, fp, space):
    """footprint_<name>.csv in the schema of the others (only if not empty) and metrics."""
    from instancespace._serialisers import _footprint_boundary_frame

    polygon = None if fp is None else fp.polygon
    record = {
        "file": None,
        "area": float(getattr(fp, "area", 0) or 0),
        "density": float(getattr(fp, "density", 0) or 0),
        "purity": float(getattr(fp, "purity", 0) or 0),
        "elements": int(getattr(fp, "elements", 0) or 0),
        "good_elements": int(getattr(fp, "good_elements", 0) or 0),
    }
    if space is not None:
        record["normalized_area"] = record["area"] / space["area"] if space["area"] else 0.0
        record["normalized_density"] = (record["density"] / space["density"]
                                        if space["density"] else 0.0)
    if polygon is not None and hasattr(polygon, "is_empty") and not polygon.is_empty:
        _footprint_boundary_frame(polygon).to_csv(outdir / f"footprint_{name}.csv", index=False)
        record["file"] = f"footprint_{name}.csv"
    return record


# --------------------------------------------------------------------------- #
# orientation
# --------------------------------------------------------------------------- #
def rotation_matrix(theta):
    """Proper 2D rotation (det = +1) by `theta` radians, counterclockwise."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def rotate(z, rot):
    """Rows of z (n, 2) rotated by `rot`, element by element: the same point
    gives bit-identical results in every file (a matrix product could round
    differently depending on the array)."""
    z = np.asarray(z, dtype=float)
    return np.column_stack([rot[0, 0] * z[:, 0] + rot[0, 1] * z[:, 1],
                            rot[1, 0] * z[:, 0] + rot[1, 1] * z[:, 1]])


def orientation(z, y_bin):
    """Rotation of pyispace's adjust_rotation on generic metadata.

    pyispace (train.py:143-150, pilot.py:97-103): an instance is "bad" when the
    mode of its row of Ybin is 0, i.e. at least half of the algorithms are bad
    (scipy.stats.mode breaks a tie towards 0); the centroid of those instances,
    seen from the origin, is rotated to 135 degrees. Here the same rule is
    written with the number of bad algorithms, n_bad = n_algos - good_algos:
    bad instance <=> n_bad >= n_algos / 2. Differences: no rotation when every
    instance is bad (the centroid of all instances is the origin, the direction
    is undefined; pyispace would rotate by noise), and the strength of the
    gradient is measured (R^2 of n_bad on z_1, z_2, a linear regression, which
    does not depend on the rotation).

    Returns (2x2 rotation or None, record for run_info["orientation"]).
    """
    z = np.asarray(z, dtype=float)
    good = np.asarray(y_bin, dtype=bool)
    n_algos = good.shape[1]
    n_bad = n_algos - good.sum(axis=1)
    bad = n_bad >= n_algos / 2
    design = np.column_stack([np.ones(len(z)), z])
    coef, *_ = np.linalg.lstsq(design, n_bad.astype(float), rcond=None)
    resid = n_bad - design @ coef
    total = ((n_bad - n_bad.mean()) ** 2).sum()
    r2 = float(1 - (resid ** 2).sum() / total) if total > 0 else 0.0
    record = {
        "enabled": True,
        "applied": False,
        "convention": "pyispace adjust_rotation: the centroid of the instances where most "
                      "algorithms are bad goes to 135 degrees (top left); proper rotation "
                      "about the origin, no reflection",
        "bad_instance_rule": "n_bad_algos >= n_algos / 2 (mode of the good/bad row is bad; "
                             "a tie counts as bad)",
        "target_angle_deg": TARGET_ANGLE_DEG,
        "n_instances_bad": int(bad.sum()),
        "gradient_r2": r2,
        "weak_gradient_r2_threshold": WEAK_GRADIENT_R2,
        "weak_gradient": r2 < WEAK_GRADIENT_R2,
        "warning": None,
    }
    if not bad.any() or bad.all():
        record["reason_not_applied"] = ("no instance has most algorithms bad" if not bad.any()
                                        else "every instance has most algorithms bad")
        return None, record
    centroid = z[bad].mean(axis=0)
    theta = np.radians(TARGET_ANGLE_DEG) - np.arctan2(centroid[1], centroid[0])
    theta = float(np.arctan2(np.sin(theta), np.cos(theta)))       # in (-pi, pi]
    rot = rotation_matrix(theta)
    grad = rot @ coef[1:]
    rms = float(np.sqrt((z ** 2).sum(axis=1).mean()))
    record.update({
        "applied": True,
        "angle_deg": float(np.degrees(theta)),
        "matrix": rot.tolist(),
        "determinant": float(np.linalg.det(rot)),
        "centroid_bad_before": centroid.tolist(),
        "centroid_bad_after": (rot @ centroid).tolist(),
        "centroid_bad_distance_over_rms": float(np.linalg.norm(centroid) / rms) if rms else None,
        "gradient_direction_after_deg": float(np.degrees(np.arctan2(grad[1], grad[0]))),
    })
    if record["weak_gradient"]:
        record["warning"] = (f"weak difficulty gradient: the number of bad algorithms depends "
                             f"little on the position in the plane (linear R^2 = {r2:.2f} < "
                             f"{WEAK_GRADIENT_R2}); the hard region is spread and the top-left "
                             "convention says little for this dataset")
    return rot, record


def _rotate_outputs(outdir, rot, projection=None):
    """Rewrite the z_1, z_2 columns of the geometric outputs rotated by `rot`,
    and projection_matrix.csv as rot @ A (A at full precision from the Model
    when given, rounded to 4 decimals like instancespace's save_to_csv)."""
    files = [outdir / f for f in ROTATED_FILES if (outdir / f).is_file()]
    for pattern in ROTATED_PATTERNS:
        files += sorted(outdir.glob(pattern))
    for path in files:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        z = df[["z_1", "z_2"]].astype(float).to_numpy()
        zr = rotate(z, rot)
        df["z_1"], df["z_2"] = zr[:, 0], zr[:, 1]
        df.to_csv(path, index=False)
    proj_path = outdir / "projection_matrix.csv"
    if proj_path.is_file():
        proj = pd.read_csv(proj_path, dtype={"Row": str}).set_index("Row")
        a = proj.to_numpy(dtype=float)
        if projection is not None:
            full = np.asarray(projection, dtype=float)
            # use the full-precision A only if it is the one written (same shape
            # and order): the file is it rounded to 4 decimals
            if full.shape == a.shape and np.allclose(np.round(full, 4), a, atol=1e-12):
                a = full
        proj.loc[:, :] = np.round(rot @ a, 4)
        proj.reset_index().to_csv(proj_path, index=False)
    return [p.name for p in files] + (["projection_matrix.csv"] if proj_path.is_file() else [])


# --------------------------------------------------------------------------- #
# output folder
# --------------------------------------------------------------------------- #
def _engine_files(outdir):
    owned = [p for p in outdir.iterdir() if p.name in SAVE_TO_CSV_FILES + EXTRA_FILES]
    for pattern in FOOTPRINT_PATTERNS:
        owned += list(outdir.glob(pattern))
    return owned


def _check_folder(outdir, metadata_path):
    """Before running, refuse the input metadata's folder and a folder with
    files of the same names but no run_info.json from the engine (output of
    another tool, e.g. resultados/isa/<name>/ from pyispace)."""
    if not outdir.exists():
        return
    if not outdir.is_dir():
        raise NotADirectoryError(f"{outdir} exists and is not a folder")
    target_meta = outdir / "metadata.csv"
    if target_meta.exists() and target_meta.samefile(metadata_path):
        raise ValueError(f"outdir {outdir} is the input metadata's folder; use another folder")
    if not _engine_files(outdir):
        return
    try:
        owner = json.loads((outdir / "run_info.json").read_text()).get("generated_by") == GENERATED_BY
    except (OSError, ValueError):
        owner = False
    if not owner:
        raise ValueError(f"{outdir} has files that did not come from {GENERATED_BY}; "
                         "use a new folder")


def _clean_folder(outdir):
    """Create the folder and delete only the files the engine writes (a
    footprint that is empty in this run must not remain from the previous one)."""
    outdir.mkdir(parents=True, exist_ok=True)
    for p in _engine_files(outdir):
        p.unlink()


def _footprint_files(outdir, algo_labels):
    """{algo: {"good": file name or None, "best": ...}} as written.

    save_to_csv names the files with _portable_stems (invalid characters
    become "_", long names are truncated); the loader uses this map instead of
    guessing.
    """
    try:
        from instancespace._serialisers import _portable_stems
        stems = _portable_stems(list(algo_labels), "algorithm")
    except ImportError:
        stems = list(algo_labels)
    out = {}
    for algo, stem in zip(algo_labels, stems):
        out[str(algo)] = {
            kind: (f"footprint_{stem}_{kind}.csv"
                   if (outdir / f"footprint_{stem}_{kind}.csv").is_file() else None)
            for kind in ("good", "best")
        }
    return out


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def run_instancespace(metadata_path, outdir, options=None, progress=None, *,
                      fix_near_duplicates=True, orient=True):
    """Run instancespace on `metadata_path` and write the folder `outdir`.

    Parameters
    ----------
    metadata_path : metadata.csv with ``instances``, optional ``source``,
        ``feature_*``, ``algo_*`` and any other columns (annotations, which
        instancespace ignores and the copy in outdir/metadata.csv keeps).
    outdir : output folder; created if needed. In a folder from a previous
        engine run only the engine's files are deleted; a folder with files
        from another origin is refused.
    options : dict per group overriding DEFAULT_OPTIONS (see build_options)
        or a ready InstanceSpaceOptions.
    progress : callable(stage_name) called BEFORE each stage, with
        "PREPROCESSING", "PRELIM", "SIFTED", "PILOT", "PYTHIA", "CLOISTER" and
        "TRACE", in that order.
    fix_near_duplicates : if False, only count the near-duplicate pairs of the
        projection and do not apply the jitter (for comparison).
    orient : standard orientation (see "Orientation" in the module docstring),
        applied after the whole pipeline; False keeps PILOT's orientation.

    Returns the dict written to run_info.json. A failing stage becomes a
    RuntimeError naming the stage; the output folder is only touched after
    every stage has finished.
    """
    metadata_path = Path(metadata_path)
    outdir = Path(outdir)
    t_start = time.perf_counter()
    _check_folder(outdir, metadata_path)
    opts = build_options(options)
    meta = _read_metadata(metadata_path)
    auxiliary, declared_types = _read_auxiliary(metadata_path)
    feats_input = [str(f) for f in meta.feature_names]
    algo_names = [str(a) for a in meta.algorithm_names]
    clash = sorted(a for a in algo_names if a.endswith(HAT_SUFFIX) and a[:-len(HAT_SUFFIX)] in algo_names)
    if clash:
        raise ValueError(f"algorithms {clash} clash with the <algo>{HAT_SUFFIX} columns of "
                         "pythia_proba.csv; rename them in the metadata")

    timings, is_warnings, robustness = {}, [], {}
    feats_pre = list(feats_input)
    labels_z = [str(x) for x in meta.instance_labels]
    sink = logger.add(lambda m: is_warnings.append(m.record["message"]), level="WARNING")
    isp = InstanceSpace(meta, opts)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            z_fixed = z_pilot = None
            for name, stage in STAGES:
                if progress is not None:
                    progress(name)
                extra = {"z": z_fixed} if name == "TRACE" and z_fixed is not None else {}
                t0 = time.perf_counter()
                try:
                    out = isp.run_stage(stage, **extra)
                except Exception as exc:
                    raise RuntimeError(f"instancespace failed in stage {name}: {exc}") from exc
                timings[name] = round(time.perf_counter() - t0, 3)
                if name == "PREPROCESSING":
                    feats_pre = [str(f) for f in out.feat_labels]
                elif name == "SIFTED":
                    # labels aligned with the rows of x (and of the PILOT z)
                    labels_z = [str(x) for x in out.inst_labels]
                elif name == "PILOT":
                    z_pilot = np.array(out.z, dtype=float)
                    t0 = time.perf_counter()
                    z_fixed, robustness = _fix_near_duplicates(out.z, labels_z, fix_near_duplicates)
                    timings["near_duplicate_check"] = round(time.perf_counter() - t0, 3)
            model = isp.model
    finally:
        logger.remove(sink)
        isp.close()
    timings["build_total"] = round(time.perf_counter() - t_start, 3)

    # ------------------------------------------------------------- writing
    t0 = time.perf_counter()
    _check_folder(outdir, metadata_path)
    _clean_folder(outdir)
    model.save_to_csv(outdir)
    algos = [str(a) for a in model.data.algo_labels]
    labels = [str(x) for x in model.data.inst_labels]
    _write_coordinates(outdir, labels, z_pilot, z_fixed)
    shutil.copyfile(metadata_path, outdir / "metadata.csv")
    for name, source in auxiliary.items():
        shutil.copyfile(source, outdir / name)
    (outdir / "run_options.json").write_text(
        json.dumps(dataclasses.asdict(opts), indent=2, default=_json_default)
    )
    report, warns = _sifted_report(model, feats_pre, feats_input, opts)
    report.to_csv(outdir / "sifted_report.csv", index=False)
    warns += _write_sifted_extras(outdir, model, feats_pre, opts)
    warns += _write_pilot_r2(outdir, model)
    pythia_info, pythia_warns = _write_pythia(outdir, model, algos, labels)
    warns += pythia_warns
    space = _write_special_footprint(outdir, "space", model.trace.space, None)
    special = {
        "space": space,
        "hard": _write_special_footprint(outdir, "hard", model.trace.hard, space),
    }
    # orientation: after everything, only on the geometric outputs
    coords = pd.read_csv(outdir / "coordinates.csv", dtype={"Row": str}).set_index("Row")
    good_bin = pd.read_csv(outdir / "algorithm_bin.csv", dtype=str).set_index("Row")
    good_bin = good_bin.reindex(coords.index).apply(lambda s: s.str.strip().str.lower() == "true")
    rot, orient_info = orientation(coords[["z_1", "z_2"]].to_numpy(), good_bin.to_numpy())
    orient_info["enabled"] = bool(orient)
    if orient and rot is not None:
        pilot_a = getattr(model.pilot, "a", None)
        orient_info["rotated_files"] = _rotate_outputs(outdir, rot, pilot_a)
    else:
        orient_info.update({"applied": False, "rotated_files": []})
        if not orient:
            orient_info["reason_not_applied"] = "disabled (orient=False)"
    timings["writing"] = round(time.perf_counter() - t0, 3)

    counts = Counter((w.category.__name__, str(w.message)) for w in caught)
    info = {
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "instancespace_version": version("instancespace"),
        "python": platform.python_version(),
        "input_metadata": str(metadata_path.resolve()),
        "n_instances_input": int(len(meta.instance_labels)),
        "n_instances": int(len(model.data.inst_labels)),
        "n_features_input": len(feats_input),
        "n_features_selected": len(model.data.feat_labels),
        "algorithms": [str(a) for a in model.data.algo_labels],
        "has_source": meta.instance_sources is not None,
        "annotation_types": {
            "file": "annotations.json" if "annotations.json" in auxiliary else None,
            "declared": declared_types,
        },
        "auxiliary_files": sorted(auxiliary),
        "timings_s": timings,
        "trace_robustness": robustness,
        "footprint_files": _footprint_files(outdir, model.data.algo_labels),
        "special_footprints": special,
        "pythia": pythia_info,
        "good_rule": _good_rule(opts.perf),
        "orientation": orient_info,
        "files": sorted({p.name for p in _engine_files(outdir)} | {"run_info.json"}),
        "warnings": warns,
        "instancespace_warnings": list(dict.fromkeys(is_warnings)),
        "python_warnings": [
            {"category": c, "message": m, "n": n} for (c, m), n in counts.most_common()
        ],
    }
    (outdir / "run_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False, default=_json_default)
    )
    return info


def main(argv=None):
    """Command line used by the UI (isaspace.ui.runner) to run the engine in a
    subprocess. On standard output, one line "@@isa stage <NAME>" before each
    stage and, at the end, "@@isa ok <outdir>" or "@@isa error <message>"
    (exit code 1); everything else is log.

    python -m isaspace.engine --metadata M.csv --outdir FOLDER [--options JSON] [--no-orient]
    """
    parser = argparse.ArgumentParser(description="Run instancespace on a metadata.csv")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--options", default="{}", help="JSON dict overriding DEFAULT_OPTIONS")
    parser.add_argument("--no-orient", action="store_true",
                        help="keep PILOT's orientation (no standard rotation)")
    args = parser.parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    def signal(text):
        print(f"{PREFIX} {text}", flush=True)

    try:
        run_instancespace(args.metadata, args.outdir, options=json.loads(args.options),
                          progress=lambda stage: signal(f"stage {stage}"),
                          orient=not args.no_orient)
    except Exception as exc:  # noqa: BLE001 -- the message goes to the UI
        traceback.print_exc()
        signal("error " + f"{type(exc).__name__}: {exc}".replace("\n", " "))
        return 1
    signal(f"ok {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
