"""Bridge between the project's per-instance table and pyispace (PILOT + TRACE).

Two public functions:

- ``to_isa_metadata``: converts ``resultados/table_<name>.csv`` into the format
  ``pyispace.train_is`` expects: numeric ``feature_*`` and ``algo_*`` columns
  and an ``instances`` index with the labels "1".."n" as text (the same layout
  as pyhard's ``Workspace``). It carries ``row_original`` and the annotations
  ``class``, ``ih`` and ``n_wrong``, which pyispace ignores and instancespace
  (``isaspace.engine``) keeps as annotations. With ``outdir``, it writes the
  metadata.csv and, next to it, the auxiliary files the engine copies
  (``write_metadata``: annotations.json, degenerate_report.csv,
  feature_info.csv; format in docs/output_format.md).
- ``run_isa``: builds the options, runs ``train_is`` (PILOT + TRACE), writes
  the CSVs in pyhard's layout via ``pyispace.utils.scriptcsv`` and checks that
  the meaning of "good" was not inverted.

Requires pyispace patched for Python 3.11 (``scripts/apply_pyispace_patch.py``).
pyispace 0.3.7 has no SIFTED, CLOISTER or PYTHIA: every ``feature_*`` that
survives ``to_isa_metadata`` goes into PILOT.
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import iqr as _iqr

_FEATURE = "feature_"
_ALGO = "algo_"
_PROBA = "proba_"
INDEX_NAME = "instances"
ROW_ORIGINAL = "row_original"
# columns of the per-instance table that travel in the metadata as
# annotations: they are neither feature_* nor algo_*, so train_is and
# instancespace do not use them
ANNOTATIONS = ("class", "ih", "n_wrong")
# class label: the nominal value of the OpenML target, always text
# (pipeline.build_instance_table writes pd.Series(y).astype(str))
LABEL = "class"
# declared type of each annotation (annotations.json): CSV does not store
# types, and "1"/"2" would come back as numbers
ANNOTATION_TYPES = {
    "row_original": "identifier",       # row index in the OpenML dataset
    "class": "categorical", "ih": "numeric", "n_wrong": "integer",
}
# family of the pyhard measures (feature_info.csv): the ones that depend on a
# fitted model are model_derived; the others, geometric
MODEL_DERIVED = ("CL", "CLD", "DS", "DCP", "TD_U", "TD_P")

# Maximum tolerated difference between pyispace's "good" rate (Ybin) and the
# true accuracy of each algorithm. Above it, perf.MaxPerf is most likely inverted.
YBIN_TOLERANCE = 0.15


def _import_pyispace():
    """Import pyispace with a helpful message if the 3.11 patch was not applied."""
    try:
        import pyispace  # noqa: F401
        from pyispace import preprocessing, utils
        from pyispace.train import train_is
    except (ValueError, ImportError) as exc:
        raise ImportError(
            "pyispace does not import on this Python (mutable defaults in "
            "pyispace/train.py). Run: python scripts/apply_pyispace_patch.py"
        ) from exc
    return train_is, preprocessing, utils


# --------------------------------------------------------------------------- #
# to_isa_metadata
# --------------------------------------------------------------------------- #
def _split_degenerate(F, min_var):
    """Separate the features that pyispace's preprocessing turns constant.

    Reproduces exactly what ``train_is`` does with ``auto.preproc=True``
    (train.py:128-134): ``bound_outliers`` (clipping at median +- 5*IQR)
    followed by ``auto_normalize`` (Yeo-Johnson + z-score). A column with
    IQR = 0 is collapsed to a constant by the clipping; after that the z-score
    leaves zero variance, PILOT returns R2 = NaN for it and the column only
    hurts the fit. Returns (kept, dropped) with the reason for each drop and
    the measured variances.
    """
    _, preprocessing, _ = _import_pyispace()
    X = F.to_numpy(dtype=float)
    n_nan = np.isnan(X).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_variance = np.nanvar(X, axis=0)
        iqr = _iqr(X, axis=0, nan_policy="omit")

    # Columns with NaN cannot go through pyispace's preprocessing:
    # bound_outliers uses np.median, which returns NaN, and np.clip with NaN
    # bounds turns the whole column into NaN; PowerTransformer then aborts
    # with scipy BracketError. They are dropped beforehand and left out of it.
    no_nan = np.flatnonzero(n_nan == 0)
    var_after = np.full(X.shape[1], np.nan)
    if no_nan.size:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Xn = preprocessing.auto_normalize(
                preprocessing.bound_outliers(X[:, no_nan])
            )
        var_after[no_nan] = np.nanvar(Xn, axis=0)

    kept, dropped = [], []
    for j, col in enumerate(F.columns):
        degenerate = not (var_after[j] >= min_var)  # NaN counts as degenerate
        if not degenerate:
            kept.append(col)
            continue
        if n_nan[j] == X.shape[0]:
            reason = "all values NaN"
        elif n_nan[j] > 0:
            reason = (
                f"{int(n_nan[j])} NaN: bound_outliers (np.median) spreads NaN "
                "to the whole column"
            )
        elif raw_variance[j] < min_var:
            reason = "raw variance ~0 (constant column)"
        elif iqr[j] == 0:
            reason = (
                "IQR = 0: bound_outliers (median +- 5*IQR) collapses the column "
                "to a constant"
            )
        else:
            reason = f"variance < {min_var:g} after Yeo-Johnson + z-score"
        dropped.append(
            {
                "feature": col,
                "raw_variance": float(raw_variance[j]),
                "iqr": float(iqr[j]),
                "var_after_preproc": float(var_after[j]),
                "reason": reason,
            }
        )
    return kept, dropped


def to_isa_metadata(
    table, drop_degenerate=True, min_var=1e-8, proba_as_performance=True,
    annotations=ANNOTATIONS, outdir=None,
):
    """Convert the per-instance table into the metadata ``train_is`` consumes.

    Parameters
    ----------
    table : DataFrame with ``feature_*``, ``algo_*`` (0/1 hit) and ``proba_*``
        (probability of the true class) columns, as produced by
        ``isaspace.pipeline.build_instance_table``.
    drop_degenerate : drop the ``feature_*`` whose variance falls below
        ``min_var`` after pyispace's own preprocessing (see
        ``_split_degenerate``).
    proba_as_performance : if True, the 0/1 hit ``algo_*`` are dropped and the
        ``proba_*`` become ``algo_<name>`` (continuous performance, higher is
        better: that is what PILOT can fit). If False, keep the original
        ``algo_*`` and drop the ``proba_*``.
    annotations : table columns copied into the metadata as annotations
        (default ``class``, ``ih``, ``n_wrong``). All of them must exist in
        the table. ``class`` goes as text: the nominal OpenML label.
    outdir : if given, write the metadata and the auxiliary files to this
        folder (``write_metadata``).

    Returns
    -------
    (metadata, info)
      metadata : DataFrame with ``row_original`` (original table index), the
          annotations, the kept ``feature_*`` and the ``algo_*``; index
          ``instances`` with the labels "1".."n" as text. Columns that do not
          start with ``feature_`` or ``algo_`` are ignored by ``train_is``
          (train.py:56-57) and treated as annotations by instancespace.
      info : dict with ``features_kept``, ``features_dropped`` (list of dicts
          with feature, raw_variance, iqr, var_after_preproc and reason),
          ``performance_source`` ("proba" or "hit"), ``algos``,
          ``annotations``, ``row_original`` (Series instances -> original
          index, for the interface's join), ``true_accuracy`` (0/1 hit rate
          per algorithm, used by ``run_isa``'s guard), ``annotation_types``
          (declared types, ANNOTATION_TYPES), ``degenerate_report``
          (DataFrame feature, raw_variance, iqr, reason) and ``feature_info``
          (DataFrame feature, family; every measure in the table).
    """
    feature_cols = [c for c in table.columns if c.startswith(_FEATURE)]
    algo_cols = [c for c in table.columns if c.startswith(_ALGO)]
    if not feature_cols:
        raise ValueError("table has no feature_* columns")
    if not algo_cols:
        raise ValueError("table has no algo_* columns")
    annotations = list(annotations)
    missing = [c for c in annotations if c not in table.columns]
    if missing:
        raise ValueError(f"table lacks the annotation columns {missing}")
    reserved = [
        c for c in annotations
        if c.casefold() in (INDEX_NAME, "source", ROW_ORIGINAL)
        or c.casefold().startswith((_FEATURE, _ALGO))
    ]
    if reserved:
        raise ValueError(f"reserved name used as an annotation: {reserved}")

    F = table[feature_cols].astype(float)
    if drop_degenerate:
        kept, dropped = _split_degenerate(F, min_var)
    else:
        kept, dropped = feature_cols, []
    if len(kept) < 2:
        raise ValueError(
            f"only {len(kept)} feature(s) survived the filter; PILOT needs "
            "at least 2"
        )

    names = [c[len(_ALGO):] for c in algo_cols]
    if proba_as_performance:
        lacking = [n for n in names if f"{_PROBA}{n}" not in table.columns]
        if lacking:
            raise ValueError(f"no proba_* column for: {lacking}")
        Y = table[[f"{_PROBA}{n}" for n in names]].astype(float)
        Y.columns = [f"{_ALGO}{n}" for n in names]
        source = "proba"
    else:
        Y = table[algo_cols].astype(float)
        source = "hit"

    n = len(table)
    # text labels; the values "1".."n" are the same as the old RangeIndex, so
    # the Row 1..n of pyispace's coordinates.csv still matches
    new_index = pd.Index(
        [str(i) for i in range(1, n + 1)], name=INDEX_NAME, dtype=object
    )
    row_original = pd.Series(
        table.index.to_numpy(), index=new_index, name=ROW_ORIGINAL
    )

    ann = table[annotations].copy()
    if LABEL in ann.columns:
        # the table read back from CSV brings the nominal labels "1"/"2"
        # (blood) and "0"/"1" (hill-valley) as integers; back to OpenML's text
        ann[LABEL] = ann[LABEL].astype(str)
    metadata = pd.concat([ann, F[kept], Y], axis=1)
    metadata.index = new_index
    metadata.insert(0, ROW_ORIGINAL, row_original.to_numpy())

    true_accuracy = {n_: float(table[f"{_ALGO}{n_}"].mean()) for n_ in names}
    info = {
        "n_instances": n,
        "features_kept": kept,
        "features_dropped": dropped,
        "performance_source": source,
        "algos": names,
        "annotations": annotations,
        "row_original": row_original,
        "true_accuracy": true_accuracy,
        "annotation_types": {c: ANNOTATION_TYPES[c] for c in [ROW_ORIGINAL, *annotations]
                             if c in ANNOTATION_TYPES},
        "degenerate_report": pd.DataFrame(
            [{"feature": d["feature"][len(_FEATURE):], "raw_variance": d["raw_variance"],
              "iqr": d["iqr"], "reason": d["reason"]} for d in dropped],
            columns=["feature", "raw_variance", "iqr", "reason"],
        ),
        "feature_info": pd.DataFrame({
            "feature": [c[len(_FEATURE):] for c in feature_cols],
            "family": ["model_derived" if c[len(_FEATURE):] in MODEL_DERIVED else "geometric"
                       for c in feature_cols],
        }),
    }
    # copy of the guard data travelling with the DataFrame, so that run_isa
    # works even without receiving the original table
    metadata.attrs["true_accuracy"] = true_accuracy
    if outdir is not None:
        write_metadata(metadata, info, outdir)
    return metadata, info


def write_metadata(metadata, info, outdir):
    """Write metadata.csv and, next to it, the files the engine copies if present.

    - annotations.json: {annotation: "categorical" | "numeric" | "integer" |
      "identifier"};
    - degenerate_report.csv: measures dropped before the engine (feature,
      raw_variance, iqr, reason); header only when none was dropped;
    - feature_info.csv: feature, family of every measure received.
    Returns the paths written.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [outdir / n for n in ("metadata.csv", "annotations.json",
                                  "degenerate_report.csv", "feature_info.csv")]
    metadata.to_csv(paths[0])
    paths[1].write_text(json.dumps(info["annotation_types"], indent=2) + "\n")
    info["degenerate_report"].to_csv(paths[2], index=False)
    info["feature_info"].to_csv(paths[3], index=False)
    return paths


# --------------------------------------------------------------------------- #
# run_isa
# --------------------------------------------------------------------------- #
def _true_accuracy(metadata, table):
    """0/1 hit rate per algorithm, from the original table or metadata.attrs."""
    algos = [c[len(_ALGO):] for c in metadata.columns if c.startswith(_ALGO)]
    if table is not None:
        lacking = [a for a in algos if f"{_ALGO}{a}" not in table.columns]
        if lacking:
            raise ValueError(f"original table has no algo_* for: {lacking}")
        return {a: float(table[f"{_ALGO}{a}"].mean()) for a in algos}
    stored = metadata.attrs.get("true_accuracy")
    if stored is None:
        raise ValueError(
            "run_isa needs the true hit rate for its guard: pass "
            "table=<original table> or use the metadata returned by "
            "to_isa_metadata"
        )
    return {a: float(stored[a]) for a in algos}


def _check_ybin(model, metadata, table, perf_epsilon):
    """Guard: pyispace's 'good' rate must match the true accuracy.

    ``train_is`` removes algorithms without any good instance (train.py:101-107);
    for those the good rate is taken as 0, which also triggers the error.
    """
    true_accuracy = _true_accuracy(metadata, table)
    good_rate = dict(zip(model.data.algolabels, model.data.Ybin.mean(axis=0)))

    rows, problems = [], []
    for algo, real in true_accuracy.items():
        good = float(good_rate.get(algo, 0.0))
        diff = abs(good - real)
        rows.append((algo, good, real, diff, algo in good_rate))
        if diff > YBIN_TOLERANCE:
            problems.append(f"{algo}: good rate={good:.3f} vs accuracy={real:.3f}")

    check = pd.DataFrame(
        rows,
        columns=[
            "algorithm", "good_rate_ybin", "true_accuracy", "difference", "in_model"
        ],
    ).set_index("algorithm")

    if problems:
        raise RuntimeError(
            "'Good' rate (Ybin) differs from the true accuracy by more than "
            f"{YBIN_TOLERANCE}: " + "; ".join(problems) + ". "
            "Classic symptom of an inverted perf.MaxPerf: with MaxPerf=False "
            "pyispace considers 'good' the performance <= epsilon, that is, the "
            "ERROR when algo_* is a hit or a probability (higher is better). "
            f"Check opts['perf'] (MaxPerf must be True) and perf_epsilon={perf_epsilon}."
        )
    return check


def run_isa(metadata, outdir, perf_epsilon=0.5, seed=42, table=None):
    """Run pyispace's PILOT + TRACE and write the CSVs in pyhard's layout.

    Parameters
    ----------
    metadata : output of ``to_isa_metadata`` (or any DataFrame with
        ``feature_*`` and ``algo_*`` where higher is better).
    outdir : output folder; receives ``metadata.csv``, ``options.json`` and
        the files of ``pyispace.utils.scriptcsv`` (coordinates.csv,
        footprint_performance.csv, algorithm_bin.csv, beta_easy.csv,
        good_algos.csv, footprint_<algo>_<good|best>.csv, model.pkl, ...).
    perf_epsilon : an instance is "good" for an algorithm if algo_* >= epsilon.
    seed : seed of PILOT (pilot.py:64) and of the best-algorithm tie break
        (train.py:119, which uses np.random without its own seed).
    table : original table, so the guard can compare the 'good' rate with the
        true accuracy. If omitted, uses ``metadata.attrs['true_accuracy']``.

    Returns the ``pyispace.train.Model``; ``model.ybin_check`` holds the
    guard's comparison table.
    """
    train_is, _, utils = _import_pyispace()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # pyispace 0.3.7 options. Only these keys are read by train_is, pilot and
    # trace; the other keys of pyhard's options.json (parallel, corr, clust,
    # cloister, pythia, selvars, outputs, auto.featsel, trace.usesim) are
    # accepted and ignored.
    opts = {
        "perf": {
            # REQUIRED: with False, "good" becomes performance <= epsilon, that
            # is, the ERROR (train.py:84-96). Our algo_* are hit/probability.
            "MaxPerf": True,
            # absolute threshold (Ybin = Y >= epsilon), not relative to the best
            "AbsPerf": True,
            "epsilon": perf_epsilon,
        },
        # an instance is beta-easy if more than 55% of the algorithms are good
        "general": {"betaThreshold": 0.55},
        # pyispace preprocessing: outlier clipping + Yeo-Johnson/z-score
        "auto": {"preproc": True},
        "bound": {"flag": True},
        "norm": {"flag": True},
        # numeric PILOT (BFGS) with 5 tries and a fixed seed
        "pilot": {"analytic": False, "ntries": 5, "seed": seed},
        # TRACE: minimum purity 0.55. parallel=True breaks: joblib's child
        # processes re-import pyispace (and the patch only exists on local disk).
        "trace": {"PI": 0.55, "parallel": False},
    }

    metadata.to_csv(outdir / "metadata.csv")
    (outdir / "options.json").write_text(json.dumps(opts, indent=2))

    logging.getLogger("pyispace").setLevel(logging.INFO)
    np.random.seed(seed)
    model = train_is(metadata, opts, rotation_adjust=True)

    # the most important guard in this file: before writing any output
    model.ybin_check = _check_ybin(model, metadata, table, perf_epsilon)

    utils.scriptcsv(model, outdir)
    return model
