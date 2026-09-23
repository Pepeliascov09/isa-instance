"""Validation of a metadata file uploaded through the UI, before the engine runs.

Only pandas and numpy (it does not import instancespace). Everything here
returns messages for the screen and never raises to the user:

- validate_metadata: instances column present, without empty or repeated
  labels; at least MIN_FEATURES feature_* and MIN_ALGOS algo_*; feature_* and
  algo_* numeric and finite; NaN reported per column;
- validate_annotations / validate_feature_info: the optional files;
- good_fraction: fraction of "good" instances per algorithm with the
  instancespace PRELIM rule (prelim.py:120-170) for the chosen direction and
  epsilon; degenerate_algorithms flags a threshold that makes almost every
  instance good or almost every instance bad (it does NOT detect a wrong
  direction: with an absolute threshold, flipping the direction turns each
  fraction f into 1 - f, so it fires in both directions or in neither);
- mean_ranking: algorithms ranked by the mean of their algo_* column under the
  chosen direction, so the user can check that the direction makes sense;
- estimated_time: from MEASURED_TIMES (measured, not guessed).
"""

import csv
import io
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from isaspace.ui.loader_is import annotation_columns, declared_type_errors

MIN_FEATURES = 3
MIN_ALGOS = 2
MIN_INSTANCES = 20           # below this PYTHIA's cross-validation is fragile
GOOD_MIN, GOOD_MAX = 0.05, 0.95   # outside this range the epsilon threshold is degenerate
MAX_EXAMPLES = 5
# Engine run times in a subprocess (s), medians of 3 repetitions of
# scripts/measure_engine_time.py (2026-09-22, Apple M5, 10 cores) on synthetic
# metadata with TIMING_FEATURES features and TIMING_ALGOS algorithms:
# {number of instances: (total, PYTHIA only)}. PYTHIA dominates and trains one
# SVM per algorithm; the rest barely depends on the number of algorithms.
MEASURED_TIMES = {500: (9.7, 4.7), 1000: (24.2, 17.7), 2000: (76.6, 66.5)}
TIMING_FEATURES, TIMING_ALGOS = 10, 6
TIME_WARNING_MIN_INSTANCES = 1000


@dataclass
class Validation:
    ok: bool = False
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    n: int = 0
    features: list = field(default_factory=list)
    algos: list = field(default_factory=list)
    annotations: list = field(default_factory=list)
    source: str | None = None
    nan: dict = field(default_factory=dict)      # {column: number of NaN}
    df: pd.DataFrame | None = None


def _examples(values) -> str:
    values = list(values)
    text = ", ".join(repr(v) for v in values[:MAX_EXAMPLES])
    return text + (f" and {len(values) - MAX_EXAMPLES} more" if len(values) > MAX_EXAMPLES else "")


def validate_metadata(content: bytes) -> Validation:
    v = Validation()
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        v.errors.append("The file is not UTF-8: save the CSV as UTF-8 and upload it again.")
        return v
    try:
        header = next(csv.reader(io.StringIO(text)))
    except StopIteration:
        v.errors.append("The file is empty.")
        return v
    except csv.Error as exc:
        v.errors.append(f"Could not read the CSV header: {exc}.")
        return v
    lower = [c.strip().casefold() for c in header]
    repeated = sorted({c for c in header if header.count(c) > 1})
    if repeated:
        v.errors.append(f"Columns with the same name: {_examples(repeated)}.")
    if lower.count("instances") == 0:
        v.errors.append("Missing the **instances** column (the label of each instance).")
    elif lower.count("instances") > 1:
        v.errors.append("There is more than one **instances** column.")
    if lower.count("source") > 1:
        v.errors.append("There is more than one **source** column.")
    features = [c for c, b in zip(header, lower) if b.startswith("feature_")]
    algos = [c for c, b in zip(header, lower) if b.startswith("algo_")]
    if len(features) < MIN_FEATURES:
        v.errors.append(f"At least {MIN_FEATURES} **feature_\\*** columns are needed; "
                        f"the file has {len(features)}.")
    if len(algos) < MIN_ALGOS:
        v.errors.append(f"At least {MIN_ALGOS} **algo_\\*** columns are needed; "
                        f"the file has {len(algos)}.")
    if v.errors:
        return v

    col_inst = header[lower.index("instances")]
    try:
        df = pd.read_csv(io.StringIO(text), dtype={col_inst: str})
    except (pd.errors.ParserError, ValueError) as exc:
        v.errors.append(f"Malformed CSV: {str(exc).splitlines()[0]}.")
        return v
    v.n, v.df = len(df), df
    v.features, v.algos = features, algos
    v.source = header[lower.index("source")] if "source" in lower else None
    v.annotations = annotation_columns([c for c in header if c != v.source])

    labels = df[col_inst]
    empty = labels.isna() | (labels.str.strip() == "")
    if empty.any():
        v.errors.append(f"{int(empty.sum())} instance(s) without a label in **instances** "
                        f"(file lines {_examples(list(np.flatnonzero(empty) + 2))}).")
    rep = labels[labels.duplicated(keep=False) & ~empty]
    if len(rep):
        v.errors.append(f"Repeated labels in **instances** ({rep.nunique()} distinct): "
                        f"{_examples(sorted(rep.unique()))}.")
    for col in features + algos:
        raw = df[col]
        numbers = pd.to_numeric(raw, errors="coerce")
        bad = raw[numbers.isna() & raw.notna()]
        if len(bad):
            v.errors.append(f"**{col}** has non-numeric values: {_examples(sorted(set(map(str, bad))))}.")
            continue
        if np.isinf(numbers).any():
            v.errors.append(f"**{col}** has infinite values.")
        n_nan = int(numbers.isna().sum())
        if n_nan:
            v.nan[col] = n_nan
    if v.nan:
        parts = [f"{c}: {n}" for c, n in sorted(v.nan.items(), key=lambda t: -t[1])]
        v.warnings.append("NaN per column (" + ", ".join(parts[:12])
                          + (f" and {len(parts) - 12} more" if len(parts) > 12 else "") + "). "
                          "PRELIM removes features with too many NaN (prelim.nan_threshold).")
    all_nan = [c for c in algos if v.nan.get(c) == v.n]
    if all_nan:
        v.errors.append(f"Algorithms without any value: {_examples(all_nan)}.")
    if 0 < v.n < MIN_INSTANCES:
        v.warnings.append(f"Only {v.n} instances: PYTHIA's cross-validation is fragile.")
    v.ok = not v.errors
    return v


def validate_annotations(content: bytes, meta: Validation) -> tuple:
    """(types, errors) of an uploaded annotations.json, checked against the metadata."""
    try:
        types = json.loads(content.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as exc:
        return None, [f"annotations.json is not valid JSON: {exc}."]
    if meta.df is None:
        return types, []
    return types, declared_type_errors(meta.df, types)


def validate_feature_info(content: bytes, meta: Validation) -> list:
    """Errors of an uploaded feature_info.csv (columns feature and family)."""
    try:
        info = pd.read_csv(io.BytesIO(content))
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        return [f"Malformed feature_info.csv: {str(exc).splitlines()[0]}."]
    missing = [c for c in ("feature", "family") if c not in info.columns]
    if missing:
        return [f"feature_info.csv is missing the columns {', '.join(missing)}."]
    return []


def prelim_rows(v: Validation) -> pd.DataFrame:
    """algo_* of the instances that reach PRELIM: PREPROCESSING removes those
    with all feature_* or all algo_* missing (preprocessing.py:437)."""
    num = v.df[v.features + v.algos].apply(pd.to_numeric, errors="coerce")
    out = num[v.features].isna().all(axis=1) | num[v.algos].isna().all(axis=1)
    return num.loc[~out, v.algos]


def _algo_names(columns) -> list:
    return [c[len("algo_"):] if c.casefold().startswith("algo_") else c for c in columns]


def good_fraction(algos: pd.DataFrame, higher_is_better: bool, absolute: bool,
                  epsilon: float) -> pd.Series:
    """Fraction of good instances per algorithm, with the instancespace PRELIM
    rule (prelim.py:116-160, copied including the NaN handling): absolute =
    algo_* >= epsilon (higher is better) or <= epsilon; relative = within
    epsilon (as a fraction) of the instance's best."""
    y = algos.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    aux = np.where(np.isnan(y), -np.inf if higher_is_better else np.inf, y)
    if absolute:
        good = aux >= epsilon if higher_is_better else aux <= epsilon
    else:
        best = aux.max(axis=1) if higher_is_better else aux.min(axis=1)
        best[best == 0] = np.finfo(float).eps
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = aux / best[:, None]
            good = (1 - ratio) <= epsilon if higher_is_better else (ratio - 1) <= epsilon
    return pd.Series(good.mean(axis=0), index=_algo_names(algos.columns))


def degenerate_algorithms(fractions: pd.Series) -> list:
    """Algorithms with less than GOOD_MIN or more than GOOD_MAX good instances:
    with this epsilon, "good" is almost constant for them."""
    return [a for a, f in fractions.items() if f < GOOD_MIN or f > GOOD_MAX]


def mean_ranking(algos: pd.DataFrame, higher_is_better: bool) -> pd.Series:
    """Mean of each algo_* column (NaN ignored), ordered from the best to the
    worst algorithm under the chosen direction. No guessing: it only shows the
    consequence of the choice. Lower-is-better is exactly the reverse of
    higher-is-better, ties included."""
    means = algos.apply(pd.to_numeric, errors="coerce").mean(axis=0)
    means.index = _algo_names(algos.columns)
    ranking = means.sort_values(ascending=False, kind="stable")
    return ranking if higher_is_better else ranking.iloc[::-1]


def _power_law(points, n):
    """Power law a*n^b fitted in log-log to the (n, t) points."""
    b, a = np.polyfit(np.log([k for k, _ in points]), np.log([v for _, v in points]), 1)
    return float(np.exp(a) * n ** b)


def estimated_time(n: int, n_algos: int) -> float | None:
    """Estimated seconds: the rest of the pipeline from the power law of the
    time measured without PYTHIA, plus PYTHIA from its own power law scaled
    linearly by the number of algorithms. A rough estimate: it extrapolates
    outside 500..2000 and ignores the number of features."""
    points = [(k, t) for k, t in MEASURED_TIMES.items() if t[0] and t[1]]
    if len(points) < 2 or n <= 0:
        return None
    rest = _power_law([(k, total - py) for k, (total, py) in points], n)
    pythia = _power_law([(k, py) for k, (_, py) in points], n)
    return rest + pythia * max(n_algos, 1) / TIMING_ALGOS
