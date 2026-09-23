"""Full pipeline: measures + portfolio performance in one per-instance table."""

import os

import pandas as pd
from joblib import parallel_backend

from isaspace.intake import load_openml_dataset, to_pyhard_frame
from isaspace.measures import instance_measures
from isaspace.performance import algo_performance


def build_instance_table(dataset_id, measures_list=None, n_folds=5, seed=42):
    """Single per-instance table: measures, portfolio performance and derived columns.

    Columns: feature_* (measures), algo_*/proba_* (out-of-fold performance),
    "class" (original label as text), "n_wrong" (algorithms that got it wrong)
    and "ih" (1 - mean probability assigned to the true class).
    Index: the original index of the instance in the dataset. Returns (table, meta).
    """
    X, y, meta = load_openml_dataset(dataset_id)
    df = to_pyhard_frame(X, y)

    # PYHARD_SEED controls the random_state of pyhard's internal trees
    # (DCP/TD_P vary between runs without it); the previous value is restored
    # so no global state leaks.
    previous_seed = os.environ.get("PYHARD_SEED")
    os.environ["PYHARD_SEED"] = str(seed)
    try:
        with parallel_backend("sequential"):
            M = instance_measures(
                df, measures_list=measures_list, ccp_alpha=0.01
            )
    finally:
        if previous_seed is None:
            del os.environ["PYHARD_SEED"]
        else:
            os.environ["PYHARD_SEED"] = previous_seed

    perf = algo_performance(df, n_folds=n_folds, seed=seed)

    table = pd.concat([M, perf], axis=1)
    table["class"] = pd.Series(y).astype(str).values

    algo_cols = [c for c in table.columns if c.startswith("algo_")]
    proba_cols = [c for c in table.columns if c.startswith("proba_")]
    table["n_wrong"] = (1 - table[algo_cols]).sum(axis=1).astype(int)
    table["ih"] = 1 - table[proba_cols].mean(axis=1)

    table.index = X.index
    return table, meta
