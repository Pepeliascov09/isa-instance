"""Per-instance performance of a portfolio of classifiers."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def _portfolio(seed):
    return {
        "knn": KNeighborsClassifier(n_neighbors=5, n_jobs=1),
        "tree": DecisionTreeClassifier(random_state=seed),
        "nb": GaussianNB(),
        "logreg": LogisticRegression(max_iter=1000, random_state=seed, n_jobs=1),
        "svm_rbf": SVC(kernel="rbf", probability=True, random_state=seed),
        "rf": RandomForestClassifier(random_state=seed, n_jobs=1),
    }


def algo_performance(df, target_col="target", n_folds=5, seed=42):
    """0/1 hit and probability of the true class, out-of-fold, per instance.

    df: numeric DataFrame with attributes and label together (PyHard format).
    Returns one row per instance with columns "algo_<name>" (hit) and
    "proba_<name>" (probability of the true class) for each algorithm.
    The StandardScaler is part of the Pipeline so that it is fitted only on
    the training part of each fold, without leakage.
    """
    X = df.drop(columns=target_col).values
    y = df[target_col].values
    n = len(y)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    out = pd.DataFrame(index=range(n))

    for name, clf in _portfolio(seed).items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        hit = np.zeros(n, dtype=int)
        proba_true = np.zeros(n, dtype=float)

        for train_idx, test_idx in skf.split(X, y):
            pipe.fit(X[train_idx], y[train_idx])
            pred = pipe.predict(X[test_idx])
            hit[test_idx] = (pred == y[test_idx]).astype(int)

            proba = pipe.predict_proba(X[test_idx])
            # classes_ is sorted; find the column of the true class
            col_true = np.searchsorted(pipe.classes_, y[test_idx])
            proba_true[test_idx] = proba[np.arange(len(test_idx)), col_true]

        out[f"algo_{name}"] = hit
        out[f"proba_{name}"] = proba_true

    return out
