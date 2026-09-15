"""Desempenho por instancia de um portfolio de classificadores."""

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
    """Acerto 0/1 e probabilidade da classe verdadeira, out-of-fold, por instancia.

    df: DataFrame numerico com atributos e rotulo juntos (formato PyHard).
    Retorna uma linha por instancia com colunas "algo_<nome>" (acerto) e
    "proba_<nome>" (probabilidade da classe verdadeira) para cada algoritmo.
    O StandardScaler entra no Pipeline para ser ajustado so no treino de cada
    fold, sem vazamento.
    """
    X = df.drop(columns=target_col).values
    y = df[target_col].values
    n = len(y)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    out = pd.DataFrame(index=range(n))

    for name, clf in _portfolio(seed).items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        acerto = np.zeros(n, dtype=int)
        proba_true = np.zeros(n, dtype=float)

        for train_idx, test_idx in skf.split(X, y):
            pipe.fit(X[train_idx], y[train_idx])
            pred = pipe.predict(X[test_idx])
            acerto[test_idx] = (pred == y[test_idx]).astype(int)

            proba = pipe.predict_proba(X[test_idx])
            # classes_ e ordenado; localiza a coluna da classe verdadeira
            col_true = np.searchsorted(pipe.classes_, y[test_idx])
            proba_true[test_idx] = proba[np.arange(len(test_idx)), col_true]

        out[f"algo_{name}"] = acerto
        out[f"proba_{name}"] = proba_true

    return out
