"""Pipeline completo: medidas + desempenho do portfolio numa tabela por instancia."""

import os

import pandas as pd
from joblib import parallel_backend

from isaspace.intake import load_openml_dataset, to_pyhard_frame
from isaspace.measures import instance_measures
from isaspace.performance import algo_performance


def build_instance_table(dataset_id, measures_list=None, n_folds=5, seed=42):
    """Tabela unica por instancia: medidas, desempenho do portfolio e derivados.

    Colunas: feature_* (medidas), algo_*/proba_* (desempenho out-of-fold),
    "class" (rotulo original em texto), "n_wrong" (algoritmos que erraram) e
    "ih" (1 - probabilidade media atribuida a classe verdadeira).
    Indice: o indice original da instancia no dataset. Retorna (tabela, meta).
    """
    X, y, meta = load_openml_dataset(dataset_id)
    df = to_pyhard_frame(X, y)

    # PYHARD_SEED controla o random_state das arvores internas do pyhard
    # (DCP/TD_P variam entre execucoes sem isso); restaura o valor anterior
    # para nao vazar estado global.
    seed_anterior = os.environ.get("PYHARD_SEED")
    os.environ["PYHARD_SEED"] = str(seed)
    try:
        with parallel_backend("sequential"):
            M = instance_measures(
                df, measures_list=measures_list, ccp_alpha=0.01
            )
    finally:
        if seed_anterior is None:
            del os.environ["PYHARD_SEED"]
        else:
            os.environ["PYHARD_SEED"] = seed_anterior

    perf = algo_performance(df, n_folds=n_folds, seed=seed)

    table = pd.concat([M, perf], axis=1)
    table["class"] = pd.Series(y).astype(str).values

    algo_cols = [c for c in table.columns if c.startswith("algo_")]
    proba_cols = [c for c in table.columns if c.startswith("proba_")]
    table["n_wrong"] = (1 - table[algo_cols]).sum(axis=1).astype(int)
    table["ih"] = 1 - table[proba_cols].mean(axis=1)

    table.index = X.index
    return table, meta
