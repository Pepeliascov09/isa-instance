"""Portfolio out-of-fold no iris e no diabetes, cruzado com kDN."""

import pandas as pd
from joblib import parallel_backend

from isaspace.intake import load_openml_dataset, to_pyhard_frame
from isaspace.measures import instance_measures
from isaspace.performance import algo_performance

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

DATASET_IDS = [61, 37]


def main():
    for did in DATASET_IDS:
        X, y, meta = load_openml_dataset(did)
        df = to_pyhard_frame(X, y)

        perf = algo_performance(df)
        with parallel_backend("sequential"):
            M = instance_measures(df, measures_list=["kDN"], ccp_alpha=0.01)

        print(f"\n=== {meta['name']} (id {meta['id']}) ===")
        algo_cols = [c for c in perf.columns if c.startswith("algo_")]
        acc = perf[algo_cols].mean().sort_values(ascending=False)
        print("acuracia media out-of-fold por algoritmo:")
        print(acc.round(4).to_string())

        n_erros = (1 - perf[algo_cols]).sum(axis=1)
        rho = M["feature_kDN"].corr(n_erros, method="spearman")
        print(f"\ninstancias erradas por 0/1/.../6 algoritmos:")
        print(n_erros.value_counts().sort_index().to_string())
        print(f"Spearman(feature_kDN, n_algoritmos_errando): {rho:.3f}")


if __name__ == "__main__":
    main()
