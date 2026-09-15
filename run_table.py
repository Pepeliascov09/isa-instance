"""Gera a tabela por instancia (medidas + desempenho) para iris e diabetes."""

from pathlib import Path

import pandas as pd

from isaspace.pipeline import build_instance_table

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)

DATASET_IDS = [61, 37, 1464, 1479]


def main():
    out_dir = Path("resultados")
    out_dir.mkdir(exist_ok=True)

    for did in DATASET_IDS:
        table, meta = build_instance_table(did)
        path = out_dir / f"table_{meta['name']}.csv"
        table.to_csv(path)

        print(f"\n=== {meta['name']} (id {meta['id']}) — {table.shape[0]} x {table.shape[1]} ===")
        print(f"salvo em {path}")

        feature_cols = [c for c in table.columns if c.startswith("feature_")]
        algo_cols = [c for c in table.columns if c.startswith("algo_")]

        corr = pd.DataFrame(
            {
                "rho_ih": [
                    table[c].corr(table["ih"], method="spearman")
                    for c in feature_cols
                ],
                "rho_n_wrong": [
                    table[c].corr(table["n_wrong"], method="spearman")
                    for c in feature_cols
                ],
            },
            index=feature_cols,
        ).sort_values("rho_ih", ascending=False)
        print("\ncorrelacao de Spearman com ih e n_wrong:")
        print(corr.round(3).to_string())

        print("\n5 instancias com maior ih:")
        top5 = table.sort_values("ih", ascending=False).head(5)
        print(top5[["class", "ih", "n_wrong"] + feature_cols].to_string())

        n_algos = len(algo_cols)
        todos_acertam = int((table["n_wrong"] == 0).sum())
        todos_erram = int((table["n_wrong"] == n_algos).sum())
        n = len(table)
        print(
            f"\nn_wrong == 0 (portfolio inteiro acerta): {todos_acertam} "
            f"({100 * todos_acertam / n:.1f}%)"
        )
        print(
            f"n_wrong == {n_algos} (portfolio inteiro erra): {todos_erram} "
            f"({100 * todos_erram / n:.1f}%)"
        )


if __name__ == "__main__":
    main()
