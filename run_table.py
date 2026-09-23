"""Build the per-instance table (measures + performance) for the IC7 datasets."""

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
        print(f"saved to {path}")

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
        print("\nSpearman correlation with ih and n_wrong:")
        print(corr.round(3).to_string())

        print("\n5 instances with the highest ih:")
        top5 = table.sort_values("ih", ascending=False).head(5)
        print(top5[["class", "ih", "n_wrong"] + feature_cols].to_string())

        n_algos = len(algo_cols)
        all_right = int((table["n_wrong"] == 0).sum())
        all_wrong = int((table["n_wrong"] == n_algos).sum())
        n = len(table)
        print(
            f"\nn_wrong == 0 (the whole portfolio is right): {all_right} "
            f"({100 * all_right / n:.1f}%)"
        )
        print(
            f"n_wrong == {n_algos} (the whole portfolio is wrong): {all_wrong} "
            f"({100 * all_wrong / n:.1f}%)"
        )


if __name__ == "__main__":
    main()
