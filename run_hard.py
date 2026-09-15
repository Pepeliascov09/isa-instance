"""Ranqueia as instancias do iris por dificuldade e cruza com a classe original."""

import pandas as pd

from isaspace.intake import load_openml_dataset, to_pyhard_frame
from isaspace.measures import instance_measures

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def main():
    X, y, meta = load_openml_dataset(61)
    print("meta:", meta)

    df = to_pyhard_frame(X, y)
    M = instance_measures(
        df, measures_list=["kDN", "DS", "DCP", "N1", "N2", "LSC"]
    )
    measure_cols = list(M.columns)

    # Agregacao provisoria e ingenua: media simples das seis medidas.
    # As medidas tem escalas/distribuicoes diferentes e sao correlacionadas
    # entre si, entao esta media nao pondera nada disso — serve apenas como
    # primeiro ranking exploratorio.
    out = M.copy()
    out["hardness"] = M.mean(axis=1)
    out["classe"] = y.reset_index(drop=True).astype(str)

    ranked = out.sort_values("hardness", ascending=False)

    print("\n10 instancias mais dificeis:")
    print(ranked.head(10).to_string())

    print("\n10 instancias mais faceis:")
    print(ranked.tail(10).sort_values("hardness").to_string())

    print("\ndistribuicao de classes entre as 20 mais dificeis:")
    print(ranked.head(20)["classe"].value_counts().to_string())

    print("\nmedia de cada medida por classe original:")
    print(out.groupby("classe")[measure_cols + ["hardness"]].mean().to_string())


if __name__ == "__main__":
    main()
