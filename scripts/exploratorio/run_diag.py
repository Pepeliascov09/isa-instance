"""Diagnostico: normalizacao do LSC, correlacoes entre medidas e histogramas."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import parallel_backend
from pyhard.measures import ClassificationMeasures

from isaspace.intake import load_openml_dataset, to_pyhard_frame

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

DATASET_IDS = [61, 37, 1464, 1479]
MEASURES = ["kDN", "N1", "N2", "LSC"]
RHO_LIMIAR = 0.7

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def ls_absoluto(cm, y_codes):
    """|LS| pela definicao: pontos mais proximos que o inimigo mais proximo.

    Conta, na ordenacao de vizinhos por Gower (self na posicao 0), quantos
    aparecem antes do primeiro inimigo — mesmo criterio do pyhard, que depois
    divide pelo tamanho da classe da instancia.
    """
    nn_labels = y_codes[cm.indices_gower]
    return np.argmax(nn_labels != y_codes[:, None], axis=1)


def main():
    out_dir = Path("resultados")
    out_dir.mkdir(exist_ok=True)
    corr_por_dataset = {}

    for did in DATASET_IDS:
        X, y, meta = load_openml_dataset(did)
        df = to_pyhard_frame(X, y)
        with parallel_backend("sequential"):
            cm = ClassificationMeasures(df, target_col="target", ccp_alpha=0.01)
            M = cm.calculate_all(measures_list=MEASURES)

        print(f"\n=== {meta['name']} (id {meta['id']}) ===")

        # --- 1) normalizacao do LSC ---
        y_codes = df["target"].values
        categorias = pd.Series(y).astype("category").cat.categories
        ls_abs = ls_absoluto(cm, y_codes)
        tamanhos = pd.Series(y_codes).value_counts().sort_index()

        linhas = []
        for classe, n_cls in tamanhos.items():
            mask = y_codes == classe
            ls = ls_abs[mask]
            mediana_ls = float(np.median(ls))
            linhas.append(
                {
                    "classe": str(categorias[classe]),
                    "n_classe": int(n_cls),
                    "LS_min": int(ls.min()),
                    "LS_mediana": mediana_ls,
                    "LS_max": int(ls.max()),
                    "LSC_def_mediana": 1 - mediana_ls / n_cls,
                    "LSC_pyhard_mediana": float(
                        np.median(M.loc[mask, "feature_LSC"])
                    ),
                }
            )
        print(pd.DataFrame(linhas).to_string(index=False))

        # --- 2) correlacao de Spearman ---
        corr = M.corr(method="spearman")
        print("\ncorrelacao de Spearman:")
        print(corr.round(3).to_string())
        corr_por_dataset[meta["name"]] = corr

        # --- 3) histogramas ---
        if HAS_MPL:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            for ax, col in zip(axes.ravel(), M.columns):
                ax.hist(M[col], bins=30)
                ax.set_title(col)
            fig.suptitle(f"{meta['name']} (id {meta['id']})")
            fig.tight_layout()
            out_png = out_dir / f"hist_{meta['name']}.png"
            fig.savefig(out_png, dpi=120)
            plt.close(fig)
            print(f"histograma salvo em {out_png}")
        else:
            print("matplotlib nao instalado — histogramas pulados")

    print(f"\n=== pares com |rho| > {RHO_LIMIAR} em pelo menos 3 dos 4 datasets ===")
    cols = list(next(iter(corr_por_dataset.values())).columns)
    algum = False
    for a, b in combinations(cols, 2):
        rhos = {nome: c.loc[a, b] for nome, c in corr_por_dataset.items()}
        hits = sum(abs(r) > RHO_LIMIAR for r in rhos.values())
        if hits >= 3:
            algum = True
            detalhe = "  ".join(f"{nome}={r:.2f}" for nome, r in rhos.items())
            print(f"{a} x {b}: {hits}/4  ({detalhe})")
    if not algum:
        print("nenhum par atingiu o criterio")


if __name__ == "__main__":
    main()
