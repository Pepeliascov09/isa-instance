"""Transferencia entre datasets: prever n_wrong a partir das medidas.

Pergunta: um modelo treinado para prever dificuldade de instancia em alguns
datasets generaliza para um dataset nunca visto? Protocolo leave-one-dataset-
out em duas condicoes: (i) medidas brutas; (ii) medidas z-scoreadas DENTRO de
cada dataset antes de juntar — testa a hipotese de que medidas como o LSC nao
sao comparaveis entre datasets (normalizacao pelo tamanho da classe).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

RES = Path("resultados")
NOMES = ["iris", "diabetes", "blood-transfusion-service-center", "hill-valley"]


def carregar():
    dados = {}
    for nome in NOMES:
        path = RES / f"table_{nome}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} nao existe — rode run_table.py antes")
        dados[nome] = pd.read_csv(path, index_col=0)
    return dados


def zscore_por_dataset(F):
    """Z-score coluna a coluna; colunas constantes viram 0 (std substituido por 1)."""
    std = F.std().replace(0, 1.0)
    return (F - F.mean()) / std


def avaliar(dados, feature_cols, zscore):
    frames = {}
    for nome in NOMES:
        F = dados[nome][feature_cols].copy()
        if zscore:
            F = zscore_por_dataset(F)
        F["n_wrong"] = dados[nome]["n_wrong"].values
        frames[nome] = F

    linhas = []
    importancias = []
    for teste in NOMES:
        treino = pd.concat([frames[n] for n in NOMES if n != teste])
        Xtr, ytr = treino[feature_cols].values, treino["n_wrong"].values
        Xte, yte = frames[teste][feature_cols].values, frames[teste]["n_wrong"].values

        rf = RandomForestRegressor(random_state=42, n_jobs=1)
        rf.fit(Xtr, ytr)
        pred = rf.predict(Xte)

        linhas.append(
            {
                "dataset_teste": teste,
                "rho": spearmanr(pred, yte)[0],
                "mae": float(np.mean(np.abs(pred - yte))),
                "mae_baseline": float(np.mean(np.abs(ytr.mean() - yte))),
            }
        )
        importancias.append(rf.feature_importances_)

    return pd.DataFrame(linhas), np.mean(importancias, axis=0)


def main():
    dados = carregar()

    feature_cols = sorted(
        set.intersection(
            *[
                {c for c in t.columns if c.startswith("feature_")}
                for t in dados.values()
            ]
        )
    )
    # medidas com NaN em algum dataset (ex.: TD_P colapsa no hill-valley
    # quando a arvore podada vira so a raiz) sao descartadas, nao imputadas
    com_nan = sorted(
        {c for t in dados.values() for c in feature_cols if t[c].isna().any()}
    )
    if com_nan:
        print(f"descartadas por NaN: {com_nan}")
        feature_cols = [c for c in feature_cols if c not in com_nan]
    print(f"{len(feature_cols)} medidas comuns aos {len(NOMES)} datasets")

    res_bruto, imp_bruto = avaliar(dados, feature_cols, zscore=False)
    res_z, imp_z = avaliar(dados, feature_cols, zscore=True)

    tabela = res_bruto.rename(
        columns={"rho": "rho_bruto", "mae": "mae_bruto"}
    ).merge(
        res_z.rename(columns={"rho": "rho_z", "mae": "mae_z"})[
            ["dataset_teste", "rho_z", "mae_z"]
        ],
        on="dataset_teste",
    )[["dataset_teste", "rho_bruto", "mae_bruto", "rho_z", "mae_z", "mae_baseline"]]

    print("\nleave-one-dataset-out (alvo: n_wrong):")
    print(tabela.round(3).to_string(index=False))

    media_bruto = res_bruto["rho"].mean()
    media_z = res_z["rho"].mean()
    print(f"\nrho medio — bruto: {media_bruto:.3f}  |  z-score: {media_z:.3f}")

    if media_z >= media_bruto:
        vencedora, imp = "z-score por dataset", imp_z
    else:
        vencedora, imp = "medidas brutas", imp_bruto
    print(f"\nimportancia media das features (condicao vencedora: {vencedora}):")
    ranking = pd.Series(imp, index=feature_cols).sort_values(ascending=False)
    print(ranking.round(4).to_string())


if __name__ == "__main__":
    main()
