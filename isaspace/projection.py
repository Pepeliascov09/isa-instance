"""Projecao 2D do espaco de instancias (versao provisoria, estilo PRELIM+PILOT)."""

import pandas as pd
from sklearn.decomposition import PCA


def instance_space(table, feature_cols=None, seed=42):
    """Projeta as medidas por instancia em 2D.

    PRELIM: limita outliers nos percentis 1/99, aplica z-score e descarta
    colunas de variancia zero.

    Projecao: PCA em 2 componentes. O PILOT verdadeiro do ISA otimiza a
    projecao para maximizar a relacao linear entre as coordenadas, as medidas
    E o desempenho dos algoritmos; o PCA aqui e um substituto temporario que
    enxerga apenas as medidas e ignora o desempenho.

    Retorna DataFrame com "z1", "z2" mais "ih", "n_wrong" e "class" copiadas
    da tabela de entrada, preservando o indice.
    """
    if feature_cols is None:
        feature_cols = [c for c in table.columns if c.startswith("feature_")]

    F = table[feature_cols].copy()

    # PRELIM: winsorizacao nos percentis 1/99 e z-score
    lo = F.quantile(0.01)
    hi = F.quantile(0.99)
    F = F.clip(lower=lo, upper=hi, axis=1)

    std = F.std()
    constantes = std[std == 0].index.tolist()
    if constantes:
        print(f"descartadas por variancia zero: {constantes}")
        F = F.drop(columns=constantes)
        std = std.drop(constantes)

    Z = (F - F.mean()) / std

    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(Z.values)

    print(
        "variancia explicada: "
        f"PC1={pca.explained_variance_ratio_[0]:.3f}  "
        f"PC2={pca.explained_variance_ratio_[1]:.3f}  "
        f"(total {pca.explained_variance_ratio_.sum():.3f})"
    )
    loadings = pd.DataFrame(
        pca.components_.T, index=Z.columns, columns=["PC1", "PC2"]
    )
    print("loadings:")
    print(loadings.round(3).to_string())

    out = pd.DataFrame(
        {"z1": coords[:, 0], "z2": coords[:, 1]}, index=table.index
    )
    for col in ["ih", "n_wrong", "class"]:
        if col in table.columns:
            out[col] = table[col]
    return out
