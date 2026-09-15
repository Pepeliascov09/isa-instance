"""Espaco de dados (PCA 2D dos atributos originais) para o scatter da esquerda.

Para cada dataset: baixa do OpenML via isaspace.intake, converte os atributos
com to_pyhard_frame (a mesma conversao numerica usada para gerar a tabela por
instancia), padroniza por z-score e aplica PCA de 2 componentes. Grava
resultados/isa/<nome>/data_space.csv com indice Row de 1 a n, colunas d_1 e
d_2 e as colunas ORIGINAIS dos atributos (valores crus, como vieram do OpenML).

Alinhamento: cada linha Row = k corresponde a instancia cujo indice original
esta em row_original[k] de metadata.csv, ou seja, a mesma linha de
coordinates.csv. O script confere isso antes de gravar.

Uso: python scripts/build_data_space.py [nome ...]   (padrao: os quatro)
"""

import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from isaspace.intake import load_openml_dataset, to_pyhard_frame  # noqa: E402

PASTA_ISA = RAIZ / "resultados" / "isa"
DATASETS = {
    "iris": 61,
    "diabetes": 37,
    "blood-transfusion-service-center": 1464,
    "hill-valley": 1479,
}
SEED = 42


def build_data_space(nome, dataset_id, pasta_isa=PASTA_ISA):
    """Gera data_space.csv de um dataset e devolve (DataFrame, variancia explicada)."""
    outdir = pasta_isa / nome
    meta_path = outdir / "metadata.csv"
    coords_path = outdir / "coordinates.csv"
    if not meta_path.is_file() or not coords_path.is_file():
        raise FileNotFoundError(
            f"{outdir} nao tem metadata.csv/coordinates.csv; rode "
            f"scripts/run_isa_all.py {nome} antes"
        )

    X, y, meta = load_openml_dataset(dataset_id)
    if meta["name"] != nome:
        print(f"aviso: OpenML chama o dataset {dataset_id} de {meta['name']!r}, nao {nome!r}")

    # conversao numerica identica a da tabela (codigos de categoria, mediana
    # para faltantes, colunas constantes fora); o indice volta a ser o de X
    numerico = to_pyhard_frame(X, y).drop(columns=["target"])
    numerico.index = X.index

    metadata = pd.read_csv(meta_path, index_col="instances")
    row_original = metadata["row_original"]
    faltam = sorted(set(row_original) - set(X.index))
    if faltam:
        raise ValueError(
            f"{nome}: {len(faltam)} valores de row_original nao existem no "
            f"dataset do OpenML (ex.: {faltam[:5]})"
        )

    numerico = numerico.loc[row_original.to_numpy()]
    Z = StandardScaler().fit_transform(numerico.to_numpy(dtype=float))
    pca = PCA(n_components=2, random_state=SEED)
    D = pca.fit_transform(Z)

    indice = pd.RangeIndex(1, len(row_original) + 1, name="Row")
    out = pd.DataFrame({"d_1": D[:, 0], "d_2": D[:, 1]}, index=indice)

    atributos = X.loc[row_original.to_numpy()].copy()
    atributos.index = indice
    colisao = [c for c in atributos.columns if c in ("d_1", "d_2", "Row")]
    if colisao:
        atributos = atributos.rename(columns={c: f"attr_{c}" for c in colisao})
    out = pd.concat([out, atributos], axis=1)

    coords = pd.read_csv(coords_path, index_col="Row")
    if not coords.index.equals(out.index):
        raise ValueError(f"{nome}: indice Row de data_space nao coincide com coordinates.csv")
    if len(out) != len(metadata):
        raise ValueError(f"{nome}: {len(out)} linhas em data_space, {len(metadata)} em metadata.csv")

    out.to_csv(outdir / "data_space.csv")
    return out, pca.explained_variance_ratio_


def main(nomes):
    for nome in nomes:
        if nome not in DATASETS:
            raise SystemExit(f"dataset desconhecido: {nome}; opcoes: {list(DATASETS)}")
        t0 = time.perf_counter()
        out, var = build_data_space(nome, DATASETS[nome])
        print(
            f"{nome:34s} data_space.csv {out.shape[0]} x {out.shape[1]} "
            f"(d_1, d_2 + {out.shape[1] - 2} atributos) | variancia explicada "
            f"PC1={var[0]:.3f} PC2={var[1]:.3f} | {time.perf_counter() - t0:.1f}s"
        )


if __name__ == "__main__":
    main(sys.argv[1:] or list(DATASETS))
