"""Carregamento de datasets do OpenML e conversao para o formato do PyHard."""

import numpy as np
import openml
import pandas as pd


def load_openml_dataset(dataset_id: int):
    """Baixa um dataset do OpenML e retorna (X, y, meta).

    X: pd.DataFrame com os atributos.
    y: pd.Series com o rotulo (default_target_attribute).
    meta: dict com id, name, n_instances, n_features, n_classes, target.
    """
    ds = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = ds.get_data(
        target=ds.default_target_attribute, dataset_format="dataframe"
    )
    meta = {
        "id": ds.dataset_id,
        "name": ds.name,
        "n_instances": X.shape[0],
        "n_features": X.shape[1],
        "n_classes": int(pd.Series(y).nunique(dropna=True)),
        "target": ds.default_target_attribute,
    }
    return X, y, meta


def to_pyhard_frame(X, y, target_col="target"):
    """Converte (X, y) para o DataFrame unico exigido pelo PyHard.

    Totalmente numerico, sem valores faltantes, com a coluna de rotulo incluida.
    - object/category/bool viram codigos de categoria (float)
    - demais colunas passam por pd.to_numeric(errors="coerce")
    - faltantes preenchidos pela mediana da coluna
    - colunas constantes (nunique <= 1) sao descartadas
    - y vira codigos inteiros de categoria
    """
    df = pd.DataFrame(X).copy()

    for col in df.columns:
        s = df[col]
        if (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_categorical_dtype(s)
            or pd.api.types.is_bool_dtype(s)
        ):
            codes = s.astype("category").cat.codes.astype(float)
            codes[s.isna().values] = np.nan
            df[col] = codes
        else:
            df[col] = pd.to_numeric(s, errors="coerce")

    df = df.fillna(df.median())
    df = df.loc[:, df.nunique() > 1]

    y_codes = pd.Series(y).astype("category").cat.codes.astype(int)
    out = df.reset_index(drop=True)
    out[target_col] = y_codes.reset_index(drop=True)
    return out
