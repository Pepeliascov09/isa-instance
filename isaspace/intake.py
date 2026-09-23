"""Loading of OpenML datasets and conversion to the PyHard format."""

import numpy as np
import openml
import pandas as pd


def load_openml_dataset(dataset_id: int):
    """Download an OpenML dataset and return (X, y, meta).

    X: pd.DataFrame with the attributes.
    y: pd.Series with the label (default_target_attribute).
    meta: dict with id, name, n_instances, n_features, n_classes, target.
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
    """Convert (X, y) to the single DataFrame PyHard requires.

    Fully numeric, without missing values, with the label column included.
    - object/category/bool become category codes (float)
    - the other columns go through pd.to_numeric(errors="coerce")
    - missing values are filled with the column median
    - constant columns (nunique <= 1) are dropped
    - y becomes integer category codes
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
