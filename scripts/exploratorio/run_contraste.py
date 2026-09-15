"""Contraste: medidas por instancia (PyHard) vs meta-features de dataset (PyMFE)."""

import pandas as pd
from pymfe.mfe import MFE

from isaspace.intake import load_openml_dataset, to_pyhard_frame
from isaspace.measures import instance_measures

pd.set_option("display.max_columns", None)


def main():
    X, y, meta = load_openml_dataset(61)
    print("meta:", meta)

    df = to_pyhard_frame(X, y)
    measures = instance_measures(
        df, measures_list=["kDN", "DS", "DCP", "N1", "N2", "LSC"]
    )

    Xn = df.drop(columns=["target"])
    yn = df["target"]

    mfe = MFE(groups=["complexity"], features=["n1", "f1", "lsc"])
    mfe.fit(Xn.values, yn.values)
    names, values = mfe.extract()
    dataset_level = dict(zip(names, values))

    print("\nmeta-features PyMFE (nivel do dataset):")
    for name, value in dataset_level.items():
        print(f"  {name}: {value}")

    print("\ncontraste dataset (PyMFE) vs instancia (PyHard):")
    rows = []
    for pymfe_name, col in [("n1", "feature_N1"), ("lsc", "feature_LSC")]:
        s = measures[col]
        rows.append(
            {
                "medida": pymfe_name.upper(),
                "pymfe_dataset": dataset_level.get(pymfe_name),
                "min": s.min(),
                "mediana": s.median(),
                "media": s.mean(),
                "p75": s.quantile(0.75),
                "max": s.max(),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

    print()
    n = len(measures)
    for col in ["feature_N1", "feature_kDN"]:
        zeros = int((measures[col] == 0).sum())
        print(f"{col} == 0: {zeros} de {n} instancias ({100 * zeros / n:.1f}%)")


if __name__ == "__main__":
    main()
