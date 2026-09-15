"""Demo: perfil de dificuldade por instancia do iris (OpenML id 61)."""

from isaspace.intake import load_openml_dataset, to_pyhard_frame
from isaspace.measures import instance_measures


def main():
    X, y, meta = load_openml_dataset(61)
    print("meta:", meta)

    df = to_pyhard_frame(X, y)
    print("frame para o PyHard:", df.shape)

    measures = instance_measures(
        df, measures_list=["kDN", "DS", "DCP", "N1", "N2", "LSC"]
    )
    print("medidas:", measures.shape)
    print()
    print(measures.head())
    print()
    print(measures.describe())


if __name__ == "__main__":
    main()
