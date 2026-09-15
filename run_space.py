"""Espaco de instancias 2D para iris e diabetes, a partir dos CSVs salvos."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from isaspace.projection import instance_space

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

NOMES = ["iris", "diabetes"]


def main():
    out_dir = Path("resultados")

    for nome in NOMES:
        csv_path = out_dir / f"table_{nome}.csv"
        table = pd.read_csv(csv_path, index_col=0)

        print(f"\n=== {nome} ({csv_path}) ===")
        space = instance_space(table)

        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(
            space["z1"], space["z2"], c=space["ih"], cmap="viridis", s=25
        )
        fig.colorbar(sc, ax=ax, label="ih")
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.set_title(f"{nome} — espaco de instancias (PCA provisorio)")
        fig.tight_layout()
        png_path = out_dir / f"space_{nome}.png"
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"scatter salvo em {png_path}")


if __name__ == "__main__":
    main()
