"""Footprints aproximadas por grade para iris e diabetes, com PNG por dataset."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from isaspace.footprint import grid_footprints
from isaspace.projection import instance_space

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

RES = Path("resultados")
NOMES = ["iris", "diabetes"]
NX = NY = 50
NX2 = NY2 = 15  # resolucao alternativa registrada no CSV
K_MIN = 3
LIMIAR = 0.8


def carregar(nome):
    table_path = RES / f"table_{nome}.csv"
    if not table_path.exists():
        return None
    table = pd.read_csv(table_path, index_col=0)
    space_path = RES / f"space_{nome}.csv"
    if space_path.exists():
        space = pd.read_csv(space_path, index_col=0)
    else:
        space = instance_space(table)
    algo_cols = [c for c in table.columns if c.startswith("algo_")]
    return space[["z1", "z2"]].join(table[algo_cols])


def plotar(nome, df, metricas, celulas, bordas, out_png):
    x_edges, y_edges = bordas
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    nomes_algo = metricas["algoritmo"].tolist()

    fig = plt.figure(figsize=(22, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    m_idx = metricas.set_index("algoritmo")

    axs = []
    pc = None
    for i, n in enumerate(nomes_algo):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        axs.append(ax)
        M = np.full((ny, nx), np.nan)
        M[celulas["iy"].to_numpy(), celulas["ix"].to_numpy()] = (
            celulas[f"acc_{n}"].to_numpy()
        )
        pc = ax.pcolormesh(
            x_edges, y_edges, np.ma.masked_invalid(M),
            cmap="RdYlGn", vmin=0, vmax=1,
        )
        acertou = df[f"algo_{n}"] == 1
        ax.scatter(
            df.loc[acertou, "z1"], df.loc[acertou, "z2"],
            s=6, c="#333333", alpha=0.45, linewidths=0,
        )
        ax.scatter(
            df.loc[~acertou, "z1"], df.loc[~acertou, "z2"],
            s=14, c="crimson", marker="x", linewidths=0.8,
        )
        m = m_idx.loc[n]
        ax.set_title(
            f"{n} — area {m['area_footprint']:.2f}, "
            f"pureza {m['pureza_media']:.2f}, "
            f"cobre {int(m['instancias_cobertas'])} inst."
        )
    fig.colorbar(pc, ax=axs, shrink=0.85, label="taxa de acerto na celula")

    ax_w = fig.add_subplot(gs[:, 3])
    categorias = nomes_algo + ["empate"]
    codigo = {n: i for i, n in enumerate(categorias)}
    W = np.full((ny, nx), np.nan)
    W[celulas["iy"].to_numpy(), celulas["ix"].to_numpy()] = (
        celulas["melhor_algo"].map(codigo).to_numpy(float)
    )
    cores = list(colormaps["tab10"].colors[: len(nomes_algo)]) + ["#b0b0b0"]
    cmap_disc = ListedColormap(cores)
    ax_w.pcolormesh(
        x_edges, y_edges, np.ma.masked_invalid(W),
        cmap=cmap_disc, vmin=-0.5, vmax=len(categorias) - 0.5,
    )
    ax_w.scatter(df["z1"], df["z2"], s=4, c="black", alpha=0.35, linewidths=0)
    ax_w.legend(
        handles=[
            Patch(color=cmap_disc(i), label=n)
            for i, n in enumerate(categorias)
        ],
        loc="best", fontsize=9,
    )
    n_emp = int(metricas["celulas_empate"].iloc[0])
    ax_w.set_title(f"melhor algoritmo por celula ({n_emp} empates em cinza)")

    fig.suptitle(
        f"{nome} — footprints aproximadas "
        f"(grade {nx}x{ny}, k>={K_MIN}, limiar {LIMIAR})"
    )
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main():
    RES.mkdir(exist_ok=True)
    for nome in NOMES:
        df = carregar(nome)
        if df is None:
            print(f"\n=== {nome}: sem resultados/table_{nome}.csv — pulado ===")
            continue

        metricas, celulas, bordas = grid_footprints(
            df, nx=NX, ny=NY, min_instances=K_MIN, threshold=LIMIAR
        )
        metricas15, _, _ = grid_footprints(
            df, nx=NX2, ny=NY2, min_instances=K_MIN, threshold=LIMIAR
        )
        print(f"\n=== {nome} ===")
        print(
            f"instancias: {len(df)}  celulas ocupadas (>= {K_MIN} inst.): "
            f"{len(celulas)}  instancias em celulas ocupadas: "
            f"{int(celulas['n'].sum())}"
        )
        print(f"grade {NX}x{NY}:")
        print(metricas.round(3).to_string(index=False))
        print(f"grade {NX2}x{NY2}:")
        print(metricas15.round(3).to_string(index=False))

        # CSV com as duas resolucoes lado a lado, para registrar a
        # dependencia da resolucao no arquivo e nao so no terminal
        m50 = metricas.rename(
            columns={
                c: f"{c}_{NX}x{NY}"
                for c in metricas.columns
                if c != "algoritmo"
            }
        )
        m15 = metricas15.rename(
            columns={
                c: f"{c}_{NX2}x{NY2}"
                for c in metricas15.columns
                if c != "algoritmo"
            }
        )
        tabela_csv = m50.merge(m15, on="algoritmo")
        csv_path = RES / f"footprint_{nome}.csv"
        tabela_csv.to_csv(csv_path, index=False)
        print(f"metricas salvas em {csv_path}")

        png_path = RES / f"footprint_{nome}.png"
        plotar(nome, df, metricas, celulas, bordas, png_path)
        print(f"figura salva em {png_path}")


if __name__ == "__main__":
    main()
