"""Aproximacao discreta das footprints do ISA no espaco projetado.

A footprint do ISA verdadeiro (etapas PYTHIA/TRACE do MATILDA) delimita
fronteiras CONTINUAS — poligonos/concave hulls construidos no espaco 2D — e
so aceita uma regiao apos testes de pureza e significancia estatistica da
densidade. A grade regular usada aqui e uma aproximacao discreta: celulas
pequenas demais ficam vazias, grandes demais misturam regioes distintas, e
nao ha teste estatistico algum. Serve como primeiro retrato, nao como TRACE.
"""

import numpy as np
import pandas as pd


def grid_footprints(df, nx=50, ny=50, min_instances=3, threshold=0.8):
    """Footprints por grade no espaco z1 x z2.

    df: DataFrame com colunas z1, z2 e algo_* (acerto 0/1 por instancia).
    Uma celula e "ocupada" se tem >= min_instances instancias; a footprint de
    um algoritmo sao as celulas ocupadas com taxa de acerto >= threshold.

    Retorna (metricas, celulas, bordas):
      metricas — linha por algoritmo: area_footprint (fracao das celulas
                 ocupadas), pureza_media (taxa media de acerto dentro da
                 footprint) e instancias_cobertas
      celulas  — linha por celula ocupada: indices ix/iy, centro cx/cy, n,
                 acc_<algo> e melhor_algo ("empate" quando mais de um
                 algoritmo atinge a taxa maxima da celula)
      bordas   — (x_edges, y_edges) da grade
    """
    algo_cols = [c for c in df.columns if c.startswith("algo_")]
    nomes = [c.replace("algo_", "") for c in algo_cols]

    x = df["z1"].to_numpy(float)
    y = df["z2"].to_numpy(float)
    x_edges = np.linspace(x.min(), x.max(), nx + 1)
    y_edges = np.linspace(y.min(), y.max(), ny + 1)
    ix = np.clip(np.digitize(x, x_edges) - 1, 0, nx - 1)
    iy = np.clip(np.digitize(y, y_edges) - 1, 0, ny - 1)

    g = df[algo_cols].copy()
    g.columns = nomes
    g["ix"] = ix
    g["iy"] = iy

    agg = g.groupby(["ix", "iy"]).agg(
        **{f"acc_{n}": (n, "mean") for n in nomes}, n=(nomes[0], "size")
    )
    celulas = agg[agg["n"] >= min_instances].reset_index()

    acc_cols = [f"acc_{n}" for n in nomes]
    acc = celulas[acc_cols].to_numpy(float)
    n_no_maximo = (acc >= acc.max(axis=1, keepdims=True) - 1e-12).sum(axis=1)
    celulas["melhor_algo"] = np.where(
        n_no_maximo > 1,
        "empate",
        np.array(nomes, dtype=object)[acc.argmax(axis=1)],
    )
    celulas_empate = int((n_no_maximo > 1).sum())
    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]
    celulas["cx"] = x_edges[celulas["ix"].to_numpy()] + dx / 2
    celulas["cy"] = y_edges[celulas["iy"].to_numpy()] + dy / 2

    linhas = []
    total = len(celulas)
    for n in nomes:
        fp = celulas[celulas[f"acc_{n}"] >= threshold]
        linhas.append(
            {
                "algoritmo": n,
                "celulas_ocupadas": total,
                "celulas_footprint": len(fp),
                "area_footprint": len(fp) / total if total else np.nan,
                "pureza_media": (
                    float(fp[f"acc_{n}"].mean()) if len(fp) else np.nan
                ),
                "instancias_cobertas": int(fp["n"].sum()),
                "celulas_empate": celulas_empate,
            }
        )
    metricas = pd.DataFrame(linhas)
    return metricas, celulas, (x_edges, y_edges)
