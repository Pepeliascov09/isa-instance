"""Roda PILOT + TRACE do pyispace para iris e diabetes.

Le resultados/table_<nome>.csv, converte com isaspace.isa.to_isa_metadata,
executa isaspace.isa.run_isa e grava em resultados/isa/<nome>/. Ao final de
cada dataset imprime: arquivos gravados e dimensoes, features descartadas com
as variancias, trace.summary, taxa 'boa' do pyispace x acuracia real (o
guarda-corpo), Spearman de z_1/z_2 com a coluna ih da tabela original e o
tempo gasto.

Uso: python scripts/run_isa_all.py [nome ...]      (padrao: iris diabetes)
"""

import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from isaspace.isa import ROW_ORIGINAL, run_isa, to_isa_metadata  # noqa: E402

DATASETS = ["iris", "diabetes"]
PASTA_TABELAS = RAIZ / "resultados"
PASTA_SAIDA = RAIZ / "resultados" / "isa"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


def listar_arquivos(outdir):
    """Uma linha por arquivo: CSVs com (linhas x colunas), demais com bytes."""
    linhas = []
    for p in sorted(outdir.iterdir()):
        if p.suffix == ".csv":
            df = pd.read_csv(p, index_col=0)
            linhas.append((p.name, f"{df.shape[0]} x {df.shape[1]}", ", ".join(map(str, df.columns))))
        else:
            linhas.append((p.name, f"{p.stat().st_size} bytes", ""))
    return pd.DataFrame(linhas, columns=["arquivo", "dimensao", "colunas"]).set_index("arquivo")


def spearman_com_ih(outdir, table, row_original):
    """rho de Spearman entre z_1/z_2 (coordinates.csv) e ih da tabela original.

    O join usa row_original (instances -> indice original) para nao depender
    da ordem das linhas.
    """
    coords = pd.read_csv(outdir / "coordinates.csv", index_col="Row")
    # o pyispace grava os rotulos "1".."n", que o read_csv le como inteiros
    coords.index = coords.index.astype(str)
    coords.index.name = row_original.index.name
    coords = coords.join(row_original)
    ih = table["ih"].reindex(coords[ROW_ORIGINAL].to_numpy()).to_numpy()
    out = {}
    for z in ("z_1", "z_2"):
        rho, p = spearmanr(coords[z].to_numpy(), ih)
        out[z] = {"rho": rho, "p_valor": p}
    return pd.DataFrame(out).T


def rodar(nome):
    tabela_csv = PASTA_TABELAS / f"table_{nome}.csv"
    outdir = PASTA_SAIDA / nome
    print(f"\n{'=' * 78}\n{nome}: {tabela_csv} -> {outdir}\n{'=' * 78}")

    table = pd.read_csv(tabela_csv, index_col=0)
    t0 = time.perf_counter()
    metadata, info = to_isa_metadata(table, outdir=outdir)
    t_meta = time.perf_counter() - t0
    print(
        f"metadata: {metadata.shape[0]} instancias, "
        f"{len(info['features_kept'])} features mantidas, "
        f"{len(info['algos'])} algoritmos (desempenho = {info['performance_source']})"
    )

    t1 = time.perf_counter()
    model = run_isa(metadata, outdir, table=table)
    t_isa = time.perf_counter() - t1

    print("\n-- arquivos gravados --")
    print(listar_arquivos(outdir).to_string())

    print("\n-- features descartadas (variancia apos o pre-processamento do pyispace) --")
    if info["features_dropped"]:
        print(pd.DataFrame(info["features_dropped"]).set_index("feature").to_string())
    else:
        print("nenhuma")

    print("\n-- trace.summary (footprint_performance.csv) --")
    print(model.trace.summary.round(4).to_string())

    print("\n-- guarda-corpo: taxa 'boa' (Ybin) x acuracia real --")
    print(model.ybin_check.round(4).to_string())
    print(f"assercao passou (tolerancia 0.15) para {len(model.ybin_check)} algoritmos")

    print("\n-- Spearman entre coordenadas e ih da tabela original --")
    print(spearman_com_ih(outdir, table, info["row_original"]).round(4).to_string())

    print(
        f"\n-- tempo: to_isa_metadata {t_meta:.1f}s | run_isa (PILOT+TRACE+CSVs) "
        f"{t_isa:.1f}s | total {t_meta + t_isa:.1f}s --"
    )
    return model


def main(nomes):
    tempos = {}
    for nome in nomes:
        t0 = time.perf_counter()
        rodar(nome)
        tempos[nome] = time.perf_counter() - t0
    print("\n== tempos por dataset ==")
    for nome, t in tempos.items():
        print(f"{nome:12s} {t:7.1f}s")


if __name__ == "__main__":
    main(sys.argv[1:] or DATASETS)
