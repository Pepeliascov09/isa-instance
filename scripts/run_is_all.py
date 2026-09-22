"""Roda o instancespace (isaspace.engine) para os quatro datasets.

Le resultados/isa/<nome>/metadata.csv (gerado por isaspace.isa.to_isa_metadata,
com as anotacoes class, ih e n_wrong) e grava resultados/is/<nome>/. A pasta
resultados/isa/ (pyispace) nao e tocada. Ao final de cada dataset imprime o
progresso por estagio, os tempos, a correcao de quase duplicatas antes do
TRACE, o sifted_report e o status das footprints.

Precisa do .venv-isa (Python 3.12, instancespace 0.3.0):
    .venv-isa/bin/python scripts/run_is_all.py [nome ...]   (padrao: os quatro)
"""

import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

from isaspace.engine import run_instancespace  # noqa: E402
from isaspace.ui.loader_is import load_is_output  # noqa: E402

DATASETS = ["iris", "diabetes", "blood-transfusion-service-center", "hill-valley"]
PASTA_ENTRADA = RAIZ / "resultados" / "isa"
PASTA_SAIDA = RAIZ / "resultados" / "is"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


def rodar(nome):
    entrada = PASTA_ENTRADA / nome / "metadata.csv"
    outdir = PASTA_SAIDA / nome
    print(f"\n{'=' * 78}\n{nome}: {entrada} -> {outdir}\n{'=' * 78}")
    t0 = time.perf_counter()
    info = run_instancespace(
        entrada, outdir,
        progress=lambda e: print(f"  [{time.perf_counter() - t0:6.1f}s] {e}", flush=True),
    )
    r = load_is_output(outdir)

    print("\n-- tempos (s) --")
    print("  " + " | ".join(f"{k} {v:.2f}" for k, v in info["tempos_s"].items()))
    rob = info["trace_robustez"]
    print("\n-- quase duplicatas na projecao (antes do TRACE) --")
    print(f"  pares < {rob['limiar']:g}: {rob['pares_quase_duplicados']} "
          f"(identicos: {rob['pares_identicos']}, entre posicoes distintas: "
          f"{rob['pares_distintos_proximos']}) | jitter aplicado: {rob['jitter_aplicado']}"
          + (f" em {rob['posicoes_perturbadas']} posicoes ({rob['pontos_perturbados']} pontos), "
             f"menor distancia entre distintos {rob['menor_distancia_distintos_antes']:.3g} -> "
             f"{rob['menor_distancia_distintos_depois']:.3g}"
             if rob["jitter_aplicado"] else ""))
    print("\n-- sifted_report --")
    print(r.sifted_report.to_string(index=False))
    print("\n-- footprints --")
    linhas = [(a, t, f.status, len(f.poligonos), round(f.area, 4), f.pureza)
              for (a, t), f in r.footprints.items()]
    print(pd.DataFrame(linhas, columns=["algo", "tipo", "status", "partes", "area", "pureza"])
          .to_string(index=False))
    for fp in (r.footprint_space, r.footprint_hard):
        print(f"  {fp.tipo}: {fp.status}, {len(fp.poligonos)} partes, area {fp.area:.4f}, "
              f"pureza {fp.pureza:.3f}")
    for aviso in info["avisos"] + info["avisos_instancespace"]:
        print(f"  aviso: {aviso}")
    return info


def main(nomes):
    # o instancespace loga em DEBUG no stderr; aqui so avisos
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    tempos = {}
    for nome in nomes:
        t0 = time.perf_counter()
        rodar(nome)
        tempos[nome] = time.perf_counter() - t0
    print("\n== tempos por dataset ==")
    for nome, t in tempos.items():
        print(f"{nome:34s} {t:7.1f}s")


if __name__ == "__main__":
    main(sys.argv[1:] or DATASETS)
