"""Mede o tempo do engine em metadata sintetico (base de upload.TEMPOS_MEDIDOS).

Gera metadata com N instancias, 10 features e 6 algoritmos (desempenho em
[0, 1], maior e melhor, dependente das features para o PYTHIA ter o que
aprender) e roda pelo mesmo caminho da interface: isaspace.ui.execucao, em
subprocesso, com as opcoes padrao do engine. Imprime o tempo total e o tempo
ate o inicio de cada estagio.

Uso (no .venv-isa): python scripts/medir_tempo_engine.py [N ...] [--repeticoes R]
    padrao: 500 1000 2000, 1 repeticao
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from isaspace.ui import execucao  # noqa: E402

N_FEATURES, N_ALGOS = 10, 6


def metadata_sintetico(n: int, semente: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(semente)
    x = rng.normal(size=(n, N_FEATURES))
    pesos = rng.normal(size=(N_FEATURES, N_ALGOS))
    y = 1 / (1 + np.exp(-(x @ pesos) / 2 + rng.normal(scale=0.5, size=(n, N_ALGOS))))
    df = pd.DataFrame({"instances": [f"i{k}" for k in range(n)]})
    for j in range(N_FEATURES):
        df[f"feature_f{j}"] = x[:, j]
    for j in range(N_ALGOS):
        df[f"algo_a{j}"] = y[:, j]
    return df


def medir(n: int, pasta: Path) -> dict:
    pasta.mkdir(parents=True, exist_ok=True)
    meta = pasta / f"metadata_{n}.csv"
    metadata_sintetico(n).to_csv(meta, index=False)
    marcas = {}
    t0 = time.perf_counter()
    exe = execucao.iniciar(meta, pasta / f"saida_{n}", {},
                           ao_estagio=lambda _e, est: marcas.setdefault(est, time.perf_counter() - t0))
    while not exe.terminou:
        time.sleep(0.2)
    if not exe.ok:
        raise SystemExit(f"n={n}: {exe.erro}")
    return {"n": n, "total_s": exe.duracao, **{f"{k}_s": v for k, v in marcas.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n", nargs="*", type=int, default=[500, 1000, 2000])
    parser.add_argument("--repeticoes", type=int, default=1)
    args = parser.parse_args()
    linhas = []
    with tempfile.TemporaryDirectory() as tmp:
        for n in args.n:
            for r in range(args.repeticoes):
                linha = medir(n, Path(tmp) / f"r{r}")
                print(f"n={n:5d} rep={r} total={linha['total_s']:.1f}s", flush=True)
                linhas.append(linha)
    tabela = pd.DataFrame(linhas).groupby("n").median()
    pd.set_option("display.width", 200)
    print("\nmediana por n (s; colunas <ESTAGIO>_s = inicio do estagio):")
    print(tabela.round(1).to_string())


if __name__ == "__main__":
    main()
