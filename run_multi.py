"""Pipeline de medidas por instancia em quatro datasets pequenos do OpenML."""

import multiprocessing as mp
import time
import traceback
from pathlib import Path
from queue import Empty

import pandas as pd

from isaspace.intake import load_openml_dataset, to_pyhard_frame

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

DATASET_IDS = [61, 37, 1464, 1479]
MEASURES = ["kDN", "N1", "N2", "LSC"]
TIMEOUT_S = 300  # Gower e O(n^2); interrompe o calculo se passar de 5 min


def _worker(df, measures_list, queue):
    try:
        from joblib import parallel_backend

        from isaspace.measures import instance_measures

        # O __init__ do pyhard usa n_jobs=-1 (CalibratedClassifierCV e, com
        # ccp_alpha=None, GridSearchCV). Dentro de um processo filho no Windows
        # o loky trava ~300s por dataset; backend sequencial elimina isso sem
        # perda real nesses tamanhos. ccp_alpha fixo pois as medidas usadas
        # aqui nao dependem da arvore podada — poupa o tuning inteiro.
        with parallel_backend("sequential"):
            result = instance_measures(
                df, measures_list=measures_list, ccp_alpha=0.01
            )
        queue.put(("ok", result))
    except Exception:
        queue.put(("err", traceback.format_exc()))


def measures_with_timeout(df, measures_list, timeout_s=TIMEOUT_S):
    """Roda instance_measures num processo filho, matando-o se estourar o tempo."""
    queue = mp.Queue()
    proc = mp.Process(target=_worker, args=(df, measures_list, queue))
    proc.start()
    try:
        status, payload = queue.get(timeout=timeout_s)
    except Empty:
        if proc.is_alive():
            proc.terminate()
            proc.join()
            raise TimeoutError(
                f"calculo excedeu {timeout_s}s e foi interrompido"
            ) from None
        raise RuntimeError("processo de calculo morreu sem retornar resultado")
    proc.join()
    if status == "err":
        raise RuntimeError(f"erro no calculo das medidas:\n{payload}")
    return payload


def main():
    rows = []
    for did in DATASET_IDS:
        t0 = time.perf_counter()
        try:
            X, y, meta = load_openml_dataset(did)
            df = to_pyhard_frame(X, y)
            M = measures_with_timeout(df, MEASURES)
            elapsed = time.perf_counter() - t0

            print(f"\n=== {meta['name']} (id {meta['id']}) ===")
            print(
                f"n_instances={meta['n_instances']}  "
                f"n_features={meta['n_features']}  n_classes={meta['n_classes']}"
            )
            print(M.agg(["mean", "median"]).T.to_string())
            pct_kdn0 = 100 * (M["feature_kDN"] == 0).mean()
            print(f"kDN == 0 (trivialmente faceis): {pct_kdn0:.1f}%")
            print(f"tempo: {elapsed:.1f}s")

            rows.append(
                {
                    "nome": meta["name"],
                    "n": meta["n_instances"],
                    "pct_kDN_0": round(pct_kdn0, 1),
                    "kDN_medio": M["feature_kDN"].mean(),
                    "N1_medio": M["feature_N1"].mean(),
                    "LSC_medio": M["feature_LSC"].mean(),
                }
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"\n=== dataset {did}: FALHOU apos {elapsed:.1f}s ===")
            print(f"{type(exc).__name__}: {exc}")

    if rows:
        tabela = pd.DataFrame(rows)
        print("\n=== tabela comparativa ===")
        print(tabela.to_string(index=False))

        out_dir = Path("resultados")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "multi.csv"
        tabela.to_csv(out_path, index=False)
        print(f"\nsalvo em {out_path}")


if __name__ == "__main__":
    main()
