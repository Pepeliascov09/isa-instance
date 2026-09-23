"""Measure the engine run time on synthetic metadata (basis of upload.MEASURED_TIMES).

Generates metadata with N instances, 10 features and 6 algorithms
(performance in [0, 1], higher is better, depending on the features so that
PYTHIA has something to learn) and runs it through the same path as the UI:
isaspace.ui.runner, in a subprocess, with the engine's default options. Prints
the total time and the time until the start of each stage.

Usage (in the .venv-isa): python scripts/measure_engine_time.py [N ...] [--repeats R]
    default: 500 1000 2000, 1 repeat
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from isaspace.ui import runner  # noqa: E402

N_FEATURES, N_ALGOS = 10, 6


def synthetic_metadata(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, N_FEATURES))
    weights = rng.normal(size=(N_FEATURES, N_ALGOS))
    y = 1 / (1 + np.exp(-(x @ weights) / 2 + rng.normal(scale=0.5, size=(n, N_ALGOS))))
    df = pd.DataFrame({"instances": [f"i{k}" for k in range(n)]})
    for j in range(N_FEATURES):
        df[f"feature_f{j}"] = x[:, j]
    for j in range(N_ALGOS):
        df[f"algo_a{j}"] = y[:, j]
    return df


def measure(n: int, folder: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    meta = folder / f"metadata_{n}.csv"
    synthetic_metadata(n).to_csv(meta, index=False)
    marks = {}
    t0 = time.perf_counter()
    run = runner.launch(meta, folder / f"output_{n}", {},
                        on_stage=lambda _r, stage: marks.setdefault(stage, time.perf_counter() - t0))
    while not run.finished:
        time.sleep(0.2)
    if not run.ok:
        raise SystemExit(f"n={n}: {run.error}")
    return {"n": n, "total_s": run.duration, **{f"{k}_s": v for k, v in marks.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n", nargs="*", type=int, default=[500, 1000, 2000])
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for n in args.n:
            for r in range(args.repeats):
                row = measure(n, Path(tmp) / f"r{r}")
                print(f"n={n:5d} rep={r} total={row['total_s']:.1f}s", flush=True)
                rows.append(row)
    table = pd.DataFrame(rows).groupby("n").median()
    pd.set_option("display.width", 200)
    print("\nmedian per n (s; columns <STAGE>_s = start of the stage):")
    print(table.round(1).to_string())


if __name__ == "__main__":
    main()
