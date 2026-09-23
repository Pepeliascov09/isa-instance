"""Run pyispace's PILOT + TRACE for iris and diabetes.

Reads resultados/table_<name>.csv, converts it with isaspace.isa.to_isa_metadata,
runs isaspace.isa.run_isa and writes to resultados/isa/<name>/. At the end of
each dataset it prints: files written and their shapes, dropped features with
their variances, trace.summary, pyispace's 'good' rate x true accuracy (the
guard), Spearman of z_1/z_2 with the ih column of the original table, and the
time spent.

Usage: python scripts/run_isa_all.py [name ...]      (default: iris diabetes)
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from isaspace.isa import ROW_ORIGINAL, run_isa, to_isa_metadata  # noqa: E402

DATASETS = ["iris", "diabetes"]
TABLES_DIR = ROOT / "resultados"
OUTPUT_DIR = ROOT / "resultados" / "isa"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


def list_files(outdir):
    """One row per file: CSVs with (rows x columns), others with bytes."""
    rows = []
    for p in sorted(outdir.iterdir()):
        if p.suffix == ".csv":
            df = pd.read_csv(p, index_col=0)
            rows.append((p.name, f"{df.shape[0]} x {df.shape[1]}", ", ".join(map(str, df.columns))))
        else:
            rows.append((p.name, f"{p.stat().st_size} bytes", ""))
    return pd.DataFrame(rows, columns=["file", "shape", "columns"]).set_index("file")


def spearman_with_ih(outdir, table, row_original):
    """Spearman rho between z_1/z_2 (coordinates.csv) and ih of the original table.

    The join uses row_original (instances -> original index) so it does not
    depend on the row order.
    """
    coords = pd.read_csv(outdir / "coordinates.csv", index_col="Row")
    # pyispace writes the labels "1".."n", which read_csv reads as integers
    coords.index = coords.index.astype(str)
    coords.index.name = row_original.index.name
    coords = coords.join(row_original)
    ih = table["ih"].reindex(coords[ROW_ORIGINAL].to_numpy()).to_numpy()
    out = {}
    for z in ("z_1", "z_2"):
        rho, p = spearmanr(coords[z].to_numpy(), ih)
        out[z] = {"rho": rho, "p_value": p}
    return pd.DataFrame(out).T


def run(name):
    table_csv = TABLES_DIR / f"table_{name}.csv"
    outdir = OUTPUT_DIR / name
    print(f"\n{'=' * 78}\n{name}: {table_csv} -> {outdir}\n{'=' * 78}")

    table = pd.read_csv(table_csv, index_col=0)
    t0 = time.perf_counter()
    metadata, info = to_isa_metadata(table, outdir=outdir)
    t_meta = time.perf_counter() - t0
    print(
        f"metadata: {metadata.shape[0]} instances, "
        f"{len(info['features_kept'])} features kept, "
        f"{len(info['algos'])} algorithms (performance = {info['performance_source']})"
    )

    t1 = time.perf_counter()
    model = run_isa(metadata, outdir, table=table)
    t_isa = time.perf_counter() - t1

    print("\n-- files written --")
    print(list_files(outdir).to_string())

    print("\n-- dropped features (variance after pyispace preprocessing) --")
    if info["features_dropped"]:
        print(pd.DataFrame(info["features_dropped"]).set_index("feature").to_string())
    else:
        print("none")

    print("\n-- trace.summary (footprint_performance.csv) --")
    print(model.trace.summary.round(4).to_string())

    print("\n-- guard: 'good' rate (Ybin) x true accuracy --")
    print(model.ybin_check.round(4).to_string())
    print(f"assertion passed (tolerance 0.15) for {len(model.ybin_check)} algorithms")

    print("\n-- Spearman between coordinates and ih of the original table --")
    print(spearman_with_ih(outdir, table, info["row_original"]).round(4).to_string())

    print(
        f"\n-- time: to_isa_metadata {t_meta:.1f}s | run_isa (PILOT+TRACE+CSVs) "
        f"{t_isa:.1f}s | total {t_meta + t_isa:.1f}s --"
    )
    return model


def main(names):
    times = {}
    for name in names:
        t0 = time.perf_counter()
        run(name)
        times[name] = time.perf_counter() - t0
    print("\n== time per dataset ==")
    for name, t in times.items():
        print(f"{name:12s} {t:7.1f}s")


if __name__ == "__main__":
    main(sys.argv[1:] or DATASETS)
