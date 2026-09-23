"""Run instancespace (isaspace.engine) for the four IC7 datasets.

Reads resultados/isa/<name>/metadata.csv (written by
isaspace.isa.to_isa_metadata, with the class, ih and n_wrong annotations) and
writes resultados/is/<name>/. The resultados/isa/ folder (pyispace) is not
touched. For each dataset it prints the progress per stage, the timings, the
near-duplicate correction before TRACE, the sifted_report and the status of
the footprints.

Needs the .venv-isa (Python 3.12, instancespace 0.3.0):
    .venv-isa/bin/python scripts/run_is_all.py [name ...]   (default: the four)
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

from isaspace.engine import run_instancespace  # noqa: E402
from isaspace.ui.loader_is import load_is_output  # noqa: E402

DATASETS = ["iris", "diabetes", "blood-transfusion-service-center", "hill-valley"]
INPUT_DIR = ROOT / "resultados" / "isa"
OUTPUT_DIR = ROOT / "resultados" / "is"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


def run(name):
    metadata = INPUT_DIR / name / "metadata.csv"
    outdir = OUTPUT_DIR / name
    print(f"\n{'=' * 78}\n{name}: {metadata} -> {outdir}\n{'=' * 78}")
    t0 = time.perf_counter()
    info = run_instancespace(
        metadata, outdir,
        progress=lambda s: print(f"  [{time.perf_counter() - t0:6.1f}s] {s}", flush=True),
    )
    r = load_is_output(outdir)

    print("\n-- timings (s) --")
    print("  " + " | ".join(f"{k} {v:.2f}" for k, v in info["timings_s"].items()))
    rob = info["trace_robustness"]
    print("\n-- near duplicates in the projection (before TRACE) --")
    print(f"  pairs < {rob['threshold']:g}: {rob['near_duplicate_pairs']} "
          f"(identical: {rob['identical_pairs']}, between distinct positions: "
          f"{rob['distinct_close_pairs']}) | jitter applied: {rob['jitter_applied']}"
          + (f" on {rob['perturbed_positions']} positions ({rob['perturbed_points']} points), "
             f"min distance between distinct {rob['min_distance_distinct_before']:.3g} -> "
             f"{rob['min_distance_distinct_after']:.3g}"
             if rob["jitter_applied"] else ""))
    print("\n-- sifted_report --")
    print(r.sifted_report.to_string(index=False))
    print("\n-- footprints --")
    rows = [(a, k, f.status, len(f.polygons), round(f.area, 4), f.purity)
            for (a, k), f in r.footprints.items()]
    print(pd.DataFrame(rows, columns=["algo", "type", "status", "parts", "area", "purity"])
          .to_string(index=False))
    for fp in (r.footprint_space, r.footprint_hard):
        print(f"  {fp.kind}: {fp.status}, {len(fp.polygons)} parts, area {fp.area:.4f}, "
              f"purity {fp.purity:.3f}")
    o = info["orientation"]
    print(f"\n-- orientation: applied {o['applied']}"
          + (f", rotated {o['angle_deg']:.1f} deg, {o['n_instances_bad']} majority-bad instances, "
             f"gradient R2 {o['gradient_r2']:.3f}" if o["applied"] else f" ({o.get('reason_not_applied')})")
          + (f" | WARNING: {o['warning']}" if o.get("warning") else ""))
    ties = int((r.instances["n_tied_best"] > 1).sum())
    print(f"\n-- ties for the best observed value: {ties} of {r.n} instances --")
    for warning in info["warnings"] + info["instancespace_warnings"]:
        print(f"  warning: {warning}")
    return info


def main(names):
    # instancespace logs at DEBUG on stderr; here only warnings
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    timings = {}
    for name in names:
        t0 = time.perf_counter()
        run(name)
        timings[name] = time.perf_counter() - t0
    print("\n== timings per dataset ==")
    for name, t in timings.items():
        print(f"{name:34s} {t:7.1f}s")


if __name__ == "__main__":
    main(sys.argv[1:] or DATASETS)
