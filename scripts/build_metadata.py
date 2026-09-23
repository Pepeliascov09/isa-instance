"""Regenerate the IC7 metadata without running pyispace.

For each dataset, read resultados/table_<name>.csv, run
isaspace.isa.to_isa_metadata and write to resultados/isa/<name>/ the
metadata.csv and the auxiliary files the engine copies (annotations.json,
degenerate_report.csv, feature_info.csv). The pyispace outputs in the same
folder are not touched. Needs the .venv (pyhard/pyispace, for the degenerate
filter).

Usage: python scripts/build_metadata.py [name ...]      (default: the four)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from isaspace.isa import to_isa_metadata  # noqa: E402

DATASETS = ["iris", "diabetes", "blood-transfusion-service-center", "hill-valley"]


def main(names):
    for name in names:
        outdir = ROOT / "resultados" / "isa" / name
        table = pd.read_csv(ROOT / "resultados" / f"table_{name}.csv", index_col=0)
        metadata, info = to_isa_metadata(table, outdir=outdir)
        dropped = ", ".join(info["degenerate_report"]["feature"]) or "none"
        print(f"{name:34s} {metadata.shape[0]} instances, {len(info['features_kept'])} "
              f"features; degenerate: {dropped}; types: {info['annotation_types']}")


if __name__ == "__main__":
    main(sys.argv[1:] or DATASETS)
