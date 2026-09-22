"""Regera o metadata do IC7 sem rodar o pyispace.

Para cada dataset, le resultados/table_<nome>.csv, roda
isaspace.isa.to_isa_metadata e grava em resultados/isa/<nome>/ o metadata.csv
e os arquivos auxiliares que o engine copia (annotations.json,
degenerate_report.csv, feature_info.csv). As saidas do pyispace na mesma pasta
nao sao tocadas. Precisa do .venv (pyhard/pyispace, para o filtro de
degeneradas).

Uso: python scripts/build_metadata.py [nome ...]      (padrao: os quatro)
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import pandas as pd  # noqa: E402

from isaspace.isa import to_isa_metadata  # noqa: E402

DATASETS = ["iris", "diabetes", "blood-transfusion-service-center", "hill-valley"]


def main(nomes):
    for nome in nomes:
        outdir = RAIZ / "resultados" / "isa" / nome
        table = pd.read_csv(RAIZ / "resultados" / f"table_{nome}.csv", index_col=0)
        metadata, info = to_isa_metadata(table, outdir=outdir)
        caidas = ", ".join(info["degenerate_report"]["feature"]) or "nenhuma"
        print(f"{nome:34s} {metadata.shape[0]} instancias, {len(info['features_kept'])} "
              f"features; degeneradas: {caidas}; tipos: {info['annotation_types']}")


if __name__ == "__main__":
    main(sys.argv[1:] or DATASETS)
