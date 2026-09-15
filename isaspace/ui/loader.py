"""Leitor da pasta de saida ISA (layout MATILDA / pyhard) para a interface.

REGRA ARQUITETURAL: este modulo nao importa pyispace, pyhard nem sklearn.
Le os CSVs de resultados/isa/<nome>/ e a tabela por instancia
resultados/table_<nome>.csv, e devolve um unico objeto IsaResult.

Arquivos lidos (todos indexados por Row = 1..n, salvo metadata.csv):
  coordinates.csv            z_1, z_2 (PILOT)
  data_space.csv             d_1, d_2 + atributos originais (build_data_space.py)
  algorithm_bin.csv          bom/ruim por algoritmo        -> algo_<x>_bin
  algorithm_raw.csv          desempenho bruto (proba)      -> algo_<x>
  feature_raw.csv            medidas cruas                 -> feature_<x>
  feature_process.csv        medidas pos Yeo-Johnson/z     -> feature_<x>_z
  beta_easy.csv              IsBetaEasy
  good_algos.csv             NumGoodAlgos
  portfolio.csv              Best_Algorithm (1-based)      + best_algo (nome)
  footprint_performance.csv  resumo do TRACE por algoritmo
  projection_matrix.csv      matriz A do PILOT (2 x features)
  metadata.csv               indice 'instances'; usa so row_original
  footprint_<algo>_<good|best>.csv   poligonos separados por linhas NaN
  options.json               (opcional) para ler trace.PI
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROW = "Row"
INDEX_META = "instances"
ROW_ORIGINAL = "row_original"
COLS_TABELA = ["class", "ih", "n_wrong"]
PI_PADRAO = 0.55
_RE_FOOTPRINT = re.compile(r"^footprint_(?P<algo>.+)_(?P<tipo>good|best)\.csv$")


@dataclass
class IsaResult:
    """Conteudo de uma pasta de saida ISA, pronto para a interface."""

    name: str
    path: Path
    instances: pd.DataFrame            # uma linha por instancia, indice Row
    footprints: dict                   # {(algo, tipo): [[(z_1, z_2), ...], ...]}
    footprint_performance: pd.DataFrame  # indice Algorithm
    projection_matrix: pd.DataFrame    # linhas z_1/z_2, colunas = features
    suspect_footprints: dict           # {(algo, tipo): purity} com purity < pi
    empty_footprints: list             # (algo, tipo) sem nenhum poligono
    pi: float                          # trace.PI usado (options.json) ou 0.55
    algos: list                        # nomes sem prefixo, na ordem do pyispace
    features: list                     # nomes das features usadas no PILOT
    data_columns: list                 # colunas dos atributos originais
    features_dropped: list = field(default_factory=list)  # medidas fora do PILOT
    options: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.instances)

    @property
    def n_features_used(self) -> int:
        return len(self.features)

    @property
    def n_features_dropped(self) -> int:
        return len(self.features_dropped)


def list_available(root) -> list:
    """Nomes das subpastas de `root` que contem um coordinates.csv."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        p.name for p in root.iterdir() if p.is_dir() and (p / "coordinates.csv").is_file()
    )


def _ler(path: Path, index_col=ROW) -> pd.DataFrame:
    if not path.is_file():
        dica = ""
        if path.name == "data_space.csv":
            dica = " (gere com scripts/build_data_space.py)"
        raise FileNotFoundError(f"{path.name} nao encontrado em {path.parent}{dica}")
    return pd.read_csv(path, index_col=index_col)


def _split_polygons(df: pd.DataFrame) -> list:
    """Quebra o CSV de footprint nas linhas NaN; poligonos com < 3 pontos caem."""
    arr = df[["z_1", "z_2"]].to_numpy(dtype=float)
    poligonos, atual = [], []
    for x, y in arr:
        if np.isnan(x) or np.isnan(y):
            if len(atual) >= 3:
                poligonos.append(atual)
            atual = []
        else:
            atual.append((float(x), float(y)))
    if len(atual) >= 3:
        poligonos.append(atual)
    return poligonos


def _juntar(base: pd.DataFrame, parte: pd.DataFrame, origem: str) -> pd.DataFrame:
    """Join 1 para 1 pelo indice Row; qualquer desalinhamento vira erro."""
    if not parte.index.equals(base.index):
        raise ValueError(
            f"{origem}: indice Row nao coincide com coordinates.csv "
            f"({len(parte)} linhas contra {len(base)})"
        )
    out = base.join(parte, how="inner")
    if len(out) != len(base):
        raise ValueError(f"{origem}: join perdeu linhas ({len(out)} de {len(base)})")
    return out


def load_isa_output(dirpath, table_path=None) -> IsaResult:
    """Le a pasta de saida ISA `dirpath` e a tabela original correspondente.

    `table_path` default: <dirpath>/../../table_<nome>.csv, ou seja,
    resultados/table_<nome>.csv para resultados/isa/<nome>/.
    """
    path = Path(dirpath)
    if not path.is_dir():
        raise NotADirectoryError(f"pasta ISA invalida: {path}")
    name = path.name
    if table_path is None:
        table_path = path.parent.parent / f"table_{name}.csv"
    table_path = Path(table_path)

    coords = _ler(path / "coordinates.csv")
    n = len(coords)
    esperado = pd.RangeIndex(1, n + 1, name=ROW)
    if not coords.index.equals(esperado):
        raise ValueError(f"coordinates.csv: esperava Row de 1 a {n} sem falhas")

    data_space = _ler(path / "data_space.csv")
    algo_bin = _ler(path / "algorithm_bin.csv")
    algo_raw = _ler(path / "algorithm_raw.csv")
    feat_raw = _ler(path / "feature_raw.csv")
    feat_proc = _ler(path / "feature_process.csv")
    beta = _ler(path / "beta_easy.csv")
    good = _ler(path / "good_algos.csv")
    portfolio = _ler(path / "portfolio.csv")
    perf = _ler(path / "footprint_performance.csv")
    perf.index.name = "Algorithm"
    proj = _ler(path / "projection_matrix.csv")
    metadata = _ler(path / "metadata.csv", index_col=INDEX_META)
    metadata.index.name = ROW
    if ROW_ORIGINAL not in metadata.columns:
        raise ValueError("metadata.csv sem a coluna row_original; regenere com run_isa_all.py")

    algos = list(algo_bin.columns)
    features = list(feat_raw.columns)

    # --- tabela original via row_original, exigindo correspondencia 1 para 1
    if not table_path.is_file():
        raise FileNotFoundError(f"tabela original nao encontrada: {table_path}")
    table = pd.read_csv(table_path, index_col=0)
    faltam = [c for c in COLS_TABELA if c not in table.columns]
    if faltam:
        raise ValueError(f"{table_path.name} sem as colunas {faltam}")
    ro = metadata[ROW_ORIGINAL]
    if not ro.is_unique:
        raise ValueError("row_original tem valores repetidos")
    if len(ro) != len(table) or set(ro.to_numpy()) != set(table.index.to_numpy()):
        raise ValueError(
            f"row_original ({len(ro)} valores) nao casa 1 para 1 com o indice de "
            f"{table_path.name} ({len(table)} linhas)"
        )
    extra = table.loc[ro.to_numpy(), COLS_TABELA].copy()
    extra.index = metadata.index

    # medidas que a tabela original tinha mas o PILOT nao usou (descartadas por
    # variancia/NaN em to_isa_metadata). Nao recalcula nada: compara os nomes.
    todas_features = [c[len("feature_"):] for c in table.columns if c.startswith("feature_")]
    usadas = set(features)
    features_dropped = [f for f in todas_features if f not in usadas]

    # --- atributos originais: renomeia colisoes com as colunas derivadas
    reservadas = set(
        list(coords.columns)
        + [f"algo_{a}" for a in algos] + [f"algo_{a}_bin" for a in algos]
        + [f"feature_{f}" for f in features] + [f"feature_{f}_z" for f in features]
        + list(beta.columns) + list(good.columns) + list(portfolio.columns)
        + [ROW_ORIGINAL, "best_algo", ROW] + COLS_TABELA
    )
    data_columns = [c for c in data_space.columns if c not in ("d_1", "d_2")]
    renomear = {c: f"attr_{c}" for c in data_columns if c in reservadas}
    if renomear:
        data_space = data_space.rename(columns=renomear)
        data_columns = [renomear.get(c, c) for c in data_columns]

    partes = [
        ("data_space.csv", data_space),
        ("algorithm_raw.csv", algo_raw.add_prefix("algo_")),
        ("algorithm_bin.csv", algo_bin.add_prefix("algo_").add_suffix("_bin")),
        ("feature_raw.csv", feat_raw.add_prefix("feature_")),
        ("feature_process.csv", feat_proc.add_prefix("feature_").add_suffix("_z")),
        ("beta_easy.csv", beta),
        ("good_algos.csv", good),
        ("portfolio.csv", portfolio),
        ("metadata.csv", metadata[[ROW_ORIGINAL]]),
        (f"{table_path.name} (via row_original)", extra),
    ]
    instances = coords
    for origem, parte in partes:
        instances = _juntar(instances, parte, origem)

    best_idx = instances["Best_Algorithm"].to_numpy()
    instances["best_algo"] = [
        algos[i - 1] if 1 <= i <= len(algos) else None for i in best_idx
    ]
    instances["class"] = instances["class"].astype(str)

    # --- footprints
    footprints, vazias = {}, []
    for arquivo in sorted(path.glob("footprint_*_*.csv")):
        m = _RE_FOOTPRINT.match(arquivo.name)
        if m is None:
            continue
        chave = (m.group("algo"), m.group("tipo"))
        poligonos = _split_polygons(pd.read_csv(arquivo, index_col=ROW))
        footprints[chave] = poligonos
        if not poligonos:
            vazias.append(chave)

    options = {}
    opts_path = path / "options.json"
    if opts_path.is_file():
        options = json.loads(opts_path.read_text())
    pi = float(options.get("trace", {}).get("PI", PI_PADRAO))

    suspeitas = {}
    for algo, tipo in footprints:
        coluna = "Purity_Good" if tipo == "good" else "Purity_Best"
        if algo in perf.index and coluna in perf.columns:
            purity = float(perf.loc[algo, coluna])
            if purity < pi:
                suspeitas[(algo, tipo)] = purity

    return IsaResult(
        name=name,
        path=path,
        instances=instances,
        footprints=footprints,
        footprint_performance=perf,
        projection_matrix=proj,
        suspect_footprints=suspeitas,
        empty_footprints=vazias,
        pi=pi,
        algos=algos,
        features=features,
        data_columns=data_columns,
        features_dropped=features_dropped,
        options=options,
    )
