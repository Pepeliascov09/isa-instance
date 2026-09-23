"""Leitor da pasta gravada por isaspace.engine (instancespace) para a interface.

REGRA ARQUITETURAL: este modulo nao importa instancespace, pyispace, pyhard nem
sklearn; so pandas e numpy. O antigo isaspace/ui/loader.py (saida do pyispace)
continua valendo para resultados/isa/.

O que muda em relacao ao loader antigo:
- Row e o rotulo da instancia (coluna instances do metadata), lido como texto,
  e nao precisa ser 1..n;
- footprints no esquema Row, Part, Ring, Vertex, z_1, z_2: um Poligono por
  Part, com os furos (Ring hole_k) preservados; footprint sem arquivo e VAZIA;
- binarios True/False (algorithm_bin, algorithm_svm, beta_easy) viram bool;
- portfolio.csv (PRELIM) e 1-based e portfolio_svm.csv (PYTHIA) e 0-based com
  -1 = nenhum: os dois viram nome de algoritmo, None para nenhum;
- anotacoes do metadata.csv (toda coluna que nao e instances, source, feature_*
  nem algo_*) entram em instances; o tipo vem de annotations.json quando a
  pasta o tem (declarado) e da heuristica _tipo_anotacao quando nao
  (inferido), com a origem em `annotation_origins`;
- degenerate_report.csv e feature_info.csv, quando existem, e a tabela
  features_table() com uma linha por feature recebida;
- CLOISTER: bounds.csv e bounds_prunned.csv como Poligono;
- coordinates.csv (z do PILOT) desenha; coordinates_trace.csv (z que o TRACE
  usou, so quando houve jitter) fica em `coordinates_trace`;
- footprints space e hard, sifted_report, sifted_correlations,
  sifted_silhouette, pilot_r2, pythia_proba (pr0_sub e pr0_hat),
  pythia_confusion, pythia_selection, svm_table, run_info e run_options expostos.

O formato de cada arquivo esta em docs/output_format.md.

Colunas de IsResult.instances (indice Row, rotulos em texto):
  z_1, z_2                     coordinates.csv
  source                       coluna source do metadata, se houver
  feature_<f>                  valores de entrada de TODAS as features do metadata
                               (as do SIFTED vem de feature_raw.csv, iguais ao metadata)
  feature_<f>_z                feature_process.csv (so as do SIFTED)
  algo_<a>                     algorithm_raw.csv
  algo_<a>_bin                 algorithm_bin.csv (bool: bom no desempenho observado)
  algo_<a>_svm                 algorithm_svm.csv (bool: bom segundo o PYTHIA)
  NumGoodAlgos, IsBetaEasy     good_algos.csv, beta_easy.csv (bool)
  best_algo, best_algo_svm     portfolio.csv, portfolio_svm.csv -> nome ou None
  <anotacoes>                  do metadata; ann_<nome> se o nome colidir
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROW = "Row"
TIPOS_FOOTPRINT = ("good", "best")
NUMERICA = "numerica"
CATEGORICA = "categorica"
INTEIRA = "numerica_inteira"     # numerica com valores inteiros (contagens)
IDENTIFICADOR = "identifier"     # chave/id: fica na tabela e na exportacao, fora dos graficos
TIPOS_ANOTACAO = (CATEGORICA, NUMERICA, INTEIRA, IDENTIFICADOR)
OK, VAZIA, SUSPEITA = "ok", "vazia", "suspeita"
COLUNAS_FOOTPRINT = ["Row", "Part", "Ring", "Vertex", "z_1", "z_2"]
_BOOL = {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False}
_RE_Z = re.compile(r"^Z_\{(\d+)\}$")
_RE_FURO = re.compile(r"^hole_(\d+)$")


def _area(anel) -> float:
    """Area pela formula do laco (sem sinal); anel (k, 2) sem repetir o 1o vertice."""
    x, y = anel[:, 0], anel[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def e_numerica(tipo: str) -> bool:
    return tipo in (NUMERICA, INTEIRA)


def colunas_de_anotacao(colunas) -> list:
    """Colunas do metadata que nao sao instances, source, feature_* nem algo_*."""
    return [c for c in colunas if str(c).casefold() not in ("instances", "source")
            and not str(c).casefold().startswith(("feature_", "algo_"))]


def erros_tipos_declarados(meta: pd.DataFrame, tipos) -> list:
    """Problemas de uma declaracao {anotacao: tipo} (annotations.json) para o
    metadata `meta`; lista vazia se estiver tudo certo. Usada pelo loader, pelo
    engine (antes de rodar) e pela validacao do upload na interface."""
    if not isinstance(tipos, dict):
        return ["annotations.json tem de ser um objeto {anotação: tipo}"]
    anotacoes = colunas_de_anotacao(meta.columns)
    erros = []
    for col, tipo in tipos.items():
        if tipo not in TIPOS_ANOTACAO:
            erros.append(f"{col!r}: tipo {tipo!r} não é um de {', '.join(TIPOS_ANOTACAO)}")
        elif col not in anotacoes:
            erros.append(f"{col!r} não é uma coluna de anotação do metadata")
        elif e_numerica(tipo):
            v = pd.to_numeric(meta[col], errors="coerce")
            if (v.isna() & meta[col].notna()).any():
                erros.append(f"{col!r} declarada {tipo} tem valores não numéricos")
            elif tipo == INTEIRA and (np.mod(v.dropna(), 1) != 0).any():
                erros.append(f"{col!r} declarada {tipo} tem valores não inteiros")
    return erros


def _no_anel(pts, anel, tol):
    """(dentro ou na borda, na borda) de cada ponto em relacao a um anel."""
    x, y = pts[:, :1], pts[:, 1:2]
    x1, y1 = anel[None, :, 0], anel[None, :, 1]
    x2, y2 = np.roll(anel[:, 0], -1)[None, :], np.roll(anel[:, 1], -1)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        cruza = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
        dx, dy = x2 - x1, y2 - y1
        l2 = dx * dx + dy * dy
        t = np.clip(((x - x1) * dx + (y - y1) * dy) / np.where(l2 == 0, 1, l2), 0, 1)
    borda = (((x1 + t * dx - x) ** 2 + (y1 + t * dy - y) ** 2) <= tol * tol).any(axis=1)
    return (cruza.sum(axis=1) % 2 == 1) | borda, borda


@dataclass
class Poligono:
    """Um poligono em z_1 x z_2: anel externo e furos, sem repetir o 1o vertice."""

    exterior: np.ndarray                       # (k, 2)
    furos: list = field(default_factory=list)  # lista de (k, 2)

    @property
    def area(self) -> float:
        return _area(self.exterior) - sum(_area(f) for f in self.furos)

    @property
    def n_vertices(self) -> int:
        return len(self.exterior)

    def contem(self, pts, tol=1e-9) -> np.ndarray:
        """Pontos (n, 2) dentro ou na borda do poligono e fora dos furos (a
        borda do furo conta como dentro), como o pointwise_covers do TRACE."""
        pts = np.asarray(pts, dtype=float)
        dentro, _ = _no_anel(pts, self.exterior, tol)
        for furo in self.furos:
            no_furo, borda = _no_anel(pts, furo, tol)
            dentro &= ~(no_furo & ~borda)
        return dentro


@dataclass
class Footprint:
    """Footprint de um algoritmo (ou do espaco / das instancias dificeis, com
    algo None); `poligonos` vazio quando nao ha arquivo."""

    algo: str | None
    tipo: str                       # "good" | "best" | "space" | "hard"
    poligonos: list                 # list[Poligono], um por Part
    status: str                     # "ok" | "vazia" | "suspeita" (pureza < trace.purity)
    arquivo: str | None
    area_normalizada: float         # de footprint_performance.csv
    densidade_normalizada: float
    pureza: float

    @property
    def vazia(self) -> bool:
        return not self.poligonos

    @property
    def area(self) -> float:
        return sum(p.area for p in self.poligonos)

    def contem(self, pts) -> np.ndarray:
        pts = np.asarray(pts, dtype=float)
        dentro = np.zeros(len(pts), dtype=bool)
        for p in self.poligonos:
            dentro |= p.contem(pts)
        return dentro


@dataclass
class IsResult:
    """Conteudo de uma pasta gravada por isaspace.engine, pronto para a interface."""

    name: str
    path: Path
    instances: pd.DataFrame          # uma linha por instancia, indice Row (texto)
    algos: list                      # ordem do instancespace (indices dos portfolios)
    features: list                   # features escolhidas pelo SIFTED
    features_input: list             # features do metadata, na ordem do sifted_report
    features_all: list               # features do metadata.csv, na ordem do arquivo
    annotations: dict                # {coluna em instances: "numerica" | "categorica"}
    annotation_renames: dict         # {nome no metadata: nome em instances}
    annotation_origins: dict         # {coluna em instances: "declarado" | "inferido" | "forcado"}
    source_column: str | None        # "source" se o metadata tinha source
    footprints: dict                 # {(algo, tipo): Footprint}, todos os algos x tipos
    footprint_space: Footprint       # TRACE: todas as instancias
    footprint_hard: Footprint        # TRACE: instancias nao beta-faceis
    footprint_performance: pd.DataFrame   # indice Algorithm
    projection_matrix: pd.DataFrame  # linhas z_1, z_2; colunas = features
    bounds: Poligono | None          # CLOISTER, bounds.csv
    bounds_pruned: Poligono | None   # CLOISTER, bounds_prunned.csv
    coordinates_trace: pd.DataFrame | None   # z_1, z_2 usados pelo TRACE (so com jitter)
    sifted_report: pd.DataFrame
    sifted_correlations: pd.DataFrame  # feature, algorithm, rho, pval (formato longo)
    sifted_silhouette: pd.DataFrame  # k, silhouette, used, best
    pilot_r2: pd.DataFrame           # variable, kind, r2
    pythia_proba: pd.DataFrame       # P(ruim) fora da amostra (pr0_sub), instancia x algoritmo
    pythia_proba_hat: pd.DataFrame   # P(ruim) dentro da amostra (pr0_hat)
    pythia_confusion: pd.DataFrame   # indice Algorithm; tn, fp, fn, tp
    pythia_selection: pd.DataFrame   # indice Row; selection0, selection1 (nome ou None)
    svm_table: pd.DataFrame          # indice Algorithm (+ Oracle, Selector)
    run_info: dict
    run_options: dict
    pi: float                        # trace.purity efetivo
    degenerate_report: pd.DataFrame | None   # feature, var_bruta, iqr, motivo (antes do engine)
    feature_info: pd.DataFrame | None        # feature, family

    @property
    def n(self) -> int:
        return len(self.instances)

    @property
    def n_features_used(self) -> int:
        return len(self.features)

    @property
    def n_features_dropped(self) -> int:
        return len(self.features_input) - len(self.features)

    @property
    def features_fora_pilot(self) -> list:
        """Features do metadata que o SIFTED nao passou ao PILOT."""
        return [f for f in self.features_all if f not in self.features]

    @property
    def anotacoes_inferidas(self) -> list:
        return [c for c, o in self.annotation_origins.items() if o == "inferido"]

    def instancias_na_footprint(self, fp) -> list:
        """Rotulos das instancias dentro (ou na borda) da footprint, no z que o
        TRACE usou (coordinates_trace quando houve jitter)."""
        z = self.coordinates_trace if self.coordinates_trace is not None else self.instances[["z_1", "z_2"]]
        return list(z.index[fp.contem(z.to_numpy())])

    def features_table(self) -> pd.DataFrame:
        """Uma linha por feature recebida: as degeneradas de degenerate_report
        (descartadas antes do engine) e as do sifted_report.

        Colunas: feature, [family], status, motivo, substituida_por,
        max_abs_rho, algoritmo_rho, pval, r2_pilot. Ordem: a de feature_info
        quando existe; senao a do metadata, com as degeneradas no fim.
        """
        rep = self.sifted_report
        sif = self.run_options.get("sifted", {})
        lim_rho, lim_p = sif.get("rho"), sif.get("pval")
        r2 = self.pilot_r2[self.pilot_r2["kind"] == "feature"].set_index("variable")["r2"]
        linhas = []
        if self.degenerate_report is not None:
            for d in self.degenerate_report.itertuples(index=False):
                linhas.append({"feature": str(d.feature), "status": "dropped_degenerate",
                               "motivo": str(d.motivo)})
        for x in rep.itertuples(index=False):
            cluster = None if pd.isna(x.cluster) else int(x.cluster)
            if x.status == "kept":
                motivo = f"mantida (cluster {cluster})" if cluster else "mantida"
            elif x.status == "dropped_correlation":
                motivo = (f"nenhum algoritmo com |rho| ≥ {lim_rho} e p ≤ {lim_p} "
                          f"(maior |rho| {abs(x.rho):.2f}, p = {x.pval:.2g})")
            elif x.status == "dropped_redundancy":
                motivo = f"redundante no cluster {cluster}: ficou {x.kept_instead}"
            elif x.status == "dropped_preprocessing":
                motivo = "removida pelo PREPROCESSING do instancespace"
            else:
                motivo = "motivo não reconstruído (ver run_info.avisos)"
            linhas.append({
                "feature": x.feature, "status": x.status, "motivo": motivo,
                "substituida_por": x.kept_instead if x.status == "dropped_redundancy" else None,
                "max_abs_rho": abs(x.rho) if pd.notna(x.rho) else np.nan,
                "algoritmo_rho": x.rho_algo, "pval": x.pval,
                "r2_pilot": float(r2[x.feature]) if x.status == "kept" and x.feature in r2 else np.nan,
            })
        colunas = ["feature", "status", "motivo", "substituida_por", "max_abs_rho",
                   "algoritmo_rho", "pval", "r2_pilot"]
        tabela = pd.DataFrame(linhas).reindex(columns=colunas)
        if self.feature_info is not None:
            familia = dict(zip(self.feature_info["feature"].astype(str), self.feature_info["family"]))
            tabela.insert(1, "family", tabela["feature"].map(familia))
            ordem = list(self.feature_info["feature"].astype(str))
        else:
            ordem = list(self.features_all)
        posicao = {f: i for i, f in enumerate(ordem)}
        tabela["_ordem"] = tabela["feature"].map(lambda f: posicao.get(f, len(ordem)))
        return tabela.sort_values("_ordem", kind="stable").drop(columns="_ordem").reset_index(drop=True)

    @property
    def empty_footprints(self) -> list:
        return [k for k, f in self.footprints.items() if f.status == VAZIA]

    @property
    def suspect_footprints(self) -> dict:
        return {k: f.pureza for k, f in self.footprints.items() if f.status == SUSPEITA}

    def _pivo(self, valor) -> pd.DataFrame:
        c = self.sifted_correlations
        tabela = c.pivot(index="feature", columns="algorithm", values=valor)
        return tabela.reindex(index=list(dict.fromkeys(c["feature"])), columns=self.algos)

    @property
    def sifted_rho(self) -> pd.DataFrame:
        """Correlacao feature x algoritmo (linhas = features de entrada do SIFTED)."""
        return self._pivo("rho")

    @property
    def sifted_pval(self) -> pd.DataFrame:
        return self._pivo("pval")


def list_available(root) -> list:
    """Nomes das subpastas de `root` com run_info.json (execucao completa do engine)."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "run_info.json").is_file())


# --------------------------------------------------------------------------- #
# leitura
# --------------------------------------------------------------------------- #
def _exigir(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{path.name} nao encontrado em {path.parent}")
    return path


def _ler_por_row(path: Path, dtype=None, skiprows=0) -> pd.DataFrame:
    """CSV indexado por Row, com Row como texto (rotulo, nunca inteiro)."""
    tipos = {ROW: str} if dtype is None else {ROW: str, **dtype}
    df = pd.read_csv(_exigir(path), dtype=tipos, skiprows=skiprows)
    if ROW not in df.columns:
        raise ValueError(f"{path.name} sem coluna {ROW}")
    df = df.set_index(ROW)
    if df.index.isna().any():
        raise ValueError(f"{path.name}: Row vazio")
    return df


def _alinhar(base: pd.Index, df: pd.DataFrame, origem: str) -> pd.DataFrame:
    """As saidas do instancespace seguem a ordem das instancias; exige a mesma."""
    if not df.index.equals(base):
        raise ValueError(f"{origem}: Row nao coincide com coordinates.csv "
                         f"({len(df)} linhas contra {len(base)})")
    return df


def _para_bool(df: pd.DataFrame, origem: str) -> pd.DataFrame:
    out = {}
    for col in df.columns:
        valores = df[col].astype(str).str.strip().str.lower().map(_BOOL)
        if valores.isna().any():
            ruins = sorted(set(df.loc[valores.isna(), col].astype(str)))[:5]
            raise ValueError(f"{origem}: valores nao binarios em {col}: {ruins}")
        out[col] = valores.astype(bool)
    return pd.DataFrame(out, index=df.index)


def _nome_do_indice(valores, algos, base, origem):
    """Indice de portfolio -> nome do algoritmo; fora de faixa = None."""
    n = len(algos)
    nomes = []
    for v in valores:
        if pd.isna(v):
            nomes.append(None)
            continue
        if int(v) != v:
            raise ValueError(f"{origem}: indice nao inteiro {v}")
        i = int(v) - base
        nomes.append(algos[i] if 0 <= i < n else None)
    return nomes


def _ler_footprint(path: Path) -> list:
    """Arquivo do esquema Row, Part, Ring, Vertex, z_1, z_2 -> list[Poligono]."""
    df = pd.read_csv(path)
    if list(df.columns) != COLUNAS_FOOTPRINT:
        raise ValueError(f"{path.name}: colunas {list(df.columns)}, esperava "
                         f"{COLUNAS_FOOTPRINT} (formato do instancespace)")
    poligonos = []
    for parte, bloco in df.groupby("Part", sort=True):
        aneis = {}
        for anel, pts in bloco.groupby("Ring", sort=False):
            xy = pts.sort_values("Vertex")[["z_1", "z_2"]].to_numpy(dtype=float)
            if len(xy) < 3:
                raise ValueError(f"{path.name}: Part {parte} Ring {anel} com {len(xy)} vertices")
            aneis[str(anel)] = xy
        if "exterior" not in aneis:
            raise ValueError(f"{path.name}: Part {parte} sem Ring exterior")
        furos = []
        for nome in aneis:
            if nome == "exterior":
                continue
            m = _RE_FURO.match(nome)
            if m is None:
                raise ValueError(f"{path.name}: Ring desconhecido {nome!r}")
            furos.append((int(m.group(1)), aneis[nome]))
        poligonos.append(Poligono(aneis["exterior"], [xy for _, xy in sorted(furos, key=lambda t: t[0])]))
    return poligonos


def _ler_bounds(path: Path) -> Poligono | None:
    if not path.is_file():
        return None
    df = pd.read_csv(path, dtype={ROW: str})
    xy = df[["z_1", "z_2"]].to_numpy(dtype=float)
    if len(xy) < 3:
        raise ValueError(f"{path.name}: fronteira com {len(xy)} pontos")
    return Poligono(xy)


def _nomes_ou_none(serie: pd.Series, algos, origem) -> pd.Series:
    """Coluna de nomes de algoritmo; vazio = None; nome desconhecido = erro."""
    out = serie.where(serie.notna(), None).astype(object)
    ruins = sorted({v for v in out if v is not None and v not in algos})
    if ruins:
        raise ValueError(f"{origem}: algoritmos desconhecidos {ruins}")
    return out


def _footprint_especial(path: Path, tipo: str, registro: dict, pi: float) -> Footprint:
    """footprint_space / footprint_hard com as metricas de run_info.json."""
    arquivo = registro.get("arquivo")
    if arquivo is not None and not (path / arquivo).is_file():
        raise FileNotFoundError(f"run_info.json cita {arquivo}, que nao existe em {path}")
    poligonos = _ler_footprint(path / arquivo) if arquivo is not None else []
    pureza = float(registro.get("pureza", np.nan))
    status = VAZIA if not poligonos else (SUSPEITA if pureza < pi else OK)
    return Footprint(
        algo=None, tipo=tipo, poligonos=poligonos, status=status, arquivo=arquivo,
        area_normalizada=float(registro.get("area_normalizada", 1.0 if poligonos else 0.0)),
        densidade_normalizada=float(registro.get("densidade_normalizada", 1.0 if poligonos else 0.0)),
        pureza=pureza,
    )


def _ler_opcional(path: Path, dtype=None) -> pd.DataFrame | None:
    return pd.read_csv(path, dtype=dtype) if path.is_file() else None


def _tipo_anotacao(serie: pd.Series) -> str:
    """categorica: texto, bool ou numero inteiro com no maximo 2 valores (codigo
    binario, como class 0/1); numerica: o resto."""
    if pd.api.types.is_bool_dtype(serie) or not pd.api.types.is_numeric_dtype(serie):
        return CATEGORICA
    v = serie.dropna()
    if len(v) and np.all(np.mod(v.to_numpy(dtype=float), 1) == 0) and v.nunique() <= 2:
        return CATEGORICA
    return NUMERICA


def _texto(v):
    if isinstance(v, (str, bool, np.bool_)):
        return str(v)
    if isinstance(v, (int, float, np.integer, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def _como_categoria(serie: pd.Series) -> pd.Series:
    """Valores em texto (1 -> "1", 2.0 -> "2"), preservando os ausentes como NaN."""
    return serie.map(_texto).where(serie.notna(), np.nan).astype(object)


def _valor(linha, coluna) -> float:
    return float(linha[coluna]) if linha is not None and coluna in linha else np.nan


def _ler_metadata(path: Path):
    """(DataFrame indexado pelo rotulo, coluna source ou None, colunas de anotacao,
    [(coluna, nome) das features])."""
    cols = list(pd.read_csv(_exigir(path), nrows=0).columns)
    baixo = [str(c).casefold() for c in cols]
    if baixo.count("instances") != 1:
        raise ValueError("metadata.csv precisa de exatamente uma coluna instances")
    col_inst = cols[baixo.index("instances")]
    col_src = cols[baixo.index("source")] if "source" in baixo else None
    meta = pd.read_csv(path, dtype={col_inst: str})
    meta = meta.set_index(col_inst)
    if meta.index.isna().any():
        raise ValueError("metadata.csv: coluna instances com rotulo vazio")
    anot = [c for c, b in zip(cols, baixo)
            if c not in (col_inst, col_src) and not b.startswith(("feature_", "algo_"))]
    feats = [(c, c[len("feature_"):]) for c, b in zip(cols, baixo) if b.startswith("feature_")]
    return meta, col_src, anot, feats


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def load_is_output(dirpath, annotation_types=None) -> IsResult:
    """Le a pasta `dirpath` gravada por isaspace.engine.run_instancespace.

    `annotation_types` ({coluna: "numerica" | "categorica"}) sobrepoe a deteccao
    automatica de tipo das anotacoes (usa o nome da coluna no metadata).
    """
    path = Path(dirpath)
    if not path.is_dir():
        raise NotADirectoryError(f"pasta invalida: {path}")
    run_info = json.loads(_exigir(path / "run_info.json").read_text())
    run_options = json.loads(_exigir(path / "run_options.json").read_text())

    coords = _ler_por_row(path / "coordinates.csv")
    if not coords.index.is_unique:
        raise ValueError("coordinates.csv: rotulos de instancia repetidos")
    base = coords.index

    algo_raw = _alinhar(base, _ler_por_row(path / "algorithm_raw.csv"), "algorithm_raw.csv")
    algos = list(algo_raw.columns)
    if run_info.get("algoritmos") not in (None, algos):
        raise ValueError(f"run_info.json lista {run_info['algoritmos']}, algorithm_raw.csv tem {algos}")
    todos = {a: str for a in algos}
    algo_bin = _para_bool(_alinhar(base, _ler_por_row(path / "algorithm_bin.csv", todos),
                                   "algorithm_bin.csv"), "algorithm_bin.csv")
    algo_svm = _para_bool(_alinhar(base, _ler_por_row(path / "algorithm_svm.csv", todos),
                                   "algorithm_svm.csv"), "algorithm_svm.csv")
    feat_raw = _alinhar(base, _ler_por_row(path / "feature_raw.csv"), "feature_raw.csv")
    feat_proc = _alinhar(base, _ler_por_row(path / "feature_process.csv"), "feature_process.csv")
    features = list(feat_raw.columns)
    good = _alinhar(base, _ler_por_row(path / "good_algos.csv"), "good_algos.csv")
    beta = _para_bool(_alinhar(base, _ler_por_row(path / "beta_easy.csv", {"IsBetaEasy": str}),
                               "beta_easy.csv"), "beta_easy.csv")
    port = _alinhar(base, _ler_por_row(path / "portfolio.csv"), "portfolio.csv")
    port_svm = _alinhar(base, _ler_por_row(path / "portfolio_svm.csv"), "portfolio_svm.csv")

    partes = [
        coords[["z_1", "z_2"]],
        feat_raw.add_prefix("feature_"),
        feat_proc.add_prefix("feature_").add_suffix("_z"),
        algo_raw.add_prefix("algo_"),
        algo_bin.add_prefix("algo_").add_suffix("_bin"),
        algo_svm.add_prefix("algo_").add_suffix("_svm"),
        good[["NumGoodAlgos"]],
        beta[["IsBetaEasy"]],
    ]
    instances = pd.concat(partes, axis=1)
    instances["best_algo"] = _nome_do_indice(port["Best_Algorithm"], algos, 1, "portfolio.csv")
    instances["best_algo_svm"] = _nome_do_indice(port_svm["Best_Algorithm"], algos, 0,
                                                 "portfolio_svm.csv")

    # --- metadata: source e anotacoes, pelo rotulo
    meta, col_src, col_anot, col_feats = _ler_metadata(path / "metadata.csv")
    if meta.index.equals(base):
        meta_al = meta
    elif meta.index.is_unique and base.isin(meta.index).all():
        meta_al = meta.loc[base]         # PRELIM/SIFTED usaram um subconjunto
    else:
        raise ValueError(f"metadata.csv: rotulos nao casam com coordinates.csv "
                         f"({len(meta)} linhas contra {len(base)})")
    # features que o SIFTED nao passou ao PILOT: valores de entrada, do metadata
    for col, nome in col_feats:
        if nome not in features:
            instances[f"feature_{nome}"] = pd.to_numeric(meta_al[col], errors="coerce")
    source_column = None
    if col_src is not None:
        instances["source"] = _como_categoria(meta_al[col_src])
        source_column = "source"
    tipos_forcados = dict(annotation_types or {})
    declarados = {}
    if (path / "annotations.json").is_file():
        declarados = json.loads((path / "annotations.json").read_text())
        erros = erros_tipos_declarados(meta.reset_index(), declarados)
        if erros:
            raise ValueError("annotations.json: " + "; ".join(erros))
    annotations, renames, origens = {}, {}, {}
    for col in col_anot:
        destino = col
        while destino in instances.columns or destino == ROW:
            destino = f"ann_{destino}"
        if destino != col:
            renames[col] = destino
        serie = meta_al[col]
        if col in tipos_forcados:
            tipo, origens[destino] = tipos_forcados[col], "forcado"
        elif col in declarados:
            tipo, origens[destino] = declarados[col], "declarado"
        else:
            tipo, origens[destino] = _tipo_anotacao(serie), "inferido"
        if tipo not in TIPOS_ANOTACAO:
            raise ValueError(f"tipo de anotacao invalido para {col}: {tipo!r}")
        if tipo == CATEGORICA:
            instances[destino] = _como_categoria(serie)
        elif tipo == IDENTIFICADOR:
            instances[destino] = serie          # como veio: so identifica a instancia
        else:
            numeros = pd.to_numeric(serie, errors="coerce")
            if (numeros.isna() & serie.notna()).any():
                raise ValueError(f"anotacao {col} ({origens[destino]} {tipo}) tem valores nao numericos")
            instances[destino] = numeros
        annotations[destino] = tipo

    # --- TRACE
    perf = _ler_por_row(path / "footprint_performance.csv")
    perf.index.name = "Algorithm"
    pi = float(run_options["trace"]["purity"])
    mapa = run_info.get("arquivos_footprint", {})
    footprints = {}
    for algo in algos:
        for tipo in TIPOS_FOOTPRINT:
            if algo in mapa:
                arquivo = mapa[algo].get(tipo)
            else:
                padrao = f"footprint_{algo}_{tipo}.csv"
                arquivo = padrao if (path / padrao).is_file() else None
            if arquivo is not None and not (path / arquivo).is_file():
                raise FileNotFoundError(f"run_info.json cita {arquivo}, que nao existe em {path}")
            poligonos = _ler_footprint(path / arquivo) if arquivo is not None else []
            sufixo = "Good" if tipo == "good" else "Best"
            linha = perf.loc[algo] if algo in perf.index else None
            pureza = _valor(linha, f"Purity_{sufixo}")
            status = VAZIA if not poligonos else (SUSPEITA if pureza < pi else OK)
            footprints[(algo, tipo)] = Footprint(
                algo=algo, tipo=tipo, poligonos=poligonos, status=status, arquivo=arquivo,
                area_normalizada=_valor(linha, f"Area_{sufixo}_Normalized"),
                densidade_normalizada=_valor(linha, f"Density_{sufixo}_Normalized"),
                pureza=pureza,
            )

    proj = _ler_por_row(path / "projection_matrix.csv")
    proj.index = [(f"z_{m.group(1)}" if (m := _RE_Z.match(i)) else i) for i in proj.index]

    svm_table = _ler_por_row(path / "svm_table.csv")
    svm_table.index.name = "Algorithm"

    sifted = pd.read_csv(_exigir(path / "sifted_report.csv"), dtype={"feature": str})
    for col in ("cluster", "n_algos_sig"):
        if col in sifted.columns:
            sifted[col] = sifted[col].astype("Int64")

    proba = _alinhar(base, _ler_por_row(path / "pythia_proba.csv"), "pythia_proba.csv")
    hats = [f"{a}_hat" for a in algos]
    if list(proba.columns) != algos + hats:
        raise ValueError(f"pythia_proba.csv: colunas {list(proba.columns)}, esperava {algos + hats}")
    proba_hat = proba[hats].rename(columns=dict(zip(hats, algos)))
    proba = proba[algos]

    conf = pd.read_csv(_exigir(path / "pythia_confusion.csv"), index_col="Algorithm",
                       dtype={"Algorithm": str})
    if list(conf.index) != algos or list(conf.columns) != ["tn", "fp", "fn", "tp"]:
        raise ValueError("pythia_confusion.csv: esperava uma linha por algoritmo e tn, fp, fn, tp")

    sel = _alinhar(base, _ler_por_row(path / "pythia_selection.csv", {"selection0": str, "selection1": str}),
                   "pythia_selection.csv")
    for col in ("selection0", "selection1"):
        sel[col] = _nomes_ou_none(sel[col], algos, f"pythia_selection.csv:{col}")

    r2 = pd.read_csv(_exigir(path / "pilot_r2.csv"), dtype={"variable": str, "kind": str})
    correl = pd.read_csv(_exigir(path / "sifted_correlations.csv"),
                         dtype={"feature": str, "algorithm": str})
    silhueta = pd.read_csv(_exigir(path / "sifted_silhouette.csv"))
    for col in ("used", "best"):
        silhueta[col] = silhueta[col].astype(str).str.lower().map(_BOOL).astype(bool)

    coords_trace = None
    if (path / "coordinates_trace.csv").is_file():
        coords_trace = _alinhar(base, _ler_por_row(path / "coordinates_trace.csv"),
                                "coordinates_trace.csv")[["z_1", "z_2"]]
    jitter = bool(run_info.get("trace_robustez", {}).get("jitter_aplicado", False))
    if jitter != (coords_trace is not None):
        raise ValueError("coordinates_trace.csv tem de existir se, e so se, "
                         "run_info.json registra jitter_aplicado")

    especiais = run_info.get("footprints_especiais", {})
    fp_space = _footprint_especial(path, "space", especiais.get("space", {}), pi)
    fp_hard = _footprint_especial(path, "hard", especiais.get("hard", {}), pi)

    return IsResult(
        name=path.name,
        path=path,
        instances=instances,
        algos=algos,
        features=features,
        features_input=list(sifted["feature"]),
        features_all=[nome for _, nome in col_feats],
        annotations=annotations,
        annotation_renames=renames,
        annotation_origins=origens,
        source_column=source_column,
        footprints=footprints,
        footprint_space=fp_space,
        footprint_hard=fp_hard,
        footprint_performance=perf,
        projection_matrix=proj,
        bounds=_ler_bounds(path / "bounds.csv"),
        bounds_pruned=_ler_bounds(path / "bounds_prunned.csv"),
        coordinates_trace=coords_trace,
        sifted_report=sifted,
        sifted_correlations=correl,
        sifted_silhouette=silhueta,
        pilot_r2=r2,
        pythia_proba=proba,
        pythia_proba_hat=proba_hat,
        pythia_confusion=conf,
        pythia_selection=sel,
        svm_table=svm_table,
        run_info=run_info,
        run_options=run_options,
        pi=pi,
        degenerate_report=_ler_opcional(path / "degenerate_report.csv", {"feature": str}),
        feature_info=_ler_opcional(path / "feature_info.csv", {"feature": str}),
    )
