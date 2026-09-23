"""Validacao de um metadata enviado pela interface, antes de rodar o engine.

So pandas e numpy (nao importa instancespace). Tudo aqui devolve mensagens para
a tela, nunca levanta excecao para o usuario:

- validar_metadata: coluna instances presente, sem rotulos vazios ou repetidos;
  pelo menos MIN_FEATURES feature_* e MIN_ALGOS algo_*; feature_* e algo_*
  numericas e finitas; NaN reportado por coluna;
- validar_anotacoes / validar_feature_info: os arquivos opcionais;
- fracao_boas: fracao de instancias "boas" por algoritmo com a regra do PRELIM
  do instancespace (prelim.py:120-170) para a direcao e o epsilon escolhidos;
- tempo_estimado: a partir de TEMPOS_MEDIDOS (medidos, nao chutados).
"""

import csv
import io
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from isaspace.ui.loader_is import colunas_de_anotacao, erros_tipos_declarados

MIN_FEATURES = 3
MIN_ALGOS = 2
MIN_INSTANCIAS = 20          # abaixo disso a validacao cruzada do PYTHIA fica fragil
FRACAO_MIN, FRACAO_MAX = 0.05, 0.95   # fora disso: conferir a direcao
MAX_EXEMPLOS = 5
# Tempos do engine em subprocesso (s), medianas de 3 repeticoes de
# scripts/medir_tempo_engine.py (2026-09-22, Apple M5, 10 nucleos) em metadata
# sintetico com TEMPOS_FEATURES features e TEMPOS_ALGOS algoritmos:
# {n de instancias: (total, so o PYTHIA)}. O PYTHIA domina e treina um SVM por
# algoritmo; o resto quase nao depende do numero de algoritmos.
TEMPOS_MEDIDOS = {500: (9.7, 4.7), 1000: (24.2, 17.7), 2000: (76.6, 66.5)}
TEMPOS_FEATURES, TEMPOS_ALGOS = 10, 6
LIMIAR_AVISO_TEMPO = 1000    # a partir de quantas instancias o aviso aparece


@dataclass
class Validacao:
    ok: bool = False
    erros: list = field(default_factory=list)
    avisos: list = field(default_factory=list)
    n: int = 0
    features: list = field(default_factory=list)
    algos: list = field(default_factory=list)
    anotacoes: list = field(default_factory=list)
    source: str | None = None
    nan: dict = field(default_factory=dict)      # {coluna: n de NaN}
    df: pd.DataFrame | None = None


def _exemplos(valores) -> str:
    valores = list(valores)
    texto = ", ".join(repr(v) for v in valores[:MAX_EXEMPLOS])
    return texto + (f" e mais {len(valores) - MAX_EXEMPLOS}" if len(valores) > MAX_EXEMPLOS else "")


def validar_metadata(conteudo: bytes) -> Validacao:
    v = Validacao()
    try:
        texto = conteudo.decode("utf-8-sig")
    except UnicodeDecodeError:
        v.erros.append("O arquivo não está em UTF-8: salve o CSV como UTF-8 e envie de novo.")
        return v
    try:
        cabecalho = next(csv.reader(io.StringIO(texto)))
    except StopIteration:
        v.erros.append("O arquivo está vazio.")
        return v
    except csv.Error as exc:
        v.erros.append(f"Não foi possível ler o cabeçalho do CSV: {exc}.")
        return v
    baixo = [c.strip().casefold() for c in cabecalho]
    repetidas = sorted({c for c in cabecalho if cabecalho.count(c) > 1})
    if repetidas:
        v.erros.append(f"Colunas com o mesmo nome: {_exemplos(repetidas)}.")
    if baixo.count("instances") == 0:
        v.erros.append("Falta a coluna **instances** (o rótulo de cada instância).")
    elif baixo.count("instances") > 1:
        v.erros.append("Há mais de uma coluna **instances**.")
    if baixo.count("source") > 1:
        v.erros.append("Há mais de uma coluna **source**.")
    features = [c for c, b in zip(cabecalho, baixo) if b.startswith("feature_")]
    algos = [c for c, b in zip(cabecalho, baixo) if b.startswith("algo_")]
    if len(features) < MIN_FEATURES:
        v.erros.append(f"São precisas pelo menos {MIN_FEATURES} colunas **feature_\\***; "
                       f"o arquivo tem {len(features)}.")
    if len(algos) < MIN_ALGOS:
        v.erros.append(f"São precisas pelo menos {MIN_ALGOS} colunas **algo_\\***; "
                       f"o arquivo tem {len(algos)}.")
    if v.erros:
        return v

    col_inst = cabecalho[baixo.index("instances")]
    try:
        df = pd.read_csv(io.StringIO(texto), dtype={col_inst: str})
    except (pd.errors.ParserError, ValueError) as exc:
        v.erros.append(f"CSV malformado: {str(exc).splitlines()[0]}.")
        return v
    v.n, v.df = len(df), df
    v.features, v.algos = features, algos
    v.source = cabecalho[baixo.index("source")] if "source" in baixo else None
    v.anotacoes = colunas_de_anotacao([c for c in cabecalho if c != v.source])

    rotulos = df[col_inst]
    vazios = rotulos.isna() | (rotulos.str.strip() == "")
    if vazios.any():
        v.erros.append(f"{int(vazios.sum())} instância(s) sem rótulo em **instances** "
                       f"(linhas {_exemplos(list(np.flatnonzero(vazios) + 2))} do arquivo).")
    rep = rotulos[rotulos.duplicated(keep=False) & ~vazios]
    if len(rep):
        v.erros.append(f"Rótulos repetidos em **instances** ({rep.nunique()} distintos): "
                       f"{_exemplos(sorted(rep.unique()))}.")
    for col in features + algos:
        bruto = df[col]
        numeros = pd.to_numeric(bruto, errors="coerce")
        ruins = bruto[numeros.isna() & bruto.notna()]
        if len(ruins):
            v.erros.append(f"**{col}** tem valores não numéricos: {_exemplos(sorted(set(map(str, ruins))))}.")
            continue
        if np.isinf(numeros).any():
            v.erros.append(f"**{col}** tem valores infinitos.")
        n_nan = int(numeros.isna().sum())
        if n_nan:
            v.nan[col] = n_nan
    if v.nan:
        partes = [f"{c}: {n}" for c, n in sorted(v.nan.items(), key=lambda t: -t[1])]
        v.avisos.append("NaN por coluna (" + ", ".join(partes[:12])
                        + (f" e mais {len(partes) - 12}" if len(partes) > 12 else "") + "). "
                        "O PRELIM remove features com NaN demais (prelim.nan_threshold).")
    todos_nan = [c for c in algos if v.nan.get(c) == v.n]
    if todos_nan:
        v.erros.append(f"Algoritmos sem nenhum valor: {_exemplos(todos_nan)}.")
    if 0 < v.n < MIN_INSTANCIAS:
        v.avisos.append(f"Só {v.n} instâncias: a validação cruzada do PYTHIA fica frágil.")
    v.ok = not v.erros
    return v


def validar_anotacoes(conteudo: bytes, meta: Validacao) -> tuple:
    """(tipos, erros) de um annotations.json enviado, conferido contra o metadata."""
    try:
        tipos = json.loads(conteudo.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as exc:
        return None, [f"annotations.json não é um JSON válido: {exc}."]
    if meta.df is None:
        return tipos, []
    return tipos, erros_tipos_declarados(meta.df, tipos)


def validar_feature_info(conteudo: bytes, meta: Validacao) -> list:
    """Erros de um feature_info.csv enviado (colunas feature e family)."""
    try:
        info = pd.read_csv(io.BytesIO(conteudo))
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        return [f"feature_info.csv malformado: {str(exc).splitlines()[0]}."]
    faltam = [c for c in ("feature", "family") if c not in info.columns]
    if faltam:
        return [f"feature_info.csv sem as colunas {', '.join(faltam)}."]
    return []


def linhas_do_prelim(v: Validacao) -> pd.DataFrame:
    """algo_* das instancias que chegam ao PRELIM: o PREPROCESSING remove as
    que tem todas as feature_* ou todos os algo_* ausentes (preprocessing.py:437)."""
    num = v.df[v.features + v.algos].apply(pd.to_numeric, errors="coerce")
    fora = num[v.features].isna().all(axis=1) | num[v.algos].isna().all(axis=1)
    return num.loc[~fora, v.algos]


def fracao_boas(algos: pd.DataFrame, maior_melhor: bool, absoluto: bool, epsilon: float) -> pd.Series:
    """Fracao de instancias boas por algoritmo, com a regra do PRELIM do
    instancespace (prelim.py:116-160, copiada inclusive no tratamento de NaN):
    absoluto = algo_* >= epsilon (maior e melhor) ou <= epsilon; relativo =
    dentro de epsilon (fracao) do melhor da instancia."""
    y = algos.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    aux = np.where(np.isnan(y), -np.inf if maior_melhor else np.inf, y)
    if absoluto:
        bom = aux >= epsilon if maior_melhor else aux <= epsilon
    else:
        melhor = aux.max(axis=1) if maior_melhor else aux.min(axis=1)
        melhor[melhor == 0] = np.finfo(float).eps
        with np.errstate(invalid="ignore", divide="ignore"):
            razao = aux / melhor[:, None]
            bom = (1 - razao) <= epsilon if maior_melhor else (razao - 1) <= epsilon
    nomes = [c[len("algo_"):] if c.casefold().startswith("algo_") else c for c in algos.columns]
    return pd.Series(bom.mean(axis=0), index=nomes)


def fora_da_faixa(fracoes: pd.Series) -> list:
    """Algoritmos com menos de FRACAO_MIN ou mais de FRACAO_MAX de boas."""
    return [a for a, f in fracoes.items() if f < FRACAO_MIN or f > FRACAO_MAX]


def _potencia(pontos, n):
    """Lei de potencia a*n^b ajustada em log-log aos pontos (n, t)."""
    b, a = np.polyfit(np.log([k for k, _ in pontos]), np.log([v for _, v in pontos]), 1)
    return float(np.exp(a) * n ** b)


def tempo_estimado(n: int, n_algos: int) -> float | None:
    """Segundos estimados: o resto do pipeline pela lei de potencia do tempo
    medido sem o PYTHIA, mais o PYTHIA pela sua lei de potencia escalada
    linearmente pelo numero de algoritmos. Estimativa grosseira: extrapola
    fora de 500..2000 e ignora o numero de features."""
    pontos = [(k, t) for k, t in TEMPOS_MEDIDOS.items() if t[0] and t[1]]
    if len(pontos) < 2 or n <= 0:
        return None
    resto = _potencia([(k, total - py) for k, (total, py) in pontos], n)
    pythia = _potencia([(k, py) for k, (_, py) in pontos], n)
    return resto + pythia * max(n_algos, 1) / TEMPOS_ALGOS
