"""Motor ISA sobre o pacote ``instancespace`` (grupo do Mario Munoz).

Roda o pipeline completo (PREPROCESSING, PRELIM, SIFTED, PILOT, PYTHIA,
CLOISTER, TRACE) estagio por estagio sobre um metadata.csv e grava a pasta que
``isaspace.ui.loader_is`` le:

- tudo o que ``Model.save_to_csv`` grava (coordinates.csv, feature_*.csv,
  algorithm_*.csv, good_algos.csv, beta_easy.csv, portfolio*.csv,
  footprint_<algo>_<good|best>.csv, footprint_performance.csv,
  projection_matrix.csv, bounds*.csv, svm_table.csv);
- ``coordinates.csv`` reescrito com o z do PILOT sem correcao e, so quando o
  jitter e aplicado, ``coordinates_trace.csv`` com o z que o TRACE usou;
- ``metadata.csv``: copia byte a byte do metadata de entrada (anotacoes e
  source inclusos) e, se existirem ao lado dele, ``annotations.json`` (tipos
  declarados das anotacoes, validados antes de rodar), ``degenerate_report.csv``
  e ``feature_info.csv``;
- ``run_options.json``: as opcoes efetivas, ``dataclasses.asdict`` de
  ``InstanceSpaceOptions`` (``InstanceSpaceOptions.from_dict`` le de volta);
- ``sifted_report.csv``, ``sifted_correlations.csv``, ``sifted_silhouette.csv``
  (``Model.sifted``), ``pilot_r2.csv`` (``Model.pilot.r2``);
- ``pythia_proba.csv`` (pr0_sub e pr0_hat), ``pythia_confusion.csv``
  (cvcmat) e ``pythia_selection.csv`` (selection0 e selection1);
- ``footprint_space.csv`` e ``footprint_hard.csv`` (``Model.trace.space`` e
  ``.hard``), no mesmo esquema das footprints dos algoritmos;
- ``run_info.json``: versoes, tempos por estagio, n de instancias, correcao de
  quase duplicatas antes do TRACE, metricas das footprints space e hard e
  avisos. E gravado por ultimo: pasta sem ele e execucao incompleta.

O formato de cada arquivo esta em docs/output_format.md.

Robustez do TRACE: o alpha shape do TRACE legado do instancespace 0.3.0 devolve
poligono vazio quando a projecao tem pontos DISTINTOS a ~1e-14 um do outro
(hill-valley: footprints good e area do espaco zeradas); pontos identicos nao
atrapalham, o TRACE os funde com np.unique. Depois do PILOT,
``run_instancespace`` conta os pares a menos de ``LIMIAR_DUPLICATA``; havendo
posicoes distintas nessa distancia, soma a cada uma um ruido normal de desvio
``ESCALA_JITTER`` com semente fixa (pontos identicos recebem o mesmo) e passa
esse z so ao TRACE, com ``run_stage(TraceStage, z=...)``. PYTHIA (que ajusta
hiperparametros sobre z e muda ate 0.08 de probabilidade com essa perturbacao)
e CLOISTER (que nao usa z) ficam como sem a correcao. O runner guarda o
override no Model, mas coordinates.csv e reescrito com o z do PILOT; o z do
TRACE vai para coordinates_trace.csv. Tudo fica registrado em
``run_info.json["trace_robustez"]``.

Requer Python 3.12 e o .venv-isa (instancespace 0.3.0); nao roda no .venv 3.11.
"""

import copy
import dataclasses
import json
import platform
import shutil
import time
import warnings
from collections import Counter
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.spatial import cKDTree

from instancespace import InstanceSpace
from instancespace.data import metadata as is_metadata
from instancespace.data.options import InstanceSpaceOptions
from instancespace.stages.cloister import CloisterStage
from instancespace.stages.pilot import PilotStage
from instancespace.stages.prelim import PrelimStage
from instancespace.stages.preprocessing import PreprocessingStage
from instancespace.stages.pythia import PythiaStage
from instancespace.stages.sifted import SiftedStage
from instancespace.stages.trace import TraceStage

from isaspace.ui.loader_is import INTEIRA, NUMERICA, TIPOS_ANOTACAO

# ordem de _BUILTIN_STAGE_ORDER do instancespace; PYTHIA e CLOISTER sao a mesma
# onda e podem rodar em qualquer ordem entre si
ESTAGIOS = [
    ("PREPROCESSING", PreprocessingStage),
    ("PRELIM", PrelimStage),
    ("SIFTED", SiftedStage),
    ("PILOT", PilotStage),
    ("PYTHIA", PythiaStage),
    ("CLOISTER", CloisterStage),
    ("TRACE", TraceStage),
]

# Opcoes padrao, com os nomes de campo dos dataclasses do instancespace (os
# mesmos do run_options.json). O que nao aparece aqui fica no padrao da
# biblioteca, inclusive todo o SIFTED (rho=0.1, pval=0.05, k=6, GA).
DEFAULT_OPTIONS = {
    "perf": {
        # algo_* e desempenho em que MAIOR e melhor (acerto ou probabilidade
        # da classe verdadeira). Com False, "bom" vira desempenho <= epsilon,
        # ou seja, o erro.
        "max_perf": True,
        # limiar absoluto: a instancia e boa para o algoritmo se algo_* >= epsilon
        "abs_perf": True,
        "epsilon": 0.5,
    },
    "trace": {
        # footprints do desempenho observado (y_bin e portfolio.csv), nao das
        # previsoes do PYTHIA; o padrao da biblioteca e True
        "use_sim": False,
    },
}

LIMIAR_DUPLICATA = 1e-6   # pares de pontos da projecao mais proximos que isso
ESCALA_JITTER = 1e-6      # desvio do ruido normal somado aos pontos envolvidos
SEMENTE_JITTER = 0

STATUS_KEPT = "kept"
STATUS_CORRELACAO = "dropped_correlation"
STATUS_REDUNDANCIA = "dropped_redundancy"
STATUS_PREPROCESSING = "dropped_preprocessing"   # removida antes do SIFTED
STATUS_INDETERMINADO = "undetermined"            # reconstrucao inconsistente

# arquivos fixos que o engine grava; so estes (e os footprint_*) sao apagados
# ao regravar uma pasta
ARQUIVOS_SAVE_TO_CSV = (
    "coordinates.csv", "bounds.csv", "bounds_prunned.csv", "feature_raw.csv",
    "feature_process.csv", "algorithm_raw.csv", "algorithm_process.csv",
    "algorithm_bin.csv", "good_algos.csv", "beta_easy.csv", "portfolio.csv",
    "algorithm_svm.csv", "portfolio_svm.csv", "footprint_performance.csv",
    "projection_matrix.csv", "svm_table.csv",
)
ARQUIVOS_EXTRAS = (
    "metadata.csv", "run_options.json", "coordinates_trace.csv",
    "sifted_report.csv", "sifted_correlations.csv", "sifted_silhouette.csv",
    "pilot_r2.csv", "pythia_proba.csv", "pythia_confusion.csv",
    "pythia_selection.csv", "footprint_space.csv", "footprint_hard.csv",
    "run_info.json",
)
# arquivos opcionais ao lado do metadata de entrada, copiados se existirem
AUXILIARES = {
    "annotations.json": None,                                  # validado a parte
    "degenerate_report.csv": ("feature", "var_bruta", "iqr", "motivo"),
    "feature_info.csv": ("feature", "family"),
}
ARQUIVOS_EXTRAS += tuple(AUXILIARES)
# pythia_proba.csv: coluna <algo> = pr0_sub (padrao), <algo>_hat = pr0_hat
SUFIXO_HAT = "_hat"
# SiftedStage.evaluate_cluster testa k = 3 .. (n de features apos a correlacao) - 1
K_MIN_SILHUETA = 3
PADROES_FOOTPRINT = (
    "footprint_*_good.csv", "footprint_*_best.csv", "footprint_*_vertices.csv",
    "footprint_*_tetrahedra.csv", "footprint_*_boundary_faces.csv",
)
GERADO_POR = "isaspace.engine.run_instancespace"


# --------------------------------------------------------------------------- #
# opcoes e metadata
# --------------------------------------------------------------------------- #
def _chave(nome):
    """Forma normalizada de uma chave de opcao: 'MaxPerf', 'max_perf' -> 'maxperf'."""
    return str(nome).casefold().replace("_", "")


def build_options(options=None):
    """InstanceSpaceOptions a partir de DEFAULT_OPTIONS sobreposto por `options`.

    `options` e um dict por grupo ({"trace": {"use_sim": True}}), com os nomes
    do run_options.json ou os do options.json do MATLAB (MaxPerf, usesim, PI);
    uma chave do usuario substitui a equivalente do padrao. Um
    InstanceSpaceOptions pronto e devolvido sem alteracao.
    """
    if isinstance(options, InstanceSpaceOptions):
        return options
    combinado = copy.deepcopy(DEFAULT_OPTIONS)
    for grupo, valores in (options or {}).items():
        base = combinado.get(grupo)
        if isinstance(valores, dict) and isinstance(base, dict):
            novas = {_chave(k) for k in valores}
            base = {k: v for k, v in base.items() if _chave(k) not in novas}
            combinado[grupo] = {**base, **valores}
        else:
            combinado[grupo] = copy.deepcopy(valores)
    return InstanceSpaceOptions.from_dict(combinado)


def _ler_metadata(path):
    """from_csv_file devolve None em erro e so loga; aqui o erro vira excecao."""
    erros = []
    sink = logger.add(lambda m: erros.append(m.record["message"]), level="ERROR")
    try:
        meta = is_metadata.from_csv_file(path)
    finally:
        logger.remove(sink)
    if meta is None:
        raise ValueError(f"metadata invalido ({path}): " + (" | ".join(erros) or "sem detalhe"))
    return meta


def _colunas_de_anotacao(colunas):
    """Colunas do metadata que nao sao instances, source, feature_* nem algo_*."""
    return [c for c in colunas if c.casefold() not in ("instances", "source")
            and not c.casefold().startswith(("feature_", "algo_"))]


def _ler_auxiliares(metadata_path):
    """Valida os arquivos auxiliares ao lado do metadata; devolve
    ({nome: caminho} dos presentes, tipos declarados). Erro vira ValueError
    antes de rodar o pipeline."""
    pasta = Path(metadata_path).parent
    presentes = {n: pasta / n for n in AUXILIARES if (pasta / n).is_file()}
    for nome, obrigatorias in AUXILIARES.items():
        if nome in presentes and obrigatorias is not None:
            cols = list(pd.read_csv(presentes[nome], nrows=0).columns)
            faltam = [c for c in obrigatorias if c not in cols]
            if faltam:
                raise ValueError(f"{nome}: faltam as colunas {faltam}")
    tipos = {}
    if "annotations.json" in presentes:
        try:
            tipos = json.loads(presentes["annotations.json"].read_text())
        except ValueError as exc:
            raise ValueError(f"annotations.json invalido: {exc}") from exc
        if not isinstance(tipos, dict):
            raise ValueError("annotations.json tem de ser um objeto {anotacao: tipo}")
        meta = pd.read_csv(metadata_path)
        anotacoes = _colunas_de_anotacao(list(meta.columns))
        for col, tipo in tipos.items():
            if tipo not in TIPOS_ANOTACAO:
                raise ValueError(f"annotations.json: tipo {tipo!r} de {col!r} nao e um de {TIPOS_ANOTACAO}")
            if col not in anotacoes:
                raise ValueError(f"annotations.json: {col!r} nao e uma coluna de anotacao do metadata")
            if tipo in (NUMERICA, INTEIRA):
                v = pd.to_numeric(meta[col], errors="coerce")
                if (v.isna() & meta[col].notna()).any():
                    raise ValueError(f"annotations.json: {col!r} declarada {tipo} tem valores nao numericos")
                if tipo == INTEIRA and (np.mod(v.dropna(), 1) != 0).any():
                    raise ValueError(f"annotations.json: {col!r} declarada {tipo} tem valores nao inteiros")
    return presentes, tipos


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, (tuple, set)):
        return list(obj)
    raise TypeError(f"nao serializavel em JSON: {type(obj).__name__}")


# --------------------------------------------------------------------------- #
# robustez do TRACE
# --------------------------------------------------------------------------- #
def _pares_proximos(z, limiar):
    """Pares (i, j) com distancia euclidiana < limiar, e as distancias."""
    pares = cKDTree(z).query_pairs(r=limiar, output_type="ndarray")
    if len(pares) == 0:
        return np.empty((0, 2), dtype=int), np.empty(0)
    d = np.linalg.norm(z[pares[:, 0]] - z[pares[:, 1]], axis=1)
    return pares[d < limiar], d[d < limiar]


def _menor_distancia(z):
    d, _ = cKDTree(z).query(z, k=2)
    return float(np.min(d[:, 1]))


def _corrigir_duplicatas(z, labels, aplicar):
    """Conta os pares quase duplicados de z e, se `aplicar`, separa-os.

    Pontos IDENTICOS nao sao perturbados: o TRACE os funde com np.unique antes
    do alpha shape e eles nao causam a degeneracao. Separa-los cria justamente
    posicoes distintas a ~1e-7, e no blood-transfusion (822 pares identicos)
    isso zerou as seis footprints good. O jitter vale para POSICOES distintas a
    menos do limiar; os pontos identicos de uma posicao recebem o mesmo
    deslocamento e continuam identicos.

    Devolve (z_corrigido ou None, registro para run_info["trace_robustez"]).
    """
    z = np.asarray(z, dtype=float)
    pares, d = _pares_proximos(z, LIMIAR_DUPLICATA)
    posicoes, inv = np.unique(z, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    pares_pos, _ = _pares_proximos(posicoes, LIMIAR_DUPLICATA)
    registro = {
        "limiar": LIMIAR_DUPLICATA,
        "pares_quase_duplicados": int(len(pares)),
        "pares_identicos": int(np.sum(d == 0)),
        "pares_distintos_proximos": int(len(pares_pos)),
        "menor_distancia_distintos_antes": (_menor_distancia(posicoes)
                                            if len(posicoes) > 1 else None),
        "regra": "jitter so em posicoes distintas a menos do limiar; pontos identicos "
                 "nao sao separados (o TRACE os funde com np.unique)",
        "jitter_aplicado": False,
    }
    if len(pares_pos) == 0:
        return None, registro
    if not aplicar:
        registro["motivo_sem_jitter"] = "fix_near_duplicates=False"
        return None, registro

    idx = np.unique(pares_pos.ravel())
    rng = np.random.default_rng(SEMENTE_JITTER)
    deslocamento = np.zeros_like(posicoes)
    deslocamento[idx] = rng.normal(scale=ESCALA_JITTER, size=(idx.size, z.shape[1]))
    z_corr = z + deslocamento[inv]
    pontos = np.flatnonzero(np.isin(inv, idx))
    pares_depois, _ = _pares_proximos(posicoes + deslocamento, LIMIAR_DUPLICATA)
    registro.update({
        "jitter_aplicado": True,
        "jitter_escala": ESCALA_JITTER,
        "jitter_semente": SEMENTE_JITTER,
        "jitter_distribuicao": "normal(0, escala) em cada coordenada, um sorteio por posicao",
        "posicoes_perturbadas": int(idx.size),
        "pontos_perturbados": int(pontos.size),
        "rotulos_perturbados": [str(labels[i]) for i in pontos],
        "deslocamento_maximo": float(np.max(np.linalg.norm(deslocamento[idx], axis=1))),
        "menor_distancia_distintos_depois": _menor_distancia(posicoes + deslocamento),
        "pares_distintos_proximos_depois": int(len(pares_depois)),
        "aplicado_em": "so no TRACE, via run_stage(TraceStage, z=...); coordinates.csv "
                       "sai com o z perturbado; PYTHIA usou o z do PILOT e CLOISTER nao usa z",
    })
    return z_corr, registro


# --------------------------------------------------------------------------- #
# sifted_report
# --------------------------------------------------------------------------- #
def _sobreviventes_correlacao(rho, pval, opts_sifted):
    """Reproduz SiftedStage.select_features_by_performance (sifted.py:774-808).

    Fica a feature mais correlacionada com cada algoritmo e toda feature com
    |rho| >= sifted.rho e p <= sifted.pval para algum algoritmo.
    """
    filtrado = np.abs(rho)
    filtrado[np.isnan(rho) | (pval > opts_sifted.pval)] = 0
    ordenado = np.sort(filtrado, axis=0)[::-1, :]
    linha = np.argsort(-filtrado, axis=0)
    manter = np.zeros(rho.shape[0], dtype=bool)
    manter[np.unique(linha[0, :])] = True
    manter[np.unique(linha[ordenado >= opts_sifted.rho])] = True
    return np.where(manter)[0]


def _sifted_report(model, feats_pre, feats_entrada, opts):
    """Uma linha por feature de entrada; devolve (DataFrame, avisos)."""
    s = model.sifted
    algos = list(model.data.algo_labels)
    selvars = [int(i) for i in np.asarray(s.selvars).ravel()]
    avisos = []
    if [feats_pre[i] for i in selvars] != list(model.data.feat_labels):
        avisos.append("sifted_report: selvars nao reproduz Model.data.feat_labels")

    rho = None if s.rho is None else np.asarray(s.rho, dtype=float)
    pval = None if s.pval is None else np.asarray(s.pval, dtype=float)
    if rho is not None and rho.shape[0] == len(feats_pre) and pval is not None:
        sobreviventes = [int(i) for i in _sobreviventes_correlacao(rho, pval, opts.sifted)]
    else:
        sobreviventes = list(range(len(feats_pre)))
        if rho is not None:
            avisos.append("sifted_report: rho/pval sem uma linha por feature; "
                          "filtro de correlacao nao reconstruido")
    if not set(selvars) <= set(sobreviventes):
        avisos.append("sifted_report: feature mantida fora dos sobreviventes da correlacao")

    cluster_de = {}
    if s.clust is not None:
        clust = np.asarray(s.clust, dtype=bool)
        if clust.shape[0] != len(sobreviventes):
            avisos.append(f"sifted_report: clust tem {clust.shape[0]} linhas e a "
                          f"reconstrucao da correlacao deu {len(sobreviventes)} features")
        else:
            for pos, i in enumerate(sobreviventes):
                cols = np.flatnonzero(clust[pos])
                if cols.size == 1:
                    cluster_de[i] = int(cols[0]) + 1
    mantidas_no_cluster = {}
    for i in selvars:
        if i in cluster_de:
            mantidas_no_cluster.setdefault(cluster_de[i], []).append(feats_pre[i])
    for c, fs in mantidas_no_cluster.items():
        if len(fs) != 1:
            avisos.append(f"sifted_report: cluster {c} com {len(fs)} features mantidas")

    linhas = []
    for f in feats_entrada:
        reg = {"feature": f, "status": None, "rho": np.nan, "rho_algo": None,
               "pval": np.nan, "n_algos_sig": np.nan, "cluster": None, "kept_instead": None}
        if f not in feats_pre:
            reg["status"] = STATUS_PREPROCESSING
            linhas.append(reg)
            continue
        i = feats_pre.index(f)
        if rho is not None and rho.shape[0] == len(feats_pre) and np.isfinite(rho[i]).any():
            j = int(np.nanargmax(np.abs(rho[i])))
            reg["rho"], reg["rho_algo"] = float(rho[i, j]), algos[j]
            if pval is not None:
                reg["pval"] = float(pval[i, j])
                sig = (np.abs(rho[i]) >= opts.sifted.rho) & (pval[i] <= opts.sifted.pval)
                reg["n_algos_sig"] = int(np.sum(sig))
        reg["cluster"] = cluster_de.get(i)
        if i in selvars:
            reg["status"] = STATUS_KEPT
        elif i not in sobreviventes:
            reg["status"] = STATUS_CORRELACAO
        elif i in cluster_de:
            reg["status"] = STATUS_REDUNDANCIA
            reg["kept_instead"] = ";".join(mantidas_no_cluster.get(cluster_de[i], [])) or None
        else:
            reg["status"] = STATUS_INDETERMINADO
            avisos.append(f"sifted_report: motivo do descarte de {f} nao reconstruido")
        linhas.append(reg)
    df = pd.DataFrame(linhas)
    df["cluster"] = df["cluster"].astype("Int64")
    df["n_algos_sig"] = df["n_algos_sig"].astype("Int64")
    return df, avisos


# --------------------------------------------------------------------------- #
# pythia_proba
# --------------------------------------------------------------------------- #
def _regra_bom(perf):
    """Texto da regra de y_bin do PRELIM (prelim.py:120-170) para estas opcoes."""
    eps = perf.epsilon
    if perf.max_perf:
        return (f"bom = algo_* >= {eps}" if perf.abs_perf
                else f"bom = 1 - algo_*/melhor <= {eps}")
    return (f"bom = algo_* <= {eps}" if perf.abs_perf
            else f"bom = algo_*/melhor - 1 <= {eps}")


def _por_instancia(valores, labels, colunas):
    """Mesmo layout do _write_array_to_csv do instancespace: indice Row = rotulo."""
    return pd.DataFrame(np.asarray(valores), columns=colunas,
                        index=pd.Index([str(x) for x in labels], name="Row"))


def _gravar_coordenadas(outdir, labels, z_pilot, z_trace):
    """coordinates.csv com o z do PILOT; coordinates_trace.csv so se houve jitter."""
    cols = [f"z_{i}" for i in range(1, z_pilot.shape[1] + 1)]
    _por_instancia(z_pilot, labels, cols).to_csv(outdir / "coordinates.csv")
    if z_trace is not None:
        _por_instancia(z_trace, labels, cols).to_csv(outdir / "coordinates_trace.csv")


def _gravar_pythia(outdir, model, algos, labels):
    """pythia_proba.csv, pythia_confusion.csv e pythia_selection.csv.

    Devolve (dict para run_info["pythia"], avisos)."""
    p = model.pythia
    sub = np.asarray(p.pr0_sub, dtype=float)
    hat = np.asarray(p.pr0_hat, dtype=float)
    pd.concat([
        _por_instancia(sub, labels, algos),
        _por_instancia(hat, labels, [f"{a}{SUFIXO_HAT}" for a in algos]),
    ], axis=1).to_csv(outdir / "pythia_proba.csv")

    # cvcmat do treino (pythia.py:861-868): [tn, fp, fn, tp] de y_bin contra
    # y_sub. O caminho de avaliacao (pythia.py:515) usa outra ordem, entao a
    # ordem e conferida contra accuracy/precision/recall do proprio Model.
    cm = np.asarray(p.cvcmat, dtype=float)
    conf = pd.DataFrame(cm, columns=["tn", "fp", "fn", "tp"],
                        index=pd.Index(algos, name="Algorithm")).astype(int)
    avisos = []
    n = conf.sum(axis=1).replace(0, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        calc = {
            "accuracy": ((conf.tp + conf.tn) / n).to_numpy(),
            "precision": (conf.tp / (conf.tp + conf.fp)).to_numpy(),
            "recall": (conf.tp / (conf.tp + conf.fn)).to_numpy(),
        }
    for nome, valores in calc.items():
        ref = np.asarray(getattr(p, nome), dtype=float)
        ok = np.isclose(valores, ref, equal_nan=True) | (np.isnan(valores) & (ref == 0))
        if not ok.all():
            avisos.append(f"pythia_confusion: {nome} recalculado da matriz nao bate com Model.pythia.{nome}")
    conf.to_csv(outdir / "pythia_confusion.csv")

    def nome(i):
        return algos[i] if 0 <= int(i) < len(algos) else None

    sel0 = np.asarray(p.selection0).ravel()
    sel1 = np.asarray(p.selection1).ravel()
    pd.DataFrame({"selection0": [nome(i) for i in sel0], "selection1": [nome(i) for i in sel1]},
                 index=pd.Index([str(x) for x in labels], name="Row")
                 ).to_csv(outdir / "pythia_selection.csv")

    y_hat = np.asarray(p.y_hat, dtype=bool)
    info = {
        "pares_instancia_algoritmo": int(y_hat.size),
        "y_hat_discorda_de_pr0_hat": int(np.sum(y_hat != (hat < 0.5))),
        "selection0_nenhum": int(np.sum(sel0 < 0)),
        "selection1_difere_de_selection0": int(np.sum(sel0 != sel1)),
    }
    return info, avisos


def _gravar_pilot_r2(outdir, model):
    """R^2 de cada coluna de [x, y] contra a reconstrucao z B' (pilot.py:478)."""
    feats = [str(f) for f in model.data.feat_labels]
    algos = [str(a) for a in model.data.algo_labels]
    r2 = np.asarray(model.pilot.r2, dtype=float).ravel()
    avisos = []
    if len(r2) != len(feats) + len(algos):
        avisos.append(f"pilot_r2: {len(r2)} valores para {len(feats)} features + {len(algos)} algoritmos")
    nomes = (feats + algos)[:len(r2)]
    tipos = (["feature"] * len(feats) + ["algorithm"] * len(algos))[:len(r2)]
    pd.DataFrame({"variable": nomes, "kind": tipos, "r2": r2[:len(nomes)]}).to_csv(
        outdir / "pilot_r2.csv", index=False)
    return avisos


def _gravar_sifted_extras(outdir, model, feats_pre, opts):
    """sifted_correlations.csv (formato longo) e sifted_silhouette.csv."""
    s = model.sifted
    algos = [str(a) for a in model.data.algo_labels]
    avisos = []
    linhas = []
    if s.rho is not None:
        rho = np.asarray(s.rho, dtype=float)
        pval = None if s.pval is None else np.asarray(s.pval, dtype=float)
        if rho.shape != (len(feats_pre), len(algos)):
            avisos.append(f"sifted_correlations: rho {rho.shape} nao e features x algoritmos")
        else:
            for i, f in enumerate(feats_pre):
                for j, a in enumerate(algos):
                    linhas.append((f, a, rho[i, j], np.nan if pval is None else pval[i, j]))
    pd.DataFrame(linhas, columns=["feature", "algorithm", "rho", "pval"]).to_csv(
        outdir / "sifted_correlations.csv", index=False)

    sil = pd.DataFrame(columns=["k", "silhouette", "used", "best"])
    if s.silhouette_scores:
        v = np.asarray(s.silhouette_scores, dtype=float)
        ks = np.arange(K_MIN_SILHUETA, K_MIN_SILHUETA + len(v))
        sil = pd.DataFrame({"k": ks, "silhouette": v, "used": ks == opts.sifted.k,
                            "best": np.arange(len(v)) == int(np.nanargmax(v))})
        if s.clust is not None and len(v) != np.asarray(s.clust).shape[0] - K_MIN_SILHUETA:
            avisos.append("sifted_silhouette: numero de k testados nao bate com clust")
        if opts.sifted.k not in ks:
            avisos.append(f"sifted_silhouette: k usado ({opts.sifted.k}) fora dos k testados")
    sil.to_csv(outdir / "sifted_silhouette.csv", index=False)
    return avisos


def _gravar_footprint_especial(outdir, nome, fp, espaco):
    """footprint_<nome>.csv no esquema das demais (so se nao vazia) e metricas."""
    from instancespace._serialisers import _footprint_boundary_frame

    poligono = None if fp is None else fp.polygon
    registro = {
        "arquivo": None,
        "area": float(getattr(fp, "area", 0) or 0),
        "densidade": float(getattr(fp, "density", 0) or 0),
        "pureza": float(getattr(fp, "purity", 0) or 0),
        "elementos": int(getattr(fp, "elements", 0) or 0),
        "elementos_bons": int(getattr(fp, "good_elements", 0) or 0),
    }
    if espaco is not None:
        registro["area_normalizada"] = registro["area"] / espaco["area"] if espaco["area"] else 0.0
        registro["densidade_normalizada"] = (registro["densidade"] / espaco["densidade"]
                                             if espaco["densidade"] else 0.0)
    if poligono is not None and hasattr(poligono, "is_empty") and not poligono.is_empty:
        _footprint_boundary_frame(poligono).to_csv(outdir / f"footprint_{nome}.csv", index=False)
        registro["arquivo"] = f"footprint_{nome}.csv"
    return registro


# --------------------------------------------------------------------------- #
# pasta de saida
# --------------------------------------------------------------------------- #
def _arquivos_do_engine(outdir):
    donos = [p for p in outdir.iterdir() if p.name in ARQUIVOS_SAVE_TO_CSV + ARQUIVOS_EXTRAS]
    for padrao in PADROES_FOOTPRINT:
        donos += list(outdir.glob(padrao))
    return donos


def _checar_pasta(outdir, metadata_path):
    """Recusa, antes de rodar, a pasta do metadata de entrada e uma pasta com
    arquivos de mesmo nome sem run_info.json do engine (saida de outra
    ferramenta, por exemplo resultados/isa/<nome>/ do pyispace)."""
    if not outdir.exists():
        return
    if not outdir.is_dir():
        raise NotADirectoryError(f"{outdir} existe e nao e pasta")
    destino_meta = outdir / "metadata.csv"
    if destino_meta.exists() and destino_meta.samefile(metadata_path):
        raise ValueError(f"outdir {outdir} e a pasta do metadata de entrada; use outra pasta")
    if not _arquivos_do_engine(outdir):
        return
    try:
        dono = json.loads((outdir / "run_info.json").read_text()).get("gerado_por") == GERADO_POR
    except (OSError, ValueError):
        dono = False
    if not dono:
        raise ValueError(f"{outdir} tem arquivos que nao vieram de {GERADO_POR}; "
                         "use uma pasta nova")


def _limpar_pasta(outdir):
    """Cria a pasta e apaga so os arquivos que o engine grava (uma footprint
    que ficou vazia nesta execucao nao pode sobrar da anterior)."""
    outdir.mkdir(parents=True, exist_ok=True)
    for p in _arquivos_do_engine(outdir):
        p.unlink()


def _arquivos_footprint(outdir, algo_labels):
    """{algo: {"good": nome do arquivo ou None, "best": ...}} como gravado.

    save_to_csv nomeia os arquivos por _portable_stems (caracteres invalidos
    viram "_", nomes longos sao truncados); o loader usa este mapa em vez de
    adivinhar.
    """
    try:
        from instancespace._serialisers import _portable_stems
        stems = _portable_stems(list(algo_labels), "algorithm")
    except ImportError:
        stems = list(algo_labels)
    out = {}
    for algo, stem in zip(algo_labels, stems):
        out[str(algo)] = {
            tipo: (f"footprint_{stem}_{tipo}.csv"
                   if (outdir / f"footprint_{stem}_{tipo}.csv").is_file() else None)
            for tipo in ("good", "best")
        }
    return out


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def run_instancespace(metadata_path, outdir, options=None, progress=None, *,
                      fix_near_duplicates=True):
    """Roda o instancespace em `metadata_path` e grava a pasta `outdir`.

    Parametros
    ----------
    metadata_path : metadata.csv com ``instances``, ``source`` opcional,
        ``feature_*``, ``algo_*`` e quaisquer outras colunas (anotacoes, que o
        instancespace ignora e a copia em outdir/metadata.csv preserva).
    outdir : pasta de saida; criada se preciso. Numa pasta de uma execucao
        anterior do engine so os arquivos do engine sao apagados; pasta com
        arquivos de outra origem e recusada.
    options : dict por grupo sobreposto a DEFAULT_OPTIONS (ver build_options)
        ou um InstanceSpaceOptions pronto.
    progress : callable(nome_do_estagio) chamado ANTES de cada estagio, com
        "PREPROCESSING", "PRELIM", "SIFTED", "PILOT", "PYTHIA", "CLOISTER" e
        "TRACE", nessa ordem.
    fix_near_duplicates : se False, so conta os pares quase duplicados da
        projecao e nao aplica o jitter (para comparacao).

    Retorna o dict gravado em run_info.json. Um estagio que falha vira
    RuntimeError com o nome do estagio; a pasta de saida so e tocada depois
    que todos os estagios terminam.
    """
    metadata_path = Path(metadata_path)
    outdir = Path(outdir)
    t_inicio = time.perf_counter()
    _checar_pasta(outdir, metadata_path)
    opts = build_options(options)
    meta = _ler_metadata(metadata_path)
    auxiliares, tipos_declarados = _ler_auxiliares(metadata_path)
    feats_entrada = [str(f) for f in meta.feature_names]
    nomes_algo = [str(a) for a in meta.algorithm_names]
    colidem = sorted(a for a in nomes_algo if a.endswith(SUFIXO_HAT) and a[:-len(SUFIXO_HAT)] in nomes_algo)
    if colidem:
        raise ValueError(f"algoritmos {colidem} colidem com as colunas <algo>{SUFIXO_HAT} de "
                         "pythia_proba.csv; renomeie-os no metadata")

    tempos, avisos_is, robustez = {}, [], {}
    feats_pre = list(feats_entrada)
    labels_z = [str(x) for x in meta.instance_labels]
    sink = logger.add(lambda m: avisos_is.append(m.record["message"]), level="WARNING")
    isp = InstanceSpace(meta, opts)
    try:
        with warnings.catch_warnings(record=True) as capturados:
            warnings.simplefilter("always")
            z_corrigido = z_pilot = None
            for nome, estagio in ESTAGIOS:
                if progress is not None:
                    progress(nome)
                extra = {"z": z_corrigido} if nome == "TRACE" and z_corrigido is not None else {}
                t0 = time.perf_counter()
                try:
                    saida = isp.run_stage(estagio, **extra)
                except Exception as exc:
                    raise RuntimeError(f"instancespace falhou no estagio {nome}: {exc}") from exc
                tempos[nome] = round(time.perf_counter() - t0, 3)
                if nome == "PREPROCESSING":
                    feats_pre = [str(f) for f in saida.feat_labels]
                elif nome == "SIFTED":
                    # rotulos alinhados com as linhas de x (e do z do PILOT)
                    labels_z = [str(x) for x in saida.inst_labels]
                elif nome == "PILOT":
                    z_pilot = np.array(saida.z, dtype=float)
                    t0 = time.perf_counter()
                    z_corrigido, robustez = _corrigir_duplicatas(
                        saida.z, labels_z, fix_near_duplicates
                    )
                    tempos["deteccao_duplicatas"] = round(time.perf_counter() - t0, 3)
            model = isp.model
    finally:
        logger.remove(sink)
        isp.close()
    tempos["build_total"] = round(time.perf_counter() - t_inicio, 3)

    # ------------------------------------------------------------- escrita
    t0 = time.perf_counter()
    _checar_pasta(outdir, metadata_path)
    _limpar_pasta(outdir)
    model.save_to_csv(outdir)
    algos = [str(a) for a in model.data.algo_labels]
    labels = [str(x) for x in model.data.inst_labels]
    _gravar_coordenadas(outdir, labels, z_pilot, z_corrigido)
    shutil.copyfile(metadata_path, outdir / "metadata.csv")
    for nome, origem in auxiliares.items():
        shutil.copyfile(origem, outdir / nome)
    (outdir / "run_options.json").write_text(
        json.dumps(dataclasses.asdict(opts), indent=2, default=_json_default)
    )
    report, avisos = _sifted_report(model, feats_pre, feats_entrada, opts)
    report.to_csv(outdir / "sifted_report.csv", index=False)
    avisos += _gravar_sifted_extras(outdir, model, feats_pre, opts)
    avisos += _gravar_pilot_r2(outdir, model)
    info_pythia, avisos_pythia = _gravar_pythia(outdir, model, algos, labels)
    avisos += avisos_pythia
    espaco = _gravar_footprint_especial(outdir, "space", model.trace.space, None)
    especiais = {
        "space": espaco,
        "hard": _gravar_footprint_especial(outdir, "hard", model.trace.hard, espaco),
    }
    tempos["escrita"] = round(time.perf_counter() - t0, 3)

    contagem = Counter((w.category.__name__, str(w.message)) for w in capturados)
    info = {
        "gerado_por": GERADO_POR,
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "instancespace_version": version("instancespace"),
        "python": platform.python_version(),
        "metadata_entrada": str(metadata_path.resolve()),
        "n_instancias_entrada": int(len(meta.instance_labels)),
        "n_instancias": int(len(model.data.inst_labels)),
        "n_features_entrada": len(feats_entrada),
        "n_features_selecionadas": len(model.data.feat_labels),
        "algoritmos": [str(a) for a in model.data.algo_labels],
        "tem_source": meta.instance_sources is not None,
        "tipos_anotacao": {
            "arquivo": "annotations.json" if "annotations.json" in auxiliares else None,
            "declarados": tipos_declarados,
        },
        "arquivos_auxiliares": sorted(auxiliares),
        "tempos_s": tempos,
        "trace_robustez": robustez,
        "arquivos_footprint": _arquivos_footprint(outdir, model.data.algo_labels),
        "footprints_especiais": especiais,
        "pythia": info_pythia,
        "regra_bom": _regra_bom(opts.perf),
        "arquivos": sorted({p.name for p in _arquivos_do_engine(outdir)} | {"run_info.json"}),
        "avisos": avisos,
        "avisos_instancespace": list(dict.fromkeys(avisos_is)),
        "warnings_python": [
            {"categoria": c, "mensagem": m, "n": n} for (c, m), n in contagem.most_common()
        ],
    }
    (outdir / "run_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False, default=_json_default)
    )
    return info
