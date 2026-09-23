"""Ponte entre a tabela por instancia do projeto e o pyispace (PILOT + TRACE).

Duas funcoes publicas:

- ``to_isa_metadata``: converte ``resultados/table_<nome>.csv`` no formato que
  ``pyispace.train_is`` espera: colunas ``feature_*`` e ``algo_*`` numericas e
  indice ``instances`` com os rotulos "1".."n" em texto (o mesmo layout do
  ``Workspace`` do pyhard). Leva junto ``row_original`` e as anotacoes
  ``class``, ``ih`` e ``n_wrong``, que o pyispace ignora e o instancespace
  (``isaspace.engine``) preserva como anotacoes. Com ``outdir``, grava o
  metadata.csv e, ao lado, os arquivos auxiliares que o engine copia
  (``write_metadata``: annotations.json, degenerate_report.csv,
  feature_info.csv; formato em docs/output_format.md).
- ``run_isa``: monta as opcoes, roda ``train_is`` (PILOT + TRACE), grava os
  CSVs no layout do pyhard via ``pyispace.utils.scriptcsv`` e verifica que o
  significado de "bom" nao foi invertido.

Requer o pyispace corrigido para Python 3.11 (``scripts/apply_pyispace_patch.py``).
O pyispace 0.3.7 nao tem SIFTED, CLOISTER nem PYTHIA: toda ``feature_*`` que
sobreviver a ``to_isa_metadata`` entra no PILOT.
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import iqr as _iqr

_FEATURE = "feature_"
_ALGO = "algo_"
_PROBA = "proba_"
INDEX_NAME = "instances"
ROW_ORIGINAL = "row_original"
# colunas da tabela por instancia que viajam no metadata como anotacoes:
# nao sao feature_* nem algo_*, entao train_is e o instancespace nao as usam
ANOTACOES = ("class", "ih", "n_wrong")
# rotulo da classe: o valor nominal do alvo no OpenML, sempre texto
# (pipeline.build_instance_table grava pd.Series(y).astype(str))
ROTULO = "class"
# tipo declarado de cada anotacao (annotations.json): o CSV nao guarda tipo e
# "1"/"2" voltaria como numero
TIPOS_ANOTACAO = {
    "row_original": "identifier",       # indice da linha no dataset do OpenML
    "class": "categorica", "ih": "numerica", "n_wrong": "numerica_inteira",
}
# familia das medidas do pyhard (feature_info.csv): as que dependem de um
# modelo ajustado sao model_derived; as demais, geometric
MODEL_DERIVED = ("CL", "CLD", "DS", "DCP", "TD_U", "TD_P")

# Diferenca maxima tolerada entre a taxa "boa" do pyispace (Ybin) e a acuracia
# real de cada algoritmo. Acima disso o mais provavel e perf.MaxPerf invertido.
TOLERANCIA_YBIN = 0.15


def _importar_pyispace():
    """Importa o pyispace com mensagem util se o patch de 3.11 nao foi aplicado."""
    try:
        import pyispace  # noqa: F401
        from pyispace import preprocessing, utils
        from pyispace.train import train_is
    except (ValueError, ImportError) as exc:
        raise ImportError(
            "pyispace nao importa neste Python (defaults mutaveis em "
            "pyispace/train.py). Rode: python scripts/apply_pyispace_patch.py"
        ) from exc
    return train_is, preprocessing, utils


# --------------------------------------------------------------------------- #
# to_isa_metadata
# --------------------------------------------------------------------------- #
def _filtrar_degeneradas(F, min_var):
    """Separa as features que o pre-processamento do pyispace torna constantes.

    Reproduz exatamente o que ``train_is`` faz com ``auto.preproc=True``
    (train.py:128-134): ``bound_outliers`` (recorte em mediana +- 5*IQR) seguido
    de ``auto_normalize`` (Yeo-Johnson + z-score). Uma coluna com IQR = 0 e
    colapsada numa constante pelo recorte; depois disso o z-score deixa a
    variancia em zero, o PILOT devolve R2 = NaN para ela e a coluna so
    atrapalha o ajuste. Devolve (mantidas, descartadas) com o motivo de cada
    descarte e as variancias medidas.
    """
    _, preprocessing, _ = _importar_pyispace()
    X = F.to_numpy(dtype=float)
    n_nan = np.isnan(X).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        var_bruta = np.nanvar(X, axis=0)
        iqr = _iqr(X, axis=0, nan_policy="omit")

    # Colunas com NaN nao podem passar pelo pre-processamento do pyispace:
    # bound_outliers usa np.median, que devolve NaN, e o np.clip com limites
    # NaN transforma a coluna inteira em NaN; o PowerTransformer entao aborta
    # com scipy BracketError. Sao descartadas antes e nao entram no calculo.
    sem_nan = np.flatnonzero(n_nan == 0)
    var_pos = np.full(X.shape[1], np.nan)
    if sem_nan.size:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Xn = preprocessing.auto_normalize(
                preprocessing.bound_outliers(X[:, sem_nan])
            )
        var_pos[sem_nan] = np.nanvar(Xn, axis=0)

    mantidas, descartadas = [], []
    for j, col in enumerate(F.columns):
        degenerada = not (var_pos[j] >= min_var)  # NaN conta como degenerada
        if not degenerada:
            mantidas.append(col)
            continue
        if n_nan[j] == X.shape[0]:
            motivo = "todos os valores NaN"
        elif n_nan[j] > 0:
            motivo = (
                f"{int(n_nan[j])} NaN: bound_outliers (np.median) propaga NaN "
                "para a coluna inteira"
            )
        elif var_bruta[j] < min_var:
            motivo = "variancia bruta ~0 (coluna constante)"
        elif iqr[j] == 0:
            motivo = (
                "IQR = 0: bound_outliers (mediana +- 5*IQR) colapsa a coluna "
                "numa constante"
            )
        else:
            motivo = f"variancia < {min_var:g} apos Yeo-Johnson + z-score"
        descartadas.append(
            {
                "feature": col,
                "var_bruta": float(var_bruta[j]),
                "iqr": float(iqr[j]),
                "var_pos_preproc": float(var_pos[j]),
                "motivo": motivo,
            }
        )
    return mantidas, descartadas


def to_isa_metadata(
    table, drop_degenerate=True, min_var=1e-8, proba_as_performance=True,
    annotations=ANOTACOES, outdir=None,
):
    """Converte a tabela por instancia no metadata que ``train_is`` consome.

    Parametros
    ----------
    table : DataFrame com colunas ``feature_*``, ``algo_*`` (acerto 0/1) e
        ``proba_*`` (probabilidade da classe verdadeira), como as geradas por
        ``isaspace.pipeline.build_instance_table``.
    drop_degenerate : descarta as ``feature_*`` cuja variancia fica abaixo de
        ``min_var`` depois do pre-processamento do proprio pyispace (ver
        ``_filtrar_degeneradas``).
    proba_as_performance : se True, as ``algo_*`` de acerto 0/1 sao
        descartadas e as ``proba_*`` viram ``algo_<nome>`` (desempenho
        continuo, maior e melhor: e o que o PILOT consegue ajustar). Se False,
        mantem as ``algo_*`` originais e descarta as ``proba_*``.
    annotations : colunas da tabela copiadas para o metadata como anotacoes
        (padrao ``class``, ``ih``, ``n_wrong``). Todas tem de existir na
        tabela. ``class`` vai como texto: o rotulo nominal do OpenML.
    outdir : se dado, grava o metadata e os arquivos auxiliares nessa pasta
        (``write_metadata``).

    Retorna
    -------
    (metadata, info)
      metadata : DataFrame com ``row_original`` (indice original da tabela),
          as anotacoes, as ``feature_*`` mantidas e as ``algo_*``; indice
          ``instances`` com os rotulos "1".."n" em texto. As colunas que nao
          comecam com ``feature_`` ou ``algo_`` sao ignoradas por
          ``train_is`` (train.py:56-57) e tratadas como anotacoes pelo
          instancespace.
      info : dict com ``features_kept``, ``features_dropped`` (lista de dicts
          com feature, var_bruta, iqr, var_pos_preproc e motivo),
          ``performance_source`` ("proba" ou "acerto"), ``algos``,
          ``annotations``, ``row_original`` (Series instances -> indice
          original, para o join da interface), ``acerto_real`` (taxa de
          acerto 0/1 por algoritmo, usada pelo guarda-corpo de ``run_isa``),
          ``annotation_types`` (tipos declarados, TIPOS_ANOTACAO),
          ``degenerate_report`` (DataFrame feature, var_bruta, iqr, motivo)
          e ``feature_info`` (DataFrame feature, family; todas as medidas da
          tabela).
    """
    feature_cols = [c for c in table.columns if c.startswith(_FEATURE)]
    algo_cols = [c for c in table.columns if c.startswith(_ALGO)]
    if not feature_cols:
        raise ValueError("tabela sem colunas feature_*")
    if not algo_cols:
        raise ValueError("tabela sem colunas algo_*")
    annotations = list(annotations)
    faltam = [c for c in annotations if c not in table.columns]
    if faltam:
        raise ValueError(f"tabela sem as colunas de anotacao {faltam}")
    reservadas = [
        c for c in annotations
        if c.casefold() in (INDEX_NAME, "source", ROW_ORIGINAL)
        or c.casefold().startswith((_FEATURE, _ALGO))
    ]
    if reservadas:
        raise ValueError(f"nome reservado usado como anotacao: {reservadas}")

    F = table[feature_cols].astype(float)
    if drop_degenerate:
        kept, dropped = _filtrar_degeneradas(F, min_var)
    else:
        kept, dropped = feature_cols, []
    if len(kept) < 2:
        raise ValueError(
            f"so {len(kept)} feature(s) sobreviveram ao filtro; o PILOT precisa "
            "de pelo menos 2"
        )

    nomes = [c[len(_ALGO):] for c in algo_cols]
    if proba_as_performance:
        faltando = [n for n in nomes if f"{_PROBA}{n}" not in table.columns]
        if faltando:
            raise ValueError(f"sem coluna proba_* para: {faltando}")
        Y = table[[f"{_PROBA}{n}" for n in nomes]].astype(float)
        Y.columns = [f"{_ALGO}{n}" for n in nomes]
        fonte = "proba"
    else:
        Y = table[algo_cols].astype(float)
        fonte = "acerto"

    n = len(table)
    # rotulos em texto; os valores "1".."n" sao os mesmos do RangeIndex antigo,
    # entao o Row 1..n do coordinates.csv do pyispace continua casando
    novo_indice = pd.Index(
        [str(i) for i in range(1, n + 1)], name=INDEX_NAME, dtype=object
    )
    row_original = pd.Series(
        table.index.to_numpy(), index=novo_indice, name=ROW_ORIGINAL
    )

    anot = table[annotations].copy()
    if ROTULO in anot.columns:
        # a tabela relida do CSV traz os rotulos nominais "1"/"2" (blood) e
        # "0"/"1" (hill-valley) como inteiros; volta para o texto do OpenML
        anot[ROTULO] = anot[ROTULO].astype(str)
    metadata = pd.concat([anot, F[kept], Y], axis=1)
    metadata.index = novo_indice
    metadata.insert(0, ROW_ORIGINAL, row_original.to_numpy())

    acerto_real = {n_: float(table[f"{_ALGO}{n_}"].mean()) for n_ in nomes}
    info = {
        "n_instances": n,
        "features_kept": kept,
        "features_dropped": dropped,
        "performance_source": fonte,
        "algos": nomes,
        "annotations": annotations,
        "row_original": row_original,
        "acerto_real": acerto_real,
        "annotation_types": {c: TIPOS_ANOTACAO[c] for c in [ROW_ORIGINAL, *annotations]
                             if c in TIPOS_ANOTACAO},
        "degenerate_report": pd.DataFrame(
            [{"feature": d["feature"][len(_FEATURE):], "var_bruta": d["var_bruta"],
              "iqr": d["iqr"], "motivo": d["motivo"]} for d in dropped],
            columns=["feature", "var_bruta", "iqr", "motivo"],
        ),
        "feature_info": pd.DataFrame({
            "feature": [c[len(_FEATURE):] for c in feature_cols],
            "family": ["model_derived" if c[len(_FEATURE):] in MODEL_DERIVED else "geometric"
                       for c in feature_cols],
        }),
    }
    # copia do guarda-corpo viajando junto com o DataFrame, para run_isa
    # funcionar mesmo sem receber a tabela original
    metadata.attrs["acerto_real"] = acerto_real
    if outdir is not None:
        write_metadata(metadata, info, outdir)
    return metadata, info


def write_metadata(metadata, info, outdir):
    """Grava metadata.csv e, ao lado, os arquivos que o engine copia se existirem.

    - annotations.json: {anotacao: "categorica" | "numerica" | "numerica_inteira" |
      "identifier"};
    - degenerate_report.csv: medidas descartadas antes do engine (feature,
      var_bruta, iqr, motivo); so o cabecalho quando nenhuma caiu;
    - feature_info.csv: feature, family de todas as medidas recebidas.
    Devolve os caminhos gravados.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    caminhos = [outdir / n for n in ("metadata.csv", "annotations.json",
                                     "degenerate_report.csv", "feature_info.csv")]
    metadata.to_csv(caminhos[0])
    caminhos[1].write_text(json.dumps(info["annotation_types"], indent=2) + "\n")
    info["degenerate_report"].to_csv(caminhos[2], index=False)
    info["feature_info"].to_csv(caminhos[3], index=False)
    return caminhos


# --------------------------------------------------------------------------- #
# run_isa
# --------------------------------------------------------------------------- #
def _acerto_real(metadata, table):
    """Taxa de acerto 0/1 por algoritmo, da tabela original ou de metadata.attrs."""
    algos = [c[len(_ALGO):] for c in metadata.columns if c.startswith(_ALGO)]
    if table is not None:
        faltando = [a for a in algos if f"{_ALGO}{a}" not in table.columns]
        if faltando:
            raise ValueError(f"tabela original sem algo_* para: {faltando}")
        return {a: float(table[f"{_ALGO}{a}"].mean()) for a in algos}
    guardado = metadata.attrs.get("acerto_real")
    if guardado is None:
        raise ValueError(
            "run_isa precisa da taxa de acerto real para o guarda-corpo: passe "
            "table=<tabela original> ou use o metadata devolvido por "
            "to_isa_metadata"
        )
    return {a: float(guardado[a]) for a in algos}


def _verificar_ybin(model, metadata, table, perf_epsilon):
    """Guarda-corpo: a taxa 'boa' do pyispace tem de bater com a acuracia real.

    ``train_is`` remove algoritmos sem nenhuma instancia boa (train.py:101-107);
    para esses a taxa boa e tratada como 0, o que tambem dispara o erro.
    """
    acerto_real = _acerto_real(metadata, table)
    taxa_boa = dict(zip(model.data.algolabels, model.data.Ybin.mean(axis=0)))

    linhas, problemas = [], []
    for algo, real in acerto_real.items():
        boa = float(taxa_boa.get(algo, 0.0))
        diff = abs(boa - real)
        linhas.append((algo, boa, real, diff, algo in taxa_boa))
        if diff > TOLERANCIA_YBIN:
            problemas.append(f"{algo}: taxa boa={boa:.3f} vs acuracia={real:.3f}")

    check = pd.DataFrame(
        linhas,
        columns=[
            "algoritmo", "taxa_boa_ybin", "acuracia_real", "diferenca", "no_modelo"
        ],
    ).set_index("algoritmo")

    if problemas:
        raise RuntimeError(
            "Taxa 'boa' (Ybin) diverge da acuracia real em mais de "
            f"{TOLERANCIA_YBIN}: " + "; ".join(problemas) + ". "
            "Sintoma classico de perf.MaxPerf invertido: com MaxPerf=False o "
            "pyispace considera 'bom' o desempenho <= epsilon, ou seja, o ERRO "
            "quando algo_* e acerto ou probabilidade (maior e melhor). Confira "
            f"opts['perf'] (MaxPerf deve ser True) e perf_epsilon={perf_epsilon}."
        )
    return check


def run_isa(metadata, outdir, perf_epsilon=0.5, seed=42, table=None):
    """Roda PILOT + TRACE do pyispace e grava os CSVs no layout do pyhard.

    Parametros
    ----------
    metadata : saida de ``to_isa_metadata`` (ou qualquer DataFrame com
        ``feature_*`` e ``algo_*`` em que maior e melhor).
    outdir : pasta de saida; recebe ``metadata.csv``, ``options.json`` e os
        arquivos de ``pyispace.utils.scriptcsv`` (coordinates.csv,
        footprint_performance.csv, algorithm_bin.csv, beta_easy.csv,
        good_algos.csv, footprint_<algo>_<good|best>.csv, model.pkl, ...).
    perf_epsilon : instancia e "boa" para um algoritmo se algo_* >= epsilon.
    seed : semente do PILOT (pilot.py:64) e do desempate de melhor algoritmo
        (train.py:119, que usa np.random sem semente propria).
    table : tabela original, para o guarda-corpo comparar a taxa 'boa' com a
        acuracia real. Se omitida, usa ``metadata.attrs['acerto_real']``.

    Retorna o ``pyispace.train.Model``; ``model.ybin_check`` guarda a tabela
    de comparacao do guarda-corpo.
    """
    train_is, _, utils = _importar_pyispace()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Opcoes do pyispace 0.3.7. So estas chaves sao lidas por train_is, pilot e
    # trace; as demais do options.json do pyhard (parallel, corr, clust,
    # cloister, pythia, selvars, outputs, auto.featsel, trace.usesim) sao
    # aceitas e ignoradas.
    opts = {
        "perf": {
            # OBRIGATORIO: com False, "bom" vira desempenho <= epsilon, ou seja,
            # o ERRO (train.py:84-96). Nossas algo_* sao acerto/probabilidade.
            "MaxPerf": True,
            # limiar absoluto (Ybin = Y >= epsilon), nao relativo ao melhor
            "AbsPerf": True,
            "epsilon": perf_epsilon,
        },
        # instancia e beta-facil se mais de 55% dos algoritmos sao bons
        "general": {"betaThreshold": 0.55},
        # pre-processamento do pyispace: recorte de outliers + Yeo-Johnson/z-score
        "auto": {"preproc": True},
        "bound": {"flag": True},
        "norm": {"flag": True},
        # PILOT numerico (BFGS) com 5 tentativas e semente fixa
        "pilot": {"analytic": False, "ntries": 5, "seed": seed},
        # TRACE: pureza minima 0.55. parallel=True quebra: os processos filhos
        # do joblib reimportam o pyispace (e o patch so vale no disco local).
        "trace": {"PI": 0.55, "parallel": False},
    }

    metadata.to_csv(outdir / "metadata.csv")
    (outdir / "options.json").write_text(json.dumps(opts, indent=2))

    logging.getLogger("pyispace").setLevel(logging.INFO)
    np.random.seed(seed)
    model = train_is(metadata, opts, rotation_adjust=True)

    # guarda-corpo mais importante do arquivo: antes de gravar qualquer saida
    model.ybin_check = _verificar_ybin(model, metadata, table, perf_epsilon)

    utils.scriptcsv(model, outdir)
    return model
