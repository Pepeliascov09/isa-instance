"""Interface do espaco de instancias (Panel 1.x) sobre isaspace.ui.loader_is.

Seis abas sobre um estado global unico (EstadoGlobal):

- "Instance Space" (0): um scatter z_1 x z_2 largo, com o lasso ativo ao abrir,
  colorido pela variavel global de cor; sobreposicoes opcionais de footprints
  (algoritmo e tipo), da fronteira do CLOISTER e da footprint hard.
- "Footprint Performance" (1): mapa das footprints com a selecao destacada e a
  tabela footprint_performance.csv com o status (ok / suspeita / vazia).
- "Algorithm Selection" (2): o que o PYTHIA e o CLOISTER produziram: mapa
  colorido pelo recomendado (selection0, "nenhum" como categoria), pela
  concordancia com o melhor observado ou pelo pr0_sub (P(ruim) fora da
  amostra) de um algoritmo, com a fronteira do CLOISTER e a selecao
  destacada; svm_table; matrizes de confusao da validacao cruzada; aviso de
  seletor quase trivial (um algoritmo em mais de SELETOR_TRIVIAL das instancias).
- "Distributions" (3): distribuicao de variaveis numericas agrupada por uma
  anotacao categorica (histograma, densidade ou violino); com selecao, cada
  grupo dividido em selecionadas e nao selecionadas.
- "Features" (4): uma linha por feature recebida, com o que foi mantido e por
  que o resto caiu (degenerate_report.csv + SIFTED), o heatmap das correlacoes
  feature x algoritmo e a silhueta por k.
- "Data Explorer" (5): scatter x-y com a cor global e a selecao destacada,
  filtro pandas.query local e o botao "Usar filtro como selecao".

Na sidebar fixa: o seletor de dataset, com resultados/is/ e runs/ (execucoes
disparadas pela interface) em grupos separados; o bloco "Novo instance space"
(isaspace.ui.novo: envio de metadata.csv, validacao, regra de desempenho e
execucao do engine em subprocesso); tipos inferidos das anotacoes (trocaveis
na sessao) e os botoes de exportacao (instancias, selecao, rotulos da
footprint ativa).

ESTADO GLOBAL. EstadoGlobal (param.Parameterized) guarda o dataset ativo, o
IsResult lido, a selecao e a variavel de cor; as abas leem so dele e se
redesenham quando ele muda. A selecao e um conjunto de rotulos Row:
    None           sem selecao
    frozenset()    selecao vazia (lasso numa area sem pontos, filtro sem linhas)
    frozenset(...) instancias selecionadas
Ela vem do stream Selection1D do scatter da aba 0. O Bokeh manda um
Selection1D a cada movimento do lasso e nenhum quando o lasso cai numa area
vazia partindo de "nada selecionado" (os indices nao mudam); por isso o
Selection1D so guarda o indice mais recente e a selecao global e gravada depois
do fim do gesto: os streams de geometria (Lasso, BoundsXY), que chegam uma vez
ao soltar o mouse, agendam a gravacao em ESPERA_GEOMETRIA_MS, e cada
Selection1D a reagenda em ESPERA_SELECAO_MS.

A variavel de cor e global: os seletores das abas 0 e 5 sao duas vistas do
mesmo EstadoGlobal.cor.

TITULO. O titulo do template do Panel 1.9 so e aplicado na primeira
renderizacao (panel/template/base.py:770-788, e o servidor desliga
use_for_title); o nome do dataset vai num pane do cabecalho e no componente
TituloAba, que escreve document.title no navegador.

REGRA ARQUITETURAL: este modulo nao importa instancespace, pyispace, pyhard nem
sklearn; le apenas as pastas de saida do engine via isaspace.ui.loader_is
(formato em docs/output_format.md). So isaspace.ui.execucao conhece o engine,
e o roda em subprocesso. O app anterior (Panel 0.14, pyispace) esta em
isaspace/ui/app_legacy.py.

Versoes alvo: Panel 1.9.4, HoloViews 1.23.2, Bokeh 3.9.2 (.venv-isa).

Uso (da raiz do projeto, com o .venv-isa):
    python -m isaspace.ui.app [--port 5006] [--no-show] [--root resultados/is] [--runs runs]
"""

import argparse
import io
import sys
from functools import partial, reduce
from pathlib import Path

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
from bokeh.models import HoverTool
from holoviews.streams import BoundsXY, Lasso, PlotReset, Selection1D
from panel.custom import JSComponent

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from isaspace.ui.execucao import PASTA_RUNS  # noqa: E402
from isaspace.ui.loader_is import (  # noqa: E402
    CATEGORICA, IDENTIFICADOR, NUMERICA, SUSPEITA, VAZIA, e_numerica, list_available,
    load_is_output,
)
from isaspace.ui.novo import NovoInstanceSpace  # noqa: E402

pn.extension("tabulator", notifications=True)
pn.config.disconnect_notification = (
    "Conexão com o servidor perdida: esta página não atualiza mais. Recarregue (F5)."
)
hv.extension("bokeh")

TITULO = "isa-instance"
TABS = ["Instance Space", "Footprint Performance", "Algorithm Selection", "Distributions",
        "Features", "Data Explorer"]
PASTA_IS = RAIZ / "resultados" / "is"
# chaves do seletor de dataset: "<origem>/<pasta>"; o rotulo e so a pasta
ORIGENS = {"is": "resultados/is", "runs": "runs (execuções pela interface)"}
CONTROL_WIDTH = 300
TODOS = "todos"
MAX_DIST_VARS = 6
R2_BAIXO = 0.3              # abaixo disso o plano 2D explica pouco da variavel
ESPERA_GEOMETRIA_MS = 150   # fim do gesto (Lasso/BoundsXY) -> grava a selecao
ESPERA_SELECAO_MS = 400     # Selection1D sem geometria (debounce)
PALETA_ALGOS = ["#e63946", "#2a9d8f", "#e9c46a", "#8338ec", "#ff7f0e", "#118ab2",
                "#6a994e", "#bc4749", "#577590", "#f15bb5"]
COR_BASE = "#5c677d"        # pontos do mapa de footprints
COR_TODAS = "#5c677d"       # serie "Todas" nas distribuicoes
COR_SEL = "#e63946"         # serie "Selecionadas"
ALPHA_SEL, ALPHA_FORA, ALPHA_NEUTRO = 0.9, 0.12, 0.7
NENHUM = "nenhum"           # valor ausente numa variavel categorica
SEM_GRUPO = "(nenhum)"      # opcao "agrupar por" sem agrupamento
TIPOS_DIST = ["histograma", "densidade", "violino"]
MIN_GRUPOS_VIOLINO = 3      # a partir daqui histogramas sobrepostos ficam ilegiveis
COR_NAO_SEL = "#adb5bd"     # parte "nao selecionadas" dos violinos
PALETA_GRUPOS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
                 "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
# graficos empilhados numa pagina que rola: sem wheel_zoom ativo, a roda do
# mouse rola a pagina em vez de dar zoom no grafico sob o cursor
SEM_ROLAGEM = dict(active_tools=["pan"])
# nomes das colunas da tabela de features na tela (o loader usa os nomes longos)
COLUNAS_FEATURES = {"substituida_por": "ficou no lugar", "max_abs_rho": "|rho| máx",
                    "algoritmo_rho": "algoritmo", "pval": "p", "r2_pilot": "r² PILOT"}
# aba Algorithm Selection
SELETOR_TRIVIAL = 0.9       # um algoritmo recomendado em mais que isso das instancias
COR_NENHUM = "#6c757d"      # sem recomendacao (selection0 = -1); cinza escuro, visivel no branco
AS_RECOMENDADO, AS_CONCORDANCIA, AS_PR0 = "recomendado", "concordancia", "pr0_sub"
AS_CORES = {"algoritmo recomendado (selection0)": AS_RECOMENDADO,
            "concordância com o melhor observado": AS_CONCORDANCIA,
            "pr0_sub de um algoritmo": AS_PR0}
IGUAL, DIFERENTE, SEM_REC = ("igual ao melhor observado", "diferente do melhor observado",
                             "sem recomendação")
CORES_CONCORDANCIA = {IGUAL: "#2a9d8f", DIFERENTE: "#f4a261", SEM_REC: COR_NENHUM}
# colunas da svm_table na tela: (nome no arquivo, nome curto)
COLUNAS_SVM = [("CV_model_accuracy", "acurácia CV %"), ("CV_model_precision", "precisão CV %"),
               ("CV_model_recall", "recall CV %"), ("Probability_of_good", "P(bom)"),
               ("Avg_Perf_all_instances", "desemp. médio"),
               ("Avg_Perf_selected_instances", "desemp. prev. boas")]
EXPORT_DERIVADAS = ["z_1", "z_2", "NumGoodAlgos", "IsBetaEasy", "best_algo", "best_algo_svm"]
# colunas derivadas de IsResult.instances oferecidas como cor: (rotulo, tipo)
DERIVADAS = {
    "NumGoodAlgos": ("n. de algoritmos bons", NUMERICA),
    "IsBetaEasy": ("beta-fácil", CATEGORICA),
    "best_algo": ("melhor algoritmo observado", CATEGORICA),
    "best_algo_svm": ("recomendado pelo PYTHIA", CATEGORICA),
}


class TituloAba(JSComponent):
    """Escreve `titulo` em document.title no navegador (ver TITULO no docstring)."""

    titulo = param.String(default="")

    _esm = """
    export function render({ model }) {
      const aplicar = () => { if (model.titulo) document.title = model.titulo }
      aplicar()
      model.on("titulo", aplicar)
    }
    """


class EstadoGlobal(param.Parameterized):
    """Estado unico da interface: todas as abas leem daqui."""

    dataset = param.Selector(default=None, objects=[], doc="""
        Chave '<origem>/<pasta>' da saida do engine (is/iris, runs/<nome>_<data>)""")
    resultado = param.Parameter(default=None, doc="IsResult do dataset ativo")
    selecao = param.Parameter(default=None, doc="""
        None = sem selecao; frozenset de rotulos Row, vazio = selecao vazia""")
    cor = param.String(default="", doc="coluna de IsResult.instances que colore os pontos")


def catalogo_de_cores(r) -> dict:
    """{coluna de r.instances: (rotulo, grupo, tipo)} das variaveis de cor."""
    cat = {}
    for col, tipo in r.annotations.items():
        if tipo != IDENTIFICADOR:           # identificadores nao colorem pontos
            cat[col] = (col, "Anotações", tipo)
    if r.source_column is not None:
        cat[r.source_column] = ("source", "Anotações", CATEGORICA)
    for f in r.features:
        cat[f"feature_{f}"] = (f, "Features no PILOT", NUMERICA)
    for f in r.features_fora_pilot:
        cat[f"feature_{f}"] = (f, "Features fora do PILOT (SIFTED)", NUMERICA)
    for a in r.algos:
        cat[f"algo_{a}"] = (f"algo_{a}", "Desempenho (algo_*)", NUMERICA)
    for col, (rotulo, tipo) in DERIVADAS.items():
        cat[col] = (rotulo, "Derivadas", tipo)
    return cat


def grupos_categoricos(r) -> list:
    """Colunas de r.instances que servem para agrupar: anotacoes categoricas e source."""
    cols = [c for c, t in r.annotations.items() if t == CATEGORICA]
    return cols + ([r.source_column] if r.source_column else [])


def cor_padrao(r, cat) -> str:
    """Primeira anotacao categorica; senao a primeira anotacao; senao NumGoodAlgos."""
    for col, tipo in r.annotations.items():
        if tipo == CATEGORICA:
            return col
    return next(iter(r.annotations), "NumGoodAlgos")


def grupos_select(cat) -> dict:
    """{grupo: {rotulo: coluna}} para pn.widgets.Select(groups=...); rotulo
    repetido entre grupos (anotacao com o nome de uma feature) vira a coluna."""
    contagem = {}
    for rotulo, _, _ in cat.values():
        contagem[rotulo] = contagem.get(rotulo, 0) + 1
    grupos = {}
    for col, (rotulo, grupo, _) in cat.items():
        grupos.setdefault(grupo, {})[rotulo if contagem[rotulo] == 1 else col] = col
    return grupos


def estilo_de_cor(valores: pd.Series, tipo: str) -> dict:
    if tipo == CATEGORICA:
        n = valores.nunique()
        cmap = "Category10" if n <= 10 else ("Category20" if n <= 20 else "glasbey")
        return dict(cmap=cmap, colorbar=False, show_legend=True)
    return dict(cmap="viridis", colorbar=True, show_legend=False)


def poligonos_hv(fp, cor, rotulo, fill_alpha=0.25, tracejado=False):
    """Footprint do loader_is -> hv.Polygons (uma geometria por Part, com furos)."""
    geoms = [{"x": p.exterior[:, 0], "y": p.exterior[:, 1], "holes": [list(p.furos)]}
             for p in fp.poligonos]
    # show_legend explicito: no HoloViews 1.23 Polygons e Path nascem sem legenda
    estilo = dict(fill_color=cor, line_color=cor, fill_alpha=fill_alpha, line_width=1.5,
                  line_alpha=0.85, show_legend=True)
    if tracejado:
        estilo.update(fill_alpha=fill_alpha / 2, line_dash="dashed", line_width=2)
    return hv.Polygons(geoms, label=rotulo).opts(**estilo)


def cloister_hv(r) -> list:
    """Fronteira do CLOISTER (e a podada, se diferente) como hv.Path."""
    if r.bounds is None:
        return []
    anel = np.vstack([r.bounds.exterior, r.bounds.exterior[:1]])
    camadas = [hv.Path([anel], label="CLOISTER").opts(
        color="black", line_width=2, line_dash="dashed", show_legend=True)]
    if r.bounds_pruned is not None and not np.array_equal(r.bounds.exterior, r.bounds_pruned.exterior):
        anel = np.vstack([r.bounds_pruned.exterior, r.bounds_pruned.exterior[:1]])
        camadas.append(hv.Path([anel], label="CLOISTER (podada)").opts(
            color="#6c757d", line_width=1.5, line_dash="dotted", show_legend=True))
    return camadas


def _pct(x) -> str:
    return f"{100 * x:.1f}%".replace(".", ",")


class IsaApp:
    """Interface do espaco de instancias: uma instancia por sessao do navegador."""

    def __init__(self, root=PASTA_IS, runs=PASTA_RUNS):
        self.root = Path(root)
        self.runs = Path(runs)
        self.estado = EstadoGlobal()
        grupos = self._listar_datasets()
        if not self.datasets:
            raise FileNotFoundError(f"nenhuma pasta com run_info.json em {self.root} nem em {self.runs}")
        self._cache = {}
        self._dados = None           # r.instances com Row como coluna (base dos plots)
        self._catalogo = {}
        self._building = False       # atualizando opcoes de widgets: ignora os eventos
        self._doc = None             # documento da sessao (debounce da selecao)
        self._geracao = 0            # streams de plots antigos sao ignorados
        self._indice = []            # ultimo Selection1D do scatter da aba 0
        self._pendente = None
        self._selecao_do_plot = None  # ultima selecao gravada pelo proprio scatter
        self._tipos_forcados = {}    # {dataset: {coluna do metadata: tipo}}, so nesta sessao
        self._grupo_escolhido = False  # "agrupar por" escolhido pelo usuario (senao, o padrao)

        # --- cabecalho: nome do dataset (pane) e titulo da aba do navegador
        self.cab = pn.pane.HTML("", margin=(0, 10))
        self.titulo_aba = TituloAba(titulo=TITULO, width=0, height=0, margin=0)

        # --- sidebar fixa
        self.w_dataset = pn.widgets.Select(name="Dataset", groups=grupos, value=self.datasets[0],
                                           width=CONTROL_WIDTH - 20)
        self.novo = NovoInstanceSpace(self.runs, CONTROL_WIDTH - 40, self._on_execucao_concluida)
        self.w_reload = pn.widgets.Button(name="Recarregar dataset", width=CONTROL_WIDTH - 20)
        self.info = pn.pane.Markdown("", width=CONTROL_WIDTH - 20)
        self.sel_info = pn.pane.Markdown("", width=CONTROL_WIDTH - 20)
        self.w_limpar = pn.widgets.Button(name="Limpar seleção", width=CONTROL_WIDTH - 20)
        self.tipos_card = pn.Card(title="Tipos inferidos", collapsed=True, visible=False,
                                  width=CONTROL_WIDTH - 20, margin=(5, 10))
        largura = dict(width=CONTROL_WIDTH - 20)
        self.w_exp_todas = pn.widgets.FileDownload(
            callback=self._csv_todas, filename="instancias.csv",
            label="Exportar instâncias (todas)", **largura)
        self.w_exp_sel = pn.widgets.FileDownload(
            callback=self._csv_selecao, filename="selecao.csv", label="Exportar seleção",
            disabled=True, **largura)
        self.w_exp_fp = pn.widgets.FileDownload(
            callback=self._csv_footprint, filename="footprint.csv",
            label="Exportar rótulos da footprint", disabled=True, **largura)
        self.exp_nota = pn.pane.Markdown("", **largura)

        # --- aba 0
        self.w_cor = pn.widgets.Select(name="Cor dos pontos", width=CONTROL_WIDTH - 110)
        self.r2 = pn.pane.Markdown("", width=90, margin=(28, 0, 0, 5))
        self.r2_aviso = pn.pane.Markdown("", width=CONTROL_WIDTH - 20)
        self.w_ov_fp = pn.widgets.Checkbox(name="Footprints")
        self.w_ov_algo = pn.widgets.Select(name="Algoritmo", options=[TODOS], width=CONTROL_WIDTH - 40)
        self.w_ov_tipo = pn.widgets.RadioButtonGroup(options=["good", "best"], value="good")
        self.w_ov_cloister = pn.widgets.Checkbox(name="Fronteira do CLOISTER")
        self.w_ov_hard = pn.widgets.Checkbox(name="Footprint hard (instâncias não beta-fáceis)")
        self.ov_nota = pn.pane.Markdown("", width=CONTROL_WIDTH - 20)
        self.espaco_status = pn.pane.Markdown("")
        self.espaco_plot = pn.pane.HoloViews(sizing_mode="stretch_width", min_height=640)

        # --- aba 1
        self.w_fp_algo = pn.widgets.Select(name="Algoritmo", options=[TODOS], width=CONTROL_WIDTH - 20)
        self.w_fp_tipo = pn.widgets.RadioButtonGroup(options=["good", "best"], value="good")
        self.fp_aviso = pn.pane.Markdown("", width=CONTROL_WIDTH - 20)
        self.fp_status = pn.pane.Markdown("")
        self.fp_plot = pn.pane.HoloViews(sizing_mode="stretch_width", min_height=460)
        self.fp_tabela = pn.widgets.Tabulator(
            pd.DataFrame(), disabled=True, layout="fit_data_stretch",
            sizing_mode="stretch_width", height=280, show_index=True)

        # --- aba Algorithm Selection
        self.w_as_cor = pn.widgets.Select(name="Colorir por", options=AS_CORES,
                                          value=AS_RECOMENDADO, width=CONTROL_WIDTH - 20)
        self.w_as_algo = pn.widgets.Select(name="Algoritmo (pr0_sub)", width=CONTROL_WIDTH - 20,
                                           visible=False)
        self.w_as_cloister = pn.widgets.Checkbox(name="Fronteira do CLOISTER", value=True)
        self.as_status = pn.pane.Markdown("")
        self.as_aviso = pn.Column(sizing_mode="stretch_width")
        self.as_resumo = pn.pane.Markdown("")
        self.as_plot = pn.pane.HoloViews(sizing_mode="stretch_width", min_height=520)
        self.as_tabela = pn.widgets.Tabulator(
            pd.DataFrame(), disabled=True, layout="fit_data_stretch", sizing_mode="stretch_width",
            show_index=False, pagination=None, selectable=False)
        self.as_nota_tabela = pn.pane.Markdown("")
        self.as_confusao = pn.FlexBox(sizing_mode="stretch_width")
        self.as_nota_confusao = pn.pane.Markdown("")

        # --- aba 2
        self.w_dist_vars = pn.widgets.MultiChoice(name="Variáveis", width=CONTROL_WIDTH - 20)
        self.w_dist_grupo = pn.widgets.Select(name="Agrupar por", options=[SEM_GRUPO],
                                              width=CONTROL_WIDTH - 20)
        self.w_dist_tipo = pn.widgets.RadioButtonGroup(options=TIPOS_DIST, value="histograma")
        self.dist_status = pn.pane.Markdown("")
        self.dist_plots = pn.Column(sizing_mode="stretch_width")

        # --- aba 3 (Features)
        self.feat_resumo = pn.pane.Markdown("")
        self.feat_tabela = pn.widgets.Tabulator(
            pd.DataFrame(), disabled=True, layout="fit_data_stretch", sizing_mode="stretch_width",
            show_index=False, pagination=None, widths={"motivo": 300},
            formatters={"motivo": {"type": "textarea"}})
        self.feat_heatmap = pn.pane.HoloViews(sizing_mode="stretch_width")
        self.feat_silhueta = pn.Column(sizing_mode="stretch_width")

        # --- aba 4
        self.w_ex_x = pn.widgets.Select(name="Eixo x", width=CONTROL_WIDTH - 20)
        self.w_ex_y = pn.widgets.Select(name="Eixo y", width=CONTROL_WIDTH - 20)
        self.w_ex_cor = pn.widgets.Select(name="Cor", width=CONTROL_WIDTH - 20)
        self.w_ex_query = pn.widgets.TextInput(
            name="Filtro (pandas query)", width=CONTROL_WIDTH - 20,
            placeholder="ex.: z_1 > 0 and NumGoodAlgos <= 3")
        self.ex_filtro = pn.pane.Markdown("", width=CONTROL_WIDTH - 20)
        self.w_ex_usar = pn.widgets.Button(name="Usar filtro como seleção", button_type="primary",
                                           width=CONTROL_WIDTH - 20, disabled=True)
        self.ex_status = pn.pane.Markdown("")
        self.ex_plot = pn.pane.HoloViews(sizing_mode="stretch_width", min_height=460)
        self.ex_titulo_tabela = pn.pane.Markdown("")
        self.ex_tabela = pn.widgets.Tabulator(
            pd.DataFrame(), disabled=True, layout="fit_data_stretch", sizing_mode="stretch_width",
            height=300, show_index=False, pagination="local", page_size=25)
        self._ex_filtradas = None    # rotulos que passam no filtro (None = filtro invalido/vazio)

        # --- sidebar e abas
        self.controls = [
            pn.Column("## Instance Space", pn.Row(self.w_cor, self.r2), self.r2_aviso,
                      "### Sobrepor", self.w_ov_fp, self.w_ov_algo, self.w_ov_tipo,
                      self.w_ov_cloister, self.w_ov_hard, self.ov_nota, width=CONTROL_WIDTH),
            pn.Column("## Footprint Performance", self.w_fp_algo, "### Tipo", self.w_fp_tipo,
                      self.fp_aviso, width=CONTROL_WIDTH),
            pn.Column("## Algorithm Selection", self.w_as_cor, self.w_as_algo, self.w_as_cloister,
                      pn.pane.Markdown(
                          "O que o **PYTHIA** e o **CLOISTER** produziram.\n\n"
                          "- **Recomendado** (selection0): entre os algoritmos cujo SVM final "
                          "prevê *bom* na instância, o de maior precisão na validação cruzada; "
                          "*nenhum* quando nenhum SVM prevê bom.\n"
                          "- **Melhor observado**: maior desempenho real (portfolio.csv; empates "
                          "desfeitos ao acaso pelo PRELIM).\n"
                          "- **Probabilidades**: pr0_sub = P(ruim) **fora da amostra** (validação "
                          "cruzada), não a do modelo final (pr0_hat).\n"
                          "- **CLOISTER**: fronteira estimada do espaço onde instâncias plausíveis "
                          "podem existir.", width=CONTROL_WIDTH - 20),
                      width=CONTROL_WIDTH),
            pn.Column("## Distributions", self.w_dist_vars,
                      pn.pane.Markdown(f"_Até {MAX_DIST_VARS} variáveis por vez._"),
                      self.w_dist_grupo, "### Tipo", self.w_dist_tipo,
                      pn.pane.Markdown(f"_Com {MIN_GRUPOS_VIOLINO} grupos ou mais o padrão "
                                       "é violino: histogramas sobrepostos ficam ilegíveis._"),
                      width=CONTROL_WIDTH),
            pn.Column("## Features", pn.pane.Markdown(
                "Uma linha por feature recebida.\n\n"
                "- **kept**: entrou no PILOT;\n"
                "- **dropped_degenerate**: descartada antes do engine, pelo gerador do "
                "metadata (degenerate_report.csv);\n"
                "- **dropped_correlation**: nenhuma correlação significativa com o "
                "desempenho (SIFTED);\n"
                "- **dropped_redundancy**: outra feature do mesmo cluster ficou no lugar "
                "(SIFTED).", width=CONTROL_WIDTH - 20), width=CONTROL_WIDTH),
            pn.Column("## Data Explorer", self.w_ex_x, self.w_ex_y, self.w_ex_cor,
                      self.w_ex_query, self.ex_filtro, self.w_ex_usar, width=CONTROL_WIDTH),
        ]
        self.sidebar = pn.Column(
            self.w_dataset, self.w_reload, self.novo.card, self.info, self.tipos_card, self.sel_info,
            self.w_limpar, pn.pane.Markdown("### Exportar", margin=(0, 10)), self.w_exp_todas,
            self.w_exp_sel, self.w_exp_fp, self.exp_nota,
            pn.layout.Divider(), self.controls[0], width=CONTROL_WIDTH,
        )
        self._swap = len(self.sidebar.objects) - 1   # bloco que troca por aba
        self.tabs = pn.Tabs(
            (TABS[0], pn.Column(self.espaco_status, self.espaco_plot, sizing_mode="stretch_width")),
            (TABS[1], pn.Column(self.fp_status, self.fp_plot,
                                pn.pane.Markdown("### footprint_performance.csv"), self.fp_tabela,
                                sizing_mode="stretch_width")),
            (TABS[2], pn.Column(self.as_status, self.as_aviso, self.as_resumo, self.as_plot,
                                pn.pane.Markdown("### Desempenho dos SVMs (svm_table.csv)"),
                                self.as_tabela, self.as_nota_tabela,
                                pn.pane.Markdown("### Matrizes de confusão da validação cruzada "
                                                 "(positivo = bom)"),
                                self.as_nota_confusao, self.as_confusao,
                                sizing_mode="stretch_width")),
            (TABS[3], pn.Column(self.dist_status, self.dist_plots, sizing_mode="stretch_width")),
            (TABS[4], pn.Column(self.feat_resumo, self.feat_tabela,
                                pn.pane.Markdown("### Correlações do SIFTED (feature × algoritmo)"),
                                self.feat_heatmap, pn.pane.Markdown("### Silhueta por k (SIFTED)"),
                                self.feat_silhueta, sizing_mode="stretch_width")),
            (TABS[5], pn.Column(self.ex_status, self.ex_plot, self.ex_titulo_tabela,
                                self.ex_tabela, sizing_mode="stretch_width")),
            dynamic=True, sizing_mode="stretch_width",
        )

        # --- ligacoes
        self.estado.param.watch(self._on_dataset, "dataset")
        self.w_dataset.param.watch(self._on_widget_dataset, "value")
        self.estado.param.watch(self._on_estado, ["resultado", "selecao", "cor"])
        self.tabs.param.watch(self._on_tab, "active")
        self.w_reload.on_click(self._on_reload)
        self.w_limpar.on_click(lambda _: setattr(self.estado, "selecao", None))
        for w in (self.w_cor, self.w_ex_cor):
            w.param.watch(self._on_widget_cor, "value")
        for w in (self.w_ov_fp, self.w_ov_algo, self.w_ov_tipo, self.w_ov_cloister, self.w_ov_hard):
            w.param.watch(lambda _: self._refresh_espaco(), "value")
        for w in (self.w_fp_algo, self.w_fp_tipo):
            w.param.watch(lambda _: (self._refresh_footprints(), self._refresh_exportacao()), "value")
        for w in (self.w_dist_vars, self.w_dist_tipo):
            w.param.watch(lambda _: None if self._building else self._refresh_distribuicoes(), "value")
        self.w_dist_grupo.param.watch(self._on_dist_grupo, "value")
        for w in (self.w_as_cor, self.w_as_algo, self.w_as_cloister):
            w.param.watch(lambda _: None if self._building else self._refresh_selecao_algo(), "value")
        for w in (self.w_ex_x, self.w_ex_y, self.w_ex_query):
            w.param.watch(lambda _: self._refresh_explorer(), "value")
        self.w_ex_usar.on_click(self._on_usar_filtro)

        self.estado.dataset = self.datasets[0]

    # ------------------------------------------------------------------ dados
    def _listar_datasets(self):
        """Relista resultados/is e runs; devolve os grupos do seletor
        {origem: {pasta: chave}} (runs: mais recente primeiro)."""
        grupos = {}
        for origem, pasta in (("is", self.root), ("runs", self.runs)):
            nomes = list_available(pasta) if pasta.is_dir() else []
            if origem == "runs":
                nomes.sort(key=lambda n: (pasta / n / "run_info.json").stat().st_mtime, reverse=True)
            if nomes:
                grupos[ORIGENS[origem]] = {n: f"{origem}/{n}" for n in nomes}
        self.datasets = [c for g in grupos.values() for c in g.values()]
        self.estado.param.dataset.objects = self.datasets
        return grupos

    def _pasta(self, chave):
        origem, _, nome = chave.partition("/")
        return (self.runs if origem == "runs" else self.root) / nome

    @staticmethod
    def _nome(chave):
        return chave.partition("/")[2]

    def _carregar(self, chave, forcar=False):
        forcados = self._tipos_forcados.get(chave, {})
        k = (chave, tuple(sorted(forcados.items())))
        if forcar:
            self._cache = {c: v for c, v in self._cache.items() if c[0] != chave}
        if k not in self._cache:
            self._cache[k] = load_is_output(self._pasta(chave), annotation_types=forcados or None)
        return self._cache[k]

    def _on_dataset(self, event):
        self._trocar_dataset(event.new)

    def _on_widget_dataset(self, event):
        if event.new in self.datasets and event.new != self.estado.dataset:
            self.estado.dataset = event.new

    def _on_reload(self, _):
        self.w_dataset.groups = self._listar_datasets()
        if self.estado.dataset not in self.datasets:
            self.estado.dataset = self.datasets[0]
        else:
            self._trocar_dataset(self.estado.dataset, forcar=True)

    def _on_execucao_concluida(self, pasta):
        """Execucao do bloco Novo instance space terminou: relista e abre."""
        self.w_dataset.groups = self._listar_datasets()
        chave = f"runs/{Path(pasta).name}"
        if chave in self.datasets:
            self.estado.dataset = chave
            if pn.state.notifications is not None:
                pn.state.notifications.success(f"Novo instance space: {Path(pasta).name}", duration=6000)

    def _trocar_dataset(self, chave, forcar=False, manter_selecao=False):
        nome = self._nome(chave)
        if self.w_dataset.value != chave:
            self.w_dataset.value = chave
        r = self._carregar(chave, forcar)
        self._dados = r.instances.reset_index()
        self._catalogo = catalogo_de_cores(r)
        cor = self.estado.cor if self.estado.cor in self._catalogo else cor_padrao(r, self._catalogo)
        self._set_options(r)
        selecao = self.estado.selecao if manter_selecao else None
        self._selecao_do_plot = None
        # uma so rodada de watchers: resultado, selecao (zerada ao trocar de
        # dataset) e cor valida
        self.estado.param.update(resultado=r, selecao=selecao, cor=cor)
        self.cab.object = f'<span style="font-size:1.25em;color:white">— {nome}</span>'
        self.titulo_aba.titulo = f"{TITULO} - {nome}"
        self._refresh_info()
        self._montar_tipos(r)

    # ---------------------------------------------- tipos inferidos
    def _montar_tipos(self, r):
        """Um seletor por anotacao cujo tipo foi inferido (ou trocado aqui)."""
        meta = {v: k for k, v in r.annotation_renames.items()}
        cols = [c for c, o in r.annotation_origins.items() if o in ("inferido", "forcado")]
        linhas = [pn.pane.Markdown(
            "_Sem declaração em annotations.json: tipo adivinhado. A troca vale só "
            "nesta sessão._", width=CONTROL_WIDTH - 50)]
        for col in cols:
            atual = r.annotations[col]
            w = pn.widgets.RadioButtonGroup(
                name=col, options={"numérica": NUMERICA, "categórica": CATEGORICA,
                                   "identificador": IDENTIFICADOR},
                value=atual if atual in (CATEGORICA, IDENTIFICADOR) else NUMERICA,
                button_type="light")
            w.param.watch(partial(self._on_tipo, meta.get(col, col)), "value")
            linhas.append(pn.Column(pn.pane.Markdown(f"`{col}`", margin=(0, 10)), w))
        self.tipos_card.objects = linhas
        self.tipos_card.title = f"Tipos inferidos ({len(cols)})"
        self.tipos_card.visible = bool(cols)

    def _on_tipo(self, coluna, event):
        chave = self.estado.dataset
        self._tipos_forcados.setdefault(chave, {})[coluna] = event.new
        self._trocar_dataset(chave, manter_selecao=True)

    def _set_options(self, r):
        """Opcoes de todos os seletores para o dataset, sem disparar redesenhos."""
        self._building = True
        try:
            grupos = grupos_select(self._catalogo)
            for w in (self.w_cor, self.w_ex_cor):
                w.groups = grupos
            algos = [TODOS] + list(r.algos)
            for w in (self.w_ov_algo, self.w_fp_algo):
                w.options = algos
                if w.value not in algos:
                    w.value = TODOS
            self.w_as_algo.options = list(r.algos)
            if self.w_as_algo.value not in r.algos:
                self.w_as_algo.value = r.algos[0]
            numericas = self._numericas(r)
            dist = [c for c in numericas if c not in ("z_1", "z_2")]
            self.w_dist_vars.options = dist
            mantidas = [v for v in self.w_dist_vars.value if v in dist]
            self.w_dist_vars.value = mantidas or [self._variavel_padrao(r)]
            grupos = [SEM_GRUPO] + grupos_categoricos(r)
            self.w_dist_grupo.options = grupos
            if not self._grupo_escolhido or self.w_dist_grupo.value not in grupos:
                # padrao: a primeira anotacao categorica
                self.w_dist_grupo.value = grupos[1] if len(grupos) > 1 else SEM_GRUPO
            self.w_dist_tipo.value = self._tipo_dist_padrao()
            for w, padrao in ((self.w_ex_x, "z_1"), (self.w_ex_y, "z_2")):
                w.options = numericas
                if w.value not in numericas:
                    w.value = padrao
        finally:
            self._building = False

    def _valores_grupo(self, col):
        """Rotulos de grupo em texto (ausente = 'nenhum')."""
        return self._dados[col].map(lambda v: NENHUM if v is None or (isinstance(v, float) and np.isnan(v))
                                    else str(v)).astype(str)

    def _tipo_dist_padrao(self):
        col = self.w_dist_grupo.value
        if col == SEM_GRUPO or col not in self._dados.columns:
            return "histograma"
        return "violino" if self._valores_grupo(col).nunique() >= MIN_GRUPOS_VIOLINO else "histograma"

    def _on_dist_grupo(self, event):
        if self._building:
            return
        self._grupo_escolhido = True
        self._building = True
        try:
            self.w_dist_tipo.value = self._tipo_dist_padrao()
        finally:
            self._building = False
        self._refresh_distribuicoes()

    def _variavel_padrao(self, r):
        """Primeira anotacao numerica continua (valores nao inteiros: evita ids e
        contagens, como row_original); senao a primeira feature do PILOT."""
        for col, tipo in r.annotations.items():
            v = pd.to_numeric(self._dados[col], errors="coerce").dropna()
            if tipo == NUMERICA and len(v) and (np.mod(v, 1) != 0).any():
                return col
        return f"feature_{r.features[0]}"

    def _numericas(self, r):
        cols = ["z_1", "z_2"]
        cols += [c for c, t in r.annotations.items() if e_numerica(t)]
        cols += [f"feature_{f}" for f in r.features_all]
        cols += [f"algo_{a}" for a in r.algos] + ["NumGoodAlgos"]
        return [c for c in dict.fromkeys(cols) if c in self._dados.columns
                and pd.api.types.is_numeric_dtype(self._dados[c])]

    def _valores_cor(self, col, linhas=None):
        """Valores da variavel de cor: texto para categoricas (ausente = 'nenhum')."""
        s = self._dados[col] if linhas is None else self._dados.loc[linhas, col]
        if self._catalogo[col][2] == CATEGORICA:
            return s.map(lambda v: NENHUM if v is None or (isinstance(v, float) and np.isnan(v))
                         else str(v)).astype(str)
        return pd.to_numeric(s, errors="coerce")

    # ------------------------------------------------------------ selecao
    def _posicoes(self, selecao):
        if not selecao:
            return []
        return np.flatnonzero(self._dados["Row"].isin(selecao)).tolist()

    def _mascara(self):
        """Array booleano por instancia, ou None sem selecao."""
        sel = self.estado.selecao
        if sel is None:
            return None
        return self._dados["Row"].isin(sel).to_numpy()

    def _alphas(self):
        m = self._mascara()
        if m is None:
            return np.full(len(self._dados), ALPHA_NEUTRO)
        return np.where(m, ALPHA_SEL, ALPHA_FORA)

    def _texto_selecao(self, contexto):
        """Linha de status comum as abas (o texto e verificado pelos testes)."""
        sel, n = self.estado.selecao, len(self._dados)
        if sel is None:
            return f"**Sem seleção.** {contexto['sem']}"
        if not sel:
            return (f"**Seleção vazia**: nenhuma instância foi selecionada (0 de {n}). "
                    f"{contexto['vazia']}")
        return f"**{len(sel)} instâncias selecionadas** de {n}. {contexto['com']}"

    def _on_indice(self, geracao, index=None, **_):
        if geracao != self._geracao:
            return
        self._indice = list(index or [])
        self._agendar(ESPERA_SELECAO_MS)

    def _on_geometria(self, geracao, **_):
        if geracao == self._geracao:
            self._agendar(ESPERA_GEOMETRIA_MS)

    def _on_reset(self, geracao, resetting=False, **_):
        if geracao == self._geracao and resetting:
            self._selecao_do_plot = None
            self.estado.selecao = None

    def _agendar(self, ms):
        if self._doc is None:
            self._gravar_selecao()
            return
        if self._pendente is not None:
            try:
                self._doc.remove_timeout_callback(self._pendente)
            except ValueError:
                pass
        self._pendente = self._doc.add_timeout_callback(self._gravar_selecao, ms)

    def _gravar_selecao(self):
        self._pendente = None
        rotulos = frozenset(self._dados["Row"].iloc[self._indice])
        self._selecao_do_plot = rotulos
        self.estado.selecao = rotulos

    def _on_usar_filtro(self, _):
        if self._ex_filtradas is not None:
            self.estado.selecao = frozenset(self._ex_filtradas)

    # --------------------------------------------------------------- cor
    def _on_widget_cor(self, event):
        if not self._building and event.new in self._catalogo:
            self.estado.cor = event.new

    def _sync_cor(self):
        self._building = True
        try:
            for w in (self.w_cor, self.w_ex_cor):
                w.value = self.estado.cor
        finally:
            self._building = False
        self._refresh_r2()

    def _refresh_r2(self):
        r, col = self.estado.resultado, self.estado.cor
        tab = r.pilot_r2
        valor = None
        if col.startswith("feature_"):
            f = col[len("feature_"):]
            if f not in r.features:
                self.r2.object = "r² —"
                self.r2_aviso.object = "_Fora do PILOT (descartada pelo SIFTED): r² não se aplica._"
                return
            linha = tab[(tab["kind"] == "feature") & (tab["variable"] == f)]
        elif col.startswith("algo_") and col[len("algo_"):] in r.algos:
            linha = tab[(tab["kind"] == "algorithm") & (tab["variable"] == col[len("algo_"):])]
        else:
            self.r2.object, self.r2_aviso.object = "", ""
            return
        if len(linha):
            valor = float(linha["r2"].iloc[0])
        if valor is None:
            self.r2.object, self.r2_aviso.object = "r² ?", ""
            return
        self.r2.object = f"r² **{valor:.2f}**".replace(".", ",")
        limite = f"{R2_BAIXO:.1f}".replace(".", ",")
        self.r2_aviso.object = (
            f"⚠ _r² do PILOT baixo (< {limite}): o plano explica pouco desta variável; "
            "o padrão de cor pode não aparecer na projeção._" if valor < R2_BAIXO else "")

    # ------------------------------------------------------------ dispatch
    def _on_estado(self, *events):
        nomes = {e.name for e in events}
        if self._dados is None:
            return
        if nomes & {"resultado", "cor"}:
            self._sync_cor()
        self._refresh_sel_info()
        externa = "selecao" in nomes and self.estado.selecao != self._selecao_do_plot
        if nomes & {"resultado", "cor"} or externa:
            self._refresh_espaco()
        else:
            self.espaco_status.object = self._status_espaco()
        if nomes & {"resultado", "selecao"}:
            self._refresh_footprints()
            self._refresh_selecao_algo()
            self._refresh_distribuicoes()
        if "resultado" in nomes:
            self._refresh_features()
        self._refresh_explorer()
        self._refresh_exportacao()

    def _on_tab(self, event):
        self.sidebar[self._swap] = self.controls[event.new]

    def _refresh_info(self):
        r = self.estado.resultado
        rob = r.run_info.get("trace_robustez", {})
        origem = self.estado.dataset.partition("/")[0]
        linhas = [
            f"**{r.name}** _({'runs' if origem == 'runs' else 'resultados/is'})_",
            f"Instâncias: **{r.n}**",
            f"Features no PILOT: **{r.n_features_used}** de {len(r.features_all)}",
            f"Algoritmos: **{len(r.algos)}**",
        ]
        if r.run_info.get("regra_bom"):
            linhas.append(f"Regra: `{r.run_info['regra_bom']}`")
        if rob.get("jitter_aplicado"):
            linhas.append(f"_TRACE com jitter em {rob['pontos_perturbados']} instâncias "
                          "(coordinates_trace.csv)_")
        self.info.object = "  \n".join(linhas)

    def _refresh_sel_info(self):
        sel = self.estado.selecao
        if sel is None:
            texto = "Seleção: **nenhuma**"
        elif not sel:
            texto = "Seleção: **vazia** (0 instâncias)"
        else:
            texto = f"Seleção: **{len(sel)}** instâncias"
        self.sel_info.object = texto

    # ----------------------------------------------------- aba 0: espaco
    def _status_espaco(self):
        return self._texto_selecao({
            "sem": "O lasso está ativo: desenhe no gráfico para selecionar.",
            "vazia": "O lasso não pegou nenhum ponto.",
            "com": "Destacadas no gráfico e usadas nas outras abas.",
        })

    def _sobreposicoes(self):
        r = self.estado.resultado
        camadas, notas = [], []
        if self.w_ov_hard.value:
            fp = r.footprint_hard
            if fp.poligonos:
                camadas.append(poligonos_hv(fp, "#343a40", "hard (não beta-fáceis)", 0.15))
            else:
                notas.append("- footprint hard vazia")
        if self.w_ov_fp.value:
            tipo = self.w_ov_tipo.value
            algos = r.algos if self.w_ov_algo.value == TODOS else [self.w_ov_algo.value]
            for a in algos:
                fp = r.footprints[(a, tipo)]
                cor = PALETA_ALGOS[r.algos.index(a) % len(PALETA_ALGOS)]
                if fp.status == VAZIA:
                    notas.append(f"- **{a} / {tipo}**: vazia")
                    continue
                if fp.status == SUSPEITA:
                    notas.append(f"- **{a} / {tipo}**: suspeita (pureza {fp.pureza:.2f}), tracejada")
                camadas.append(poligonos_hv(fp, cor, f"{a} {tipo}", 0.22, fp.status == SUSPEITA))
        if self.w_ov_cloister.value:
            camadas += cloister_hv(r)
        self.ov_nota.object = "\n".join(notas)
        return camadas

    def _refresh_espaco(self):
        """Reconstroi o scatter da aba 0 (cor, dataset, sobreposicoes ou selecao
        vinda de fora). A selecao atual volta como `selected`, e o Selection1D
        novo comeca com esses indices."""
        col = self.estado.cor
        rotulo, _, tipo = self._catalogo[col]
        dados = self._dados[["Row", "z_1", "z_2"]].copy()
        dados["cor_valor"] = self._valores_cor(col).to_numpy()
        posicoes = self._posicoes(self.estado.selecao)
        self._geracao += 1
        geracao = self._geracao
        self._indice = posicoes
        hover = HoverTool(tooltips=[("Row", "@Row"), (rotulo, "@cor_valor")])
        estilo = dict(
            color="cor_valor", size=6, alpha=0.85, nonselection_alpha=0.12, line_color=None,
            tools=["lasso_select", "box_select", hover], active_tools=["lasso_select"],
            responsive=True, min_height=620, show_grid=True, xlabel="z_1", ylabel="z_2",
            title=f"Espaço de instâncias (PILOT) — cor: {rotulo}",
        )
        estilo.update(estilo_de_cor(dados["cor_valor"], tipo))
        if posicoes:
            estilo["selected"] = posicoes
        pontos = hv.Points(dados, ["z_1", "z_2"], [hv.Dimension("cor_valor", label=rotulo), "Row"]
                           ).opts(**estilo)
        Selection1D(source=pontos, index=posicoes).add_subscriber(partial(self._on_indice, geracao))
        for stream in (Lasso(source=pontos), BoundsXY(source=pontos)):
            stream.add_subscriber(partial(self._on_geometria, geracao))
        PlotReset(source=pontos).add_subscriber(partial(self._on_reset, geracao))
        camadas = self._sobreposicoes() + [pontos]
        self.espaco_plot.object = reduce(lambda a, b: a * b, camadas).opts(
            legend_position="right", legend_opts={"click_policy": "hide"},
            responsive=True, min_height=620)
        self.espaco_status.object = self._status_espaco()

    # ------------------------------------------------ aba 1: footprints
    def _refresh_footprints(self):
        r = self.estado.resultado
        algo, tipo = self.w_fp_algo.value, self.w_fp_tipo.value
        dados = self._dados[["Row", "z_1", "z_2"]].assign(opacidade=self._alphas())
        hover = HoverTool(tooltips=[("Row", "@Row")])
        camadas = [hv.Points(dados, ["z_1", "z_2"], ["Row", "opacidade"]).opts(
            color=COR_BASE, alpha="opacidade", size=6, line_color=None, tools=[hover],
            show_legend=False)]
        avisos = []
        algos = r.algos if algo == TODOS else [algo]
        for a in algos:
            fp = r.footprints[(a, tipo)]
            cor = PALETA_ALGOS[r.algos.index(a) % len(PALETA_ALGOS)]
            if fp.status == VAZIA:
                avisos.append(f"- **{a} / {tipo}**: footprint vazia, nada a desenhar.")
                continue
            if fp.status == SUSPEITA:
                avisos.append(f"- **{a} / {tipo}**: suspeita, pureza {fp.pureza:.3f} < "
                              f"trace.purity {r.pi}; desenhada tracejada.")
            camadas.append(poligonos_hv(fp, cor, a, 0.25, fp.status == SUSPEITA))
        self.fp_aviso.object = ("Footprints suspeitas/vazias:\n" + "\n".join(avisos) if avisos
                                else "Nenhuma footprint suspeita ou vazia para esta escolha.")
        self.fp_plot.object = reduce(lambda a, b: a * b, camadas).opts(
            responsive=True, min_height=460, show_grid=True, legend_position="right",
            legend_opts={"click_policy": "hide"}, title=f"Footprints {tipo} — {algo}")
        tabela = r.footprint_performance.copy().round(4)
        tabela["status"] = [r.footprints[(a, tipo)].status if (a, tipo) in r.footprints else "?"
                            for a in tabela.index]
        self.fp_tabela.value = tabela
        self.fp_status.object = self._texto_selecao({
            "sem": "Use o lasso na aba Instance Space para destacar um subconjunto no mapa.",
            "vazia": "Nenhum ponto destacado no mapa.",
            "com": "Destacadas no mapa; as demais aparecem apagadas.",
        })

    # ------------------------------------- aba 2: algorithm selection
    def _dados_selecao_algo(self):
        """Uma linha por instancia: recomendado (selection0, 'nenhum'), melhor
        observado, concordancia, pr0_sub do recomendado e se ele e bom."""
        r = self.estado.resultado
        rows = self._dados["Row"]
        rec = r.pythia_selection["selection0"].reindex(rows.to_numpy()).to_numpy(dtype=object)
        melhor = self._dados["best_algo"].to_numpy(dtype=object)
        tem = np.array([v is not None and not (isinstance(v, float) and np.isnan(v)) for v in rec])
        rec_txt = np.where(tem, rec, NENHUM).astype(str)
        concorda = np.where(~tem, SEM_REC, np.where(rec == melhor, IGUAL, DIFERENTE))
        pr0 = r.pythia_proba.reindex(rows.to_numpy())
        pr0_rec = np.full(len(rows), np.nan)
        rec_bom = np.zeros(len(rows), dtype=bool)
        for i, a in enumerate(r.algos):
            m = rec_txt == a
            pr0_rec[m] = pr0[a].to_numpy(dtype=float)[m]
            rec_bom[m] = self._dados[f"algo_{a}_bin"].to_numpy(dtype=bool)[m]
        return pd.DataFrame({
            "Row": rows.to_numpy(), "z_1": self._dados["z_1"].to_numpy(),
            "z_2": self._dados["z_2"].to_numpy(), "recomendado": rec_txt,
            "melhor": pd.Series(melhor).fillna(NENHUM).astype(str).to_numpy(),
            "concordancia": concorda, "pr0_rec": pr0_rec,
            "rec_bom": np.where(tem, np.where(rec_bom, "sim", "não"), "—"),
        })

    def _refresh_selecao_algo(self):
        r = self.estado.resultado
        if r is None or self._dados is None:
            return
        d = self._dados_selecao_algo()
        n = len(d)
        d["opacidade"] = self._alphas()
        modo = self.w_as_cor.value
        self.w_as_algo.visible = modo == AS_PR0
        contagem = d["recomendado"].value_counts()

        # aviso de seletor quase trivial (so algoritmos, nao 'nenhum')
        algos_rec = contagem.drop(NENHUM, errors="ignore")
        avisos = []
        if len(algos_rec) and algos_rec.iloc[0] / n > SELETOR_TRIVIAL:
            a, k = algos_rec.index[0], int(algos_rec.iloc[0])
            avisos.append(pn.pane.Alert(
                f"⚠ **Seletor quase trivial:** o PYTHIA recomenda **{a}** para {k} de {n} "
                f"instâncias ({_pct(k / n)}, acima de {_pct(SELETOR_TRIVIAL)}). Recomendar "
                f"sempre {a} daria quase o mesmo resultado: o mapa diz pouco sobre regiões "
                "em que cada algoritmo é melhor.", alert_type="warning",
                css_classes=["as-trivial"], sizing_mode="stretch_width"))
        self.as_aviso.objects = avisos

        # resumo da concordancia
        c = d["concordancia"].value_counts()
        dif = d["concordancia"] == DIFERENTE
        dif_bom = int((dif & (d["rec_bom"] == "sim")).sum())
        self.as_resumo.object = (
            f"Recomendado = melhor observado em **{int(c.get(IGUAL, 0))}** de {n} instâncias; "
            f"diferente em **{int(c.get(DIFERENTE, 0))}** (em {dif_bom} delas o recomendado "
            f"também é bom); sem recomendação em **{int(c.get(SEM_REC, 0))}**.  \n"
            "_Probabilidades nesta aba: **pr0_sub**, P(ruim) fora da amostra (validação cruzada "
            "do PYTHIA)._")

        # mapa
        cores_algo = {a: PALETA_ALGOS[i % len(PALETA_ALGOS)] for i, a in enumerate(r.algos)}
        hover = HoverTool(tooltips=[
            ("Row", "@Row"), ("recomendado", "@recomendado"), ("melhor observado", "@melhor"),
            ("recomendado é bom", "@rec_bom"), ("pr0_sub do recomendado", "@pr0_rec{0.000}")])
        vdims = ["Row", "recomendado", "melhor", "rec_bom", "pr0_rec", "opacidade"]
        estilo = dict(alpha="opacidade", size=6, line_color=None, tools=[hover],
                      responsive=True, min_height=520, show_grid=True)
        if modo == AS_PR0:
            a = self.w_as_algo.value
            d["cor_valor"] = r.pythia_proba[a].reindex(d["Row"].to_numpy()).to_numpy(dtype=float)
            rotulo = f"pr0_sub {a}: P(ruim) fora da amostra"
            estilo.update(color="cor_valor", cmap="RdYlGn_r", clim=(0, 1), colorbar=True,
                          show_legend=False, colorbar_opts={"title": "pr0_sub"})
            hover.tooltips = hover.tooltips + [(f"pr0_sub {a}", "@cor_valor{0.000}")]
            titulo = f"pr0_sub de {a} (P(ruim) fora da amostra); verde = provável bom"
        else:
            if modo == AS_CONCORDANCIA:
                col, cores = "concordancia", CORES_CONCORDANCIA
                titulo = "Recomendado (selection0) × melhor observado"
            else:
                col, cores = "recomendado", {**cores_algo, NENHUM: COR_NENHUM}
                titulo = "Algoritmo recomendado pelo PYTHIA (selection0)"
            # legenda e ordem de desenho: da categoria mais frequente para a
            # mais rara (as raras ficam por cima e continuam visiveis)
            cont = d[col].value_counts()
            ordem = list(cont.index)
            rot = {k: f"{k} ({int(cont[k])})" for k in ordem}
            d["cor_valor"] = d[col].map(rot)
            d = d.iloc[np.argsort(d[col].map({k: i for i, k in enumerate(ordem)}).to_numpy(),
                                  kind="stable")]
            cmap = {rot[k]: cores.get(k, COR_NENHUM) for k in ordem}
            rotulo = "cor"
            estilo.update(color="cor_valor", cmap=cmap, show_legend=True)
        pontos = hv.Points(d, ["z_1", "z_2"], [hv.Dimension("cor_valor", label=rotulo), *vdims]
                           ).opts(**estilo)
        camadas = [pontos]
        if self.w_as_cloister.value:
            camadas += cloister_hv(r)
        self.as_plot.object = reduce(lambda a, b: a * b, camadas).opts(
            responsive=True, min_height=520, legend_position="right", title=titulo,
            legend_opts={"click_policy": "hide"}, **SEM_ROLAGEM)
        self.as_status.object = self._texto_selecao({
            "sem": "Use o lasso na aba Instance Space para destacar um subconjunto no mapa.",
            "vazia": "Nenhum ponto destacado no mapa.",
            "com": "Destacadas no mapa; as demais aparecem apagadas. " + self._resumo_sel_algo(d),
        })
        self._refresh_svm_table(r, contagem)
        self._refresh_confusao(r)

    def _resumo_sel_algo(self, d):
        m = self._mascara()
        if m is None or not m.any():
            return ""
        c = d.loc[m, "concordancia"].value_counts()
        top = d.loc[m, "recomendado"].value_counts()
        return (f"Na seleção: mais recomendado **{top.index[0]}** ({int(top.iloc[0])}); igual ao "
                f"melhor observado {int(c.get(IGUAL, 0))}, diferente {int(c.get(DIFERENTE, 0))}, "
                f"sem recomendação {int(c.get(SEM_REC, 0))}.")

    def _refresh_svm_table(self, r, contagem):
        tab = r.svm_table
        linhas = []
        for nome in tab.index:
            linha = {"algoritmo": str(nome)}
            for col, curto in COLUNAS_SVM:
                v = tab.loc[nome, col] if col in tab.columns else np.nan
                linha[curto] = "—" if pd.isna(v) else (f"{v:.1f}" if curto.endswith("%") else f"{v:.3f}")
            if nome in r.algos:
                linha["recomendado em"] = int(contagem.get(nome, 0))
            elif nome == "Selector":
                linha["recomendado em"] = int(contagem.drop(NENHUM, errors="ignore").sum())
            else:
                linha["recomendado em"] = "—"
            linhas.append(linha)
        df = pd.DataFrame(linhas)
        ordem = ["algoritmo", "acurácia CV %", "precisão CV %", "recall CV %",
                 "recomendado em", "P(bom)", "desemp. médio", "desemp. prev. boas"]
        self.as_tabela.value = df[ordem].astype(str)
        self.as_tabela.height = 40 + 31 * len(df)
        self.as_nota_tabela.object = (
            "- **acurácia, precisão e recall**: validação cruzada de cada SVM (positivo = bom).\n"
            "- **recomendado em**: instâncias em que o algoritmo é o selection0.\n"
            "- **P(bom)**: fração de instâncias em que o algoritmo é bom (Selector: com "
            "selection1, que troca *nenhum* pelo algoritmo de maior P(bom)).\n"
            "- **desemp. prev. boas**: média de `algo_*` onde o SVM prevê bom (Selector: a do "
            "recomendado).\n"
            "- **Oracle**: sempre o melhor observado de cada instância.\n"
            "- **Selector**: o recomendado. Precisão = fração das instâncias recomendadas em "
            "que o recomendado é bom. O recall segue a definição do PYTHIA/MATLAB, que conta "
            "como perda toda instância com algum algoritmo bom não recomendado; por isso fica "
            "perto de 50% quando há vários algoritmos bons por instância.")

    def _refresh_confusao(self, r):
        conf = r.pythia_confusion
        paineis = []
        for a in r.algos:
            if a not in conf.index:
                continue
            tn, fp, fn, tp = (int(conf.loc[a, k]) for k in ("tn", "fp", "fn", "tp"))
            celulas = []
            for obs, prev, k, nome in (("bom", "bom", tp, "VP"), ("bom", "ruim", fn, "FN"),
                                       ("ruim", "bom", fp, "FP"), ("ruim", "ruim", tn, "VN")):
                total = tp + fn if obs == "bom" else fp + tn
                frac = k / total if total else 0.0
                celulas.append({"previsto": prev, "observado": obs, "fracao": frac, "n": k,
                                "texto": f"{nome} {k}\n{_pct(frac)}" if total else f"{nome} 0",
                                "cor_texto": "white" if frac >= 0.6 else "black"})
            c = pd.DataFrame(celulas)
            acc = (tp + tn) / max(tp + tn + fp + fn, 1)
            mapa = hv.HeatMap(c, ["previsto", "observado"], ["fracao", "n"]).opts(
                cmap="Blues", clim=(0, 1), colorbar=False, tools=["hover"], width=230,
                height=190, xlabel="previsto (CV)", ylabel="observado", toolbar=None,
                invert_yaxis=True, default_tools=[])
            rotulos = hv.Labels(c, ["previsto", "observado"], ["texto", "cor_texto"]).opts(
                text_font_size="11pt", text_color="cor_texto")
            paineis.append(pn.Column(
                pn.pane.Markdown(f"**{a}** · acurácia {_pct(acc)}", width=230, margin=(0, 10)),
                pn.pane.HoloViews(mapa * rotulos, width=230, height=190),
                css_classes=["as-confusao"], margin=(5, 5)))
        self.as_confusao.objects = paineis or [pn.pane.Markdown("Sem matrizes de confusão.")]
        self.as_nota_confusao.object = (
            "Linhas = desempenho observado, colunas = previsão do SVM na validação cruzada. "
            "Cor e porcentagem: fração dentro da linha (a diagonal escura indica acerto nas duas "
            "classes). VP/FN/FP/VN: verdadeiro positivo, falso negativo, falso positivo, "
            "verdadeiro negativo.")

    # --------------------------------------------- aba 3: distribuicoes
    def _series_dist(self, vals, grupo_col, mascara):
        """[(rotulo, valores finitos, cor, parte)] por grupo; parte e True
        (selecionadas), False (nao selecionadas) ou None (sem divisao)."""
        n = len(vals)
        if grupo_col is None:
            grupos = [("todas", np.ones(n, dtype=bool), COR_TODAS)]
        else:
            g = self._valores_grupo(grupo_col).to_numpy()
            grupos = [(c, g == c, PALETA_GRUPOS[i % len(PALETA_GRUPOS)])
                      for i, c in enumerate(sorted(set(g)))]
        dividir = mascara is not None and bool(mascara.any())
        series = []
        for nome, idx, cor in grupos:
            partes = ((True, "selecionadas"), (False, "não selecionadas")) if dividir else ((None, None),)
            for parte, texto in partes:
                m = idx if parte is None else idx & (mascara if parte else ~mascara)
                v = vals[m]
                prefixo = nome if texto is None else (texto if grupo_col is None else f"{nome} · {texto}")
                if grupo_col is None and parte is not None:
                    cor = COR_SEL if parte else COR_TODAS
                series.append((f"{prefixo} (n={int(m.sum())})", v[np.isfinite(v)], cor, parte))
        return grupos, series

    def _grafico_dist(self, var, grupo_col, tipo, mascara):
        vals = self._dados[var].to_numpy(dtype=float)
        titulo = f"{var}  (Todas n={len(vals)}"
        titulo += f", Selecionadas n={int(mascara.sum())})" if mascara is not None else ")"
        if grupo_col is not None:
            titulo += f" — por {grupo_col}"
        grupos, series = self._series_dist(vals, grupo_col, mascara)
        comum = dict(title=titulo, responsive=True, show_grid=True, **SEM_ROLAGEM)
        fin = vals[np.isfinite(vals)]
        if fin.size == 0:
            return hv.Curve([]).opts(title=f"{var}: sem valores finitos")
        if tipo == "violino":
            dividir = mascara is not None and bool(mascara.any())
            linhas = []
            for nome, idx, _ in grupos:
                rotulo = f"{nome} (n={int(idx.sum())}"
                rotulo += f"; {int((idx & mascara).sum())} sel.)" if dividir else ")"
                for parte in ((True, False) if dividir else (None,)):
                    m = idx if parte is None else idx & (mascara if parte else ~mascara)
                    v = vals[m]
                    v = v[np.isfinite(v)]
                    texto = "todas" if parte is None else ("selecionadas" if parte else "não selecionadas")
                    linhas.append(pd.DataFrame({"grupo": rotulo, "parte": texto, "valor": v}))
            dfl = pd.concat(linhas, ignore_index=True)
            if dividir:
                return hv.Violin(dfl, ["grupo", "parte"], "valor").opts(
                    split="parte", violin_fill_color="parte",
                    cmap={"selecionadas": COR_SEL, "não selecionadas": COR_NAO_SEL},
                    height=300, ylabel=var, xlabel="", **comum)
            cores = {r: (COR_TODAS if grupo_col is None else PALETA_GRUPOS[i % len(PALETA_GRUPOS)])
                     for i, r in enumerate(dict.fromkeys(dfl["grupo"]))}
            return hv.Violin(dfl, ["grupo"], "valor").opts(
                violin_fill_color="grupo", cmap=cores, height=300, ylabel=var, xlabel="", **comum)
        camadas = []
        if tipo == "histograma":
            edges = np.histogram_bin_edges(fin, bins=25)
            for rotulo, v, cor, parte in series:
                if not v.size:
                    continue
                dens, _ = np.histogram(v, bins=edges, density=True)
                camadas.append(hv.Histogram((edges, dens), label=rotulo).opts(
                    fill_color=cor, line_color=cor, line_alpha=0.6,
                    fill_alpha={True: 0.6, False: 0.15, None: 0.4}[parte],
                    line_dash="dashed" if parte is False else "solid"))
            ylabel = "densidade"
        else:  # densidade (KDE)
            for rotulo, v, cor, parte in series:
                if v.size < 2:
                    continue
                camadas.append(hv.Distribution(v, label=rotulo).opts(
                    fill_color=cor, line_color=cor, line_width=2,
                    fill_alpha={True: 0.4, False: 0.05, None: 0.25}[parte],
                    line_dash="dashed" if parte is False else "solid"))
            ylabel = "densidade (KDE)"
        if not camadas:
            return hv.Curve([]).opts(title=f"{var}: nenhum grupo com valores suficientes")
        return hv.Overlay(camadas).opts(height=260, xlabel=var, ylabel=ylabel,
                                        legend_position="right", **comum)

    def _refresh_distribuicoes(self):
        variaveis = list(self.w_dist_vars.value)
        mascara = self._mascara()
        grupo = self.w_dist_grupo.value
        grupo_col = None if grupo == SEM_GRUPO or grupo not in self._dados.columns else grupo
        tipo = self.w_dist_tipo.value
        cores = (" Nos violinos, vermelho = selecionadas e cinza = não selecionadas."
                 if tipo == "violino" else " Linha tracejada = não selecionadas.")
        self.dist_status.object = self._texto_selecao({
            "sem": "Só os grupos aparecem. Use o lasso na aba Instance Space para comparar.",
            "vazia": "Só os grupos aparecem (Selecionadas n=0).",
            "com": "Cada grupo aparece dividido em selecionadas e não selecionadas." + cores,
        })
        if not variaveis:
            self.dist_plots.objects = [pn.pane.Markdown("Escolha ao menos uma variável na barra lateral.")]
            return
        if len(variaveis) > MAX_DIST_VARS:
            self.dist_plots.objects = [pn.pane.Markdown(
                f"**{len(variaveis)} variáveis escolhidas.** O limite é {MAX_DIST_VARS} por vez.")]
            return
        self.dist_plots.objects = [
            pn.pane.HoloViews(self._grafico_dist(v, grupo_col, tipo, mascara),
                              sizing_mode="stretch_width")
            for v in variaveis
        ]

    # ---------------------------------------------------- aba 4: features
    def _refresh_features(self):
        r = self.estado.resultado
        tabela = r.features_table()
        n = tabela["status"].value_counts()
        partes = [f"**{len(tabela)} features recebidas**: {n.get('kept', 0)} no PILOT"]
        for status, texto in (("dropped_degenerate", "degeneradas"),
                              ("dropped_correlation", "sem correlação"),
                              ("dropped_redundancy", "redundantes")):
            if n.get(status, 0):
                partes.append(f"{n[status]} {texto}")
        resumo = ", ".join(partes) + "."
        if r.degenerate_report is None:
            resumo += ("  \n_Sem degenerate_report.csv: a tabela mostra só o que o SIFTED fez "
                       "com as features do metadata._")
        self.feat_resumo.object = resumo
        exibir = tabela.copy()
        exibir["substituida_por"] = exibir["substituida_por"].fillna("")
        exibir["algoritmo_rho"] = exibir["algoritmo_rho"].fillna("")
        for col in ("max_abs_rho", "r2_pilot"):
            exibir[col] = exibir[col].round(3)
        exibir["pval"] = exibir["pval"].map(lambda p: "" if pd.isna(p) else f"{p:.2g}")
        self.feat_tabela.value = exibir.rename(columns=COLUNAS_FEATURES)
        self.feat_tabela.height = min(40 + 31 * len(exibir), 640)

        c = r.sifted_correlations
        if c.empty:
            self.feat_heatmap.object = hv.Curve([]).opts(title="SIFTED sem correlações calculadas")
        else:
            mantidas = set(r.features)
            ordem = [f for f in tabela["feature"] if f in set(c["feature"])]
            c = c.assign(
                feature=c["feature"].map(lambda f: f"{f} ✓" if f in mantidas else f),
                texto=c["rho"].map(lambda v: f"{v:.2f}"),
                cor_texto=np.where(c["rho"].abs() >= 0.6, "white", "black"))
            ordem = [f"{f} ✓" if f in mantidas else f for f in ordem]
            c = c.set_index("feature").loc[ordem].reset_index()
            mapa = hv.HeatMap(c, ["algorithm", "feature"], ["rho", "pval"]).opts(
                cmap="RdBu_r", clim=(-1, 1), colorbar=True, tools=["hover"], responsive=True,
                height=90 + 26 * len(ordem), xlabel="algoritmo", ylabel="feature (✓ = no PILOT)",
                invert_yaxis=True, title="rho de Pearson entre a feature processada e o desempenho",
                **SEM_ROLAGEM)
            rotulos = hv.Labels(c, ["algorithm", "feature"], ["texto", "cor_texto"]).opts(
                text_font_size="8pt", text_color="cor_texto")
            self.feat_heatmap.object = mapa * rotulos

        sil = r.sifted_silhouette
        if sil.empty:
            self.feat_silhueta.objects = [pn.pane.Markdown(
                "_O SIFTED não clusterizou (poucas features depois do filtro de correlação): "
                "silhueta não calculada._")]
            return
        usado = sil.loc[sil["used"], "k"].tolist()
        melhor = sil.loc[sil["best"], "k"].tolist()
        camadas = [hv.Curve(sil, "k", "silhouette").opts(color=COR_BASE),
                   hv.Scatter(sil, "k", "silhouette").opts(color=COR_BASE, size=7)]
        partes = []
        if usado:
            camadas.append(hv.VLine(usado[0]).opts(color=COR_SEL, line_width=2))
            partes.append(f"k usado = {usado[0]} (vermelho)")
        if melhor:
            camadas.append(hv.VLine(melhor[0]).opts(color="#2a9d8f", line_dash="dashed", line_width=2))
            partes.append(f"maior silhueta: k = {melhor[0]} (tracejado)")
        self.feat_silhueta.objects = [pn.pane.HoloViews(
            reduce(lambda a, b: a * b, camadas).opts(
                responsive=True, height=260, show_grid=True, xlabel="k (clusters)",
                ylabel="silhueta", title="; ".join(partes), **SEM_ROLAGEM),
            sizing_mode="stretch_width")]

    # ------------------------------------------------------ exportacao
    def _tabela_exportacao(self, rotulos=None):
        """Rotulo (instances), source, anotacoes, todas as features, algo_*, z e
        derivadas; so as linhas de `rotulos` se dado."""
        r = self.estado.resultado
        cols = (["Row"] + ([r.source_column] if r.source_column else []) + list(r.annotations)
                + [f"feature_{f}" for f in r.features_all] + [f"algo_{a}" for a in r.algos]
                + EXPORT_DERIVADAS)
        dados = self._dados if rotulos is None else self._dados[self._dados["Row"].isin(rotulos)]
        return dados[cols].rename(columns={"Row": "instances"})

    @staticmethod
    def _csv(df):
        return io.BytesIO(df.to_csv(index=False).encode("utf-8"))

    def _csv_todas(self):
        return self._csv(self._tabela_exportacao())

    def _csv_selecao(self):
        return self._csv(self._tabela_exportacao(self.estado.selecao or frozenset()))

    def _footprint_ativa(self):
        """Footprint escolhida na aba Footprint Performance, ou None ('todos')."""
        algo, tipo = self.w_fp_algo.value, self.w_fp_tipo.value
        if algo == TODOS or self.estado.resultado is None:
            return None
        return self.estado.resultado.footprints.get((algo, tipo))

    def _csv_footprint(self):
        fp = self._footprint_ativa()
        rotulos = [] if fp is None else self.estado.resultado.instancias_na_footprint(fp)
        return self._csv(pd.DataFrame({"instances": rotulos}))

    def _refresh_exportacao(self):
        nome, sel = self._nome(self.estado.dataset), self.estado.selecao
        self.w_exp_todas.filename = f"{nome}_instancias.csv"
        self.w_exp_sel.filename = f"{nome}_selecao.csv"
        self.w_exp_sel.disabled = sel is None
        self.w_exp_sel.label = "Exportar seleção" if sel is None else f"Exportar seleção ({len(sel)})"
        fp = self._footprint_ativa()
        algo, tipo = self.w_fp_algo.value, self.w_fp_tipo.value
        self.w_exp_fp.disabled = fp is None or fp.status == VAZIA
        self.w_exp_fp.filename = f"{nome}_footprint_{algo}_{tipo}.csv"
        self.w_exp_fp.label = ("Exportar rótulos da footprint" if fp is None
                               else f"Exportar rótulos da footprint {algo}/{tipo}")
        self.exp_nota.object = (
            "_Footprint ativa: a da aba Footprint Performance; escolha um algoritmo lá._"
            if fp is None else (f"_Footprint {algo}/{tipo} vazia._" if fp.status == VAZIA else ""))

    # --------------------------------------------- aba 5: data explorer
    def _refresh_explorer(self):
        if self._dados is None:
            return
        dados = self._dados
        x, y, col = self.w_ex_x.value, self.w_ex_y.value, self.estado.cor
        if x is None or y is None or col not in self._catalogo:
            return
        rotulo, _, tipo = self._catalogo[col]
        q = (self.w_ex_query.value or "").strip()
        erro = None
        if q:
            try:
                filtrado = dados.query(q)
            except Exception as exc:  # noqa: BLE001 -- entrada do usuario
                erro, filtrado = f"{type(exc).__name__}: {exc}", dados
        else:
            filtrado = dados
        if erro:
            self.ex_filtro.object = f"**Query inválida — filtro ignorado.**  \n`{erro}`"
            self._ex_filtradas = None
        else:
            self.ex_filtro.object = f"**{len(filtrado)} de {len(dados)} linhas** passam no filtro."
            self._ex_filtradas = list(filtrado["Row"]) if q else None
        self.w_ex_usar.disabled = self._ex_filtradas is None

        linhas = filtrado.index
        mascara = self._mascara()
        alpha = self._alphas()[linhas]
        plot = pd.DataFrame({
            "eixo_x": filtrado[x].to_numpy(dtype=float), "eixo_y": filtrado[y].to_numpy(dtype=float),
            "cor_valor": self._valores_cor(col, linhas).to_numpy(), "Row": filtrado["Row"].to_numpy(),
            "opacidade": alpha,
        })
        hover = HoverTool(tooltips=[("Row", "@Row"), (x, "@eixo_x"), (y, "@eixo_y"), (rotulo, "@cor_valor")])
        estilo = dict(color="cor_valor", alpha="opacidade", size=6, line_color=None, tools=[hover],
                      responsive=True, min_height=460, show_grid=True, legend_position="right",
                      title=f"{x} x {y} — cor: {rotulo}")
        estilo.update(estilo_de_cor(plot["cor_valor"], tipo))
        self.ex_plot.object = hv.Points(
            plot, [hv.Dimension("eixo_x", label=x), hv.Dimension("eixo_y", label=y)],
            [hv.Dimension("cor_valor", label=rotulo), "Row", "opacidade"]).opts(**estilo)

        ids = [c for c, t in self.estado.resultado.annotations.items() if t == IDENTIFICADOR]
        cols = list(dict.fromkeys(["Row", *ids, x, y, col, "best_algo", "best_algo_svm"]))
        tabela = filtrado[cols].copy()
        tabela.insert(1, "selecionada", False if mascara is None else mascara[linhas])
        self.ex_tabela.value = tabela.round(4)
        self.ex_titulo_tabela.object = f"### Linhas filtradas ({len(filtrado)})"
        self.ex_status.object = self._texto_selecao({
            "sem": "Todos os pontos com a mesma opacidade.",
            "vazia": "Todos os pontos aparecem apagados.",
            "com": "Destacadas no gráfico e marcadas na coluna *selecionada*.",
        })

    # --------------------------------------------------------------- template
    def render(self):
        self._doc = pn.state.curdoc
        return pn.template.FastListTemplate(
            site="", title=TITULO, header=[self.cab, self.titulo_aba],
            sidebar=[self.sidebar], main=[self.tabs], sidebar_width=CONTROL_WIDTH + 30,
            header_background="#0466C8", theme_toggle=False,
        )


def make_app(root=PASTA_IS, runs=PASTA_RUNS):
    """Fabrica usada por pn.serve: uma instancia por sessao do navegador."""
    return IsaApp(root, runs).render()


def start(port=5006, show=True, threaded=False, root=PASTA_IS, runs=PASTA_RUNS):
    """Sobe o servidor. Com threaded=True devolve a thread do servidor."""
    return pn.serve(
        lambda: make_app(root, runs), port=port, show=show, title=TITULO,
        websocket_origin=[f"localhost:{port}", f"127.0.0.1:{port}"], threaded=threaded,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Interface do espaco de instancias")
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument("--no-show", action="store_true", help="nao abre o navegador")
    parser.add_argument("--root", default=str(PASTA_IS), help="pasta resultados/is")
    parser.add_argument("--runs", default=str(PASTA_RUNS),
                        help="pasta das execucoes disparadas pela interface")
    args = parser.parse_args(argv)
    start(port=args.port, show=not args.no_show, root=Path(args.root), runs=Path(args.runs))


if __name__ == "__main__":
    main()
