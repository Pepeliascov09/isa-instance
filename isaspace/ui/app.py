"""Interface do espaco de instancias (Panel 1.x) sobre isaspace.ui.loader_is.

Quatro abas sobre um estado global unico (EstadoGlobal):

- "Instance Space" (0): um scatter z_1 x z_2 largo, com o lasso ativo ao abrir,
  colorido pela variavel global de cor; sobreposicoes opcionais de footprints
  (algoritmo e tipo), da fronteira do CLOISTER e da footprint hard.
- "Footprint Performance" (1): mapa das footprints com a selecao destacada e a
  tabela footprint_performance.csv com o status (ok / suspeita / vazia).
- "Distributions" (2): distribuicao de variaveis numericas, todas as instancias
  contra as selecionadas (histograma ou boxplot).
- "Data Explorer" (3): scatter x-y com a cor global e a selecao destacada,
  filtro pandas.query local e o botao "Usar filtro como selecao".

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

A variavel de cor e global: os seletores das abas 0 e 3 sao duas vistas do
mesmo EstadoGlobal.cor.

TITULO. O titulo do template do Panel 1.9 so e aplicado na primeira
renderizacao (panel/template/base.py:770-788, e o servidor desliga
use_for_title); o nome do dataset vai num pane do cabecalho e no componente
TituloAba, que escreve document.title no navegador.

REGRA ARQUITETURAL: este modulo nao importa instancespace, pyispace, pyhard nem
sklearn; le apenas resultados/is/<nome>/ via isaspace.ui.loader_is (formato em
docs/output_format.md). O app anterior (Panel 0.14, pyispace) esta em
isaspace/ui/app_legacy.py.

Versoes alvo: Panel 1.9.4, HoloViews 1.23.2, Bokeh 3.9.2 (.venv-isa).

Uso (da raiz do projeto, com o .venv-isa):
    python -m isaspace.ui.app [--port 5006] [--no-show] [--root resultados/is]
"""

import argparse
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

from isaspace.ui.loader_is import (  # noqa: E402
    CATEGORICA, NUMERICA, SUSPEITA, VAZIA, list_available, load_is_output,
)

pn.extension("tabulator", notifications=True)
pn.config.disconnect_notification = (
    "Conexão com o servidor perdida: esta página não atualiza mais. Recarregue (F5)."
)
hv.extension("bokeh")

TITULO = "isa-instance"
TABS = ["Instance Space", "Footprint Performance", "Distributions", "Data Explorer"]
PASTA_IS = RAIZ / "resultados" / "is"
CONTROL_WIDTH = 300
SWAP = 6                    # indice, na sidebar, do bloco que troca por aba
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

    dataset = param.Selector(default=None, objects=[], doc="Pasta de saída do engine (resultados/is/<nome>)")
    resultado = param.Parameter(default=None, doc="IsResult do dataset ativo")
    selecao = param.Parameter(default=None, doc="""
        None = sem selecao; frozenset de rotulos Row, vazio = selecao vazia""")
    cor = param.String(default="", doc="coluna de IsResult.instances que colore os pontos")


def catalogo_de_cores(r) -> dict:
    """{coluna de r.instances: (rotulo, grupo, tipo)} das variaveis de cor."""
    cat = {}
    for col, tipo in r.annotations.items():
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


class IsaApp:
    """Interface do espaco de instancias: uma instancia por sessao do navegador."""

    def __init__(self, root=PASTA_IS):
        self.root = Path(root)
        self.datasets = list_available(self.root)
        if not self.datasets:
            raise FileNotFoundError(f"nenhuma pasta com run_info.json em {self.root}")

        self.estado = EstadoGlobal()
        self.estado.param.dataset.objects = self.datasets
        self._cache = {}
        self._dados = None           # r.instances com Row como coluna (base dos plots)
        self._catalogo = {}
        self._building = False       # atualizando opcoes de widgets: ignora os eventos
        self._doc = None             # documento da sessao (debounce da selecao)
        self._geracao = 0            # streams de plots antigos sao ignorados
        self._indice = []            # ultimo Selection1D do scatter da aba 0
        self._pendente = None
        self._selecao_do_plot = None  # ultima selecao gravada pelo proprio scatter

        # --- cabecalho: nome do dataset (pane) e titulo da aba do navegador
        self.cab = pn.pane.HTML("", margin=(0, 10))
        self.titulo_aba = TituloAba(titulo=TITULO, width=0, height=0, margin=0)

        # --- sidebar fixa
        self.w_dataset = pn.widgets.Select.from_param(
            self.estado.param.dataset, name="Dataset", width=CONTROL_WIDTH - 20)
        self.w_reload = pn.widgets.Button(name="Recarregar dataset", width=CONTROL_WIDTH - 20)
        self.info = pn.pane.Markdown("", width=CONTROL_WIDTH - 20)
        self.sel_info = pn.pane.Markdown("", width=CONTROL_WIDTH - 20)
        self.w_limpar = pn.widgets.Button(name="Limpar seleção", width=CONTROL_WIDTH - 20)

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

        # --- aba 2
        self.w_dist_vars = pn.widgets.MultiChoice(name="Variáveis", width=CONTROL_WIDTH - 20)
        self.w_dist_tipo = pn.widgets.RadioButtonGroup(options=["histograma", "boxplot"],
                                                       value="histograma")
        self.dist_status = pn.pane.Markdown("")
        self.dist_plots = pn.Column(sizing_mode="stretch_width")

        # --- aba 3
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
            pn.Column("## Distributions", self.w_dist_vars,
                      pn.pane.Markdown(f"_Até {MAX_DIST_VARS} variáveis por vez._"),
                      "### Tipo", self.w_dist_tipo, width=CONTROL_WIDTH),
            pn.Column("## Data Explorer", self.w_ex_x, self.w_ex_y, self.w_ex_cor,
                      self.w_ex_query, self.ex_filtro, self.w_ex_usar, width=CONTROL_WIDTH),
        ]
        self.sidebar = pn.Column(
            self.w_dataset, self.w_reload, self.info, self.sel_info, self.w_limpar,
            pn.layout.Divider(), self.controls[0], width=CONTROL_WIDTH,
        )
        self.tabs = pn.Tabs(
            (TABS[0], pn.Column(self.espaco_status, self.espaco_plot, sizing_mode="stretch_width")),
            (TABS[1], pn.Column(self.fp_status, self.fp_plot,
                                pn.pane.Markdown("### footprint_performance.csv"), self.fp_tabela,
                                sizing_mode="stretch_width")),
            (TABS[2], pn.Column(self.dist_status, self.dist_plots, sizing_mode="stretch_width")),
            (TABS[3], pn.Column(self.ex_status, self.ex_plot, self.ex_titulo_tabela,
                                self.ex_tabela, sizing_mode="stretch_width")),
            dynamic=True, sizing_mode="stretch_width",
        )

        # --- ligacoes
        self.estado.param.watch(self._on_dataset, "dataset")
        self.estado.param.watch(self._on_estado, ["resultado", "selecao", "cor"])
        self.tabs.param.watch(self._on_tab, "active")
        self.w_reload.on_click(self._on_reload)
        self.w_limpar.on_click(lambda _: setattr(self.estado, "selecao", None))
        for w in (self.w_cor, self.w_ex_cor):
            w.param.watch(self._on_widget_cor, "value")
        for w in (self.w_ov_fp, self.w_ov_algo, self.w_ov_tipo, self.w_ov_cloister, self.w_ov_hard):
            w.param.watch(lambda _: self._refresh_espaco(), "value")
        for w in (self.w_fp_algo, self.w_fp_tipo):
            w.param.watch(lambda _: self._refresh_footprints(), "value")
        for w in (self.w_dist_vars, self.w_dist_tipo):
            w.param.watch(lambda _: self._refresh_distribuicoes(), "value")
        for w in (self.w_ex_x, self.w_ex_y, self.w_ex_query):
            w.param.watch(lambda _: self._refresh_explorer(), "value")
        self.w_ex_usar.on_click(self._on_usar_filtro)

        self.estado.dataset = self.datasets[0]

    # ------------------------------------------------------------------ dados
    def _carregar(self, nome, forcar=False):
        if forcar or nome not in self._cache:
            self._cache[nome] = load_is_output(self.root / nome)
        return self._cache[nome]

    def _on_dataset(self, event):
        self._trocar_dataset(event.new)

    def _on_reload(self, _):
        self._trocar_dataset(self.estado.dataset, forcar=True)

    def _trocar_dataset(self, nome, forcar=False):
        r = self._carregar(nome, forcar)
        self._dados = r.instances.reset_index()
        self._catalogo = catalogo_de_cores(r)
        cor = self.estado.cor if self.estado.cor in self._catalogo else cor_padrao(r, self._catalogo)
        self._set_options(r)
        self._selecao_do_plot = None
        # uma so rodada de watchers: resultado, selecao zerada e cor valida
        self.estado.param.update(resultado=r, selecao=None, cor=cor)
        self.cab.object = f'<span style="font-size:1.25em;color:white">— {nome}</span>'
        self.titulo_aba.titulo = f"{TITULO} - {nome}"
        self._refresh_info()

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
            numericas = self._numericas(r)
            dist = [c for c in numericas if c not in ("z_1", "z_2")]
            self.w_dist_vars.options = dist
            mantidas = [v for v in self.w_dist_vars.value if v in dist]
            self.w_dist_vars.value = mantidas or [self._variavel_padrao(r)]
            for w, padrao in ((self.w_ex_x, "z_1"), (self.w_ex_y, "z_2")):
                w.options = numericas
                if w.value not in numericas:
                    w.value = padrao
        finally:
            self._building = False

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
        cols += [c for c, t in r.annotations.items() if t == NUMERICA]
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
            self._refresh_distribuicoes()
        self._refresh_explorer()

    def _on_tab(self, event):
        self.sidebar[SWAP] = self.controls[event.new]

    def _refresh_info(self):
        r = self.estado.resultado
        rob = r.run_info.get("trace_robustez", {})
        linhas = [
            f"**{r.name}**",
            f"Instâncias: **{r.n}**",
            f"Features no PILOT: **{r.n_features_used}** de {len(r.features_all)}",
            f"Algoritmos: **{len(r.algos)}**",
        ]
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
        if self.w_ov_cloister.value and r.bounds is not None:
            anel = np.vstack([r.bounds.exterior, r.bounds.exterior[:1]])
            camadas.append(hv.Path([anel], label="CLOISTER").opts(
                color="black", line_width=2, line_dash="dashed", show_legend=True))
            if r.bounds_pruned is not None and not np.array_equal(r.bounds.exterior,
                                                                   r.bounds_pruned.exterior):
                anel = np.vstack([r.bounds_pruned.exterior, r.bounds_pruned.exterior[:1]])
                camadas.append(hv.Path([anel], label="CLOISTER (podada)").opts(
                    color="#6c757d", line_width=1.5, line_dash="dotted", show_legend=True))
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

    # --------------------------------------------- aba 2: distribuicoes
    def _grafico_dist(self, var, tipo, mascara, n_sel):
        vals = self._dados[var].to_numpy(dtype=float)
        n_all = len(vals)
        titulo = f"{var}  (Todas n={n_all}" + (f", Selecionadas n={n_sel})" if mascara is not None else ")")
        if tipo == "histograma":
            fin = vals[np.isfinite(vals)]
            if fin.size == 0:
                return hv.Curve([]).opts(title=f"{var}: sem valores finitos")
            edges = np.histogram_bin_edges(fin, bins=25)
            fa, _ = np.histogram(fin, bins=edges, density=True)
            camadas = [hv.Histogram((edges, fa), label="Todas").opts(
                fill_color=COR_TODAS, line_alpha=0, fill_alpha=0.5)]
            if mascara is not None and n_sel:
                sv = vals[mascara]
                sv = sv[np.isfinite(sv)]
                if sv.size:
                    fs, _ = np.histogram(sv, bins=edges, density=True)
                    camadas.append(hv.Histogram((edges, fs), label="Selecionadas").opts(
                        fill_color=COR_SEL, line_alpha=0, fill_alpha=0.5))
            return hv.Overlay(camadas).opts(title=titulo, responsive=True, height=240,
                                            xlabel=var, ylabel="densidade",
                                            legend_position="top_right", show_grid=True)
        partes = [pd.DataFrame({"grupo": "Todas", "valor": vals})]
        if mascara is not None and n_sel:
            partes.append(pd.DataFrame({"grupo": "Selecionadas", "valor": vals[mascara]}))
        dfl = pd.concat(partes, ignore_index=True)
        dfl = dfl[np.isfinite(dfl["valor"])]
        return hv.BoxWhisker(dfl, "grupo", "valor").opts(
            title=titulo, responsive=True, height=240, box_color="grupo",
            cmap={"Todas": COR_TODAS, "Selecionadas": COR_SEL}, ylabel=var, xlabel="",
            show_legend=False)

    def _refresh_distribuicoes(self):
        variaveis = list(self.w_dist_vars.value)
        mascara = self._mascara()
        n_sel = 0 if mascara is None else int(mascara.sum())
        self.dist_status.object = self._texto_selecao({
            "sem": "Mostrando só *Todas*. Use o lasso na aba Instance Space para comparar.",
            "vazia": "Não há o que comparar: só *Todas* aparece (Selecionadas n=0).",
            "com": "Cada gráfico compara *Todas* com *Selecionadas*.",
        })
        if not variaveis:
            self.dist_plots.objects = [pn.pane.Markdown("Escolha ao menos uma variável na barra lateral.")]
            return
        if len(variaveis) > MAX_DIST_VARS:
            self.dist_plots.objects = [pn.pane.Markdown(
                f"**{len(variaveis)} variáveis escolhidas.** O limite é {MAX_DIST_VARS} por vez.")]
            return
        self.dist_plots.objects = [
            pn.pane.HoloViews(self._grafico_dist(v, self.w_dist_tipo.value, mascara, n_sel),
                              sizing_mode="stretch_width")
            for v in variaveis
        ]

    # --------------------------------------------- aba 3: data explorer
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

        cols = list(dict.fromkeys(["Row", x, y, col, "best_algo", "best_algo_svm"]))
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


def make_app(root=PASTA_IS):
    """Fabrica usada por pn.serve: uma instancia por sessao do navegador."""
    return IsaApp(root).render()


def start(port=5006, show=True, threaded=False, root=PASTA_IS):
    """Sobe o servidor. Com threaded=True devolve a thread do servidor."""
    return pn.serve(
        lambda: make_app(root), port=port, show=show, title=TITULO,
        websocket_origin=[f"localhost:{port}", f"127.0.0.1:{port}"], threaded=threaded,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Interface do espaco de instancias")
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument("--no-show", action="store_true", help="nao abre o navegador")
    parser.add_argument("--root", default=str(PASTA_IS), help="pasta resultados/is")
    args = parser.parse_args(argv)
    start(port=args.port, show=not args.no_show, root=Path(args.root))


if __name__ == "__main__":
    main()
