"""Interface Panel do espaco de instancias: as quatro abas implementadas.

Arquitetura copiada de pyhard.app.ClassificationApp: FastListTemplate subido
por pn.serve(), pn.Tabs com quatro abas e uma unica sidebar cuja parte de
baixo troca conforme a aba ativa (tabs.param.active). No topo da sidebar, fixos:
o seletor de dataset, um botao de recarregar e um bloco com os numeros do
IsaResult.

- "Instance Space" (0): dois hv.Points lado a lado (d_1 x d_2 do espaco de
  dados, z_1 x z_2 do espaco de instancias) sobre o MESMO DataFrame, num
  hv.Layout com shared_datasource=True, de modo que a selecao num destaca os
  mesmos pontos no outro. Os streams SelectionExpr so guardam a geometria em
  self.bbox e a expressao em self.selection_expr e atualizam o contador.
- "Footprint Performance" (1): mapa z_1 x z_2 com as footprints (hv.Polygons) e
  a tabela footprint_performance.csv num Tabulator.
- "Distributions" (2): compara a distribuicao de cada variavel escolhida entre
  todas as instancias e as selecionadas (histograma ou boxplot).
- "Data Explorer" (3): scatter x-y colorido com filtro pandas.query e Tabulator.

As abas 1, 2 e 3 consomem selected_mask(), calculado de forma PREGUICOSA: so
quando a aba e ativada (tabs.param.active), nunca no callback da selecao.

REGRA ARQUITETURAL: este modulo nao importa pyispace, pyhard nem sklearn; le
apenas resultados/isa/<nome>/ via isaspace.ui.loader.

Versoes alvo: Panel 0.14.4, HoloViews 1.17.1, Bokeh 2.4.3.

Copia preservada do app anterior ao porte para Panel 1.x (isaspace/ui/app.py).
Roda so no .venv (Python 3.11, Panel 0.14.4) e le resultados/isa/ (pyispace).

Uso (da raiz do projeto, com o .venv):
    python -m isaspace.ui.app_legacy [--port 5006] [--no-show]
"""

import argparse
import sys
import time
from functools import reduce
from pathlib import Path

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
from bokeh.models import HoverTool
from holoviews import opts
from holoviews.streams import PlotReset, SelectionExpr

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from isaspace.ui.loader import list_available, load_isa_output  # noqa: E402

pn.extension("tabulator")
hv.extension("bokeh")

TABS = ["Instance Space", "Footprint Performance", "Distributions", "Data Explorer"]
PASTA_ISA = RAIZ / "resultados" / "isa"
CONTROL_WIDTH = 280
COR_PADRAO = "ih"
HOVER_COLS = ["Row", "class", "ih", "n_wrong"]
TODOS = "todos"
MAX_DIST_VARS = 6
# cores por algoritmo no mapa de footprints (modo "todos")
PALETA = ["#e63946", "#2a9d8f", "#e9c46a", "#8338ec", "#ff7f0e", "#118ab2"]
COR_BASE = "#5c677d"        # pontos do mapa de footprints
COR_UM = "#e63946"          # footprint de um unico algoritmo
COR_TODAS = "#5c677d"       # serie "Todas" nas distribuicoes
COR_SEL = "#e63946"         # serie "Selecionadas"
SWAP = 4                    # indice, na sidebar, do bloco que troca por aba
# jscallback do pane tab_title (ver _set_title): roda no navegador a cada
# mudanca do texto e o copia para document.title. O Panel manda o texto do
# pane HTML escapado, por isso decodifica num elemento antes de usar.
TAB_TITLE_JS = """
const d = document.createElement('div');
d.innerHTML = cb_obj.text;
document.title = d.textContent.trim();
"""
# Batimento servidor -> navegador (ver _beat). A cada batimento o jscallback
# abaixo rearma um watchdog no cliente; se os batimentos param, o watchdog
# vence e mostra uma faixa vermelha fixa no topo da pagina.
HEARTBEAT_MS = 5000
WATCHDOG_MS = 20000
HEARTBEAT_JS = """
const id = 'isa-conexao-perdida';
const antigo = document.getElementById(id);
if (antigo) antigo.remove();
clearTimeout(window._isaWatchdog);
window._isaWatchdog = setTimeout(() => {
  const b = document.createElement('div');
  b.id = id;
  b.textContent = 'Conexao com o servidor perdida: esta pagina nao atualiza mais. Recarregue (F5).';
  b.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:100000;' +
    'background:#e63946;color:#fff;padding:10px;text-align:center;font:15px sans-serif';
  document.body.appendChild(b);
}, __WATCHDOG_MS__);
""".replace("__WATCHDOG_MS__", str(WATCHDOG_MS))


class IsaApp:
    """Interface do espaco de instancias com as quatro abas."""

    def __init__(self, root=PASTA_ISA):
        self.root = Path(root)
        self.datasets = list_available(self.root)
        if not self.datasets:
            raise FileNotFoundError(f"nenhuma pasta com coordinates.csv em {self.root}")

        self.result = None
        self.template = None
        self.bbox = None             # geometria da ultima selecao (SelectionExpr.bbox)
        self.selection_expr = None   # expressao hv.dim da ultima selecao
        self.load_times = {}
        self._cache = {}
        self._data = None            # DataFrame com Row como coluna (base dos plots)
        self._hv_dataset = None      # hv.Dataset(self._data) para aplicar a expressao
        self._explained = {"d_1": float("nan"), "d_2": float("nan")}
        self._streams = []
        self._building = False

        # --- widgets fixos e da aba 0
        self.tabs = pn.Tabs(
            *[(nome, pn.pane.Markdown(f"**{nome}**: em construcao.")) for nome in TABS],
            dynamic=False,
        )
        self.w_dataset = pn.widgets.Select(
            name="Dataset", options=self.datasets, value=self.datasets[0]
        )
        self.w_reload = pn.widgets.Button(
            name="Recarregar dataset", button_type="default", width=CONTROL_WIDTH - 20
        )
        self.info_fixed = pn.pane.Markdown("", width=CONTROL_WIDTH - 20)
        # Titulo vivo (ver _set_title): o nome do dataset no cabecalho e um
        # pane (modelo Bokeh sincronizado); o <title> da aba e copiado, no
        # navegador, do texto de um pane invisivel via jscallback.
        self.header_name = pn.pane.HTML("", margin=0)
        self.tab_title = pn.pane.HTML("", visible=False, margin=0)
        self.tab_title.jscallback(object=TAB_TITLE_JS)
        self.heartbeat = pn.pane.HTML("", visible=False, margin=0)
        self.heartbeat.jscallback(object=HEARTBEAT_JS)

        self.w_color = pn.widgets.Select(
            name="Cor dos pontos", options=[COR_PADRAO], value=COR_PADRAO
        )
        # Panel 0.14.4 nao tem tooltip real em Select (nem TooltipIcon), entao o
        # aviso vai como legenda abaixo do seletor.
        self.color_note = pn.pane.Markdown(
            "_Trocar a cor limpa o realce visual; a selecao em si e mantida._",
            width=CONTROL_WIDTH - 20,
        )
        self.w_selected = pn.indicators.Number(
            name="Instancias selecionadas", value=0, format="{value}",
            font_size="30pt", title_size="12pt", width=CONTROL_WIDTH - 20, height=90,
        )

        # --- widgets aba 1 (Footprint Performance)
        self.w_fp_algo = pn.widgets.Select(name="Algoritmo", options=[TODOS], value=TODOS)
        self.w_fp_type = pn.widgets.RadioButtonGroup(
            name="Tipo", options=["good", "best"], value="good"
        )
        self.fp_warning = pn.pane.Markdown("", width=CONTROL_WIDTH - 20)

        # --- widgets aba 2 (Distributions)
        self.w_dist_vars = pn.widgets.MultiChoice(
            name="Variaveis", options=["ih"], value=["ih"], width=CONTROL_WIDTH - 20
        )
        self.w_dist_type = pn.widgets.RadioButtonGroup(
            name="Tipo de grafico", options=["histograma", "boxplot"], value="histograma"
        )
        self.dist_note = pn.pane.Markdown(
            f"_Ate {MAX_DIST_VARS} variaveis por vez._", width=CONTROL_WIDTH - 20
        )

        # --- widgets aba 3 (Data Explorer)
        self.w_ex_x = pn.widgets.Select(name="Eixo x", options=["z_1"], value="z_1")
        self.w_ex_y = pn.widgets.Select(name="Eixo y", options=["z_2"], value="z_2")
        self.w_ex_color = pn.widgets.Select(name="Cor", options=["ih"], value="ih")
        self.w_ex_query = pn.widgets.TextInput(
            name="Filtro (pandas query)", placeholder="ex.: ih > 0.3 and n_wrong >= 2",
            width=CONTROL_WIDTH - 20,
        )
        self.ex_status = pn.pane.Markdown("", width=CONTROL_WIDTH - 20)

        # --- sidebar: bloco fixo no topo, controle da aba ativa embaixo (SWAP)
        self.controls = [
            pn.Column(
                "## Instance Space", self.w_color, self.color_note, self.w_selected,
                pn.pane.Markdown(
                    "Use *lasso* ou *box select* nos dois graficos; a selecao "
                    "alimenta as outras abas."
                ),
                width=CONTROL_WIDTH,
            ),
            pn.Column(
                "## Footprint Performance", self.w_fp_algo, "### Tipo", self.w_fp_type,
                self.fp_warning, width=CONTROL_WIDTH,
            ),
            pn.Column(
                "## Distributions", self.w_dist_vars, self.dist_note,
                "### Tipo", self.w_dist_type, width=CONTROL_WIDTH,
            ),
            pn.Column(
                "## Data Explorer", self.w_ex_x, self.w_ex_y, self.w_ex_color,
                self.w_ex_query, self.ex_status, width=CONTROL_WIDTH,
            ),
        ]
        self.sidebar = pn.Column(
            self.w_dataset, self.w_reload, self.info_fixed,
            pn.layout.Divider(), self.controls[0], width=CONTROL_WIDTH,
        )

        @pn.depends(a=self.tabs.param.active, watch=True)
        def on_tab(a):
            self.sidebar[SWAP] = self.controls[a]
            if a in (1, 2, 3):        # propagacao preguicosa da mascara
                self._refresh_lazy_tab(a)

        self._on_tab = on_tab

        self.w_dataset.param.watch(self._on_dataset, "value")
        self.w_color.param.watch(self._on_color, "value")
        self.w_reload.on_click(self._on_reload)
        self.w_fp_algo.param.watch(self._on_lazy_control, "value")
        self.w_fp_type.param.watch(self._on_lazy_control, "value")
        self.w_dist_vars.param.watch(self._on_lazy_control, "value")
        self.w_dist_type.param.watch(self._on_lazy_control, "value")
        for w in (self.w_ex_x, self.w_ex_y, self.w_ex_color, self.w_ex_query):
            w.param.watch(self._on_lazy_control, "value")
        self.load_dataset(self.w_dataset.value)

    # ------------------------------------------------------------------ dados
    def load_dataset(self, nome):
        t0 = time.perf_counter()
        if nome not in self._cache:
            self._cache[nome] = load_isa_output(self.root / nome)
            self.load_times[nome] = time.perf_counter() - t0
        r = self._cache[nome]
        self.result = r
        self._data = r.instances.reset_index()   # Row vira coluna, para o hover
        self._hv_dataset = hv.Dataset(self._data)

        # variancia explicada dos 2 PCs: variancia de cada escore / n de
        # atributos (a variancia total dos atributos padronizados = n). So numpy.
        p = len(r.data_columns)
        self._explained = {
            "d_1": float(np.var(self._data["d_1"], ddof=0) / p) if p else float("nan"),
            "d_2": float(np.var(self._data["d_2"], ddof=0) / p) if p else float("nan"),
        }

        self._set_options(r)
        self._refresh_space_tab(preserve_selection=False)   # troca de dataset zera
        if self.tabs.active in (1, 2, 3):
            self._refresh_lazy_tab(self.tabs.active)

        # bloco fixo da sidebar (numeros lidos do IsaResult, sem recalcular)
        self.info_fixed.object = (
            f"**{r.name}**  \n"
            f"Instancias: **{r.n}**  \n"
            f"Features no PILOT: **{r.n_features_used}**  \n"
            f"Features descartadas: **{r.n_features_dropped}**  \n"
            f"Algoritmos: **{len(r.algos)}**"
        )
        self._set_title(r.name)

    def _set_title(self, nome):
        """Sincroniza o cabecalho e o <title> da aba com o dataset.

        template.title NAO serve para isso depois de servido: ele so alimenta
        o Jinja do cabecalho ({{ app_title }}) e o Document.title na primeira
        renderizacao (BasicTemplate._init_doc); o Panel 0.14.4 nem observa o
        param depois disso, entao atribuir template.title numa sessao viva
        nao gera evento algum para o navegador. Document.title tambem nao
        resolve: o Panel 0.14.4 serve a pagina com use_for_title=False
        (panel/io/server.py), e ai o BokehJS ignora o evento TitleChanged.
        O que sincroniza e:
          (1) header_name, pane no header do template (modelo Bokeh vivo);
          (2) tab_title, pane invisivel cujo jscallback (TAB_TITLE_JS)
              escreve o texto em document.title, no navegador.
        """
        self.header_name.object = f'<span class="title">- {nome}</span>'
        self.tab_title.object = f"isa-instance - {nome}"

    def _beat(self):
        """Batimento: texto novo (hora) no pane invisivel heartbeat, a cada
        HEARTBEAT_MS. No navegador, HEARTBEAT_JS rearma um watchdog a cada
        batimento; se a sessao morre (servidor reiniciado, websocket fechado),
        os batimentos param, o watchdog vence e aparece a faixa vermelha.

        Sem isso uma pagina com sessao morta e indistinguivel de uma viva: as
        abas ainda trocam (isso e do lado do cliente) e mostram o ultimo
        conteudo recebido, mas sidebar, cabecalho e dados nao respondem mais.
        Panel 0.14.4 / Bokeh 2.4.3 nao tem evento ConnectionLost nem
        disconnect_notification; o batimento e o que ha.
        """
        self.heartbeat.object = time.strftime("%H:%M:%S")

    def _on_load(self):
        """Roda quando a pagina carregou (pn.state.onload): preenche o canal do
        <title> (primeira MUDANCA, que dispara o jscallback) e da o primeiro
        batimento, que arma o watchdog."""
        self._set_title(self.result.name)
        self._beat()

    def _set_options(self, r):
        """Atualiza as opcoes de todos os seletores para o dataset, sem disparar
        rebuilds (guardado por _building)."""
        self._building = True
        try:
            cor = ["ih", "n_wrong", "class"] + [f"algo_{a}" for a in r.algos]
            self.w_color.options = cor
            if self.w_color.value not in cor:
                self.w_color.value = COR_PADRAO

            self.w_fp_algo.options = [TODOS] + list(r.algos)
            if self.w_fp_algo.value not in self.w_fp_algo.options:
                self.w_fp_algo.value = TODOS

            dist_opts = (
                [f"feature_{f}" for f in r.features] + ["ih", "n_wrong"]
                + [f"algo_{a}" for a in r.algos]
            )
            self.w_dist_vars.options = dist_opts
            mantidas = [v for v in self.w_dist_vars.value if v in dist_opts]
            self.w_dist_vars.value = mantidas or ["ih"]

            numericas = self._numeric_cols(r)
            for w, padrao in (
                (self.w_ex_x, "z_1"), (self.w_ex_y, "z_2"), (self.w_ex_color, "ih")
            ):
                opcoes = numericas + (["class"] if w is self.w_ex_color else [])
                w.options = opcoes
                if w.value not in opcoes:
                    w.value = padrao if padrao in opcoes else opcoes[0]
        finally:
            self._building = False

    def _numeric_cols(self, r):
        data = self._data
        candidatas = (
            [f"feature_{f}" for f in r.features] + ["z_1", "z_2", "d_1", "d_2"]
            + list(r.data_columns) + ["ih", "n_wrong"] + [f"algo_{a}" for a in r.algos]
        )
        vistos, out = set(), []
        for c in candidatas:
            if c in data.columns and c not in vistos and pd.api.types.is_numeric_dtype(data[c]):
                vistos.add(c)
                out.append(c)
        return out

    # ------------------------------------------------------------ selecao/mask
    def count_selected(self) -> int:
        if self.selection_expr is None or self._hv_dataset is None:
            return 0
        return int(self.selected_mask().sum())

    def selected_mask(self) -> np.ndarray:
        """Array booleano por instancia a partir de self.selection_expr.

        Unica fonte de verdade da selecao para as abas 1, 2 e 3. Sem selecao (ou
        sem dados) devolve tudo True. Calculada sob demanda; nunca no callback
        da selecao.
        """
        n = 0 if self._data is None else len(self._data)
        if self.selection_expr is None or self._hv_dataset is None:
            return np.ones(n, dtype=bool)
        return np.asarray(self.selection_expr.apply(self._hv_dataset), dtype=bool)

    def selected_rows(self):
        if self.selection_expr is None:
            return []
        return self._data.loc[self.selected_mask(), "Row"].tolist()

    def _has_selection(self, mask=None):
        """True se ha selecao ativa que nao cobre tudo."""
        if self.selection_expr is None:
            return False
        m = self.selected_mask() if mask is None else mask
        return not bool(m.all())

    # ----------------------------------------------------------- aba 0: plots
    @staticmethod
    def _scatter(data, kdims, vdims, color, title, xlabel=None, ylabel=None):
        hover = HoverTool(tooltips=[
            ("Row", "@Row"), ("class", "@class"),
            ("ih", "@ih{0.000}"), ("n_wrong", "@n_wrong"),
        ])
        categorico = color == "class"
        vd = list(dict.fromkeys(vdims + [color]))
        estilo = dict(
            title=title, responsive=True, aspect=1.15,
            color=color, cmap="Category10" if categorico else "viridis",
            colorbar=not categorico, size=6, alpha=0.85, nonselection_alpha=0.15,
            tools=["lasso_select", "box_select", hover],
            active_tools=["lasso_select"], show_grid=True, framewise=True,
        )
        if xlabel is not None:
            estilo["xlabel"] = xlabel
        if ylabel is not None:
            estilo["ylabel"] = ylabel
        return hv.Points(data, kdims=kdims, vdims=vd).opts(**estilo)

    def _refresh_space_tab(self, preserve_selection=False):
        """(Re)constroi a aba 0 do zero com a cor atual e a repoe em self.tabs.

        Um DynamicMap com stream/pn.depends so empurra DADOS ao atualizar; nao
        troca o tipo de color mapper nem liga/desliga a colorbar. Colorir por
        'class' (categorico) exige CategoricalColorMapper, entao a cada troca de
        cor OU de dataset o plot inteiro e reconstruido.
        """
        r = self.result
        data = self._data
        color = self.w_color.value
        vdims = HOVER_COLS + [f"algo_{a}" for a in r.algos]

        ev = self._explained
        xl = f"d_1 ({ev['d_1'] * 100:.1f}%)".replace(".", ",")
        yl = f"d_2 ({ev['d_2'] * 100:.1f}%)".replace(".", ",")
        pts_data = self._scatter(
            data, ["d_1", "d_2"], vdims, color,
            "Espaco de dados (PCA dos atributos)", xlabel=xl, ylabel=yl
        )
        pts_is = self._scatter(
            data, ["z_1", "z_2"], vdims, color, "Espaco de instancias (PILOT)"
        )

        if not preserve_selection:
            # troca de dataset: zera tudo (os indices mudam de dataset)
            self.bbox = None
            self.selection_expr = None
            self.w_selected.value = 0
        else:
            # DECISAO TOMADA: mantem shared_datasource=True. Na troca de COR
            # reaplica a expressao guardada e restaura o contador; o DESTAQUE
            # visual (Bokeh CDS.selected.indices) NAO e reaplicado. Motivo,
            # verificado neste stack:
            #   * a unica forma limpa de semear e hv.Points(...).opts(selected=),
            #     mas shared_datasource=True funde os dois plots num CDS unico e
            #     DESCARTA esse opts -- os indices saem vazios;
            #   * o pane servido nao expoe o CDS por-sessao no servidor
            #     (pane._plots vazio), entao nao ha handle limpo para reaplicar.
            # Reaplicar exigiria varrer o documento por next-tick e casar o CDS
            # por colunas -- fragil e dependente de timing. O tooltip/legenda no
            # seletor de cor avisa que trocar a cor limpa o realce. Contador e
            # selected_mask() (o que as abas 1-3 consomem) seguem corretos.
            self.w_selected.value = self.count_selected()

        self._streams = []
        for pts in (pts_data, pts_is):
            sel = SelectionExpr(source=pts)
            sel.add_subscriber(self._on_selection)
            reset = PlotReset(source=pts)
            reset.add_subscriber(self._on_reset)
            self._streams += [sel, reset]

        layout = (pts_data + pts_is).cols(2).opts(
            opts.Layout(shared_axes=False, shared_datasource=True, framewise=True)
        )
        self.tabs[0] = (TABS[0], pn.pane.HoloViews(layout, sizing_mode="stretch_both"))

    # ------------------------------------------------- aba 1: footprint perf.
    def _footprint_map(self):
        r, data = self.result, self._data
        algo, tipo = self.w_fp_algo.value, self.w_fp_type.value
        mask = self.selected_mask()

        d = data.assign(_alpha=np.where(mask, 0.9, 0.12))
        hover = HoverTool(tooltips=[
            ("Row", "@Row"), ("class", "@class"),
            ("ih", "@ih{0.000}"), ("n_wrong", "@n_wrong"),
        ])
        base = hv.Points(
            d, ["z_1", "z_2"], ["Row", "class", "ih", "n_wrong", "_alpha"]
        ).opts(color=COR_BASE, alpha="_alpha", line_color=None, size=6,
               tools=[hover], show_legend=False)

        algos = list(r.algos) if algo == TODOS else [algo]
        camadas, avisos = [base], []
        for i, a in enumerate(algos):
            key = (a, tipo)
            polys = r.footprints.get(key, [])
            cor = PALETA[i % len(PALETA)] if algo == TODOS else COR_UM
            if key in r.empty_footprints or not polys:
                avisos.append(f"- **{a} / {tipo}**: footprint vazia, nada a desenhar.")
                continue
            pg = hv.Polygons([np.array(p) for p in polys]).relabel(a)
            if key in r.suspect_footprints:
                pur = r.suspect_footprints[key]
                avisos.append(
                    f"- **{a} / {tipo}**: suspeita, pureza medida {pur:.3f} "
                    f"< trace.PI {r.pi}; desenhada tracejada."
                )
                pg = pg.opts(fill_color=cor, fill_alpha=0.12, line_color=cor,
                             line_dash="dashed", line_width=2, line_alpha=0.7,
                             show_legend=(algo == TODOS))
            else:
                pg = pg.opts(fill_color=cor, fill_alpha=0.28, line_color=cor,
                             line_width=1.5, line_alpha=0.85, show_legend=(algo == TODOS))
            camadas.append(pg)

        n_sel = int(mask.sum()) if self._has_selection(mask) else 0
        cabec = "" if not self._has_selection(mask) else f"**{n_sel} instancias na selecao destacadas.**\n\n"
        self.fp_warning.object = cabec + (
            "Footprints suspeitas/vazias:\n" + "\n".join(avisos) if avisos
            else "Nenhuma footprint suspeita ou vazia para esta escolha."
        )

        overlay = reduce(lambda x, y: x * y, camadas)
        return overlay.opts(responsive=True, height=430, show_grid=True,
                            legend_position="right", title=f"Footprints {tipo} -- {algo}")

    def _footprint_table(self):
        r = self.result
        tipo = self.w_fp_type.value
        df = r.footprint_performance.copy().round(4)
        df.insert(len(df.columns), "status", [
            "vazia" if (a, tipo) in r.empty_footprints
            else "suspeita" if (a, tipo) in r.suspect_footprints
            else "ok"
            for a in df.index
        ])
        return pn.widgets.Tabulator(
            df, disabled=True, layout="fit_data_stretch",
            sizing_mode="stretch_width", height=280, show_index=True,
        )

    def _refresh_footprint_tab(self):
        if self.result is None:
            return
        conteudo = pn.Column(
            pn.pane.HoloViews(self._footprint_map(), sizing_mode="stretch_width"),
            pn.pane.Markdown("### footprint_performance.csv"),
            self._footprint_table(),
            sizing_mode="stretch_both",
        )
        self.tabs[1] = (TABS[1], conteudo)

    # -------------------------------------------------- aba 2: distributions
    def _dist_plot(self, var, tipo, mask, tem_sel, n_all, n_sel):
        data = self._data
        vals = data[var].to_numpy(dtype=float)
        titulo = f"{var}  (Todas n={n_all}" + (f", Selecionadas n={n_sel})" if tem_sel else ")")
        if tipo == "histograma":
            fin = vals[np.isfinite(vals)]
            if fin.size == 0:
                return hv.Overlay([hv.Curve([])]).opts(title=f"{var}: sem valores finitos")
            edges = np.histogram_bin_edges(fin, bins=25)
            fa, _ = np.histogram(fin, bins=edges, density=True)
            camadas = [hv.Histogram((edges, fa)).relabel("Todas").opts(
                fill_color=COR_TODAS, line_alpha=0, fill_alpha=0.5)]
            if tem_sel:
                sv = data.loc[mask, var].to_numpy(dtype=float)
                sv = sv[np.isfinite(sv)]
                if sv.size:
                    fs, _ = np.histogram(sv, bins=edges, density=True)
                    camadas.append(hv.Histogram((edges, fs)).relabel("Selecionadas").opts(
                        fill_color=COR_SEL, line_alpha=0, fill_alpha=0.5))
            return hv.Overlay(camadas).opts(
                title=titulo, responsive=True, height=240, xlabel=var,
                ylabel="densidade", legend_position="top_right", show_grid=True)
        # boxplot
        partes = [pd.DataFrame({"grupo": "Todas", "valor": vals})]
        if tem_sel:
            partes.append(pd.DataFrame(
                {"grupo": "Selecionadas", "valor": data.loc[mask, var].to_numpy(dtype=float)}))
        dfl = pd.concat(partes, ignore_index=True)
        dfl = dfl[np.isfinite(dfl["valor"])]
        return hv.BoxWhisker(dfl, "grupo", "valor").opts(
            title=titulo, responsive=True, height=240, box_color="grupo",
            cmap={"Todas": COR_TODAS, "Selecionadas": COR_SEL}, ylabel=var,
            xlabel="", show_legend=False)

    def _distributions_panel(self):
        data = self._data
        variaveis = list(self.w_dist_vars.value)
        tipo = self.w_dist_type.value
        if not variaveis:
            return pn.pane.Markdown("Escolha ao menos uma variavel na barra lateral.")
        if len(variaveis) > MAX_DIST_VARS:
            return pn.pane.Markdown(
                f"**{len(variaveis)} variaveis escolhidas.** O limite e "
                f"{MAX_DIST_VARS} por vez; reduza a selecao para renderizar."
            )
        mask = self.selected_mask()
        tem_sel = self._has_selection(mask)
        n_all = len(data)
        n_sel = int(mask.sum()) if tem_sel else 0

        faixas = []
        if not tem_sel:
            faixas.append(pn.pane.Markdown(
                "**Sem selecao ativa.** Mostrando so *Todas*. Use o *lasso* na "
                "aba Instance Space para comparar o subconjunto selecionado."))
        for var in variaveis:
            faixas.append(pn.pane.HoloViews(
                self._dist_plot(var, tipo, mask, tem_sel, n_all, n_sel),
                sizing_mode="stretch_width"))
        return pn.Column(*faixas, sizing_mode="stretch_both")

    def _refresh_distributions_tab(self):
        if self.result is None:
            return
        self.tabs[2] = (TABS[2], self._distributions_panel())

    # --------------------------------------------------- aba 3: data explorer
    def _explorer_panel(self):
        data = self._data
        x, y, color = self.w_ex_x.value, self.w_ex_y.value, self.w_ex_color.value
        q = (self.w_ex_query.value or "").strip()

        # filtro pandas.query -- entrada do usuario, captura qualquer excecao
        erro = None
        if q:
            try:
                df_f = data.query(q)
            except Exception as e:  # noqa: BLE001 -- mostra o erro, nao derruba
                erro, df_f = f"{type(e).__name__}: {e}", data
        else:
            df_f = data
        if erro:
            self.ex_status.object = f"**Query invalida - filtro ignorado.**  \n`{erro}`"
        else:
            self.ex_status.object = f"**{len(df_f)} de {len(data)} linhas** passam no filtro."

        # scatter: filtro aplicado ao que se ve; a selecao do lasso (selected_mask)
        # aparece como destaque/apagado entre os pontos filtrados (igual a aba 2)
        mask = self.selected_mask()
        tem_sel = self._has_selection(mask)
        pos = df_f.index.to_numpy()
        alpha = np.where(mask[pos], 0.9, 0.12) if tem_sel else np.full(len(df_f), 0.85)
        dplot = df_f.assign(_alpha=alpha)

        categorico = color == "class"
        vdims = list(dict.fromkeys([color, "Row", "class", "ih", "n_wrong", x, y]))
        hover = HoverTool(tooltips=[
            ("Row", "@Row"), ("class", "@class"),
            (x, "@{%s}" % x), (y, "@{%s}" % y), (color, "@{%s}" % color),
        ])
        scatter = hv.Points(dplot, [x, y], vdims).opts(
            color=color, cmap="Category10" if categorico else "viridis",
            colorbar=not categorico, alpha="_alpha", size=6, line_color=None,
            tools=[hover], responsive=True, height=430, show_grid=True,
            title=f"{x} x {y} (cor: {color})")

        # tabela: linhas filtradas, colunas uteis (evita 100+ colunas)
        cols = []
        for c in ["Row", "class", "ih", "n_wrong", x, y, color, "best_algo"]:
            if c in df_f.columns and c not in cols:
                cols.append(c)
        tabela = pn.widgets.Tabulator(
            df_f[cols].round(4), disabled=True, layout="fit_data_stretch",
            sizing_mode="stretch_width", height=280, show_index=False,
            pagination="local", page_size=25,
        )
        return pn.Column(
            pn.pane.HoloViews(scatter, sizing_mode="stretch_width"),
            pn.pane.Markdown(f"### Linhas filtradas ({len(df_f)})"),
            tabela, sizing_mode="stretch_both",
        )

    def _refresh_explorer_tab(self):
        if self.result is None:
            return
        self.tabs[3] = (TABS[3], self._explorer_panel())

    # ------------------------------------------------------------- dispatch
    def _refresh_lazy_tab(self, i):
        if i == 1:
            self._refresh_footprint_tab()
        elif i == 2:
            self._refresh_distributions_tab()
        elif i == 3:
            self._refresh_explorer_tab()

    # -------------------------------------------------------------- callbacks
    def _on_selection(self, selection_expr=None, bbox=None, region_element=None, **_):
        """So guarda geometria e atualiza o contador; NAO calcula a mascara aqui
        (propagacao preguicosa) nem redesenha as outras abas."""
        self.bbox = bbox
        self.selection_expr = selection_expr
        self.w_selected.value = self.count_selected()

    def _on_reset(self, **_):
        self.bbox = None
        self.selection_expr = None
        self.w_selected.value = 0

    def _on_color(self, event):
        if not self._building:
            self._refresh_space_tab(preserve_selection=True)

    def _on_lazy_control(self, event):
        if not self._building and self.tabs.active in (1, 2, 3):
            self._refresh_lazy_tab(self.tabs.active)

    def _on_reload(self, event):
        nome = self.w_dataset.value
        self._cache.pop(nome, None)   # forca releitura do disco
        self.load_dataset(nome)

    def _on_dataset(self, event):
        self.load_dataset(event.new)

    # --------------------------------------------------------------- template
    def render(self):
        self.template = pn.template.FastListTemplate(
            site="",
            title="isa-instance",            # parte fixa; o dataset vai em header_name
            header=[self.header_name, self.tab_title, self.heartbeat],
            sidebar=[self.sidebar],
            main=[self.tabs],
            sidebar_width=CONTROL_WIDTH + 20,
            header_background="#0466C8",
            theme_toggle=False,
        )
        # O jscallback de tab_title so roda quando o texto MUDA no navegador:
        # zera o canal aqui e preenche em _on_load, para o <title> ficar certo
        # tambem na carga inicial (ate la a pagina mostra template.title).
        self.tab_title.object = ""
        pn.state.onload(self._on_load)
        if pn.state.curdoc is not None:          # so numa sessao servida
            pn.state.add_periodic_callback(self._beat, period=HEARTBEAT_MS)
        return self.template


def make_app(root=PASTA_ISA):
    """Fabrica usada por pn.serve: uma instancia por sessao do navegador."""
    return IsaApp(root).render()


def start(port=5006, show=True, threaded=False, root=PASTA_ISA):
    """Sobe o servidor. Com threaded=True devolve a thread do servidor."""
    return pn.serve(
        lambda: make_app(root),
        port=port,
        show=show,
        title="isa-instance",
        websocket_origin=[f"localhost:{port}", f"127.0.0.1:{port}"],
        threaded=threaded,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Interface do espaco de instancias")
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument("--no-show", action="store_true", help="nao abre o navegador")
    parser.add_argument("--root", default=str(PASTA_ISA), help="pasta resultados/isa")
    args = parser.parse_args(argv)
    start(port=args.port, show=not args.no_show, root=Path(args.root))


if __name__ == "__main__":
    main()
