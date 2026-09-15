"""App Panel/Bokeh do espaco de instancias.

Servir a partir da raiz do projeto:
    .venv\\Scripts\\panel.exe serve isaspace/app.py --show

Sintaxe do Panel 0.14.4 / Bokeh 2.4.3 (versoes instaladas). O script roda uma
vez por sessao do navegador. A figura, o source e o colorbar sao criados UMA
vez; trocas de dataset/cor atualizam propriedades in place (sem recriar
modelos, preservando zoom e o callback de selecao).
"""

import sys
from pathlib import Path

import pandas as pd
import panel as pn
from bokeh.models import ColorBar, ColumnDataSource, HoverTool, LinearColorMapper
from bokeh.palettes import Viridis256
from bokeh.plotting import figure

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaspace.projection import instance_space  # noqa: E402

pn.extension()

RESULTS_DIR = ROOT / "resultados"
DATASETS = ["iris", "diabetes"]
MSG_CLIQUE = "Clique num ponto do scatter para inspecionar a instancia."
CDS_KEYS = ["z1", "z2", "ih", "n_wrong", "classe", "cor", "indice"]

_cache = {}
state = {"df": None}
_guard = {"on": False}


def load_dataset(nome):
    """Carrega table_<nome>.csv + space_<nome>.csv (ou projeta na hora)."""
    if nome in _cache:
        return _cache[nome]

    table_path = RESULTS_DIR / f"table_{nome}.csv"
    if not table_path.exists():
        return None
    table = pd.read_csv(table_path, index_col=0)

    space_path = RESULTS_DIR / f"space_{nome}.csv"
    if space_path.exists():
        space = pd.read_csv(space_path, index_col=0)
    else:
        space = instance_space(table)

    extra = [
        c for c in table.columns
        if c.startswith(("feature_", "algo_", "proba_"))
    ]
    df = space.join(table[extra])
    _cache[nome] = df
    return df


# --- widgets e paineis (persistentes) ---
w_dataset = pn.widgets.Select(name="Dataset", options=DATASETS, value=DATASETS[0])
w_color = pn.widgets.Select(name="Colorir por", options=["ih", "n_wrong"], value="ih")
status = pn.pane.Markdown("")
details = pn.Column(pn.pane.Markdown(MSG_CLIQUE))

# --- figura unica ---
mapper = LinearColorMapper(palette=Viridis256, low=0.0, high=1.0)
source = ColumnDataSource({k: [] for k in CDS_KEYS})

fig = figure(
    tools="pan,wheel_zoom,box_zoom,reset,tap",
    active_scroll="wheel_zoom",
    sizing_mode="stretch_both",
    x_axis_label="z1",
    y_axis_label="z2",
)
fig.scatter(
    "z1",
    "z2",
    source=source,
    size=9,
    line_color=None,
    fill_color={"field": "cor", "transform": mapper},
    nonselection_fill_alpha=0.25,
)
fig.add_tools(
    HoverTool(
        tooltips=[
            ("instancia", "@indice"),
            ("classe", "@classe"),
            ("ih", "@ih{0.000}"),
            ("n_wrong", "@n_wrong"),
        ]
    )
)
colorbar = ColorBar(color_mapper=mapper, title="ih")
fig.add_layout(colorbar, "right")


def update_details(indices):
    df = state["df"]
    if df is None or not indices:
        details.objects = [pn.pane.Markdown(MSG_CLIQUE)]
        return

    row = df.iloc[indices[0]]
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    algo_cols = [c for c in df.columns if c.startswith("algo_")]

    medidas = pd.DataFrame(
        {"valor": row[feature_cols].astype(float).round(4).values},
        index=[c.replace("feature_", "") for c in feature_cols],
    )
    algos = pd.DataFrame(
        {
            "acertou": ["sim" if row[c] == 1 else "NAO" for c in algo_cols],
            "proba_classe_verdadeira": [
                round(float(row["proba_" + c[len("algo_"):]]), 3)
                for c in algo_cols
            ],
        },
        index=[c.replace("algo_", "") for c in algo_cols],
    )

    cabecalho = (
        f"### Instancia {row.name}\n"
        f"**classe:** {row['class']}  \n"
        f"**ih:** {float(row['ih']):.3f} &nbsp;&nbsp; "
        f"**n_wrong:** {int(row['n_wrong'])}"
    )
    details.objects = [
        pn.pane.Markdown(cabecalho),
        pn.pane.Markdown("#### Medidas"),
        pn.pane.DataFrame(medidas, sizing_mode="stretch_width"),
        pn.pane.Markdown("#### Portfolio"),
        pn.pane.DataFrame(algos, sizing_mode="stretch_width"),
    ]


def _on_select(attr, old, new):
    update_details(list(new))


source.selected.on_change("indices", _on_select)


def _apply_color(df, col):
    valores = df[col].astype(float)
    low, high = float(valores.min()), float(valores.max())
    if low == high:
        high = low + 1e-9
    source.data["cor"] = valores.values
    mapper.low = low
    mapper.high = high
    colorbar.title = col
    fig.title.text = f"{w_dataset.value} — cor: {col}"


def update_color(event=None):
    if _guard["on"] or state["df"] is None:
        return
    _apply_color(state["df"], w_color.value)


def update_data(event=None):
    nome = w_dataset.value
    df = load_dataset(nome)

    if df is None:
        state["df"] = None
        status.object = (
            f"**Sem resultados para '{nome}'.** Gere o CSV com "
            f"`python run_table.py` (esperado: resultados/table_{nome}.csv)."
        )
        source.selected.indices = []
        source.data = {k: [] for k in CDS_KEYS}
        fig.title.text = f"{nome} — sem dados"
        details.objects = [pn.pane.Markdown(MSG_CLIQUE)]
        return

    state["df"] = df
    status.object = ""
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    opcoes = ["ih", "n_wrong"] + feature_cols

    _guard["on"] = True
    try:
        if list(w_color.options) != opcoes:
            w_color.options = opcoes
        if w_color.value not in opcoes:
            w_color.value = "ih"
    finally:
        _guard["on"] = False

    col = w_color.value
    source.selected.indices = []
    source.data = {
        "z1": df["z1"].values,
        "z2": df["z2"].values,
        "ih": df["ih"].values,
        "n_wrong": df["n_wrong"].values,
        "classe": df["class"].astype(str).values,
        "cor": df[col].astype(float).values,
        "indice": df.index.astype(str).values,
    }
    _apply_color(df, col)
    details.objects = [pn.pane.Markdown(MSG_CLIQUE)]


w_dataset.param.watch(update_data, "value")
w_color.param.watch(update_color, "value")
update_data()

template = pn.template.FastListTemplate(
    title="isa-instance — espaco de instancias",
    sidebar=[w_dataset, w_color, "## Detalhes da instancia", details],
    main=[pn.Column(status, pn.pane.Bokeh(fig, sizing_mode="stretch_both"),
                    sizing_mode="stretch_both")],
    theme_toggle=False,
)
template.servable()
