"""Instance space user interface (Panel 1.x) on top of isaspace.ui.loader_is.

Six tabs over a single global state (GlobalState):

- "Instance Space" (0): a wide z_1 x z_2 scatter, with the lasso active on
  open, colored by the global color variable; optional overlays of footprints
  (algorithm and type), the CLOISTER boundary and the hard footprint.
- "Footprint Performance" (1): map of the footprints with the selection
  highlighted and the footprint_performance.csv table with the status
  (ok / suspect / empty).
- "Algorithm Selection" (2): what PYTHIA and CLOISTER produced: map colored by
  the recommended algorithm (selection0, "none" as a category), by whether the
  recommended algorithm is good for the instance (algorithm_bin.csv) or by an
  algorithm's pr0_sub (P(bad) out of sample), with the CLOISTER boundary and
  the selection highlighted; svm_table; cross-validation confusion matrices; a
  nearly-trivial-selector warning (one algorithm in more than TRIVIAL_SELECTOR
  of the instances).
- "Distributions" (3): distribution of numeric variables grouped by a
  categorical annotation (histogram, density or violin); with a selection,
  each group is split into selected and not selected.
- "Features" (4): one row per received feature, with what was kept and why
  the rest was dropped (degenerate_report.csv + SIFTED), the feature x
  algorithm correlation heatmap and the silhouette per k.
- "Data Explorer" (5): x-y scatter with the global color and the selection
  highlighted, a local pandas.query filter and the "Use filter as selection"
  button.

The fixed sidebar has the dataset selector, with resultados/is/ and runs/
(runs launched from the UI) in separate groups; the "New instance space" block
(isaspace.ui.new_space: metadata.csv upload, validation, performance rule and
engine run in a subprocess); the inferred annotation types (changeable for the
session) and the export buttons (instances, selection, labels of the active
footprint).

GLOBAL STATE. GlobalState (param.Parameterized) holds the active dataset, the
loaded IsResult, the selection and the color variable; the tabs only read
from it and redraw when it changes. The selection is a set of Row labels:
    None           no selection
    frozenset()    empty selection (lasso over an area without points, filter without rows)
    frozenset(...) selected instances
It comes from the Selection1D stream of the tab 0 scatter. Bokeh sends a
Selection1D on every lasso move and none when the lasso falls on an empty area
starting from "nothing selected" (the indices do not change); so Selection1D
only stores the latest index and the global selection is written after the
gesture ends: the geometry streams (Lasso, BoundsXY), which arrive once when
the mouse is released, schedule the write in GEOMETRY_WAIT_MS, and each
Selection1D reschedules it in SELECTION_WAIT_MS.

The color variable is global: the selectors of tabs 0 and 5 are two views of
the same GlobalState.color.

TITLE. The Panel 1.9 template title is only applied on the first render
(panel/template/base.py:770-788, and the server turns use_for_title off); the
dataset name goes to a header pane and to the TabTitle component, which
writes document.title in the browser.

ARCHITECTURAL RULE: this module does not import instancespace, pyispace,
pyhard or sklearn; it only reads the engine's output folders via
isaspace.ui.loader_is (format in docs/output_format.md). Only
isaspace.ui.runner knows the engine, and runs it in a subprocess. The previous
app (Panel 0.14, pyispace; legacy, in Portuguese) is isaspace/ui/app_legacy.py.

Target versions: Panel 1.9.4, HoloViews 1.23.2, Bokeh 3.9.2 (.venv-isa).

Usage (from the project root, with the .venv-isa):
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
from bokeh.models import FixedTicker, HoverTool
from holoviews.streams import BoundsXY, Lasso, PlotReset, Selection1D
from panel.custom import JSComponent

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaspace.ui.loader_is import (  # noqa: E402
    CATEGORICAL, EMPTY, FORCED, IDENTIFIER, INFERRED, NUMERIC, SUSPECT, TIE, is_numeric_type,
    list_available, load_is_output,
)
from isaspace.ui.new_space import NewSpacePanel  # noqa: E402
from isaspace.ui.runner import RUNS_DIR  # noqa: E402

pn.extension("tabulator", notifications=True)
pn.config.disconnect_notification = (
    "Connection to the server lost: this page no longer updates. Reload it (F5)."
)
hv.extension("bokeh")

TITLE = "isa-instance"
TABS = ["Instance Space", "Footprint Performance", "Algorithm Selection", "Distributions",
        "Features", "Data Explorer"]
IS_DIR = ROOT / "resultados" / "is"
# dataset selector keys: "<origin>/<folder>"; the label is just the folder
ORIGINS = {"is": "resultados/is", "runs": "runs (launched from this interface)"}
CONTROL_WIDTH = 300
ALL = "all"
MAX_DIST_VARS = 6
LOW_R2 = 0.3                # below this the 2D plane explains little of the variable
GEOMETRY_WAIT_MS = 150      # end of the gesture (Lasso/BoundsXY) -> store the selection
SELECTION_WAIT_MS = 400     # Selection1D without geometry (debounce)
ALGO_PALETTE = ["#e63946", "#2a9d8f", "#e9c46a", "#8338ec", "#ff7f0e", "#118ab2",
                "#6a994e", "#bc4749", "#577590", "#f15bb5"]
BASE_COLOR = "#5c677d"      # points of the footprint map
ALL_COLOR = "#5c677d"       # "all" series in the distributions
SEL_COLOR = "#e63946"       # "selected" series
ALPHA_SEL, ALPHA_OUT, ALPHA_NEUTRAL = 0.9, 0.12, 0.7
NONE = "none"               # missing value of a categorical variable
NO_GROUP = "(none)"         # "group by" option without grouping
DIST_TYPES = ["histogram", "density", "violin"]
MIN_VIOLIN_GROUPS = 3       # from here on overlaid histograms become unreadable
NOT_SEL_COLOR = "#adb5bd"   # "not selected" half of the violins
GROUP_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
                 "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
NONE_COLOR = "#6c757d"      # no value / no recommendation: dark gray, visible on white
TIE_COLOR = "#b0b7bf"       # tie for the best observed value: light gray
# stacked plots on a scrolling page: without an active wheel_zoom, the mouse
# wheel scrolls the page instead of zooming the plot under the cursor
NO_SCROLL_ZOOM = dict(active_tools=["pan"])
# tables: no sort arrows (a few rows) and wrapped headers, so they fit the width
TABLE_CONFIG = {"columnDefaults": {"headerSort": False, "headerWordWrap": True}}
MAX_DISCRETE_LEVELS = 12    # integer color variables with up to this many values: one color each
REASON_WIDTH = 260          # px of the "reason" column of the features table
REASON_CHARS_PER_LINE = 34  # characters that fit in one line of that column
# on-screen names of the features table columns (the loader uses the long names)
FEATURE_COLUMNS = {"replaced_by": "kept instead", "max_abs_rho": "max |rho|",
                   "rho_algorithm": "algorithm", "pval": "p", "r2_pilot": "PILOT r²"}
# Algorithm Selection tab
TRIVIAL_SELECTOR = 0.9      # one algorithm recommended in more than this fraction of instances
AS_RECOMMENDED, AS_GOODNESS, AS_PR0 = "recommended", "goodness", "pr0_sub"
AS_COLORS = {"recommended algorithm (selection0)": AS_RECOMMENDED,
             "recommended good / bad": AS_GOODNESS,
             "pr0_sub of an algorithm": AS_PR0}
REC_GOOD, REC_BAD, NO_REC = "recommended good", "recommended bad", "no recommendation"
GOODNESS_COLORS = {REC_GOOD: "#2a9d8f", REC_BAD: "#e76f51", NO_REC: NONE_COLOR}
# svm_table columns on screen: (name in the file, short name)
SVM_COLUMNS = [("CV_model_accuracy", "CV accuracy %"), ("CV_model_precision", "CV precision %"),
               ("CV_model_recall", "CV recall %"), ("Probability_of_good", "P(good)"),
               ("Avg_Perf_all_instances", "mean perf."),
               ("Avg_Perf_selected_instances", "mean perf. pred. good")]
EXPORT_DERIVED = ["z_1", "z_2", "NumGoodAlgos", "IsBetaEasy", "best_algo", "best_algo_or_tie",
                  "n_tied_best", "best_algo_svm"]
# derived columns of IsResult.instances offered as colors: (label, type)
DERIVED = {
    "NumGoodAlgos": ("number of good algorithms", NUMERIC),
    "IsBetaEasy": ("beta-easy", CATEGORICAL),
    "best_algo_or_tie": ("best observed algorithm", CATEGORICAL),
    "n_tied_best": ("algorithms tied for the best", NUMERIC),
    "best_algo_svm": ("recommended by PYTHIA", CATEGORICAL),
}
# categorical colors whose values are algorithm names: fixed algorithm colors
ALGO_VALUED = ("best_algo_or_tie", "best_algo_svm")


class TabTitle(JSComponent):
    """Writes `text` into document.title in the browser (see TITLE in the docstring)."""

    text = param.String(default="")

    _esm = """
    export function render({ model }) {
      const apply = () => { if (model.text) document.title = model.text }
      apply()
      model.on("text", apply)
    }
    """


class GlobalState(param.Parameterized):
    """Single state of the interface: every tab reads from here."""

    dataset = param.Selector(default=None, objects=[], doc="""
        Key '<origin>/<folder>' of the engine output (is/iris, runs/<name>_<date>)""")
    result = param.Parameter(default=None, doc="IsResult of the active dataset")
    selection = param.Parameter(default=None, doc="""
        None = no selection; frozenset of Row labels, empty = empty selection""")
    color = param.String(default="", doc="column of IsResult.instances that colors the points")


def color_catalog(r) -> dict:
    """{column of r.instances: (label, group, type)} of the color variables."""
    cat = {}
    for col, kind in r.annotations.items():
        if kind != IDENTIFIER:              # identifiers do not color points
            cat[col] = (col, "Annotations", kind)
    if r.source_column is not None:
        cat[r.source_column] = ("source", "Annotations", CATEGORICAL)
    for f in r.features:
        cat[f"feature_{f}"] = (f, "Features in PILOT", NUMERIC)
    for f in r.features_outside_pilot:
        cat[f"feature_{f}"] = (f, "Features outside PILOT (SIFTED)", NUMERIC)
    for a in r.algos:
        cat[f"algo_{a}"] = (f"algo_{a}", "Performance (algo_*)", NUMERIC)
    for col, (label, kind) in DERIVED.items():
        cat[col] = (label, "Derived", kind)
    return cat


def categorical_groups(r) -> list:
    """Columns of r.instances usable for grouping: categorical annotations and source."""
    cols = [c for c, t in r.annotations.items() if t == CATEGORICAL]
    return cols + ([r.source_column] if r.source_column else [])


def default_color(r, cat) -> str:
    """First categorical annotation; else the first colorable annotation; else NumGoodAlgos."""
    for col, kind in r.annotations.items():
        if kind == CATEGORICAL and col in cat:
            return col
    return next((c for c in r.annotations if c in cat), "NumGoodAlgos")


def select_groups(cat) -> dict:
    """{group: {label: column}} for pn.widgets.Select(groups=...); a label
    repeated across groups (an annotation named like a feature) becomes the column."""
    count = {}
    for label, _, _ in cat.values():
        count[label] = count.get(label, 0) + 1
    groups = {}
    for col, (label, group, _) in cat.items():
        groups.setdefault(group, {})[label if count[label] == 1 else col] = col
    return groups


def color_style(values: pd.Series, kind: str, fixed=None) -> dict:
    if kind == CATEGORICAL:
        if fixed is not None:
            present = set(values)
            return dict(cmap={k: v for k, v in fixed.items() if k in present},
                        colorbar=False, show_legend=True)
        n = values.nunique()
        cmap = "Category10" if n <= 10 else ("Category20" if n <= 20 else "glasbey")
        return dict(cmap=cmap, colorbar=False, show_legend=True)
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(v) and bool(np.all(np.mod(v, 1) == 0)):
        lo, hi = int(v.min()), int(v.max())
        if 0 < hi - lo < MAX_DISCRETE_LEVELS:
            # integer values (counts): one color per value, ticks only on integers
            return dict(cmap="viridis", colorbar=True, show_legend=False, clim=(lo - 0.5, hi + 0.5),
                        color_levels=hi - lo + 1,
                        colorbar_opts={"ticker": FixedTicker(ticks=list(range(lo, hi + 1)))})
    return dict(cmap="viridis", colorbar=True, show_legend=False)


def footprint_polygons(fp, color, label, fill_alpha=0.25, dashed=False):
    """loader_is Footprint -> hv.Polygons (one geometry per Part, with holes)."""
    geoms = [{"x": p.exterior[:, 0], "y": p.exterior[:, 1], "holes": [list(p.holes)]}
             for p in fp.polygons]
    # explicit show_legend: in HoloViews 1.23 Polygons and Path start without a legend
    style = dict(fill_color=color, line_color=color, fill_alpha=fill_alpha, line_width=1.5,
                 line_alpha=0.85, show_legend=True)
    if dashed:
        style.update(fill_alpha=fill_alpha / 2, line_dash="dashed", line_width=2)
    return hv.Polygons(geoms, label=label).opts(**style)


def cloister_hv(r) -> list:
    """CLOISTER boundary (and the pruned one, if different) as hv.Path."""
    if r.bounds is None:
        return []
    ring = np.vstack([r.bounds.exterior, r.bounds.exterior[:1]])
    layers = [hv.Path([ring], label="CLOISTER").opts(
        color="black", line_width=2, line_dash="dashed", show_legend=True)]
    if r.bounds_pruned is not None and not np.array_equal(r.bounds.exterior, r.bounds_pruned.exterior):
        ring = np.vstack([r.bounds_pruned.exterior, r.bounds_pruned.exterior[:1]])
        layers.append(hv.Path([ring], label="CLOISTER (pruned)").opts(
            color="#6c757d", line_width=1.5, line_dash="dotted", show_legend=True))
    return layers


def _pct(x) -> str:
    return f"{100 * x:.1f}%"


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


class IsaApp:
    """Instance space interface: one instance per browser session."""

    def __init__(self, root=IS_DIR, runs=RUNS_DIR):
        self.root = Path(root)
        self.runs = Path(runs)
        self.state = GlobalState()
        groups = self._list_datasets()
        if not self.datasets:
            raise FileNotFoundError(f"no folder with run_info.json in {self.root} or {self.runs}")
        self._cache = {}
        self._data = None            # r.instances with Row as a column (base of the plots)
        self._catalog = {}
        self._building = False       # updating widget options: ignore the events
        self._doc = None             # session document (selection debounce)
        self._generation = 0         # streams of old plots are ignored
        self._index = []             # latest Selection1D of the tab 0 scatter
        self._pending = None
        self._plot_selection = None  # latest selection written by the scatter itself
        self._forced_types = {}      # {dataset: {metadata column: type}}, this session only
        self._group_chosen = False   # "group by" chosen by the user (otherwise, the default)

        # --- header: dataset name (pane) and browser tab title
        self.header = pn.pane.HTML("", margin=(0, 10))
        self.tab_title = TabTitle(text=TITLE, width=0, height=0, margin=0)

        # --- fixed sidebar
        self.w_dataset = pn.widgets.Select(name="Dataset", groups=groups, value=self.datasets[0],
                                           width=CONTROL_WIDTH - 20)
        self.new_space = NewSpacePanel(self.runs, CONTROL_WIDTH - 40, self._on_run_done)
        self.w_reload = pn.widgets.Button(name="Reload dataset", width=CONTROL_WIDTH - 20)
        self.info = pn.pane.Markdown("", width=CONTROL_WIDTH - 20)
        self.sel_info = pn.pane.Markdown("", width=CONTROL_WIDTH - 20)
        self.w_clear = pn.widgets.Button(name="Clear selection", width=CONTROL_WIDTH - 20)
        self.types_card = pn.Card(title="Inferred types", collapsed=True, visible=False,
                                  width=CONTROL_WIDTH - 20, margin=(5, 10))
        width = dict(width=CONTROL_WIDTH - 20)
        self.w_exp_all = pn.widgets.FileDownload(
            callback=self._csv_all, filename="instances.csv",
            label="Export instances (all)", **width)
        self.w_exp_sel = pn.widgets.FileDownload(
            callback=self._csv_selection, filename="selection.csv", label="Export selection",
            disabled=True, **width)
        self.w_exp_fp = pn.widgets.FileDownload(
            callback=self._csv_footprint, filename="footprint.csv",
            label="Export footprint labels", disabled=True, **width)
        self.exp_note = pn.pane.Markdown("", **width)

        # --- tab 0
        self.w_color = pn.widgets.Select(name="Point color", width=CONTROL_WIDTH - 110)
        self.r2 = pn.pane.Markdown("", width=90, margin=(28, 0, 0, 5))
        self.r2_warning = pn.pane.Markdown("", width=CONTROL_WIDTH - 20)
        self.w_ov_fp = pn.widgets.Checkbox(name="Footprints")
        self.w_ov_algo = pn.widgets.Select(name="Algorithm", options=[ALL], width=CONTROL_WIDTH - 40)
        self.w_ov_type = pn.widgets.RadioButtonGroup(options=["good", "best"], value="good")
        self.w_ov_cloister = pn.widgets.Checkbox(name="CLOISTER boundary")
        self.w_ov_hard = pn.widgets.Checkbox(name="Hard footprint (instances that are not beta-easy)")
        self.ov_note = pn.pane.Markdown("", width=CONTROL_WIDTH - 20)
        self.space_status = pn.pane.Markdown("")
        self.space_plot = pn.pane.HoloViews(sizing_mode="stretch_width", min_height=640)

        # --- tab 1
        self.w_fp_algo = pn.widgets.Select(name="Algorithm", options=[ALL], width=CONTROL_WIDTH - 20)
        self.w_fp_type = pn.widgets.RadioButtonGroup(options=["good", "best"], value="good")
        self.fp_warning = pn.pane.Markdown("", width=CONTROL_WIDTH - 20)
        self.fp_status = pn.pane.Markdown("")
        self.fp_plot = pn.pane.HoloViews(sizing_mode="stretch_width", min_height=460)
        self.fp_table = pn.widgets.Tabulator(
            pd.DataFrame(), disabled=True, layout="fit_columns", configuration=TABLE_CONFIG,
            sizing_mode="stretch_width", height=280, show_index=True, selectable=False)

        # --- tab 2 (Algorithm Selection)
        self.w_as_color = pn.widgets.Select(name="Color by", options=AS_COLORS,
                                            value=AS_RECOMMENDED, width=CONTROL_WIDTH - 20)
        self.w_as_algo = pn.widgets.Select(name="Algorithm (pr0_sub)", width=CONTROL_WIDTH - 20,
                                           visible=False)
        self.w_as_cloister = pn.widgets.Checkbox(name="CLOISTER boundary", value=True)
        self.as_status = pn.pane.Markdown("")
        self.as_warning = pn.Column(sizing_mode="stretch_width")
        self.as_summary = pn.pane.Markdown("")
        self.as_plot = pn.pane.HoloViews(sizing_mode="stretch_width", min_height=520)
        self.as_table = pn.widgets.Tabulator(
            pd.DataFrame(), disabled=True, layout="fit_columns", configuration=TABLE_CONFIG,
            sizing_mode="stretch_width", show_index=False, pagination=None, selectable=False)
        self.as_table_note = pn.pane.Markdown("")
        self.as_confusion = pn.FlexBox(sizing_mode="stretch_width")
        self.as_confusion_note = pn.pane.Markdown("")

        # --- tab 3 (Distributions)
        self.w_dist_vars = pn.widgets.MultiChoice(name="Variables", width=CONTROL_WIDTH - 20)
        self.w_dist_group = pn.widgets.Select(name="Group by", options=[NO_GROUP],
                                              width=CONTROL_WIDTH - 20)
        self.w_dist_type = pn.widgets.RadioButtonGroup(options=DIST_TYPES, value="histogram")
        self.dist_status = pn.pane.Markdown("")
        self.dist_plots = pn.Column(sizing_mode="stretch_width")

        # --- tab 4 (Features)
        self.feat_summary = pn.pane.Markdown("")
        self.feat_table = pn.widgets.Tabulator(
            pd.DataFrame(), disabled=True, layout="fit_data_stretch", configuration=TABLE_CONFIG,
            sizing_mode="stretch_width", show_index=False, pagination=None, selectable=False,
            widths={"reason": REASON_WIDTH}, formatters={"reason": {"type": "textarea"}})
        self.feat_heatmap = pn.pane.HoloViews(sizing_mode="stretch_width")
        self.feat_silhouette = pn.Column(sizing_mode="stretch_width")

        # --- tab 5 (Data Explorer)
        self.w_ex_x = pn.widgets.Select(name="X axis", width=CONTROL_WIDTH - 20)
        self.w_ex_y = pn.widgets.Select(name="Y axis", width=CONTROL_WIDTH - 20)
        self.w_ex_color = pn.widgets.Select(name="Color", width=CONTROL_WIDTH - 20)
        self.w_ex_query = pn.widgets.TextInput(
            name="Filter (pandas query)", width=CONTROL_WIDTH - 20,
            placeholder="e.g. z_1 > 0 and NumGoodAlgos <= 3")
        self.ex_filter = pn.pane.Markdown("", width=CONTROL_WIDTH - 20)
        self.w_ex_use = pn.widgets.Button(name="Use filter as selection", button_type="primary",
                                          width=CONTROL_WIDTH - 20, disabled=True)
        self.ex_status = pn.pane.Markdown("")
        self.ex_plot = pn.pane.HoloViews(sizing_mode="stretch_width", min_height=460)
        self.ex_table_title = pn.pane.Markdown("")
        self.ex_table = pn.widgets.Tabulator(
            pd.DataFrame(), disabled=True, layout="fit_data_stretch", sizing_mode="stretch_width",
            height=300, show_index=False, pagination="local", page_size=25)
        self._ex_filtered = None     # labels that pass the filter (None = invalid/empty filter)

        # --- sidebar and tabs
        self.controls = [
            pn.Column("## Instance Space", pn.Row(self.w_color, self.r2), self.r2_warning,
                      "### Overlay", self.w_ov_fp, self.w_ov_algo, self.w_ov_type,
                      self.w_ov_cloister, self.w_ov_hard, self.ov_note, width=CONTROL_WIDTH),
            pn.Column("## Footprint Performance", self.w_fp_algo, "### Type", self.w_fp_type,
                      self.fp_warning, width=CONTROL_WIDTH),
            pn.Column("## Algorithm Selection", self.w_as_color, self.w_as_algo, self.w_as_cloister,
                      pn.pane.Markdown(
                          "What **PYTHIA** and **CLOISTER** produced.\n\n"
                          "- **Recommended** (selection0): among the algorithms whose final SVM "
                          "predicts *good* for the instance, the one with the highest "
                          "cross-validation precision; *none* when no SVM predicts good.\n"
                          "- **Recommended good / bad**: whether the recommended algorithm is "
                          "good for the instance in the observed performance "
                          "(algorithm_bin.csv).\n"
                          "- **Best observed**: highest observed performance; *tie* when several "
                          "algorithms share it (portfolio.csv breaks those ties at random).\n"
                          "- **Probabilities**: pr0_sub = P(bad) **out of sample** "
                          "(cross-validation), not the final model's (pr0_hat).\n"
                          "- **CLOISTER**: estimated boundary of the region where plausible "
                          "instances can exist.", width=CONTROL_WIDTH - 20),
                      width=CONTROL_WIDTH),
            pn.Column("## Distributions", self.w_dist_vars,
                      pn.pane.Markdown(f"_Up to {MAX_DIST_VARS} variables at a time._"),
                      self.w_dist_group, "### Type", self.w_dist_type,
                      pn.pane.Markdown(f"_With {MIN_VIOLIN_GROUPS} groups or more the default "
                                       "is violin: overlaid histograms become unreadable._"),
                      width=CONTROL_WIDTH),
            pn.Column("## Features", pn.pane.Markdown(
                "One row per received feature.\n\n"
                "- **kept**: went into PILOT;\n"
                "- **dropped_degenerate**: dropped before the engine by the metadata "
                "generator (degenerate_report.csv);\n"
                "- **dropped_correlation**: no significant correlation with the performance "
                "(SIFTED);\n"
                "- **dropped_redundancy**: another feature of the same cluster was kept "
                "(SIFTED).", width=CONTROL_WIDTH - 20), width=CONTROL_WIDTH),
            pn.Column("## Data Explorer", self.w_ex_x, self.w_ex_y, self.w_ex_color,
                      self.w_ex_query, self.ex_filter, self.w_ex_use, width=CONTROL_WIDTH),
        ]
        self.sidebar = pn.Column(
            self.w_dataset, self.w_reload, self.new_space.card, self.info, self.types_card,
            self.sel_info, self.w_clear, pn.pane.Markdown("### Export", margin=(0, 10)),
            self.w_exp_all, self.w_exp_sel, self.w_exp_fp, self.exp_note,
            pn.layout.Divider(), self.controls[0], width=CONTROL_WIDTH,
        )
        self._swap = len(self.sidebar.objects) - 1   # block that changes with the tab
        self.tabs = pn.Tabs(
            (TABS[0], pn.Column(self.space_status, self.space_plot, sizing_mode="stretch_width")),
            (TABS[1], pn.Column(self.fp_status, self.fp_plot,
                                pn.pane.Markdown("### footprint_performance.csv"), self.fp_table,
                                sizing_mode="stretch_width")),
            (TABS[2], pn.Column(self.as_status, self.as_warning, self.as_summary, self.as_plot,
                                pn.pane.Markdown("### SVM performance (svm_table.csv)"),
                                self.as_table, self.as_table_note,
                                pn.pane.Markdown("### Cross-validation confusion matrices "
                                                 "(positive = good)"),
                                self.as_confusion_note, self.as_confusion,
                                sizing_mode="stretch_width")),
            (TABS[3], pn.Column(self.dist_status, self.dist_plots, sizing_mode="stretch_width")),
            (TABS[4], pn.Column(self.feat_summary, self.feat_table,
                                pn.pane.Markdown("### SIFTED correlations (feature × algorithm)"),
                                self.feat_heatmap, pn.pane.Markdown("### Silhouette per k (SIFTED)"),
                                self.feat_silhouette, sizing_mode="stretch_width")),
            (TABS[5], pn.Column(self.ex_status, self.ex_plot, self.ex_table_title,
                                self.ex_table, sizing_mode="stretch_width")),
            dynamic=True, sizing_mode="stretch_width",
        )

        # --- wiring
        self.state.param.watch(self._on_dataset, "dataset")
        self.w_dataset.param.watch(self._on_widget_dataset, "value")
        self.state.param.watch(self._on_state, ["result", "selection", "color"])
        self.tabs.param.watch(self._on_tab, "active")
        self.w_reload.on_click(self._on_reload)
        self.w_clear.on_click(lambda _: setattr(self.state, "selection", None))
        for w in (self.w_color, self.w_ex_color):
            w.param.watch(self._on_widget_color, "value")
        for w in (self.w_ov_fp, self.w_ov_algo, self.w_ov_type, self.w_ov_cloister, self.w_ov_hard):
            w.param.watch(lambda _: self._refresh_space(), "value")
        for w in (self.w_fp_algo, self.w_fp_type):
            w.param.watch(lambda _: (self._refresh_footprints(), self._refresh_export()), "value")
        for w in (self.w_dist_vars, self.w_dist_type):
            w.param.watch(lambda _: None if self._building else self._refresh_distributions(), "value")
        self.w_dist_group.param.watch(self._on_dist_group, "value")
        for w in (self.w_as_color, self.w_as_algo, self.w_as_cloister):
            w.param.watch(lambda _: None if self._building else self._refresh_algo_selection(), "value")
        for w in (self.w_ex_x, self.w_ex_y, self.w_ex_query):
            w.param.watch(lambda _: self._refresh_explorer(), "value")
        self.w_ex_use.on_click(self._on_use_filter)

        self.state.dataset = self.datasets[0]

    # ------------------------------------------------------------------ data
    def _list_datasets(self):
        """Relist resultados/is and runs; return the selector groups
        {origin: {folder: key}} (runs: most recent first)."""
        groups = {}
        for origin, folder in (("is", self.root), ("runs", self.runs)):
            names = list_available(folder) if folder.is_dir() else []
            if origin == "runs":
                names.sort(key=lambda n: (folder / n / "run_info.json").stat().st_mtime, reverse=True)
            if names:
                groups[ORIGINS[origin]] = {n: f"{origin}/{n}" for n in names}
        self.datasets = [k for g in groups.values() for k in g.values()]
        self.state.param.dataset.objects = self.datasets
        return groups

    def _folder(self, key):
        origin, _, name = key.partition("/")
        return (self.runs if origin == "runs" else self.root) / name

    @staticmethod
    def _name(key):
        return key.partition("/")[2]

    def _load(self, key, force=False):
        forced = self._forced_types.get(key, {})
        k = (key, tuple(sorted(forced.items())))
        if force:
            self._cache = {c: v for c, v in self._cache.items() if c[0] != key}
        if k not in self._cache:
            self._cache[k] = load_is_output(self._folder(key), annotation_types=forced or None)
        return self._cache[k]

    def _on_dataset(self, event):
        self._switch_dataset(event.new)

    def _on_widget_dataset(self, event):
        if event.new in self.datasets and event.new != self.state.dataset:
            self.state.dataset = event.new

    def _on_reload(self, _):
        self.w_dataset.groups = self._list_datasets()
        if self.state.dataset not in self.datasets:
            self.state.dataset = self.datasets[0]
        else:
            self._switch_dataset(self.state.dataset, force=True)

    def _on_run_done(self, path):
        """A run of the New instance space block finished: relist and open it."""
        self.w_dataset.groups = self._list_datasets()
        key = f"runs/{Path(path).name}"
        if key in self.datasets:
            self.state.dataset = key
            if pn.state.notifications is not None:
                pn.state.notifications.success(f"New instance space: {Path(path).name}", duration=6000)

    def _switch_dataset(self, key, force=False, keep_selection=False):
        name = self._name(key)
        if self.w_dataset.value != key:
            self.w_dataset.value = key
        r = self._load(key, force)
        self._data = r.instances.reset_index()
        self._catalog = color_catalog(r)
        color = self.state.color if self.state.color in self._catalog else default_color(r, self._catalog)
        self._set_options(r)
        selection = self.state.selection if keep_selection else None
        self._plot_selection = None
        # a single round of watchers: result, selection (reset when the
        # dataset changes) and a valid color
        self.state.param.update(result=r, selection=selection, color=color)
        self.header.object = f'<span style="font-size:1.25em;color:white">— {name}</span>'
        self.tab_title.text = f"{TITLE} - {name}"
        self._refresh_info()
        self._build_types(r)

    # ---------------------------------------------------------- inferred types
    def _build_types(self, r):
        """One selector per annotation whose type was inferred (or changed here)."""
        meta = {v: k for k, v in r.annotation_renames.items()}
        cols = [c for c, o in r.annotation_origins.items() if o in (INFERRED, FORCED)]
        rows = [pn.pane.Markdown(
            "_Not declared in annotations.json: the type was guessed. A change only "
            "applies to this session._", width=CONTROL_WIDTH - 50)]
        for col in cols:
            current = r.annotations[col]
            w = pn.widgets.RadioButtonGroup(
                name=col, options={"numeric": NUMERIC, "categorical": CATEGORICAL,
                                   "identifier": IDENTIFIER},
                value=current if current in (CATEGORICAL, IDENTIFIER) else NUMERIC,
                button_type="light")
            w.param.watch(partial(self._on_type, meta.get(col, col)), "value")
            rows.append(pn.Column(pn.pane.Markdown(f"`{col}`", margin=(0, 10)), w))
        self.types_card.objects = rows
        self.types_card.title = f"Inferred types ({len(cols)})"
        self.types_card.visible = bool(cols)

    def _on_type(self, column, event):
        key = self.state.dataset
        self._forced_types.setdefault(key, {})[column] = event.new
        self._switch_dataset(key, keep_selection=True)

    def _set_options(self, r):
        """Options of every selector for the dataset, without triggering redraws."""
        self._building = True
        try:
            groups = select_groups(self._catalog)
            for w in (self.w_color, self.w_ex_color):
                w.groups = groups
            algos = [ALL] + list(r.algos)
            for w in (self.w_ov_algo, self.w_fp_algo):
                w.options = algos
                if w.value not in algos:
                    w.value = ALL
            self.w_as_algo.options = list(r.algos)
            if self.w_as_algo.value not in r.algos:
                self.w_as_algo.value = r.algos[0]
            numeric = self._numeric_columns(r)
            dist = [c for c in numeric if c not in ("z_1", "z_2")]
            self.w_dist_vars.options = dist
            kept = [v for v in self.w_dist_vars.value if v in dist]
            self.w_dist_vars.value = kept or [self._default_variable(r)]
            groups = [NO_GROUP] + categorical_groups(r)
            self.w_dist_group.options = groups
            if not self._group_chosen or self.w_dist_group.value not in groups:
                # default: the first categorical annotation
                self.w_dist_group.value = groups[1] if len(groups) > 1 else NO_GROUP
            self.w_dist_type.value = self._default_dist_type()
            for w, default in ((self.w_ex_x, "z_1"), (self.w_ex_y, "z_2")):
                w.options = numeric
                if w.value not in numeric:
                    w.value = default
        finally:
            self._building = False

    def _group_values(self, col):
        """Group labels as text (missing = 'none')."""
        return self._data[col].map(lambda v: NONE if _is_missing(v) else str(v)).astype(str)

    def _default_dist_type(self):
        col = self.w_dist_group.value
        if col == NO_GROUP or col not in self._data.columns:
            return "histogram"
        return "violin" if self._group_values(col).nunique() >= MIN_VIOLIN_GROUPS else "histogram"

    def _on_dist_group(self, event):
        if self._building:
            return
        self._group_chosen = True
        self._building = True
        try:
            self.w_dist_type.value = self._default_dist_type()
        finally:
            self._building = False
        self._refresh_distributions()

    def _default_variable(self, r):
        """First continuous numeric annotation (non-integer values: avoids ids
        and counts, such as row_original); otherwise the first PILOT feature."""
        for col, kind in r.annotations.items():
            v = pd.to_numeric(self._data[col], errors="coerce").dropna()
            if kind == NUMERIC and len(v) and (np.mod(v, 1) != 0).any():
                return col
        return f"feature_{r.features[0]}"

    def _numeric_columns(self, r):
        cols = ["z_1", "z_2"]
        cols += [c for c, t in r.annotations.items() if is_numeric_type(t)]
        cols += [f"feature_{f}" for f in r.features_all]
        cols += [f"algo_{a}" for a in r.algos] + ["NumGoodAlgos", "n_tied_best"]
        return [c for c in dict.fromkeys(cols) if c in self._data.columns
                and pd.api.types.is_numeric_dtype(self._data[c])]

    def _color_values(self, col, rows=None):
        """Values of the color variable: text for categorical ones (missing = 'none')."""
        s = self._data[col] if rows is None else self._data.loc[rows, col]
        if self._catalog[col][2] == CATEGORICAL:
            return s.map(lambda v: NONE if _is_missing(v) else str(v)).astype(str)
        return pd.to_numeric(s, errors="coerce")

    def _fixed_colors(self, col):
        """Algorithm colors (the same as the footprints) for columns whose values
        are algorithm names; gray for tie and none."""
        if col not in ALGO_VALUED:
            return None
        algos = self.state.result.algos
        colors = {a: ALGO_PALETTE[i % len(ALGO_PALETTE)] for i, a in enumerate(algos)}
        return {**colors, TIE: TIE_COLOR, NONE: NONE_COLOR}

    # ------------------------------------------------------------- selection
    def _positions(self, selection):
        if not selection:
            return []
        return np.flatnonzero(self._data["Row"].isin(selection)).tolist()

    def _mask(self):
        """Boolean array per instance, or None without a selection."""
        sel = self.state.selection
        if sel is None:
            return None
        return self._data["Row"].isin(sel).to_numpy()

    def _alphas(self):
        m = self._mask()
        if m is None:
            return np.full(len(self._data), ALPHA_NEUTRAL)
        return np.where(m, ALPHA_SEL, ALPHA_OUT)

    def _selection_text(self, context):
        """Status line shared by the tabs (the text is checked by the tests)."""
        sel, n = self.state.selection, len(self._data)
        if sel is None:
            return f"**No selection.** {context['none']}"
        if not sel:
            return (f"**Empty selection**: no instance was selected (0 of {n}). "
                    f"{context['empty']}")
        return f"**{len(sel)} instances selected** of {n}. {context['some']}"

    def _on_index(self, generation, index=None, **_):
        if generation != self._generation:
            return
        self._index = list(index or [])
        self._schedule(SELECTION_WAIT_MS)

    def _on_geometry(self, generation, **_):
        if generation == self._generation:
            self._schedule(GEOMETRY_WAIT_MS)

    def _on_reset(self, generation, resetting=False, **_):
        if generation == self._generation and resetting:
            self._plot_selection = None
            self.state.selection = None

    def _schedule(self, ms):
        if self._doc is None:
            self._store_selection()
            return
        if self._pending is not None:
            try:
                self._doc.remove_timeout_callback(self._pending)
            except ValueError:
                pass
        self._pending = self._doc.add_timeout_callback(self._store_selection, ms)

    def _store_selection(self):
        self._pending = None
        labels = frozenset(self._data["Row"].iloc[self._index])
        self._plot_selection = labels
        self.state.selection = labels

    def _on_use_filter(self, _):
        if self._ex_filtered is not None:
            self.state.selection = frozenset(self._ex_filtered)

    # ----------------------------------------------------------------- color
    def _on_widget_color(self, event):
        if not self._building and event.new in self._catalog:
            self.state.color = event.new

    def _sync_color(self):
        self._building = True
        try:
            for w in (self.w_color, self.w_ex_color):
                w.value = self.state.color
        finally:
            self._building = False
        self._refresh_r2()

    def _refresh_r2(self):
        r, col = self.state.result, self.state.color
        tab = r.pilot_r2
        value = None
        if col.startswith("feature_"):
            f = col[len("feature_"):]
            if f not in r.features:
                self.r2.object = "r² —"
                self.r2_warning.object = "_Outside PILOT (dropped by SIFTED): r² does not apply._"
                return
            row = tab[(tab["kind"] == "feature") & (tab["variable"] == f)]
        elif col.startswith("algo_") and col[len("algo_"):] in r.algos:
            row = tab[(tab["kind"] == "algorithm") & (tab["variable"] == col[len("algo_"):])]
        else:
            self.r2.object, self.r2_warning.object = "", ""
            return
        if len(row):
            value = float(row["r2"].iloc[0])
        if value is None:
            self.r2.object, self.r2_warning.object = "r² ?", ""
            return
        self.r2.object = f"r² **{value:.2f}**"
        self.r2_warning.object = (
            f"⚠ _Low PILOT r² (< {LOW_R2:.1f}): the plane explains little of this variable; "
            "the color pattern may not show in the projection._" if value < LOW_R2 else "")

    # -------------------------------------------------------------- dispatch
    def _on_state(self, *events):
        names = {e.name for e in events}
        if self._data is None:
            return
        if names & {"result", "color"}:
            self._sync_color()
        self._refresh_sel_info()
        external = "selection" in names and self.state.selection != self._plot_selection
        if names & {"result", "color"} or external:
            self._refresh_space()
        else:
            self.space_status.object = self._space_status()
        if names & {"result", "selection"}:
            self._refresh_footprints()
            self._refresh_algo_selection()
            self._refresh_distributions()
        if "result" in names:
            self._refresh_features()
        self._refresh_explorer()
        self._refresh_export()

    def _on_tab(self, event):
        self.sidebar[self._swap] = self.controls[event.new]

    def _refresh_info(self):
        r = self.state.result
        rob = r.run_info.get("trace_robustness", {})
        origin = self.state.dataset.partition("/")[0]
        rows = [
            f"**{r.name}** _({'runs' if origin == 'runs' else 'resultados/is'})_",
            f"Instances: **{r.n}**",
            f"Features in PILOT: **{r.n_features_used}** of {len(r.features_all)}",
            f"Algorithms: **{len(r.algos)}**",
        ]
        if r.run_info.get("good_rule"):
            rows.append(f"Rule: `{r.run_info['good_rule']}`")
        if rob.get("jitter_applied"):
            rows.append(f"_TRACE with jitter on {rob['perturbed_points']} instances "
                        "(coordinates_trace.csv)_")
        self.info.object = "  \n".join(rows)

    def _refresh_sel_info(self):
        sel = self.state.selection
        if sel is None:
            text = "Selection: **none**"
        elif not sel:
            text = "Selection: **empty** (0 instances)"
        else:
            text = f"Selection: **{len(sel)}** instances"
        self.sel_info.object = text

    # --------------------------------------------------- tab 0: instance space
    def _space_status(self):
        return self._selection_text({
            "none": "The lasso is active: draw on the plot to select.",
            "empty": "The lasso did not catch any point.",
            "some": "Highlighted in the plot and used in the other tabs.",
        })

    def _overlays(self):
        r = self.state.result
        layers, notes = [], []
        if self.w_ov_hard.value:
            fp = r.footprint_hard
            if fp.polygons:
                layers.append(footprint_polygons(fp, "#343a40", "hard (not beta-easy)", 0.15))
            else:
                notes.append("- hard footprint empty")
        if self.w_ov_fp.value:
            kind = self.w_ov_type.value
            algos = r.algos if self.w_ov_algo.value == ALL else [self.w_ov_algo.value]
            for a in algos:
                fp = r.footprints[(a, kind)]
                color = ALGO_PALETTE[r.algos.index(a) % len(ALGO_PALETTE)]
                if fp.status == EMPTY:
                    notes.append(f"- **{a} / {kind}**: empty")
                    continue
                if fp.status == SUSPECT:
                    notes.append(f"- **{a} / {kind}**: suspect (purity {fp.purity:.2f}), dashed")
                layers.append(footprint_polygons(fp, color, f"{a} {kind}", 0.22, fp.status == SUSPECT))
        if self.w_ov_cloister.value:
            layers += cloister_hv(r)
        self.ov_note.object = "\n".join(notes)
        return layers

    def _refresh_space(self):
        """Rebuild the tab 0 scatter (color, dataset, overlays or a selection
        coming from outside). The current selection comes back as `selected`,
        and the new Selection1D starts with those indices."""
        col = self.state.color
        label, _, kind = self._catalog[col]
        data = self._data[["Row", "z_1", "z_2"]].copy()
        data["color_value"] = self._color_values(col).to_numpy()
        positions = self._positions(self.state.selection)
        self._generation += 1
        generation = self._generation
        self._index = positions
        hover = HoverTool(tooltips=[("Row", "@Row"), (label, "@color_value")])
        style = dict(
            color="color_value", size=6, alpha=0.85, nonselection_alpha=0.12, line_color=None,
            tools=["lasso_select", "box_select", hover], active_tools=["lasso_select"],
            responsive=True, min_height=620, show_grid=True, xlabel="z_1", ylabel="z_2",
            title=f"Instance space (PILOT) — color: {label}",
        )
        style.update(color_style(data["color_value"], kind, self._fixed_colors(col)))
        if positions:
            style["selected"] = positions
        points = hv.Points(data, ["z_1", "z_2"], [hv.Dimension("color_value", label=label), "Row"]
                           ).opts(**style)
        Selection1D(source=points, index=positions).add_subscriber(partial(self._on_index, generation))
        for stream in (Lasso(source=points), BoundsXY(source=points)):
            stream.add_subscriber(partial(self._on_geometry, generation))
        PlotReset(source=points).add_subscriber(partial(self._on_reset, generation))
        layers = self._overlays() + [points]
        self.space_plot.object = reduce(lambda a, b: a * b, layers).opts(
            legend_position="right", legend_opts={"click_policy": "hide"},
            responsive=True, min_height=620)
        self.space_status.object = self._space_status()

    # ------------------------------------------------------ tab 1: footprints
    def _refresh_footprints(self):
        r = self.state.result
        algo, kind = self.w_fp_algo.value, self.w_fp_type.value
        data = self._data[["Row", "z_1", "z_2"]].assign(opacity=self._alphas())
        hover = HoverTool(tooltips=[("Row", "@Row")])
        layers = [hv.Points(data, ["z_1", "z_2"], ["Row", "opacity"]).opts(
            color=BASE_COLOR, alpha="opacity", size=6, line_color=None, tools=[hover],
            show_legend=False)]
        warns = []
        algos = r.algos if algo == ALL else [algo]
        for a in algos:
            fp = r.footprints[(a, kind)]
            color = ALGO_PALETTE[r.algos.index(a) % len(ALGO_PALETTE)]
            if fp.status == EMPTY:
                warns.append(f"- **{a} / {kind}**: empty footprint, nothing to draw.")
                continue
            if fp.status == SUSPECT:
                warns.append(f"- **{a} / {kind}**: suspect, purity {fp.purity:.3f} < "
                             f"trace.purity {r.pi}; drawn dashed.")
            layers.append(footprint_polygons(fp, color, a, 0.25, fp.status == SUSPECT))
        self.fp_warning.object = ("Suspect/empty footprints:\n" + "\n".join(warns) if warns
                                  else "No suspect or empty footprint for this choice.")
        self.fp_plot.object = reduce(lambda a, b: a * b, layers).opts(
            responsive=True, min_height=460, show_grid=True, legend_position="right",
            legend_opts={"click_policy": "hide"}, title=f"Footprints {kind} — {algo}")
        table = r.footprint_performance.copy().round(4)
        table["status"] = [r.footprints[(a, kind)].status if (a, kind) in r.footprints else "?"
                           for a in table.index]
        # "Area_Good_Normalized" -> "Area Good Normalized": the headers can wrap
        self.fp_table.value = table.rename(columns=lambda c: c.replace("_", " "))
        self.fp_status.object = self._selection_text({
            "none": "Use the lasso in the Instance Space tab to highlight a subset on the map.",
            "empty": "No point highlighted on the map.",
            "some": "Highlighted on the map; the others appear faded.",
        })

    # --------------------------------------------- tab 2: algorithm selection
    def _algo_selection_data(self):
        """One row per instance: recommended (selection0, 'none'), best observed
        (or 'tie'), whether the recommended algorithm is good, its pr0_sub."""
        r = self.state.result
        rows = self._data["Row"]
        rec = r.pythia_selection["selection0"].reindex(rows.to_numpy()).to_numpy(dtype=object)
        has = np.array([not _is_missing(v) for v in rec])
        rec_txt = np.where(has, rec, NONE).astype(str)
        pr0 = r.pythia_proba.reindex(rows.to_numpy())
        pr0_rec = np.full(len(rows), np.nan)
        rec_good = np.zeros(len(rows), dtype=bool)
        for a in r.algos:
            m = rec_txt == a
            pr0_rec[m] = pr0[a].to_numpy(dtype=float)[m]
            rec_good[m] = self._data[f"algo_{a}_bin"].to_numpy(dtype=bool)[m]
        goodness = np.where(~has, NO_REC, np.where(rec_good, REC_GOOD, REC_BAD))
        best = self._data["best_algo_or_tie"].map(lambda v: NONE if _is_missing(v) else str(v))
        return pd.DataFrame({
            "Row": rows.to_numpy(), "z_1": self._data["z_1"].to_numpy(),
            "z_2": self._data["z_2"].to_numpy(), "recommended": rec_txt,
            "best": best.to_numpy(), "goodness": goodness, "pr0_rec": pr0_rec,
            "rec_good": np.where(has, np.where(rec_good, "yes", "no"), "—"),
        })

    def _refresh_algo_selection(self):
        r = self.state.result
        if r is None or self._data is None:
            return
        d = self._algo_selection_data()
        n = len(d)
        d["opacity"] = self._alphas()
        mode = self.w_as_color.value
        self.w_as_algo.visible = mode == AS_PR0
        counts = d["recommended"].value_counts()

        # nearly trivial selector (algorithms only, not 'none')
        rec_algos = counts.drop(NONE, errors="ignore")
        warns = []
        if len(rec_algos) and rec_algos.iloc[0] / n > TRIVIAL_SELECTOR:
            a, k = rec_algos.index[0], int(rec_algos.iloc[0])
            warns.append(pn.pane.Alert(
                f"⚠ **Nearly trivial selector:** PYTHIA recommends **{a}** for {k} of {n} "
                f"instances ({_pct(k / n)}, above {_pct(TRIVIAL_SELECTOR)}). Always "
                f"recommending {a} would give almost the same result: the map says little "
                "about regions where each algorithm is better.", alert_type="warning",
                css_classes=["as-trivial"], sizing_mode="stretch_width"))
        self.as_warning.objects = warns

        c = d["goodness"].value_counts()
        ties = int((self._data["n_tied_best"] > 1).sum())
        self.as_summary.object = (
            f"The recommended algorithm is good for the instance in **{int(c.get(REC_GOOD, 0))}** "
            f"of {n} instances, bad in **{int(c.get(REC_BAD, 0))}**; no recommendation in "
            f"**{int(c.get(NO_REC, 0))}**. Ties for the best observed value: **{ties}** instances "
            "(shown as *tie*; portfolio.csv breaks them at random).  \n"
            "_Probabilities in this tab: **pr0_sub**, P(bad) out of sample (PYTHIA "
            "cross-validation)._")

        # map
        algo_colors = {a: ALGO_PALETTE[i % len(ALGO_PALETTE)] for i, a in enumerate(r.algos)}
        hover = HoverTool(tooltips=[
            ("Row", "@Row"), ("recommended", "@recommended"), ("best observed", "@best"),
            ("recommended is good", "@rec_good"), ("pr0_sub of the recommended", "@pr0_rec{0.000}")])
        vdims = ["Row", "recommended", "best", "rec_good", "pr0_rec", "opacity"]
        style = dict(alpha="opacity", size=6, line_color=None, tools=[hover],
                     responsive=True, min_height=520, show_grid=True)
        if mode == AS_PR0:
            a = self.w_as_algo.value
            d["color_value"] = r.pythia_proba[a].reindex(d["Row"].to_numpy()).to_numpy(dtype=float)
            label = f"pr0_sub {a}: P(bad) out of sample"
            style.update(color="color_value", cmap="RdYlGn_r", clim=(0, 1), colorbar=True,
                         show_legend=False, colorbar_opts={"title": "pr0_sub"})
            hover.tooltips = hover.tooltips + [(f"pr0_sub {a}", "@color_value{0.000}")]
            title = f"pr0_sub of {a} (P(bad) out of sample); green = likely good"
        else:
            if mode == AS_GOODNESS:
                col, colors = "goodness", GOODNESS_COLORS
                title = "Recommended algorithm (selection0): good or bad for the instance"
            else:
                col, colors = "recommended", {**algo_colors, NONE: NONE_COLOR}
                title = "Recommended algorithm (PYTHIA selection0)"
            # legend and drawing order: from the most frequent category to the
            # rarest (the rare ones stay on top and remain visible)
            cont = d[col].value_counts()
            order = list(cont.index)
            lab = {k: f"{k} ({int(cont[k])})" for k in order}
            d["color_value"] = d[col].map(lab)
            d = d.iloc[np.argsort(d[col].map({k: i for i, k in enumerate(order)}).to_numpy(),
                                  kind="stable")]
            cmap = {lab[k]: colors.get(k, NONE_COLOR) for k in order}
            label = "color"
            style.update(color="color_value", cmap=cmap, show_legend=True)
        points = hv.Points(d, ["z_1", "z_2"], [hv.Dimension("color_value", label=label), *vdims]
                           ).opts(**style)
        layers = [points]
        if self.w_as_cloister.value:
            layers += cloister_hv(r)
        self.as_plot.object = reduce(lambda a, b: a * b, layers).opts(
            responsive=True, min_height=520, legend_position="right", title=title,
            legend_opts={"click_policy": "hide"}, **NO_SCROLL_ZOOM)
        self.as_status.object = self._selection_text({
            "none": "Use the lasso in the Instance Space tab to highlight a subset on the map.",
            "empty": "No point highlighted on the map.",
            "some": "Highlighted on the map; the others appear faded. " + self._selection_summary(d),
        })
        self._refresh_svm_table(r, counts)
        self._refresh_confusion(r)

    def _selection_summary(self, d):
        m = self._mask()
        if m is None or not m.any():
            return ""
        c = d.loc[m, "goodness"].value_counts()
        top = d.loc[m, "recommended"].value_counts()
        return (f"In the selection: most recommended **{top.index[0]}** ({int(top.iloc[0])}); "
                f"recommended good {int(c.get(REC_GOOD, 0))}, bad {int(c.get(REC_BAD, 0))}, "
                f"no recommendation {int(c.get(NO_REC, 0))}.")

    def _refresh_svm_table(self, r, counts):
        tab = r.svm_table
        rows = []
        for name in tab.index:
            row = {"algorithm": str(name)}
            for col, short in SVM_COLUMNS:
                v = tab.loc[name, col] if col in tab.columns else np.nan
                row[short] = "—" if pd.isna(v) else (f"{v:.1f}" if short.endswith("%") else f"{v:.3f}")
            if name in r.algos:
                row["recommended in"] = int(counts.get(name, 0))
            elif name == "Selector":
                row["recommended in"] = int(counts.drop(NONE, errors="ignore").sum())
            else:
                row["recommended in"] = "—"
            rows.append(row)
        df = pd.DataFrame(rows)
        shorts = [short for _, short in SVM_COLUMNS]
        order = ["algorithm", *shorts[:3], "recommended in", *shorts[3:]]
        self.as_table.value = df[order].astype(str)
        self.as_table.height = 40 + 31 * len(df)
        self.as_table_note.object = (
            "- **accuracy, precision and recall**: cross-validation of each SVM (positive = good).\n"
            "- **recommended in**: instances where the algorithm is selection0.\n"
            "- **P(good)**: fraction of instances where the algorithm is good (Selector: with "
            "selection1, which replaces *none* by the algorithm with the highest P(good)).\n"
            "- **mean perf. pred. good**: mean `algo_*` where the SVM predicts good "
            "(Selector: that of the recommended algorithm).\n"
            "- **Oracle**: always the best observed algorithm of each instance.\n"
            "- **Selector**: the recommended algorithm. Precision = fraction of the instances with "
            "a recommendation where the recommended algorithm is good. Recall follows the "
            "PYTHIA/MATLAB definition, which counts as a miss every instance with some good "
            "algorithm that was not recommended; so it stays close to 50% when there are "
            "several good algorithms per instance.")

    def _refresh_confusion(self, r):
        conf = r.pythia_confusion
        panels = []
        for a in r.algos:
            if a not in conf.index:
                continue
            tn, fp, fn, tp = (int(conf.loc[a, k]) for k in ("tn", "fp", "fn", "tp"))
            cells = []
            for obs, pred, k, name in (("good", "good", tp, "TP"), ("good", "bad", fn, "FN"),
                                       ("bad", "good", fp, "FP"), ("bad", "bad", tn, "TN")):
                total = tp + fn if obs == "good" else fp + tn
                frac = k / total if total else 0.0
                cells.append({"predicted": pred, "observed": obs, "fraction": frac, "n": k,
                              "text": f"{name} {k}\n{_pct(frac)}" if total else f"{name} 0",
                              "text_color": "white" if frac >= 0.6 else "black"})
            c = pd.DataFrame(cells)
            acc = (tp + tn) / max(tp + tn + fp + fn, 1)
            heat = hv.HeatMap(c, ["predicted", "observed"], ["fraction", "n"]).opts(
                cmap="Blues", clim=(0, 1), colorbar=False, tools=["hover"], width=230,
                height=190, xlabel="predicted (CV)", ylabel="observed", toolbar=None,
                invert_yaxis=True, default_tools=[])
            labels = hv.Labels(c, ["predicted", "observed"], ["text", "text_color"]).opts(
                text_font_size="11pt", text_color="text_color")
            panels.append(pn.Column(
                pn.pane.Markdown(f"**{a}** · accuracy {_pct(acc)}", width=230, margin=(0, 10)),
                pn.pane.HoloViews(heat * labels, width=230, height=190),
                css_classes=["as-confusion"], margin=(5, 5)))
        self.as_confusion.objects = panels or [pn.pane.Markdown("No confusion matrices.")]
        self.as_confusion_note.object = (
            "Rows = observed performance, columns = the SVM's prediction in cross-validation. "
            "Color and percentage: fraction within the row (a dark diagonal means both classes "
            "are right). TP/FN/FP/TN: true positive, false negative, false positive, "
            "true negative.")

    # --------------------------------------------------- tab 3: distributions
    def _dist_series(self, vals, group_col, mask):
        """[(label, finite values, color, part)] per group; part is True
        (selected), False (not selected) or None (no split)."""
        n = len(vals)
        if group_col is None:
            groups = [("all", np.ones(n, dtype=bool), ALL_COLOR)]
        else:
            g = self._group_values(group_col).to_numpy()
            groups = [(c, g == c, GROUP_PALETTE[i % len(GROUP_PALETTE)])
                      for i, c in enumerate(sorted(set(g)))]
        split = mask is not None and bool(mask.any())
        series = []
        for name, idx, color in groups:
            parts = ((True, "selected"), (False, "not selected")) if split else ((None, None),)
            for part, text in parts:
                m = idx if part is None else idx & (mask if part else ~mask)
                v = vals[m]
                prefix = name if text is None else (text if group_col is None else f"{name} · {text}")
                if group_col is None and part is not None:
                    color = SEL_COLOR if part else ALL_COLOR
                series.append((f"{prefix} (n={int(m.sum())})", v[np.isfinite(v)], color, part))
        return groups, series

    def _dist_plot(self, var, group_col, kind, mask):
        vals = self._data[var].to_numpy(dtype=float)
        title = f"{var}  (All n={len(vals)}"
        title += f", Selected n={int(mask.sum())})" if mask is not None else ")"
        if group_col is not None:
            title += f" — by {group_col}"
        groups, series = self._dist_series(vals, group_col, mask)
        common = dict(title=title, responsive=True, show_grid=True, **NO_SCROLL_ZOOM)
        fin = vals[np.isfinite(vals)]
        if fin.size == 0:
            return hv.Curve([]).opts(title=f"{var}: no finite values")
        if kind == "violin":
            split = mask is not None and bool(mask.any())
            frames = []
            for name, idx, _ in groups:
                label = f"{name} (n={int(idx.sum())}"
                label += f"; {int((idx & mask).sum())} sel.)" if split else ")"
                for part in ((True, False) if split else (None,)):
                    m = idx if part is None else idx & (mask if part else ~mask)
                    v = vals[m]
                    v = v[np.isfinite(v)]
                    text = "all" if part is None else ("selected" if part else "not selected")
                    frames.append(pd.DataFrame({"group": label, "part": text, "value": v}))
            dfl = pd.concat(frames, ignore_index=True)
            if split:
                return hv.Violin(dfl, ["group", "part"], "value").opts(
                    split="part", violin_fill_color="part",
                    cmap={"selected": SEL_COLOR, "not selected": NOT_SEL_COLOR},
                    height=300, ylabel=var, xlabel="", **common)
            colors = {g: (ALL_COLOR if group_col is None else GROUP_PALETTE[i % len(GROUP_PALETTE)])
                      for i, g in enumerate(dict.fromkeys(dfl["group"]))}
            return hv.Violin(dfl, ["group"], "value").opts(
                violin_fill_color="group", cmap=colors, height=300, ylabel=var, xlabel="", **common)
        layers = []
        if kind == "histogram":
            edges = np.histogram_bin_edges(fin, bins=25)
            for label, v, color, part in series:
                if not v.size:
                    continue
                dens, _ = np.histogram(v, bins=edges, density=True)
                layers.append(hv.Histogram((edges, dens), label=label).opts(
                    fill_color=color, line_color=color, line_alpha=0.6,
                    fill_alpha={True: 0.6, False: 0.15, None: 0.4}[part],
                    line_dash="dashed" if part is False else "solid"))
            ylabel = "density"
        else:  # density (KDE)
            for label, v, color, part in series:
                if v.size < 2:
                    continue
                layers.append(hv.Distribution(v, label=label).opts(
                    fill_color=color, line_color=color, line_width=2,
                    fill_alpha={True: 0.4, False: 0.05, None: 0.25}[part],
                    line_dash="dashed" if part is False else "solid"))
            ylabel = "density (KDE)"
        if not layers:
            return hv.Curve([]).opts(title=f"{var}: no group with enough values")
        return hv.Overlay(layers).opts(height=260, xlabel=var, ylabel=ylabel,
                                       legend_position="right", **common)

    def _refresh_distributions(self):
        variables = list(self.w_dist_vars.value)
        mask = self._mask()
        group = self.w_dist_group.value
        group_col = None if group == NO_GROUP or group not in self._data.columns else group
        kind = self.w_dist_type.value
        colors = (" In the violins, red = selected and gray = not selected."
                  if kind == "violin" else " Dashed line = not selected.")
        self.dist_status.object = self._selection_text({
            "none": "Only the groups are shown. Use the lasso in the Instance Space tab to compare.",
            "empty": "Only the groups are shown (Selected n=0).",
            "some": "Each group is split into selected and not selected." + colors,
        })
        if not variables:
            self.dist_plots.objects = [pn.pane.Markdown("Choose at least one variable in the sidebar.")]
            return
        if len(variables) > MAX_DIST_VARS:
            self.dist_plots.objects = [pn.pane.Markdown(
                f"**{len(variables)} variables chosen.** The limit is {MAX_DIST_VARS} at a time.")]
            return
        self.dist_plots.objects = [
            pn.pane.HoloViews(self._dist_plot(v, group_col, kind, mask),
                              sizing_mode="stretch_width")
            for v in variables
        ]

    # -------------------------------------------------------- tab 4: features
    def _refresh_features(self):
        r = self.state.result
        table = r.features_table()
        n = table["status"].value_counts()
        parts = [f"**{len(table)} features received**: {n.get('kept', 0)} in PILOT"]
        for status, text in (("dropped_degenerate", "degenerate"),
                             ("dropped_correlation", "without correlation"),
                             ("dropped_redundancy", "redundant")):
            if n.get(status, 0):
                parts.append(f"{n[status]} {text}")
        summary = ", ".join(parts) + "."
        if r.degenerate_report is None:
            summary += ("  \n_No degenerate_report.csv: the table only shows what SIFTED did "
                        "with the metadata features._")
        self.feat_summary.object = summary
        shown = table.copy()
        shown["replaced_by"] = shown["replaced_by"].fillna("")
        shown["rho_algorithm"] = shown["rho_algorithm"].fillna("")
        for col in ("max_abs_rho", "r2_pilot"):
            shown[col] = shown[col].round(3)
        shown["pval"] = shown["pval"].map(lambda p: "" if pd.isna(p) else f"{p:.2g}")
        self.feat_table.value = shown.rename(columns=FEATURE_COLUMNS)
        # the reason column wraps: the height follows the number of lines
        lines = shown["reason"].fillna("").map(lambda s: max(1, -(-len(s) // REASON_CHARS_PER_LINE)))
        self.feat_table.height = int(45 + sum(8 + 20 * k for k in lines))

        c = r.sifted_correlations
        if c.empty:
            self.feat_heatmap.object = hv.Curve([]).opts(title="SIFTED without computed correlations")
        else:
            kept = set(r.features)
            order = [f for f in table["feature"] if f in set(c["feature"])]
            c = c.assign(
                feature=c["feature"].map(lambda f: f"{f} ✓" if f in kept else f),
                text=c["rho"].map(lambda v: f"{v:.2f}"),
                text_color=np.where(c["rho"].abs() >= 0.6, "white", "black"))
            order = [f"{f} ✓" if f in kept else f for f in order]
            c = c.set_index("feature").loc[order].reset_index()
            heat = hv.HeatMap(c, ["algorithm", "feature"], ["rho", "pval"]).opts(
                cmap="RdBu_r", clim=(-1, 1), colorbar=True, tools=["hover"], responsive=True,
                height=90 + 26 * len(order), xlabel="algorithm", ylabel="feature (✓ = in PILOT)",
                invert_yaxis=True, title="Pearson rho between the processed feature and the performance",
                **NO_SCROLL_ZOOM)
            labels = hv.Labels(c, ["algorithm", "feature"], ["text", "text_color"]).opts(
                text_font_size="8pt", text_color="text_color")
            self.feat_heatmap.object = heat * labels

        sil = r.sifted_silhouette
        if sil.empty:
            self.feat_silhouette.objects = [pn.pane.Markdown(
                "_SIFTED did not cluster (few features after the correlation filter): "
                "silhouette not computed._")]
            return
        used = sil.loc[sil["used"], "k"].tolist()
        best = sil.loc[sil["best"], "k"].tolist()
        layers = [hv.Curve(sil, "k", "silhouette").opts(color=BASE_COLOR),
                  hv.Scatter(sil, "k", "silhouette").opts(color=BASE_COLOR, size=7)]
        parts = []
        if used:
            layers.append(hv.VLine(used[0]).opts(color=SEL_COLOR, line_width=2))
            parts.append(f"k used = {used[0]} (red)")
        if best:
            layers.append(hv.VLine(best[0]).opts(color="#2a9d8f", line_dash="dashed", line_width=2))
            parts.append(f"highest silhouette: k = {best[0]} (dashed)")
        self.feat_silhouette.objects = [pn.pane.HoloViews(
            reduce(lambda a, b: a * b, layers).opts(
                responsive=True, height=260, show_grid=True, xlabel="k (clusters)",
                ylabel="silhouette", title="; ".join(parts), **NO_SCROLL_ZOOM),
            sizing_mode="stretch_width")]

    # ---------------------------------------------------------------- export
    def _export_table(self, labels=None):
        """Label (instances), source, annotations, every feature, algo_*, z and
        derived columns; only the rows of `labels` if given."""
        r = self.state.result
        cols = (["Row"] + ([r.source_column] if r.source_column else []) + list(r.annotations)
                + [f"feature_{f}" for f in r.features_all] + [f"algo_{a}" for a in r.algos]
                + EXPORT_DERIVED)
        data = self._data if labels is None else self._data[self._data["Row"].isin(labels)]
        return data[cols].rename(columns={"Row": "instances"})

    @staticmethod
    def _csv(df):
        return io.BytesIO(df.to_csv(index=False).encode("utf-8"))

    def _csv_all(self):
        return self._csv(self._export_table())

    def _csv_selection(self):
        return self._csv(self._export_table(self.state.selection or frozenset()))

    def _active_footprint(self):
        """Footprint chosen in the Footprint Performance tab, or None ('all')."""
        algo, kind = self.w_fp_algo.value, self.w_fp_type.value
        if algo == ALL or self.state.result is None:
            return None
        return self.state.result.footprints.get((algo, kind))

    def _csv_footprint(self):
        fp = self._active_footprint()
        labels = [] if fp is None else self.state.result.instances_in_footprint(fp)
        return self._csv(pd.DataFrame({"instances": labels}))

    def _refresh_export(self):
        name, sel = self._name(self.state.dataset), self.state.selection
        self.w_exp_all.filename = f"{name}_instances.csv"
        self.w_exp_sel.filename = f"{name}_selection.csv"
        self.w_exp_sel.disabled = sel is None
        self.w_exp_sel.label = "Export selection" if sel is None else f"Export selection ({len(sel)})"
        fp = self._active_footprint()
        algo, kind = self.w_fp_algo.value, self.w_fp_type.value
        self.w_exp_fp.disabled = fp is None or fp.status == EMPTY
        self.w_exp_fp.filename = f"{name}_footprint_{algo}_{kind}.csv"
        self.w_exp_fp.label = ("Export footprint labels" if fp is None
                               else f"Export footprint labels {algo}/{kind}")
        self.exp_note.object = (
            "_Active footprint: the one in the Footprint Performance tab; choose an algorithm there._"
            if fp is None else (f"_Footprint {algo}/{kind} is empty._" if fp.status == EMPTY else ""))

    # --------------------------------------------------- tab 5: data explorer
    def _refresh_explorer(self):
        if self._data is None:
            return
        data = self._data
        x, y, col = self.w_ex_x.value, self.w_ex_y.value, self.state.color
        if x is None or y is None or col not in self._catalog:
            return
        label, _, kind = self._catalog[col]
        q = (self.w_ex_query.value or "").strip()
        error = None
        if q:
            try:
                filtered = data.query(q)
            except Exception as exc:  # noqa: BLE001 -- user input
                error, filtered = f"{type(exc).__name__}: {exc}", data
        else:
            filtered = data
        if error:
            self.ex_filter.object = f"**Invalid query — filter ignored.**  \n`{error}`"
            self._ex_filtered = None
        else:
            self.ex_filter.object = f"**{len(filtered)} of {len(data)} rows** pass the filter."
            self._ex_filtered = list(filtered["Row"]) if q else None
        self.w_ex_use.disabled = self._ex_filtered is None

        rows = filtered.index
        mask = self._mask()
        alpha = self._alphas()[rows]
        plot = pd.DataFrame({
            "x_value": filtered[x].to_numpy(dtype=float), "y_value": filtered[y].to_numpy(dtype=float),
            "color_value": self._color_values(col, rows).to_numpy(), "Row": filtered["Row"].to_numpy(),
            "opacity": alpha,
        })
        hover = HoverTool(tooltips=[("Row", "@Row"), (x, "@x_value"), (y, "@y_value"),
                                    (label, "@color_value")])
        style = dict(color="color_value", alpha="opacity", size=6, line_color=None, tools=[hover],
                     responsive=True, min_height=460, show_grid=True, legend_position="right",
                     title=f"{x} x {y} — color: {label}")
        style.update(color_style(plot["color_value"], kind, self._fixed_colors(col)))
        self.ex_plot.object = hv.Points(
            plot, [hv.Dimension("x_value", label=x), hv.Dimension("y_value", label=y)],
            [hv.Dimension("color_value", label=label), "Row", "opacity"]).opts(**style)

        ids = [c for c, t in self.state.result.annotations.items() if t == IDENTIFIER]
        cols = list(dict.fromkeys(["Row", *ids, x, y, col, "best_algo_or_tie", "n_tied_best",
                                   "best_algo_svm"]))
        table = filtered[cols].copy()
        table.insert(1, "selected", False if mask is None else mask[rows])
        self.ex_table.value = table.round(4)
        self.ex_table_title.object = f"### Filtered rows ({len(filtered)})"
        self.ex_status.object = self._selection_text({
            "none": "All points with the same opacity.",
            "empty": "All points appear faded.",
            "some": "Highlighted in the plot and marked in the *selected* column.",
        })

    # -------------------------------------------------------------- template
    def render(self):
        self._doc = pn.state.curdoc
        return pn.template.FastListTemplate(
            site="", title=TITLE, header=[self.header, self.tab_title],
            sidebar=[self.sidebar], main=[self.tabs], sidebar_width=CONTROL_WIDTH + 30,
            header_background="#0466C8", theme_toggle=False,
        )


def make_app(root=IS_DIR, runs=RUNS_DIR):
    """Factory used by pn.serve: one instance per browser session."""
    return IsaApp(root, runs).render()


def start(port=5006, show=True, threaded=False, root=IS_DIR, runs=RUNS_DIR):
    """Start the server. With threaded=True, return the server thread."""
    return pn.serve(
        lambda: make_app(root, runs), port=port, show=show, title=TITLE,
        websocket_origin=[f"localhost:{port}", f"127.0.0.1:{port}"], threaded=threaded,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Instance space interface")
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument("--no-show", action="store_true", help="do not open the browser")
    parser.add_argument("--root", default=str(IS_DIR), help="resultados/is folder")
    parser.add_argument("--runs", default=str(RUNS_DIR),
                        help="folder for the runs launched from the interface")
    args = parser.parse_args(argv)
    start(port=args.port, show=not args.no_show, root=Path(args.root), runs=Path(args.runs))


if __name__ == "__main__":
    main()
