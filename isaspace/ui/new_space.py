"""The "New instance space" sidebar block: upload of a metadata.csv, immediate
validation, choice of the performance rule and engine run in a subprocess.

- Validation (isaspace.ui.upload) on every uploaded file; errors and warnings
  become text on screen, never a traceback.
- The performance direction (higher or lower is better) and the threshold
  type (absolute or relative) start WITHOUT a value: the Run button is only
  enabled after the choice and an epsilon. With the rule chosen, the preview
  shows the fraction of good instances per algorithm (the PRELIM rule) and
  warns when some algorithm is below 5% or above 95% good: a degenerate
  epsilon threshold. Before the Run button, the algorithms are ranked by the
  mean of their algo_* column under the chosen direction, so the user can see
  the consequence of the choice; nothing tries to guess the direction.
- The run happens in a subprocess (isaspace.ui.runner) and writes
  runs/<name>_<YYYYMMDD-HHMMSS>/; the progress per stage comes from the
  subprocess output, read by a thread that hands the updates to the session's
  document (Document.add_next_tick_callback, the only Bokeh method that is
  safe outside the server thread). The UI stays usable while it runs.

This module does not import the engine or instancespace.
"""

from pathlib import Path

import numpy as np
import panel as pn
from panel.io.state import set_curdoc

from isaspace.ui import runner
from isaspace.ui.upload import (
    GOOD_MAX, GOOD_MIN, TIME_WARNING_MIN_INSTANCES, TIMING_ALGOS, TIMING_FEATURES,
    degenerate_algorithms, estimated_time, good_fraction, mean_ranking, prelim_rows,
    validate_annotations, validate_feature_info, validate_metadata,
)

CHOOSE = "— choose —"
DIRECTIONS = {CHOOSE: "", "higher is better": "max", "lower is better": "min"}
THRESHOLDS = {CHOOSE: "", "absolute": "abs", "relative to the instance's best": "rel"}
K_DEFAULT = 6                # SiftedOptions.k in instancespace 0.3.0
RULES = {
    ("max", "abs"): "good for the algorithm if `algo_*` ≥ ε",
    ("min", "abs"): "good for the algorithm if `algo_*` ≤ ε",
    ("max", "rel"): "good if `algo_*` is at most ε·100% below the instance's best: "
                    "`(1 − algo_*/best) ≤ ε`",
    ("min", "rel"): "good if `algo_*` is at most ε·100% above the instance's best: "
                    "`(algo_*/best − 1) ≤ ε`",
}
UPSIDE_DOWN = "If this ranking looks upside down for your problem, the direction is probably wrong."


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def format_time(s: float) -> str:
    if s < 90:
        return f"{max(5, round(s / 5) * 5):.0f} s"
    return f"{s / 60:.0f} min"


class NewSpacePanel:
    """Sidebar card; `on_done(path)` runs in the session when a run finishes well."""

    def __init__(self, runs, width, on_done):
        self.runs = Path(runs)
        self.on_done = on_done
        self.validation = None       # upload.Validation of the current metadata
        self._aux_errors = []        # annotations.json / feature_info.csv
        self.run = None              # runner.Run in progress or the last one
        self._clock = None
        W = dict(width=width)

        self.w_meta = pn.widgets.FileInput(accept=".csv", css_classes=["new-metadata"], **W)
        self.w_ann = pn.widgets.FileInput(accept=".json", css_classes=["new-annotations"], **W)
        self.w_finfo = pn.widgets.FileInput(accept=".csv", css_classes=["new-feature-info"], **W)
        self.w_clear = pn.widgets.Button(name="Clear files", button_type="light", **W)
        self.validation_msgs = pn.Column(**W)
        self.summary = pn.pane.Markdown("", **W)
        self.w_name = pn.widgets.TextInput(name="Run name", **W)
        self.w_direction = pn.widgets.Select(name="Performance direction", options=DIRECTIONS, **W)
        self.w_threshold = pn.widgets.Select(name="Threshold", options=THRESHOLDS, **W)
        self.w_eps = pn.widgets.FloatInput(name="ε (epsilon)", value=None, step=0.05, **W)
        self.rule = pn.pane.Markdown("", **W)
        self.preview = pn.pane.Markdown("", **W)
        self.preview_warning = pn.Column(**W)
        self.w_k = pn.widgets.IntInput(name="SIFTED k (feature clusters)", value=K_DEFAULT,
                                       start=2, **W)
        self.w_usesim = pn.widgets.Checkbox(
            name="trace.usesim: footprints from the PYTHIA predictions", value=False)
        self.advanced = pn.Card(
            self.w_k, pn.pane.Markdown(
                f"_instancespace default: k = {K_DEFAULT}. With fewer surviving features "
                "than k, SIFTED does not cluster._", width=width - 20),
            self.w_usesim, pn.pane.Markdown(
                "_Off (this interface's default): footprints of the observed performance._",
                width=width - 20),
            title="Advanced options", collapsed=True, width=width, margin=(5, 0))
        self.time = pn.Column(**W)
        self.ranking = pn.pane.Markdown("", css_classes=["new-ranking"], **W)
        self.w_run = pn.widgets.Button(name="Run ISA", button_type="primary", disabled=True, **W)
        self.missing = pn.pane.Markdown("", **W)
        self.progress = pn.indicators.Progress(max=len(runner.STAGES), value=0,
                                               visible=False, **W)
        self.status = pn.pane.Markdown("", css_classes=["new-status"], **W)

        def label(text):
            return pn.pane.Markdown(text, margin=(8, 10, 0, 10), **W)

        self.card = pn.Card(
            label("**metadata.csv** (required)"), self.w_meta,
            label("annotations.json (optional)"), self.w_ann,
            label("feature_info.csv (optional)"), self.w_finfo, self.w_clear,
            self.validation_msgs, self.summary, self.w_name,
            pn.pane.Markdown("### Performance rule", margin=(0, 10)),
            self.w_direction, self.w_threshold, self.w_eps, self.rule, self.preview,
            self.preview_warning, self.advanced, self.time, self.ranking, self.w_run,
            self.missing, self.progress, self.status,
            title="New instance space", collapsed=True, width=width + 20, margin=(5, 10),
            css_classes=["new-card"])

        for w in (self.w_meta, self.w_ann, self.w_finfo):
            w.param.watch(lambda _: self._validate(), "value")
        for w in (self.w_direction, self.w_threshold, self.w_eps):
            w.param.watch(lambda _: self._update_rule(), "value")
        self.w_name.param.watch(lambda _: self._update_button(), "value")
        self.w_clear.on_click(self._on_clear)
        self.w_run.on_click(self._on_run)
        self._validate()

    # ------------------------------------------------------------ validation
    def _on_clear(self, _):
        for w in (self.w_meta, self.w_ann, self.w_finfo):
            w.clear()
        self._validate()

    def _validate(self):
        self.validation, self._aux_errors = None, []
        errors, warns = [], []
        if self.w_meta.value:
            try:
                v = validate_metadata(self.w_meta.value)
            except Exception as exc:  # noqa: BLE001 -- never a traceback on screen
                v = None
                errors.append(f"Could not read the metadata ({type(exc).__name__}: {exc}).")
            if v is not None:
                self.validation = v
                errors += v.errors
                warns += v.warnings
                if self.w_ann.value:
                    try:
                        _, e = validate_annotations(self.w_ann.value, v)
                    except Exception as exc:  # noqa: BLE001
                        e = [f"Unreadable annotations.json ({type(exc).__name__}: {exc})."]
                    self._aux_errors += e
                if self.w_finfo.value:
                    try:
                        e = validate_feature_info(self.w_finfo.value, v)
                    except Exception as exc:  # noqa: BLE001
                        e = [f"Unreadable feature_info.csv ({type(exc).__name__}: {exc})."]
                    self._aux_errors += e
                errors += self._aux_errors
            if not self.w_name.value and self.w_meta.filename:
                self.w_name.value = runner.safe_name(Path(self.w_meta.filename).stem)
        elif self.w_ann.value or self.w_finfo.value:
            warns.append("Upload the metadata.csv too.")
        blocks = []
        if errors:
            blocks.append(pn.pane.Alert(
                "**Invalid metadata, fix it and upload it again:**\n\n"
                + "\n".join(f"- {e}" for e in errors), alert_type="danger",
                css_classes=["new-errors"], sizing_mode="stretch_width"))
        if warns:
            blocks.append(pn.pane.Alert("\n".join(f"- {w}" for w in warns), alert_type="warning",
                                        css_classes=["new-warnings"], sizing_mode="stretch_width"))
        self.validation_msgs.objects = blocks
        v = self.validation
        if v is not None and not errors:
            ann = ", ".join(v.annotations) if v.annotations else "none"
            self.summary.object = (
                f"✔ **{v.n} instances**, {len(v.features)} features, {len(v.algos)} "
                f"algorithms. Annotations: {ann}."
                + (f" Source: `{v.source}`." if v.source else ""))
        else:
            self.summary.object = ""
        self._update_time()
        self._update_rule()

    def _ok(self) -> bool:
        return self.validation is not None and self.validation.ok and not self._aux_errors

    # ------------------------------------------------------------------ rule
    def chosen_rule(self):
        """(higher_is_better, absolute, epsilon), or None while something is missing."""
        d, lim, eps = self.w_direction.value, self.w_threshold.value, self.w_eps.value
        if not d or not lim or eps is None or not np.isfinite(eps):
            return None
        return d == "max", lim == "abs", float(eps)

    def options(self) -> dict:
        higher, absolute, eps = self.chosen_rule()
        return {"perf": {"max_perf": higher, "abs_perf": absolute, "epsilon": eps},
                "sifted": {"k": int(self.w_k.value or K_DEFAULT)},
                "trace": {"use_sim": bool(self.w_usesim.value)}}

    def _update_rule(self):
        d, lim = self.w_direction.value, self.w_threshold.value
        v = self.validation
        text = RULES.get((d, lim), "")
        if text and self._ok():
            y = prelim_rows(v).to_numpy(dtype=float)
            if np.isfinite(y).any():
                text += f"  \n`algo_*` ranges from {np.nanmin(y):.4g} to {np.nanmax(y):.4g}."
            if lim == "rel" and (y < 0).any():
                text += "  \n⚠ Some `algo_*` values are negative: the ratio to the best loses its meaning."
        self.rule.object = text
        self._update_ranking()
        rule = self.chosen_rule()
        if rule is None or not self._ok():
            self.preview.object = ""
            self.preview_warning.objects = []
            self._update_button()
            return
        fr = good_fraction(prelim_rows(v), *rule)
        degenerate = degenerate_algorithms(fr)
        rows = ["| algorithm | good instances |", "|:--|--:|"]
        rows += [f"| {a} | {_pct(f)}{' ⚠' if a in degenerate else ''} |" for a, f in fr.items()]
        self.preview.object = ("**Preview (PRELIM rule):** fraction of good instances\n\n"
                               + "\n".join(rows))
        if degenerate:
            self.preview_warning.objects = [pn.pane.Alert(
                f"⚠ **Degenerate ε threshold.** {', '.join(degenerate)}: fewer than "
                f"{_pct(GOOD_MIN)} or more than {_pct(GOOD_MAX)} of the instances are good with "
                "this ε, so *good* is almost constant for them and their footprints and PYTHIA "
                "classifiers carry little information. Consider another ε. This check says "
                "nothing about the direction: see the ranking before the Run button.",
                alert_type="warning", css_classes=["new-degenerate-eps"],
                sizing_mode="stretch_width")]
        else:
            self.preview_warning.objects = []
        self._update_button()

    def _update_ranking(self):
        """Ranking by the mean of each algo_* column under the chosen direction."""
        d, v = self.w_direction.value, self.validation
        if not d or not self._ok():
            self.ranking.object = ""
            return
        higher = d == "max"
        rank = mean_ranking(prelim_rows(v), higher)
        rows = ["| # | algorithm | mean `algo_*` |", "|--:|:--|--:|"]
        rows += [f"| {i} | {a} | {m:.4g} |" for i, (a, m) in enumerate(rank.items(), 1)]
        best, worst = rank.index[0], rank.index[-1]
        self.ranking.object = (
            f"**Consequence of the direction** ({'higher' if higher else 'lower'} is better): "
            f"algorithms ranked by their mean `algo_*`.\n\n" + "\n".join(rows) + "\n\n"
            f"Best under this direction: **{best}** (mean {rank.iloc[0]:.4g}); "
            f"worst: **{worst}** (mean {rank.iloc[-1]:.4g}).  \n"
            f"⚠ _{UPSIDE_DOWN}_")

    def _update_time(self):
        v = self.validation
        if not self._ok():
            self.time.objects = []
            return
        t = estimated_time(v.n, len(v.algos))
        if t is None:
            self.time.objects = []
            return
        basis = (f"measured on this computer on synthetic metadata with {TIMING_FEATURES} "
                 f"features and {TIMING_ALGOS} algorithms; PYTHIA dominates and grows with the "
                 "number of algorithms")
        if v.n >= TIME_WARNING_MIN_INSTANCES:
            self.time.objects = [pn.pane.Alert(
                f"⏱ **Large file: {v.n} instances × {len(v.algos)} algorithms.** Estimated "
                f"time: **~{format_time(t)}** ({basis}). The interface stays usable while it "
                "runs.", alert_type="info", css_classes=["new-time"],
                sizing_mode="stretch_width")]
        else:
            self.time.objects = [pn.pane.Markdown(
                f"_Estimated time: ~{format_time(t)} ({basis})._", css_classes=["new-time"])]

    def _update_button(self):
        running = self.run is not None and not self.run.finished
        missing = []
        if not self._ok():
            missing.append("a valid metadata")
        if not self.w_direction.value:
            missing.append("the performance direction")
        if not self.w_threshold.value:
            missing.append("the threshold type")
        if self.w_eps.value is None:
            missing.append("ε")
        self.w_run.disabled = bool(missing) or running
        self.missing.object = ("_To run, still missing: " + ", ".join(missing) + "._"
                               if missing and not running else "")

    # ------------------------------------------------------------------- run
    def _in_session(self, doc, function):
        """Wrap `function` so it can be called from the reader thread and run
        in the server thread, inside the session's document."""
        def schedule(*args):
            def cb():
                with set_curdoc(doc):
                    function(*args)
            try:
                doc.add_next_tick_callback(cb)
            except Exception:  # noqa: BLE001 -- session closed: nothing to update
                pass
        return schedule if doc is not None else function

    def _on_run(self, _):
        if self.w_run.disabled or self.chosen_rule() is None or not self._ok():
            return
        try:
            path = runner.new_run_dir(self.w_name.value or "metadata", self.runs)
            meta = runner.write_inputs(
                path, self.w_meta.value, self.w_ann.value or None, self.w_finfo.value or None)
        except OSError as exc:
            self.status.object = f"**Could not create the run folder:** {exc}"
            return
        doc = pn.state.curdoc
        self.run = runner.launch(
            meta, path, self.options(),
            on_stage=self._in_session(doc, self._on_stage),
            on_finish=self._in_session(doc, self._on_finish))
        self.progress.value, self.progress.visible = 0, True
        self.status.object = f"Starting in `runs/{path.name}`…"
        if doc is not None and doc.session_context is not None:
            self._clock = pn.state.add_periodic_callback(self._tick, period=1000)
        self._update_button()

    def _stage_text(self, run):
        stage = run.stage
        if stage is None:
            return f"Starting… · {run.duration:.0f} s"
        i = runner.STAGES.index(stage) if stage in runner.STAGES else 0
        return f"Running **{stage}** (stage {i + 1} of {len(runner.STAGES)}) · {run.duration:.0f} s"

    def _tick(self):
        run = self.run
        if run is not None and not run.finished:
            self.status.object = self._stage_text(run)

    def _on_stage(self, run, stage):
        if run is not self.run:
            return
        if stage in runner.STAGES:
            self.progress.value = runner.STAGES.index(stage)
        self.status.object = self._stage_text(run)

    def _on_finish(self, run):
        if run is not self.run:
            return
        if self._clock is not None:
            self._clock.stop()
            self._clock = None
        path = f"runs/{run.path.name}"
        if run.ok:
            self.progress.value = len(runner.STAGES)
            self.status.object = f"✔ **Done** in {run.duration:.0f} s: `{path}`."
            self.on_done(run.path)
        else:
            self.progress.visible = False
            self.status.object = (f"✖ **Failed** ({run.stage or 'before the first stage'}): "
                                  f"{run.error}  \n_Full output in `{path}/{runner.LOG}`._")
        self._update_button()
