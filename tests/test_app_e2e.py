"""End-to-end tests of isaspace.ui.app in a real Chromium (Playwright).

Starts the app once (python -m isaspace.ui.app, with the pytest interpreter)
and opens a NEW page for each test, hence a new server session and a new
global state. Most tests run on the four datasets of resultados/is/. The
lasso is drawn with the mouse, inside the plot area, from screen coordinates
computed by BokehJS itself.

The server uses a temporary runs/ folder (--runs): the tests of the "New
instance space" block run the real engine, in a subprocess, and the runs stay
there. The instancespace example metadata is downloaded from GitHub (tag
v0.3.0) to the pytest cache; without network, that test is skipped.

Needs the .venv-isa with playwright and the Playwright Chromium
(python -m playwright install chromium). Usage, from the root:
    .venv-isa/bin/python -m pytest tests/test_app_e2e.py
    .venv-isa/bin/python -m pytest tests/ -m "not e2e and not slow"   (only the fast ones)
"""

import json
import math
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sync_api = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parents[1]
IS_DIR = ROOT / "resultados" / "is"
DATASETS = ["iris", "diabetes", "blood-transfusion-service-center", "hill-valley"]
TITLE = "isa-instance"
TIMEOUT = 25  # s per wait
RUN_TIMEOUT = 180   # s for the engine to finish (example ~5 s, cycle ~10 s)
EXAMPLE_URL = ("https://raw.githubusercontent.com/andremun/pyInstanceSpace/v0.3.0/"
               "examples/data/metadata.csv")
# instances with a tie for the best observed value
TIES = {"iris": 123, "diabetes": 217, "blood-transfusion-service-center": 215, "hill-valley": 54}
UPSIDE_DOWN = "If this ranking looks upside down for your problem, the direction is probably wrong."

pytestmark = pytest.mark.e2e

# rendered Bokeh plots (only the active tab is in the DOM: Tabs dynamic=True)
PLOTS_JS = r"""() => {
  const out = [];
  for (const v of Bokeh.index.all_views()) {
    const m = v.model;
    if (!m || !v.frame || !m.renderers || !m.title) continue;
    const rs = [];
    for (const r of m.renderers) {
      if (!r.data_source || !r.glyph || !r.data_source.data) continue;
      const ds = r.data_source;
      const data = ds.data instanceof Map ? Object.fromEntries(ds.data) : ds.data;
      const cols = Object.keys(data);
      const n = cols.length ? data[cols[0]].length : 0;
      const g = r.glyph, fa = g.fill_alpha, fc = g.fill_color;
      let hi = null;
      // per-point opacity; if all are equal, HoloViews writes a scalar
      if (fa && fa.field !== undefined && data[fa.field]) hi = [...data[fa.field]].filter(a => a > 0.5).length;
      else if (fa && typeof fa.value === 'number') hi = fa.value > 0.5 ? n : 0;
      rs.push({glyph: g.type, n, sel: [...ds.selected.indices].length, hi, has_row: 'Row' in data,
               mapper: fc && fc.transform ? fc.transform.type : null});
    }
    const tb = m.toolbar;
    const xr = m.x_range;
    out.push({title: m.title.text || "", rs,
              factors: xr && xr.factors ? [...xr.factors].map(String) : null,
              active: tb && tb.active_drag && tb.active_drag.type ? tb.active_drag.type : null});
  }
  return out;
}"""
# points of the tab 0 scatter in page coordinates, and the frame rectangle
SCREEN_JS = r"""(prefix) => {
  for (const v of Bokeh.index.all_views()) {
    const m = v.model;
    if (!m || !v.frame || !m.title || !(m.title.text || "").startsWith(prefix)) continue;
    for (const r of m.renderers) {
      const d0 = r.data_source && r.data_source.data;
      if (!d0) continue;
      const data = d0 instanceof Map ? Object.fromEntries(d0) : d0;
      if (!('Row' in data) || !('z_1' in data)) continue;
      const fb = v.frame.bbox, cb = v.canvas_view.el.getBoundingClientRect();
      const pts = [...data.z_1].map((x, i) => [cb.left + v.frame.x_scale.compute(x),
                                               cb.top + v.frame.y_scale.compute(data.z_2[i])]);
      return {frame: [cb.left + fb.x0, cb.top + fb.y0, cb.left + fb.x1, cb.top + fb.y1], pts};
    }
  }
  return null;
}"""
# Row labels of the instances selected in the Instance Space scatter
SELECTED_JS = r"""(prefix) => {
  for (const v of Bokeh.index.all_views()) {
    const m = v.model;
    if (!m || !v.frame || !m.title || !(m.title.text || "").startsWith(prefix)) continue;
    for (const r of m.renderers) {
      const ds = r.data_source;
      if (!ds || !ds.data) continue;
      const data = ds.data instanceof Map ? Object.fromEntries(ds.data) : ds.data;
      if (!('Row' in data)) continue;
      return [...ds.selected.indices].map(i => String(data.Row[i]));
    }
  }
  return null;
}"""
# values of a column of the points renderer (the one with Row) of a plot
COLUMN_JS = r"""([prefix, col]) => {
  for (const v of Bokeh.index.all_views()) {
    const m = v.model;
    if (!m || !v.frame || !m.title || !(m.title.text || "").startsWith(prefix)) continue;
    for (const r of m.renderers) {
      const d0 = r.data_source && r.data_source.data;
      if (!d0) continue;
      const data = d0 instanceof Map ? Object.fromEntries(d0) : d0;
      if (!('Row' in data) || !(col in data)) continue;
      return [...data[col]].map(x => x === null ? null : String(x));
    }
  }
  return null;
}"""
# columns of every table (Tabulator) of the document that has `column`
TABLE_JS = r"""(column) => {
  for (const m of Bokeh.documents[0]._all_models.values()) {
    if (m.type !== 'ColumnDataSource' || !m.data) continue;
    const data = m.data instanceof Map ? Object.fromEntries(m.data) : m.data;
    if (column in data && 'status' in data) {
      const out = {};
      for (const k of Object.keys(data)) out[k] = [...data[k]].map(v => v === null ? null : String(v));
      return out;
    }
  }
  return null;
}"""
# text of a Panel component, going through the shadow roots (inner_text does not)
TEXT_JS = r"""(e) => {
  const f = (n) => {
    let s = n.shadowRoot ? f(n.shadowRoot) : "";
    for (const c of n.childNodes) {
      if (c.nodeType === 3) s += c.textContent;
      else if (c.nodeType === 1 && ["STYLE", "SCRIPT"].includes(c.tagName)) continue;
      else if (c.nodeType === 1 || c.nodeType === 11) s += (c.tagName === "BR" ? "\n" : "") + f(c) + " ";
    }
    return s;
  };
  return f(e).replace(/[ \t]+/g, " ");
}"""
# rows (cell texts) of the Markdown tables inside a Panel component
TABLE_ROWS_JS = r"""(e) => {
  const roots = [e], rows = [];
  while (roots.length) {
    const n = roots.shift();
    if (n.shadowRoot) roots.push(n.shadowRoot);
    for (const c of n.children || []) roots.push(c);
    if (n.tagName === "TR" && n.querySelector("td"))
      rows.push([...n.querySelectorAll("td")].map(td => td.textContent.trim()));
  }
  return rows;
}"""
RE_STATUS = re.compile(r"No selection\.|Empty selection|\d+ instances selected")
SPACE = "Instance space"
EXPLORER = "z_1 x z_2"   # title of the Data Explorer scatter with the default axes
RECOMMENDED = "Recommended algorithm"


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def runs_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("runs")


@pytest.fixture(scope="session")
def server(runs_dir):
    if not all((IS_DIR / d / "run_info.json").is_file() for d in DATASETS):
        pytest.skip("resultados/is incomplete (run scripts/run_is_all.py in the .venv-isa)")
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "isaspace.ui.app", "--no-show", "--port", str(port),
         "--runs", str(runs_dir)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    url = f"http://localhost:{port}/"
    start = time.time()
    while True:
        if proc.poll() is not None:
            pytest.fail("the app died on startup:\n" + proc.stdout.read().decode()[-3000:])
        try:
            if urllib.request.urlopen(url, timeout=2).status == 200:
                break
        except OSError:
            pass
        if time.time() - start > 90:
            proc.kill()
            pytest.fail("the app did not answer in 90 s")
        time.sleep(0.5)
    yield url
    proc.terminate()
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"Playwright Chromium unavailable: {exc}")
        yield b
        b.close()


class Screen:
    """App page with the steps the tests use."""

    def __init__(self, page):
        self.page = page

    # ---------------------------------------------------------------- waits
    def wait_for(self, cond, msg, timeout=TIMEOUT):
        end = time.time() + timeout
        last = None
        while time.time() < end:
            try:
                last = cond()
                if last:
                    return last
            except Exception as exc:  # noqa: BLE001 -- DOM in transition
                last = exc
            self.page.wait_for_timeout(200)
        raise AssertionError(f"{msg} (last value: {last!r})")

    # -------------------------------------------------------------- reading
    def plots(self):
        return self.page.evaluate(PLOTS_JS)

    def plot(self, prefix):
        found = [p for p in self.plots() if p["title"].startswith(prefix)]
        return found[0] if found else None

    def points(self, prefix):
        """Points renderer (it has the Row column) of the plot with this title."""
        p = self.plot(prefix)
        if p is None:
            return None
        return next((r for r in p["rs"] if r["has_row"]), None)

    def column(self, prefix, col):
        return self.page.evaluate(COLUMN_JS, [prefix, col])

    def status(self):
        loc = self.page.get_by_text(RE_STATUS)
        return loc.first.inner_text() if loc.count() else ""

    def n_status(self):
        m = re.search(r"(\d+) instances selected", self.status())
        if m:
            return int(m.group(1))
        return 0 if "Empty selection" in self.status() else None

    def sidebar_selection(self):
        return self.page.get_by_text(re.compile(r"^Selection: ")).first.inner_text()

    def select_value(self, label):
        return self.page.get_by_label(label, exact=True).evaluate(
            "e => e.options[e.selectedIndex].text")

    def text(self, css_class):
        """Text of the first component with the css_class ('' if none)."""
        loc = self.page.locator(f".{css_class}")
        return loc.first.evaluate(TEXT_JS) if loc.count() else ""

    def table_rows(self, css_class):
        loc = self.page.locator(f".{css_class}")
        return loc.first.evaluate(TABLE_ROWS_JS) if loc.count() else []

    # -------------------------------------------------------------- actions
    def open(self, url, dataset):
        self.page.goto(url)
        self.page.wait_for_function("window.Bokeh && Bokeh.documents.length > 0", timeout=60000)
        self.wait_for(lambda: self.points(SPACE), "Instance Space scatter did not appear", 60)
        if self.page.evaluate("document.title") != f"{TITLE} - {dataset}":
            self.choose_dataset(dataset)

    def choose_dataset(self, dataset):
        n = len(pd.read_csv(IS_DIR / dataset / "coordinates.csv"))
        self.page.get_by_label("Dataset", exact=True).select_option(dataset)
        self.wait_for(lambda: self.page.evaluate("document.title") == f"{TITLE} - {dataset}",
                      f"title did not become {dataset}")
        self.wait_for(lambda: (self.points(SPACE) or {}).get("n") == n,
                      f"scatter did not get {n} points")
        return n

    def tab(self, name):
        self.page.locator(".bk-tab", has_text=name).first.click()
        self.wait_for(lambda: "bk-active" in (self.page.locator(".bk-tab", has_text=name).first
                                              .get_attribute("class") or ""), f"tab {name} not active")
        self.page.wait_for_timeout(600)

    def _draw(self, cx, cy, r, steps=28):
        m = self.page.mouse
        m.move(cx + r, cy)
        m.down()
        for i in range(1, steps + 1):
            a = 2 * math.pi * i / steps
            m.move(cx + r * math.cos(a), cy + r * math.sin(a), steps=2)
        m.up()

    def lasso_with_points(self):
        """Circular lasso around the median point; return how many points it covers."""
        screen = self.page.evaluate(SCREEN_JS, SPACE)
        x0, y0, x1, y1 = screen["frame"]
        pts = np.array(screen["pts"])
        cx, cy = np.median(pts[:, 0]), np.median(pts[:, 1])
        cx, cy = min(max(cx, x0 + 60), x1 - 60), min(max(cy, y0 + 60), y1 - 60)
        r = min(90, cx - x0 - 20, x1 - cx - 20, cy - y0 - 20, y1 - cy - 20)
        inside = int(np.sum(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) < r * 0.95))
        assert inside > 0, "no point under the planned lasso"
        self._draw(float(cx), float(cy), float(r))
        return inside

    def empty_lasso(self):
        """Small lasso on the frame point farthest from the data."""
        screen = self.page.evaluate(SCREEN_JS, SPACE)
        x0, y0, x1, y1 = screen["frame"]
        pts = np.array(screen["pts"])
        gx, gy = np.meshgrid(np.arange(x0 + 40, x1 - 40, 6), np.arange(y0 + 40, y1 - 40, 6))
        grid = np.column_stack([gx.ravel(), gy.ravel()])
        dist = np.min(np.hypot(grid[:, None, 0] - pts[None, :, 0],
                               grid[:, None, 1] - pts[None, :, 1]), axis=1)
        i = int(np.argmax(dist))
        assert dist[i] > 12, "no empty area inside the frame"
        self._draw(float(grid[i, 0]), float(grid[i, 1]), float(min(0.45 * dist[i], 20)))

    def choose_option(self, select_label, option_label):
        self.page.get_by_label(select_label, exact=True).select_option(label=option_label)

    # --------------------------------------------------- New instance space
    def open_new_space(self):
        header = self.page.locator(".card-header", has_text="New instance space").first
        header.click()
        self.wait_for(lambda: self.page.get_by_role("button", name="Run ISA").is_visible(),
                      "the New instance space block did not open")

    def upload(self, css_class, path):
        self.page.locator(f".{css_class} input[type=file]").set_input_files(str(path))

    def choose_rule(self, direction, threshold, eps):
        self.choose_option("Performance direction", direction)
        self.choose_option("Threshold", threshold)
        field = self.page.get_by_label("ε (epsilon)", exact=True)
        field.fill(str(eps))
        field.press("Tab")

    def name_run(self, name):
        field = self.page.get_by_label("Run name", exact=True)
        field.fill(name)
        field.press("Tab")

    def run(self, name, tabs_during=("Features",)):
        """Click Run ISA, switch tabs while it runs (the interface stays
        usable) and wait for the new dataset to open; return (folder name,
        progress messages seen)."""
        button = self.page.get_by_role("button", name="Run ISA")
        self.wait_for(lambda: button.is_enabled(), "Run ISA button not enabled")
        button.click()
        seen, end = [], time.time() + RUN_TIMEOUT
        pending = list(tabs_during)
        while time.time() < end:
            status = self.text("new-status")
            if status and (not seen or seen[-1] != status):
                seen.append(status)
            if pending and "Running" in status:
                self.tab(pending.pop(0))
            title = self.page.evaluate("document.title")
            if title.startswith(f"{TITLE} - {name}_"):
                return title[len(f"{TITLE} - "):], seen
            assert "Failed" not in status, status
            self.page.wait_for_timeout(250)
        raise AssertionError(f"run did not finish in {RUN_TIMEOUT} s: {seen[-3:]}")


@pytest.fixture
def screen(browser, server):
    ctx = browser.new_context(viewport={"width": 1500, "height": 1000}, accept_downloads=True)
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    yield Screen(page), server
    ctx.close()
    assert not errors, f"JavaScript errors on the page: {errors[:3]}"


def _n_distributions(t):
    """'Selected n=K' count of the Distributions titles (None if absent)."""
    ks = {int(m.group(1)) for p in t.plots() for m in [re.search(r"Selected n=(\d+)", p["title"])] if m}
    return ks.pop() if len(ks) == 1 else (None if not ks else ks)


def _check_tabs(t, n):
    """Footprint Performance, Algorithm Selection, Distributions and Data
    Explorer show the same selection of n points (n=0: empty selection)."""
    t.tab("Footprint Performance")
    t.wait_for(lambda: t.n_status() == n, f"Footprint Performance: status does not show {n}")
    t.wait_for(lambda: (t.points("Footprints") or {}).get("hi") == n,
               f"Footprint Performance: map does not highlight {n} points")
    t.tab("Algorithm Selection")
    t.wait_for(lambda: t.n_status() == n, f"Algorithm Selection: status does not show {n}")
    t.wait_for(lambda: (t.points(RECOMMENDED) or {}).get("hi") == n,
               f"Algorithm Selection: map does not highlight {n} points")
    t.tab("Distributions")
    t.wait_for(lambda: t.n_status() == n, f"Distributions: status does not show {n}")
    t.wait_for(lambda: _n_distributions(t) == n, f"Distributions: titles do not show Selected n={n}")
    t.tab("Data Explorer")
    t.wait_for(lambda: t.n_status() == n, f"Data Explorer: status does not show {n}")
    t.wait_for(lambda: (t.points(EXPLORER) or {}).get("hi") == n,
               f"Data Explorer: scatter does not highlight {n} points")


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dataset", DATASETS)
def test_lasso_active_on_open(screen, dataset):
    t, url = screen
    t.open(url, dataset)
    assert t.plot(SPACE)["active"] == "LassoSelectTool"
    assert "No selection" in t.status()
    t.lasso_with_points()            # without clicking any tool
    n = t.wait_for(lambda: t.n_status(), "the lasso selected nothing")
    assert n > 0
    assert t.points(SPACE)["sel"] == n


@pytest.mark.parametrize("dataset", DATASETS)
def test_lasso_shows_in_the_other_tabs(screen, dataset):
    t, url = screen
    t.open(url, dataset)
    t.lasso_with_points()
    n = t.wait_for(lambda: t.n_status(), "the lasso selected nothing")
    assert t.points(SPACE)["sel"] == n
    assert t.sidebar_selection() == f"Selection: {n} instances"
    _check_tabs(t, n)
    t.tab("Instance Space")
    t.wait_for(lambda: (t.points(SPACE) or {}).get("sel") == n, "the selection vanished from tab 0")


@pytest.mark.parametrize("dataset", DATASETS)
def test_global_color_between_instance_space_and_explorer(screen, dataset):
    t, url = screen
    t.open(url, dataset)
    categorical, numeric = "best observed algorithm", "number of good algorithms"
    t.choose_option("Point color", categorical)
    t.wait_for(lambda: (t.plot(SPACE) or {}).get("title", "").endswith(f"color: {categorical}"),
               "Instance Space did not change color")
    assert t.points(SPACE)["mapper"] == "CategoricalColorMapper"
    t.tab("Data Explorer")
    t.wait_for(lambda: t.select_value("Color") == categorical, "the Data Explorer selector did not follow")
    t.wait_for(lambda: (t.plot(EXPLORER) or {}).get("title", "").endswith(f"color: {categorical}"),
               "Data Explorer scatter did not change color")
    assert t.points(EXPLORER)["mapper"] == "CategoricalColorMapper"
    # and the other way round
    t.choose_option("Color", numeric)
    t.wait_for(lambda: (t.plot(EXPLORER) or {}).get("title", "").endswith(f"color: {numeric}"),
               "Data Explorer did not change color")
    assert t.points(EXPLORER)["mapper"] == "LinearColorMapper"
    t.tab("Instance Space")
    t.wait_for(lambda: t.select_value("Point color") == numeric, "the tab 0 selector did not follow")
    t.wait_for(lambda: (t.plot(SPACE) or {}).get("title", "").endswith(f"color: {numeric}"),
               "Instance Space scatter did not change color")
    assert t.points(SPACE)["mapper"] == "LinearColorMapper"


@pytest.mark.parametrize("dataset", DATASETS)
def test_empty_lasso_gives_an_empty_selection(screen, dataset):
    t, url = screen
    t.open(url, dataset)
    t.empty_lasso()
    t.wait_for(lambda: "Empty selection" in t.status(), "tab 0 did not show an empty selection")
    assert t.sidebar_selection() == "Selection: empty (0 instances)"
    assert t.points(SPACE)["sel"] == 0
    _check_tabs(t, 0)
    for tab in ("Footprint Performance", "Algorithm Selection", "Distributions", "Data Explorer"):
        t.tab(tab)
        assert "Empty selection" in t.status(), tab


@pytest.mark.parametrize("dataset", DATASETS)
def test_switching_dataset_resets_selection_and_updates_everything(screen, dataset):
    t, url = screen
    t.open(url, dataset)
    t.lasso_with_points()
    t.wait_for(lambda: t.n_status(), "the lasso selected nothing")
    other = DATASETS[(DATASETS.index(dataset) + 1) % len(DATASETS)]
    n = t.choose_dataset(other)
    assert t.page.evaluate("document.title") == f"{TITLE} - {other}"
    assert t.page.get_by_text(f"— {other}").count() == 1                  # header
    assert t.page.get_by_text(f"Instances: {n}").count() == 1            # sidebar
    assert "No selection" in t.status()
    assert t.sidebar_selection() == "Selection: none"
    assert t.points(SPACE)["sel"] == 0
    t.tab("Footprint Performance")
    t.wait_for(lambda: (t.points("Footprints") or {}).get("n") == n, "footprint map did not update")
    assert "No selection" in t.status()
    t.tab("Algorithm Selection")
    t.wait_for(lambda: (t.points(RECOMMENDED) or {}).get("n") == n, "Algorithm Selection did not update")
    t.tab("Distributions")
    t.wait_for(lambda: any(f"All n={n}" in p["title"] for p in t.plots()), "distributions did not update")
    assert _n_distributions(t) is None
    t.tab("Data Explorer")
    t.wait_for(lambda: (t.points(EXPLORER) or {}).get("n") == n, "Data Explorer did not update")
    assert "No selection" in t.status()


@pytest.mark.parametrize("dataset", DATASETS)
def test_use_filter_as_selection(screen, dataset):
    t, url = screen
    t.open(url, dataset)
    coords = pd.read_csv(IS_DIR / dataset / "coordinates.csv")
    k = int((coords["z_1"] > 0).sum())
    t.tab("Data Explorer")
    field = t.page.get_by_label("Filter (pandas query)", exact=True)
    field.fill("z_1 > 0")
    field.press("Enter")
    t.wait_for(lambda: t.page.get_by_text(f"{k} of {len(coords)} rows").count() == 1,
               f"the filter did not show {k} rows")
    assert "No selection" in t.status()                  # the filter alone is local
    button = t.page.get_by_role("button", name="Use filter as selection")
    t.wait_for(lambda: button.is_enabled(), "button not enabled")
    button.click()
    t.wait_for(lambda: t.n_status() == k, f"Data Explorer: status does not show {k}")
    assert t.sidebar_selection() == f"Selection: {k} instances"
    t.wait_for(lambda: (t.points(EXPLORER) or {}).get("hi") == k, "Data Explorer did not highlight")
    t.tab("Instance Space")
    t.wait_for(lambda: t.n_status() == k, f"Instance Space: status does not show {k}")
    t.wait_for(lambda: (t.points(SPACE) or {}).get("sel") == k, "Instance Space did not highlight")
    t.tab("Footprint Performance")
    t.wait_for(lambda: (t.points("Footprints") or {}).get("hi") == k, "Footprint Performance did not highlight")
    t.tab("Distributions")
    t.wait_for(lambda: _n_distributions(t) == k, "Distributions did not show the selection")


def test_group_by_class_on_iris_gives_3_groups(screen):
    t, url = screen
    t.open(url, "iris")
    t.tab("Distributions")
    t.choose_option("Group by", "class")
    plot = t.wait_for(lambda: next((p for p in t.plots() if "by class" in p["title"]
                                    and p["factors"]), None), "distribution by class did not appear")
    classes = sorted(pd.read_csv(ROOT / "resultados" / "table_iris.csv")["class"].unique())
    assert len(plot["factors"]) == 3                      # 3 groups -> violin by default
    assert [f.split(" (n=")[0] for f in plot["factors"]] == classes
    assert all("(n=50)" in f for f in plot["factors"])


@pytest.mark.parametrize("dataset", DATASETS)
def test_exported_selection_csv_has_the_selected_rows(screen, dataset):
    t, url = screen
    t.open(url, dataset)
    t.lasso_with_points()
    n = t.wait_for(lambda: t.n_status(), "the lasso selected nothing")
    labels = t.wait_for(lambda: (r := t.page.evaluate(SELECTED_JS, SPACE)) and len(r) == n and r,
                        "selected labels do not match the status")
    button = t.page.get_by_role("button", name=f"Export selection ({n})")
    t.wait_for(lambda: button.is_enabled(), "export selection button not enabled")
    with t.page.expect_download() as info:
        button.click()
    csv = pd.read_csv(info.value.path(), dtype={"instances": str})
    assert info.value.suggested_filename == f"{dataset}_selection.csv"
    assert len(csv) == n and sorted(csv["instances"]) == sorted(labels)
    meta = pd.read_csv(IS_DIR / dataset / "metadata.csv", nrows=1)
    features = [c for c in meta.columns if c.startswith("feature_")]
    algos = [c for c in meta.columns if c.startswith("algo_")]
    for col in ["instances", "class", "ih", "n_wrong", *features, *algos, "z_1", "z_2",
                "NumGoodAlgos", "IsBetaEasy", "best_algo", "best_algo_or_tie", "n_tied_best",
                "best_algo_svm"]:
        assert col in csv.columns, col


def test_features_tab_of_iris_lists_every_feature_and_the_degenerate_ones(screen):
    t, url = screen
    t.open(url, "iris")
    t.tab("Features")
    data = t.wait_for(lambda: t.page.evaluate(TABLE_JS, "feature"), "features table did not appear")
    table = pd.read_csv(ROOT / "resultados" / "table_iris.csv", nrows=1)
    received = [c[len("feature_"):] for c in table.columns if c.startswith("feature_")]
    assert sorted(data["feature"]) == sorted(received) and len(data["feature"]) == 19
    degenerate = {f for f, st in zip(data["feature"], data["status"]) if st == "dropped_degenerate"}
    assert degenerate == {"kDN", "MV", "CB", "N1", "Harmfulness"}
    assert t.page.get_by_text("19 features received").count() == 1
    assert any(p["title"].startswith("Pearson rho") for p in t.plots())      # heatmap
    assert any("k used = 6" in p["title"] for p in t.plots())                # silhouette


@pytest.mark.parametrize("dataset", DATASETS)
def test_orientation_note_and_bad_algorithms_color(screen, dataset):
    t, url = screen
    t.open(url, dataset)
    note = t.wait_for(lambda: t.text("space-orientation"), "orientation note did not appear")
    assert "top left" in note
    assert ("Weak difficulty gradient" in note) is (dataset == "hill-valley")
    t.choose_option("Point color", "number of bad algorithms")
    t.wait_for(lambda: (t.plot(SPACE) or {}).get("title", "").endswith("color: number of bad algorithms"),
               "color did not change")
    assert t.points(SPACE)["mapper"] == "LinearColorMapper"
    good = pd.read_csv(IS_DIR / dataset / "algorithm_bin.csv", dtype={"Row": str}).set_index("Row")
    expected = (~good.astype(bool)).sum(axis=1)
    values = t.wait_for(lambda: t.column(SPACE, "color_value"), "color column not found")
    got = pd.Series([float(v) for v in values], index=t.column(SPACE, "Row"))
    assert (got.reindex(expected.index).to_numpy() == expected.to_numpy()).all()


# --------------------------------------------------------------------------- #
# ties for the best observed value (Task 2)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dataset", DATASETS)
def test_best_observed_color_marks_ties_instead_of_a_random_pick(screen, dataset):
    t, url = screen
    t.open(url, dataset)
    raw = pd.read_csv(IS_DIR / dataset / "algorithm_raw.csv", dtype={"Row": str}).set_index("Row")
    n_tied = (raw.to_numpy() == raw.to_numpy().max(axis=1, keepdims=True)).sum(axis=1)
    t.choose_option("Point color", "best observed algorithm")
    t.wait_for(lambda: (t.plot(SPACE) or {}).get("title", "").endswith("color: best observed algorithm"),
               "color did not change")
    values = t.wait_for(lambda: t.column(SPACE, "color_value"), "color column not found")
    assert values.count("tie") == TIES[dataset] == int((n_tied > 1).sum())
    assert set(values) - {"tie"} <= set(raw.columns)
    t.choose_option("Point color", "algorithms tied for the best")
    t.wait_for(lambda: (t.plot(SPACE) or {}).get("title", "").endswith("color: algorithms tied for the best"),
               "color did not change")
    assert t.points(SPACE)["mapper"] == "LinearColorMapper"
    values = t.wait_for(lambda: t.column(SPACE, "color_value"), "color column not found")
    rows = t.column(SPACE, "Row")
    got = pd.Series([float(v) for v in values], index=rows)
    assert (got.reindex(raw.index).to_numpy() == n_tied).all()


def test_algorithm_selection_good_bad_categories_follow_algorithm_bin(screen):
    t, url = screen
    t.open(url, "iris")
    folder = IS_DIR / "iris"
    sel = pd.read_csv(folder / "pythia_selection.csv", dtype={"Row": str}).set_index("Row")["selection0"]
    good = pd.read_csv(folder / "algorithm_bin.csv", dtype={"Row": str}).set_index("Row")
    rec_good = [bool(good.loc[r, a]) if isinstance(a, str) else None for r, a in sel.items()]
    expected = {"recommended good": rec_good.count(True), "recommended bad": rec_good.count(False),
                "no recommendation": rec_good.count(None)}
    t.tab("Algorithm Selection")
    t.choose_option("Color by", "recommended good / bad")
    title = "Recommended algorithm (selection0): good or bad"
    t.wait_for(lambda: t.plot(title), "good/bad map did not appear")
    values = t.wait_for(lambda: t.column(title, "color_value"), "color column not found")
    got = pd.Series([v.rsplit(" (", 1)[0] for v in values]).value_counts().to_dict()
    assert got == {k: v for k, v in expected.items() if v}
    for v in set(values):                                 # the legend count matches
        name, count = v.rsplit(" (", 1)
        assert int(count.rstrip(")")) == expected[name]
    summary = t.page.get_by_text(re.compile(r"The recommended algorithm is good")).first.inner_text()
    assert f"good for the instance in {expected['recommended good']} of 150" in summary
    assert f"Ties for the best observed value: {TIES['iris']} instances" in summary


# --------------------------------------------------------------------------- #
# New instance space (upload, validation, engine run in a subprocess)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def example_metadata(request):
    """Example metadata.csv of the instancespace repository (tag v0.3.0),
    downloaded once to the pytest cache (non-commercial license: it does not
    go into this repository)."""
    target = request.config.cache.mkdir("instancespace_v0.3.0") / "metadata.csv"
    if not target.is_file():
        try:
            with urllib.request.urlopen(EXAMPLE_URL, timeout=30) as resp:
                target.write_bytes(resp.read())
        except OSError as exc:
            pytest.skip(f"no access to the example metadata ({exc})")
    return target


def _selector_groups(t):
    return t.page.get_by_label("Dataset", exact=True).evaluate(
        "e => Object.fromEntries([...e.querySelectorAll('optgroup')].map("
        "g => [g.label, [...g.querySelectorAll('option')].map(o => o.text)]))")


def _check_all_tabs(t, n):
    """Open the six tabs of the active dataset; the scatters have n points and
    no tab shows a traceback."""
    t.tab("Instance Space")
    t.wait_for(lambda: (t.points(SPACE) or {}).get("n") == n, f"scatter does not have {n} points")
    t.tab("Footprint Performance")
    t.wait_for(lambda: (t.points("Footprints") or {}).get("n") == n, "footprint map")
    t.tab("Algorithm Selection")
    t.wait_for(lambda: (t.points(RECOMMENDED) or {}).get("n") == n, "Algorithm Selection map")
    t.tab("Distributions")
    t.wait_for(lambda: len(t.plots()) >= 1, "Distributions without plots")
    t.tab("Features")
    t.wait_for(lambda: t.page.get_by_text(re.compile(r"\d+ features received")).count() == 1,
               "Features without summary")
    t.tab("Data Explorer")
    t.wait_for(lambda: (t.points(EXPLORER) or {}).get("n") == n, "Data Explorer")
    assert t.page.get_by_text("Traceback").count() == 0


def test_broken_upload_shows_an_error_without_traceback(screen, tmp_path):
    t, url = screen
    t.open(url, "iris")
    t.open_new_space()
    broken = tmp_path / "broken.csv"
    broken.write_text("instances,feature_a,feature_b,algo_x,algo_y\n1,1,2,0.1,0.2\n1,x,4,0.5,0.6\n")
    t.upload("new-metadata", broken)
    text = t.wait_for(lambda: t.text("new-errors"), "the validation error did not appear")
    assert "At least 3 feature_* columns" in text and "the file has 2" in text
    assert t.page.get_by_text("Traceback").count() == 0
    assert t.page.get_by_role("button", name="Run ISA").is_disabled()


def test_upload_of_the_instancespace_example_runs_and_opens_the_tabs(screen, runs_dir, example_metadata):
    """Metadata of the instancespace repository, without annotations: choose
    the direction, run, and open every tab of the result."""
    t, url = screen
    t.open(url, "iris")
    t.open_new_space()
    t.upload("new-metadata", example_metadata)
    t.wait_for(lambda: t.page.get_by_text(re.compile(r"212 instances, 10 features, 10 algorithms"))
               .count() == 1, "metadata summary did not appear")
    button = t.page.get_by_role("button", name="Run ISA")
    assert button.is_disabled()                       # direction not chosen yet
    assert t.page.get_by_text(re.compile("still missing: the performance direction")).count() == 1
    t.choose_rule("lower is better", "absolute", 0.2)          # the example's options.json
    t.wait_for(lambda: t.page.get_by_text("Preview (PRELIM rule):", exact=False).count() >= 1,
               "good-fraction preview did not appear")
    t.name_run("example_e2e")
    name, seen = t.run("example_e2e")
    assert any(re.search(r"stage \d of 7", s) for s in seen), seen
    t.wait_for(lambda: "Done" in t.text("new-status"), "status does not show Done")
    folder = runs_dir / name
    assert (folder / "run_info.json").is_file() and (folder / "run.log").is_file()
    assert (folder / "input" / "metadata.csv").read_bytes() == example_metadata.read_bytes()
    options = json.loads((folder / "run_options.json").read_text())
    assert options["perf"] == {**options["perf"], "max_perf": False, "abs_perf": True, "epsilon": 0.2}
    assert options["trace"]["use_sim"] is False and options["sifted"]["k"] == 6
    groups = _selector_groups(t)
    assert name in groups["runs (launched from this interface)"]
    assert "iris" in groups["resultados/is"] and name not in groups["resultados/is"]
    n = len(pd.read_csv(folder / "coordinates.csv"))
    _check_all_tabs(t, n)


def test_cycle_export_selection_and_upload_it_as_metadata(screen, runs_dir, tmp_path):
    """Export the selection (filter z_1 < 0 on diabetes), upload the exported
    CSV as new metadata, run, and check that the instance count matches."""
    t, url = screen
    t.open(url, "diabetes")
    t.tab("Data Explorer")
    t.page.get_by_label("Filter (pandas query)", exact=True).fill("z_1 < 0")
    t.page.get_by_label("Filter (pandas query)", exact=True).press("Enter")
    use = t.page.get_by_role("button", name="Use filter as selection")
    t.wait_for(lambda: use.is_enabled(), "use filter button not enabled")
    use.click()
    n = t.wait_for(lambda: t.n_status(), "the filter did not become a selection")
    expected = int((pd.read_csv(IS_DIR / "diabetes" / "coordinates.csv")["z_1"] < 0).sum())
    assert n == expected
    button = t.page.get_by_role("button", name=f"Export selection ({n})")
    t.wait_for(lambda: button.is_enabled(), "export selection button not enabled")
    with t.page.expect_download() as info:
        button.click()
    exported = tmp_path / "diabetes_selection.csv"
    info.value.save_as(exported)
    assert len(pd.read_csv(exported)) == n

    t.open_new_space()
    t.upload("new-metadata", exported)
    t.wait_for(lambda: t.page.get_by_text(re.compile(rf"{n} instances, ")).count() == 1,
               "summary of the exported metadata did not appear")
    t.choose_rule("higher is better", "absolute", 0.5)
    t.name_run("cycle_e2e")
    name, _ = t.run("cycle_e2e")
    folder = runs_dir / name
    assert len(pd.read_csv(folder / "coordinates.csv")) == n
    assert json.loads((folder / "run_info.json").read_text())["n_instances"] == n
    t.wait_for(lambda: t.page.get_by_text(re.compile(rf"Instances: {n}\b")).count() >= 1,
               f"sidebar does not show {n} instances")
    t.tab("Instance Space")
    t.wait_for(lambda: (t.points(SPACE) or {}).get("n") == n, f"scatter does not have {n} points")


def test_flipping_the_direction_flips_the_ranking(screen):
    """Task 1: the direction is not guessed; the block shows its consequence.
    Flipping the direction reverses the mean ranking shown before the Run
    button (best and worst swap). The degenerate-ε warning is about the
    threshold: on iris with absolute ε = 0.5 it fires in both directions."""
    t, url = screen
    t.open(url, "iris")
    t.open_new_space()
    t.upload("new-metadata", IS_DIR / "iris" / "metadata.csv")
    t.wait_for(lambda: t.page.get_by_text(re.compile(r"150 instances, ")).count() == 1,
               "iris summary did not appear")
    assert t.table_rows("new-ranking") == []               # no direction yet: no ranking
    t.choose_option("Performance direction", "higher is better")
    higher = t.wait_for(lambda: t.table_rows("new-ranking"), "ranking did not appear")
    text = t.text("new-ranking")
    assert UPSIDE_DOWN in " ".join(text.split())
    order = [row[1] for row in higher]
    assert f"Best under this direction: {order[0]}" in " ".join(text.split())
    means = pd.read_csv(IS_DIR / "iris" / "metadata.csv").filter(like="algo_").mean()
    assert order[0] == means.idxmax().removeprefix("algo_")
    t.choose_option("Performance direction", "lower is better")
    lower = t.wait_for(lambda: (r := t.table_rows("new-ranking")) and [x[1] for x in r] != order and r,
                       "ranking did not change")
    assert [row[1] for row in lower] == order[::-1]
    assert f"Best under this direction: {order[-1]}" in " ".join(t.text("new-ranking").split())
    t.choose_option("Threshold", "absolute")
    field = t.page.get_by_label("ε (epsilon)", exact=True)
    field.fill("0.5")
    field.press("Tab")
    for direction in ("lower is better", "higher is better"):
        t.choose_option("Performance direction", direction)
        warning = t.wait_for(lambda: t.text("new-degenerate-eps"), f"no ε warning with {direction}")
        assert "Degenerate ε threshold" in warning and "says nothing about the direction" in warning
    assert t.page.get_by_role("button", name="Run ISA").is_enabled()


def test_algorithm_selection_warns_about_a_trivial_selector_on_iris(screen):
    t, url = screen
    t.open(url, "iris")
    t.tab("Algorithm Selection")
    text = t.wait_for(lambda: t.text("as-trivial"), "trivial selector warning did not appear on iris")
    assert "Nearly trivial selector" in text and "logreg" in text and "143 of 150" in text
    assert t.page.get_by_text(re.compile(r"pr0_sub.*out of sample")).count() >= 1
    t.wait_for(lambda: (t.points(RECOMMENDED) or {}).get("n") == 150, "iris map")
    assert len(t.page.locator(".as-confusion").all()) == 6
    t.tab("Instance Space")               # choose_dataset waits for the tab 0 scatter
    t.choose_dataset("hill-valley")       # logreg in 853 of 1212 (70%): no warning
    t.tab("Algorithm Selection")
    t.wait_for(lambda: (t.points(RECOMMENDED) or {}).get("n") == 1212, "hill-valley map")
    assert t.page.locator(".as-trivial").count() == 0
