"""Testes de ponta a ponta de isaspace.ui.app num Chromium real (Playwright).

Sobe o app uma vez (python -m isaspace.ui.app, com o interpretador do pytest)
e abre uma pagina NOVA a cada teste, portanto uma sessao nova do servidor e um
estado global novo. A maioria dos testes roda nos quatro datasets de
resultados/is/. O lasso e desenhado com o mouse, dentro da area de plotagem, a
partir das coordenadas de tela calculadas pelo proprio BokehJS.

O servidor usa uma pasta runs/ temporaria (--runs): os testes do bloco "Novo
instance space" rodam o engine de verdade, em subprocesso, e as execucoes
ficam nela. O metadata de exemplo do instancespace e baixado do GitHub (tag
v0.3.0) para o cache do pytest; sem rede, esse teste e pulado.

Precisa do .venv-isa com playwright e do Chromium do Playwright
(python -m playwright install chromium). Uso, da raiz:
    .venv-isa/bin/python -m pytest tests/test_app_e2e.py
    .venv-isa/bin/python -m pytest tests/ -m "not e2e"     (so os rapidos)
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

RAIZ = Path(__file__).resolve().parents[1]
PASTA_IS = RAIZ / "resultados" / "is"
DATASETS = ["iris", "diabetes", "blood-transfusion-service-center", "hill-valley"]
TITULO = "isa-instance"
TIMEOUT = 25  # s por espera
TIMEOUT_EXECUCAO = 180   # s para o engine terminar (exemplo ~5 s, ciclo ~10 s)
URL_EXEMPLO = ("https://raw.githubusercontent.com/andremun/pyInstanceSpace/v0.3.0/"
               "examples/data/metadata.csv")

pytestmark = pytest.mark.e2e

# plots Bokeh renderizados (so a aba ativa existe no DOM: Tabs dynamic=True)
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
      // opacidade por ponto; se todos tem a mesma, o HoloViews grava um escalar
      if (fa && fa.field !== undefined && data[fa.field]) hi = [...data[fa.field]].filter(a => a > 0.5).length;
      else if (fa && typeof fa.value === 'number') hi = fa.value > 0.5 ? n : 0;
      rs.push({glyph: g.type, n, sel: [...ds.selected.indices].length, hi, tem_row: 'Row' in data,
               mapper: fc && fc.transform ? fc.transform.type : null});
    }
    const tb = m.toolbar;
    const xr = m.x_range;
    out.push({titulo: m.title.text || "", rs,
              fatores: xr && xr.factors ? [...xr.factors].map(String) : null,
              ativo: tb && tb.active_drag && tb.active_drag.type ? tb.active_drag.type : null});
  }
  return out;
}"""
# pontos do scatter da aba 0 em coordenadas de pagina, e o retangulo do frame
TELA_JS = r"""(prefixo) => {
  for (const v of Bokeh.index.all_views()) {
    const m = v.model;
    if (!m || !v.frame || !m.title || !(m.title.text || "").startsWith(prefixo)) continue;
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
# rotulos Row das instancias selecionadas no scatter da aba Instance Space
SELECIONADAS_JS = r"""(prefixo) => {
  for (const v of Bokeh.index.all_views()) {
    const m = v.model;
    if (!m || !v.frame || !m.title || !(m.title.text || "").startsWith(prefixo)) continue;
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
# colunas de todas as tabelas (Tabulator) do documento que tenham `coluna`
TABELA_JS = r"""(coluna) => {
  for (const m of Bokeh.documents[0]._all_models.values()) {
    if (m.type !== 'ColumnDataSource' || !m.data) continue;
    const data = m.data instanceof Map ? Object.fromEntries(m.data) : m.data;
    if (coluna in data && 'status' in data) {
      const out = {};
      for (const k of Object.keys(data)) out[k] = [...data[k]].map(v => v === null ? null : String(v));
      return out;
    }
  }
  return null;
}"""
# texto de um componente Panel, atravessando os shadow roots (inner_text nao entra neles)
TEXTO_JS = r"""(e) => {
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
RE_STATUS = re.compile(r"Sem seleção\.|Seleção vazia|\d+ instâncias selecionadas")
ESPACO = "Espaço de instâncias"
EXPLORER = "z_1 x z_2"   # titulo do scatter do Data Explorer com os eixos padrao


def _porta_livre():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def pasta_runs(tmp_path_factory):
    return tmp_path_factory.mktemp("runs")


@pytest.fixture(scope="session")
def servidor(pasta_runs):
    if not all((PASTA_IS / d / "run_info.json").is_file() for d in DATASETS):
        pytest.skip("resultados/is incompleto (rode scripts/run_is_all.py no .venv-isa)")
    porta = _porta_livre()
    proc = subprocess.Popen(
        [sys.executable, "-m", "isaspace.ui.app", "--no-show", "--port", str(porta),
         "--runs", str(pasta_runs)],
        cwd=RAIZ, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    url = f"http://localhost:{porta}/"
    inicio = time.time()
    while True:
        if proc.poll() is not None:
            pytest.fail("o app morreu ao subir:\n" + proc.stdout.read().decode()[-3000:])
        try:
            if urllib.request.urlopen(url, timeout=2).status == 200:
                break
        except OSError:
            pass
        if time.time() - inicio > 90:
            proc.kill()
            pytest.fail("o app nao respondeu em 90 s")
        time.sleep(0.5)
    yield url
    proc.terminate()
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def navegador():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"Chromium do Playwright indisponivel: {exc}")
        yield b
        b.close()


class Tela:
    """Pagina do app com os passos que os testes usam."""

    def __init__(self, page):
        self.page = page

    # -------------------------------------------------------------- esperas
    def esperar(self, cond, msg, timeout=TIMEOUT):
        fim = time.time() + timeout
        ultimo = None
        while time.time() < fim:
            try:
                ultimo = cond()
                if ultimo:
                    return ultimo
            except Exception as exc:  # noqa: BLE001 -- DOM em transicao
                ultimo = exc
            self.page.wait_for_timeout(200)
        raise AssertionError(f"{msg} (ultimo valor: {ultimo!r})")

    # -------------------------------------------------------------- leitura
    def plots(self):
        return self.page.evaluate(PLOTS_JS)

    def plot(self, prefixo):
        achados = [p for p in self.plots() if p["titulo"].startswith(prefixo)]
        return achados[0] if achados else None

    def pontos(self, prefixo):
        """Renderer de pontos (tem a coluna Row) do plot com esse titulo."""
        p = self.plot(prefixo)
        if p is None:
            return None
        return next((r for r in p["rs"] if r["tem_row"]), None)

    def status(self):
        loc = self.page.get_by_text(RE_STATUS)
        return loc.first.inner_text() if loc.count() else ""

    def n_status(self):
        m = re.search(r"(\d+) instâncias selecionadas", self.status())
        if m:
            return int(m.group(1))
        return 0 if "Seleção vazia" in self.status() else None

    def sidebar_selecao(self):
        return self.page.get_by_text(re.compile(r"^Seleção: ")).first.inner_text()

    def valor_select(self, rotulo):
        return self.page.get_by_label(rotulo, exact=True).evaluate(
            "e => e.options[e.selectedIndex].text")

    # -------------------------------------------------------------- acoes
    def abrir(self, url, dataset):
        self.page.goto(url)
        self.page.wait_for_function("window.Bokeh && Bokeh.documents.length > 0", timeout=60000)
        self.esperar(lambda: self.pontos(ESPACO), "scatter da aba Instance Space nao apareceu", 60)
        if self.page.evaluate("document.title") != f"{TITULO} - {dataset}":
            self.escolher_dataset(dataset)

    def escolher_dataset(self, dataset):
        n = len(pd.read_csv(PASTA_IS / dataset / "coordinates.csv"))
        self.page.get_by_label("Dataset", exact=True).select_option(dataset)
        self.esperar(lambda: self.page.evaluate("document.title") == f"{TITULO} - {dataset}",
                     f"titulo nao virou {dataset}")
        self.esperar(lambda: (self.pontos(ESPACO) or {}).get("n") == n,
                     f"scatter nao passou a ter {n} pontos")
        return n

    def aba(self, nome):
        self.page.locator(".bk-tab", has_text=nome).first.click()
        self.esperar(lambda: "bk-active" in (self.page.locator(".bk-tab", has_text=nome).first
                                             .get_attribute("class") or ""), f"aba {nome} nao ativou")
        self.page.wait_for_timeout(600)

    def _desenhar(self, cx, cy, r, passos=28):
        m = self.page.mouse
        m.move(cx + r, cy)
        m.down()
        for i in range(1, passos + 1):
            a = 2 * math.pi * i / passos
            m.move(cx + r * math.cos(a), cy + r * math.sin(a), steps=2)
        m.up()

    def lasso_com_pontos(self):
        """Lasso circular em torno do ponto mediano; devolve quantos pontos ele cobre."""
        tela = self.page.evaluate(TELA_JS, ESPACO)
        x0, y0, x1, y1 = tela["frame"]
        pts = np.array(tela["pts"])
        cx, cy = np.median(pts[:, 0]), np.median(pts[:, 1])
        cx, cy = min(max(cx, x0 + 60), x1 - 60), min(max(cy, y0 + 60), y1 - 60)
        r = min(90, cx - x0 - 20, x1 - cx - 20, cy - y0 - 20, y1 - cy - 20)
        dentro = int(np.sum(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) < r * 0.95))
        assert dentro > 0, "nenhum ponto sob o lasso planejado"
        self._desenhar(float(cx), float(cy), float(r))
        return dentro

    def lasso_vazio(self):
        """Lasso pequeno no ponto do frame mais distante dos dados."""
        tela = self.page.evaluate(TELA_JS, ESPACO)
        x0, y0, x1, y1 = tela["frame"]
        pts = np.array(tela["pts"])
        gx, gy = np.meshgrid(np.arange(x0 + 40, x1 - 40, 6), np.arange(y0 + 40, y1 - 40, 6))
        grade = np.column_stack([gx.ravel(), gy.ravel()])
        dist = np.min(np.hypot(grade[:, None, 0] - pts[None, :, 0],
                               grade[:, None, 1] - pts[None, :, 1]), axis=1)
        i = int(np.argmax(dist))
        assert dist[i] > 12, "nao ha area vazia dentro do frame"
        self._desenhar(float(grade[i, 0]), float(grade[i, 1]), float(min(0.45 * dist[i], 20)))

    def escolher_cor(self, rotulo_select, rotulo_opcao):
        self.page.get_by_label(rotulo_select, exact=True).select_option(label=rotulo_opcao)

    def texto(self, classe):
        """Texto do primeiro componente com a css_class `classe` ('' se nao ha)."""
        loc = self.page.locator(f".{classe}")
        return loc.first.evaluate(TEXTO_JS) if loc.count() else ""

    # ------------------------------------------------- Novo instance space
    def abrir_novo(self):
        cab = self.page.locator(".card-header", has_text="Novo instance space").first
        cab.click()
        self.esperar(lambda: self.page.get_by_role("button", name="Rodar ISA").is_visible(),
                     "o bloco Novo instance space nao abriu")

    def enviar(self, classe, caminho):
        self.page.locator(f".{classe} input[type=file]").set_input_files(str(caminho))

    def escolher_regra(self, direcao, limiar, eps):
        self.page.get_by_label("Direção do desempenho", exact=True).select_option(label=direcao)
        self.page.get_by_label("Limiar", exact=True).select_option(label=limiar)
        campo = self.page.get_by_label("ε (epsilon)", exact=True)
        campo.fill(str(eps))
        campo.press("Tab")

    def nomear(self, nome):
        campo = self.page.get_by_label("Nome da execução", exact=True)
        campo.fill(nome)
        campo.press("Tab")

    def rodar(self, nome, abas_durante=("Features",)):
        """Clica em Rodar ISA, troca de aba durante a execucao (a interface
        continua usavel) e espera o dataset novo abrir; devolve (pasta,
        mensagens de progresso vistas)."""
        botao = self.page.get_by_role("button", name="Rodar ISA")
        self.esperar(lambda: botao.is_enabled(), "botao Rodar ISA nao habilitou")
        botao.click()
        vistos, fim = [], time.time() + TIMEOUT_EXECUCAO
        pendentes = list(abas_durante)
        while time.time() < fim:
            status = self.texto("novo-status")
            if status and (not vistos or vistos[-1] != status):
                vistos.append(status)
            if pendentes and "Rodando" in status:
                self.aba(pendentes.pop(0))
            titulo = self.page.evaluate("document.title")
            if titulo.startswith(f"{TITULO} - {nome}_"):
                return titulo[len(f"{TITULO} - "):], vistos
            assert "Falhou" not in status, status
            self.page.wait_for_timeout(250)
        raise AssertionError(f"execucao nao terminou em {TIMEOUT_EXECUCAO} s: {vistos[-3:]}")


@pytest.fixture
def tela(navegador, servidor):
    ctx = navegador.new_context(viewport={"width": 1500, "height": 1000}, accept_downloads=True)
    page = ctx.new_page()
    erros = []
    page.on("pageerror", lambda e: erros.append(str(e)))
    yield Tela(page), servidor
    ctx.close()
    assert not erros, f"erros de JavaScript na pagina: {erros[:3]}"


def _n_distribuicoes(t):
    """Contagem 'Selecionadas n=K' dos titulos da aba Distributions (None se ausente)."""
    ks = {int(m.group(1)) for p in t.plots() for m in [re.search(r"Selecionadas n=(\d+)", p["titulo"])] if m}
    return ks.pop() if len(ks) == 1 else (None if not ks else ks)


def _conferir_abas(t, n):
    """As abas 1, 2 e 3 mostram a mesma selecao de n pontos (n=0: selecao vazia)."""
    t.aba("Footprint Performance")
    t.esperar(lambda: t.n_status() == n, f"Footprint Performance: status nao mostra {n}")
    t.esperar(lambda: (t.pontos("Footprints") or {}).get("hi") == n,
              f"Footprint Performance: mapa nao destaca {n} pontos")
    t.aba("Distributions")
    t.esperar(lambda: t.n_status() == n, f"Distributions: status nao mostra {n}")
    t.esperar(lambda: _n_distribuicoes(t) == n, f"Distributions: titulos nao mostram Selecionadas n={n}")
    t.aba("Data Explorer")
    t.esperar(lambda: t.n_status() == n, f"Data Explorer: status nao mostra {n}")
    t.esperar(lambda: (t.pontos(EXPLORER) or {}).get("hi") == n,
              f"Data Explorer: scatter nao destaca {n} pontos")


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dataset", DATASETS)
def test_lasso_ativo_ao_abrir(tela, dataset):
    t, url = tela
    t.abrir(url, dataset)
    assert t.plot(ESPACO)["ativo"] == "LassoSelectTool"
    assert "Sem seleção" in t.status()
    t.lasso_com_pontos()            # sem clicar em nenhuma ferramenta
    n = t.esperar(lambda: t.n_status(), "o lasso nao selecionou nada")
    assert n > 0
    assert t.pontos(ESPACO)["sel"] == n


@pytest.mark.parametrize("dataset", DATASETS)
def test_lasso_aparece_nas_outras_abas(tela, dataset):
    t, url = tela
    t.abrir(url, dataset)
    t.lasso_com_pontos()
    n = t.esperar(lambda: t.n_status(), "o lasso nao selecionou nada")
    assert t.pontos(ESPACO)["sel"] == n
    assert t.sidebar_selecao() == f"Seleção: {n} instâncias"
    _conferir_abas(t, n)
    t.aba("Instance Space")
    t.esperar(lambda: (t.pontos(ESPACO) or {}).get("sel") == n, "a selecao sumiu da aba 1")


@pytest.mark.parametrize("dataset", DATASETS)
def test_cor_global_entre_abas_1_e_4(tela, dataset):
    t, url = tela
    t.abrir(url, dataset)
    categorica, numerica = "melhor algoritmo observado", "n. de algoritmos bons"
    t.escolher_cor("Cor dos pontos", categorica)
    t.esperar(lambda: (t.plot(ESPACO) or {}).get("titulo", "").endswith(f"cor: {categorica}"),
              "aba 1 nao trocou a cor")
    assert t.pontos(ESPACO)["mapper"] == "CategoricalColorMapper"
    t.aba("Data Explorer")
    t.esperar(lambda: t.valor_select("Cor") == categorica, "seletor da aba 4 nao acompanhou")
    t.esperar(lambda: (t.plot(EXPLORER) or {}).get("titulo", "").endswith(f"cor: {categorica}"),
              "scatter da aba 4 nao trocou a cor")
    assert t.pontos(EXPLORER)["mapper"] == "CategoricalColorMapper"
    # e o caminho inverso
    t.escolher_cor("Cor", numerica)
    t.esperar(lambda: (t.plot(EXPLORER) or {}).get("titulo", "").endswith(f"cor: {numerica}"),
              "aba 4 nao trocou a cor")
    assert t.pontos(EXPLORER)["mapper"] == "LinearColorMapper"
    t.aba("Instance Space")
    t.esperar(lambda: t.valor_select("Cor dos pontos") == numerica, "seletor da aba 1 nao acompanhou")
    t.esperar(lambda: (t.plot(ESPACO) or {}).get("titulo", "").endswith(f"cor: {numerica}"),
              "scatter da aba 1 nao trocou a cor")
    assert t.pontos(ESPACO)["mapper"] == "LinearColorMapper"


@pytest.mark.parametrize("dataset", DATASETS)
def test_lasso_em_area_vazia_e_selecao_vazia(tela, dataset):
    t, url = tela
    t.abrir(url, dataset)
    t.lasso_vazio()
    t.esperar(lambda: "Seleção vazia" in t.status(), "aba 1 nao mostrou seleção vazia")
    assert t.sidebar_selecao() == "Seleção: vazia (0 instâncias)"
    assert t.pontos(ESPACO)["sel"] == 0
    _conferir_abas(t, 0)
    for aba in ("Footprint Performance", "Distributions", "Data Explorer"):
        t.aba(aba)
        assert "Seleção vazia" in t.status(), aba


@pytest.mark.parametrize("dataset", DATASETS)
def test_trocar_dataset_zera_selecao_e_atualiza_tudo(tela, dataset):
    t, url = tela
    t.abrir(url, dataset)
    t.lasso_com_pontos()
    t.esperar(lambda: t.n_status(), "o lasso nao selecionou nada")
    outro = DATASETS[(DATASETS.index(dataset) + 1) % len(DATASETS)]
    n = t.escolher_dataset(outro)
    assert t.page.evaluate("document.title") == f"{TITULO} - {outro}"
    assert t.page.get_by_text(f"— {outro}").count() == 1                  # cabecalho
    assert t.page.get_by_text(f"Instâncias: {n}").count() == 1           # sidebar
    assert "Sem seleção" in t.status()
    assert t.sidebar_selecao() == "Seleção: nenhuma"
    assert t.pontos(ESPACO)["sel"] == 0
    t.aba("Footprint Performance")
    t.esperar(lambda: (t.pontos("Footprints") or {}).get("n") == n, "mapa de footprints nao atualizou")
    assert "Sem seleção" in t.status()
    t.aba("Distributions")
    t.esperar(lambda: any(f"Todas n={n}" in p["titulo"] for p in t.plots()), "distribuicoes nao atualizaram")
    assert _n_distribuicoes(t) is None
    t.aba("Data Explorer")
    t.esperar(lambda: (t.pontos(EXPLORER) or {}).get("n") == n, "Data Explorer nao atualizou")
    assert "Sem seleção" in t.status()


@pytest.mark.parametrize("dataset", DATASETS)
def test_usar_filtro_como_selecao(tela, dataset):
    t, url = tela
    t.abrir(url, dataset)
    coords = pd.read_csv(PASTA_IS / dataset / "coordinates.csv")
    k = int((coords["z_1"] > 0).sum())
    t.aba("Data Explorer")
    campo = t.page.get_by_label("Filtro (pandas query)", exact=True)
    campo.fill("z_1 > 0")
    campo.press("Enter")
    t.esperar(lambda: t.page.get_by_text(f"{k} de {len(coords)} linhas").count() == 1,
              f"filtro nao mostrou {k} linhas")
    assert "Sem seleção" in t.status()                  # o filtro sozinho e local
    botao = t.page.get_by_role("button", name="Usar filtro como seleção")
    t.esperar(lambda: botao.is_enabled(), "botao nao habilitou")
    botao.click()
    t.esperar(lambda: t.n_status() == k, f"Data Explorer: status nao mostra {k}")
    assert t.sidebar_selecao() == f"Seleção: {k} instâncias"
    t.esperar(lambda: (t.pontos(EXPLORER) or {}).get("hi") == k, "Data Explorer nao destacou")
    t.aba("Instance Space")
    t.esperar(lambda: t.n_status() == k, f"Instance Space: status nao mostra {k}")
    t.esperar(lambda: (t.pontos(ESPACO) or {}).get("sel") == k, "Instance Space nao destacou")
    t.aba("Footprint Performance")
    t.esperar(lambda: (t.pontos("Footprints") or {}).get("hi") == k, "Footprint Performance nao destacou")
    t.aba("Distributions")
    t.esperar(lambda: _n_distribuicoes(t) == k, "Distributions nao mostrou a selecao")


def test_agrupar_por_class_no_iris_produz_3_grupos(tela):
    t, url = tela
    t.abrir(url, "iris")
    t.aba("Distributions")
    t.page.get_by_label("Agrupar por", exact=True).select_option(label="class")
    plot = t.esperar(lambda: next((p for p in t.plots() if "por class" in p["titulo"]
                                   and p["fatores"]), None), "distribuicao por class nao apareceu")
    classes = sorted(pd.read_csv(RAIZ / "resultados" / "table_iris.csv")["class"].unique())
    assert len(plot["fatores"]) == 3                      # 3 grupos -> violino por padrao
    assert [f.split(" (n=")[0] for f in plot["fatores"]] == classes
    assert all("(n=50)" in f for f in plot["fatores"])


@pytest.mark.parametrize("dataset", DATASETS)
def test_csv_exportado_da_selecao_tem_as_linhas_selecionadas(tela, dataset):
    t, url = tela
    t.abrir(url, dataset)
    t.lasso_com_pontos()
    n = t.esperar(lambda: t.n_status(), "o lasso nao selecionou nada")
    rotulos = t.esperar(lambda: (r := t.page.evaluate(SELECIONADAS_JS, ESPACO)) and len(r) == n and r,
                        "rotulos selecionados nao conferem com o status")
    botao = t.page.get_by_role("button", name=f"Exportar seleção ({n})")
    t.esperar(lambda: botao.is_enabled(), "botao de exportar a selecao nao habilitou")
    with t.page.expect_download() as info:
        botao.click()
    csv = pd.read_csv(info.value.path(), dtype={"instances": str})
    assert info.value.suggested_filename == f"{dataset}_selecao.csv"
    assert len(csv) == n and sorted(csv["instances"]) == sorted(rotulos)
    meta = pd.read_csv(PASTA_IS / dataset / "metadata.csv", nrows=1)
    features = [c for c in meta.columns if c.startswith("feature_")]
    algos = [c for c in meta.columns if c.startswith("algo_")]
    for col in ["instances", "class", "ih", "n_wrong", *features, *algos, "z_1", "z_2",
                "NumGoodAlgos", "IsBetaEasy", "best_algo", "best_algo_svm"]:
        assert col in csv.columns, col


def test_aba_features_do_iris_lista_todas_as_features_e_as_degeneradas(tela):
    t, url = tela
    t.abrir(url, "iris")
    t.aba("Features")
    dados = t.esperar(lambda: t.page.evaluate(TABELA_JS, "feature"), "tabela de features nao apareceu")
    tabela = pd.read_csv(RAIZ / "resultados" / "table_iris.csv", nrows=1)
    recebidas = [c[len("feature_"):] for c in tabela.columns if c.startswith("feature_")]
    assert sorted(dados["feature"]) == sorted(recebidas) and len(dados["feature"]) == 19
    degeneradas = {f for f, st in zip(dados["feature"], dados["status"]) if st == "dropped_degenerate"}
    assert degeneradas == {"kDN", "MV", "CB", "N1", "Harmfulness"}
    assert t.page.get_by_text("19 features recebidas").count() == 1
    assert any(p["titulo"].startswith("rho de Pearson") for p in t.plots())      # heatmap
    assert any("k usado = 6" in p["titulo"] for p in t.plots())                  # silhueta


# --------------------------------------------------------------------------- #
# Novo instance space (upload, validacao, execucao em subprocesso)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def metadata_exemplo(request):
    """metadata.csv de exemplo do repositorio do instancespace (tag v0.3.0),
    baixado uma vez para o cache do pytest (licenca nao comercial: nao vai
    para o repositorio)."""
    destino = request.config.cache.mkdir("instancespace_v0.3.0") / "metadata.csv"
    if not destino.is_file():
        try:
            with urllib.request.urlopen(URL_EXEMPLO, timeout=30) as resp:
                destino.write_bytes(resp.read())
        except OSError as exc:
            pytest.skip(f"sem acesso ao metadata de exemplo ({exc})")
    return destino


def _grupos_do_seletor(t):
    return t.page.get_by_label("Dataset", exact=True).evaluate(
        "e => Object.fromEntries([...e.querySelectorAll('optgroup')].map("
        "g => [g.label, [...g.querySelectorAll('option')].map(o => o.text)]))")


def _conferir_todas_as_abas(t, n):
    """Abre as seis abas do dataset ativo; o scatter tem n pontos e nenhuma
    aba mostra traceback."""
    t.aba("Instance Space")
    t.esperar(lambda: (t.pontos(ESPACO) or {}).get("n") == n, f"scatter nao tem {n} pontos")
    t.aba("Footprint Performance")
    t.esperar(lambda: (t.pontos("Footprints") or {}).get("n") == n, "mapa de footprints")
    t.aba("Algorithm Selection")
    t.esperar(lambda: (t.pontos("Algoritmo recomendado") or {}).get("n") == n,
              "mapa da Algorithm Selection")
    t.aba("Distributions")
    t.esperar(lambda: len(t.plots()) >= 1, "Distributions sem graficos")
    t.aba("Features")
    t.esperar(lambda: t.page.get_by_text(re.compile(r"\d+ features recebidas")).count() == 1,
              "Features sem resumo")
    t.aba("Data Explorer")
    t.esperar(lambda: (t.pontos(EXPLORER) or {}).get("n") == n, "Data Explorer")
    assert t.page.get_by_text("Traceback").count() == 0


def test_upload_quebrado_mostra_erro_sem_traceback(tela, tmp_path):
    t, url = tela
    t.abrir(url, "iris")
    t.abrir_novo()
    quebrado = tmp_path / "quebrado.csv"
    quebrado.write_text("instances,feature_a,feature_b,algo_x,algo_y\n1,1,2,0.1,0.2\n1,x,4,0.5,0.6\n")
    t.enviar("novo-metadata", quebrado)
    texto = t.esperar(lambda: t.texto("novo-erros"), "o erro de validacao nao apareceu")
    assert "pelo menos 3 colunas feature_*" in texto and "o arquivo tem 2" in texto
    assert t.page.get_by_text("Traceback").count() == 0
    assert t.page.get_by_role("button", name="Rodar ISA").is_disabled()


def test_upload_do_exemplo_do_instancespace_roda_e_abre_as_abas(tela, pasta_runs, metadata_exemplo):
    """Metadata do repositorio do instancespace, sem anotacoes: escolher a
    direcao, rodar, e abrir todas as abas do resultado."""
    t, url = tela
    t.abrir(url, "iris")
    t.abrir_novo()
    t.enviar("novo-metadata", metadata_exemplo)
    t.esperar(lambda: t.page.get_by_text(re.compile(r"212 instâncias, 10 features, 10 algoritmos"))
              .count() == 1, "resumo do metadata nao apareceu")
    botao = t.page.get_by_role("button", name="Rodar ISA")
    assert botao.is_disabled()                       # direcao ainda nao escolhida
    assert t.page.get_by_text(re.compile("falta escolher: a direção do desempenho")).count() == 1
    t.escolher_regra("menor é melhor", "absoluto", 0.2)          # options.json do exemplo
    t.esperar(lambda: t.page.get_by_text("Prévia (regra do PRELIM):", exact=False).count() >= 1,
              "previa da fracao de boas nao apareceu")
    t.nomear("exemplo_e2e")
    nome, vistos = t.rodar("exemplo_e2e")
    assert any(re.search(r"estágio \d de 7", v) for v in vistos), vistos
    t.esperar(lambda: "Concluído" in t.texto("novo-status"), "status nao mostra Concluído")
    pasta = pasta_runs / nome
    assert (pasta / "run_info.json").is_file() and (pasta / "execucao.log").is_file()
    assert (pasta / "entrada" / "metadata.csv").read_bytes() == metadata_exemplo.read_bytes()
    opcoes = json.loads((pasta / "run_options.json").read_text())
    assert opcoes["perf"] == {**opcoes["perf"], "max_perf": False, "abs_perf": True, "epsilon": 0.2}
    assert opcoes["trace"]["use_sim"] is False and opcoes["sifted"]["k"] == 6
    grupos = _grupos_do_seletor(t)
    assert nome in grupos["runs (execuções pela interface)"]
    assert "iris" in grupos["resultados/is"] and nome not in grupos["resultados/is"]
    n = len(pd.read_csv(pasta / "coordinates.csv"))
    _conferir_todas_as_abas(t, n)


def test_ciclo_exportar_selecao_e_subir_como_metadata(tela, pasta_runs, tmp_path):
    """Exporta a selecao (filtro z_1 < 0 do diabetes), sobe o CSV exportado
    como metadata novo, roda, e confere que a contagem de instancias bate."""
    t, url = tela
    t.abrir(url, "diabetes")
    t.aba("Data Explorer")
    t.page.get_by_label("Filtro (pandas query)", exact=True).fill("z_1 < 0")
    t.page.get_by_label("Filtro (pandas query)", exact=True).press("Enter")
    usar = t.page.get_by_role("button", name="Usar filtro como seleção")
    t.esperar(lambda: usar.is_enabled(), "botao usar filtro nao habilitou")
    usar.click()
    n = t.esperar(lambda: t.n_status(), "o filtro nao virou selecao")
    esperado = int((pd.read_csv(PASTA_IS / "diabetes" / "coordinates.csv")["z_1"] < 0).sum())
    assert n == esperado
    botao = t.page.get_by_role("button", name=f"Exportar seleção ({n})")
    t.esperar(lambda: botao.is_enabled(), "botao de exportar a selecao nao habilitou")
    with t.page.expect_download() as info:
        botao.click()
    exportado = tmp_path / "diabetes_selecao.csv"
    info.value.save_as(exportado)
    assert len(pd.read_csv(exportado)) == n

    t.abrir_novo()
    t.enviar("novo-metadata", exportado)
    t.esperar(lambda: t.page.get_by_text(re.compile(rf"{n} instâncias, ")).count() == 1,
              "resumo do metadata exportado nao apareceu")
    t.escolher_regra("maior é melhor", "absoluto", 0.5)
    t.nomear("ciclo_e2e")
    nome, _ = t.rodar("ciclo_e2e")
    pasta = pasta_runs / nome
    assert len(pd.read_csv(pasta / "coordinates.csv")) == n
    assert json.loads((pasta / "run_info.json").read_text())["n_instancias"] == n
    t.esperar(lambda: t.page.get_by_text(re.compile(rf"Instâncias: {n}\b")).count() >= 1,
              f"sidebar nao mostra {n} instancias")
    t.aba("Instance Space")
    t.esperar(lambda: (t.pontos(ESPACO) or {}).get("n") == n, f"scatter nao tem {n} pontos")


def test_direcao_invertida_dispara_aviso(tela):
    """iris com 'menor é melhor' (o certo e maior): as frações de boas caem
    abaixo de 5% e o aviso aparece, sem bloquear o botao."""
    t, url = tela
    t.abrir(url, "iris")
    t.abrir_novo()
    t.enviar("novo-metadata", PASTA_IS / "iris" / "metadata.csv")
    t.esperar(lambda: t.page.get_by_text(re.compile(r"150 instâncias, ")).count() == 1,
              "resumo do iris nao apareceu")
    t.escolher_regra("menor é melhor", "absoluto", 0.5)
    texto = t.esperar(lambda: t.texto("novo-aviso-direcao"), "aviso de direcao nao apareceu")
    assert "Confira a direção" in texto and "knn" in texto and "menos de 5,0%" in texto
    assert t.page.get_by_role("button", name="Rodar ISA").is_enabled()


def test_algorithm_selection_avisa_seletor_trivial_no_iris(tela):
    t, url = tela
    t.abrir(url, "iris")
    t.aba("Algorithm Selection")
    texto = t.esperar(lambda: t.texto("as-trivial"), "aviso de seletor trivial nao apareceu no iris")
    assert "Seletor quase trivial" in texto and "logreg" in texto and "143 de 150" in texto
    assert t.page.get_by_text(re.compile(r"pr0_sub.*fora da amostra")).count() >= 1
    t.esperar(lambda: (t.pontos("Algoritmo recomendado") or {}).get("n") == 150, "mapa do iris")
    assert len(t.page.locator(".as-confusao").all()) == 6
    t.aba("Instance Space")               # escolher_dataset espera o scatter da aba 0
    t.escolher_dataset("hill-valley")     # logreg em 853 de 1212 (70%): sem aviso
    t.aba("Algorithm Selection")
    t.esperar(lambda: (t.pontos("Algoritmo recomendado") or {}).get("n") == 1212,
              "mapa do hill-valley")
    assert t.page.locator(".as-trivial").count() == 0


@pytest.mark.parametrize("dataset", DATASETS)
def test_algorithm_selection_destaca_a_selecao(tela, dataset):
    t, url = tela
    t.abrir(url, dataset)
    t.lasso_com_pontos()
    n = t.esperar(lambda: t.n_status(), "o lasso nao selecionou nada")
    t.aba("Algorithm Selection")
    t.esperar(lambda: t.n_status() == n, f"Algorithm Selection: status nao mostra {n}")
    t.esperar(lambda: (t.pontos("Algoritmo recomendado") or {}).get("hi") == n,
              f"Algorithm Selection: mapa nao destaca {n} pontos")
