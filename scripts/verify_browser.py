"""Teste de regressao da interface (isaspace.ui.app) num navegador real.

Controla um Edge ou Chrome headless pelo DevTools Protocol (so stdlib +
tornado, que ja e dependencia do Bokeh) e, contra um servidor JA RODANDO,
executa os cenarios:

  datasets    troca o dataset pelo seletor da pagina, ida e volta pelos que
              existem em resultados/isa/, e confere <title>, cabecalho,
              seletor, bloco de informacoes e n de pontos nos graficos;
  abas        em cada dataset, percorre as quatro abas nos dois sentidos e
              confere que o bloco inferior da sidebar (h2 e labels) e o
              conteudo do painel correspondem a aba ativa;
  controles   mexe nos controles de cada aba (algoritmo, good/best, tipo de
              grafico, eixo x, filtro) e confere o efeito no painel;
  titulo      navega pelas abas e SO DEPOIS troca o dataset, caminho que ja
              dessincronizou o titulo no passado;
  desconexao  (opcional, --desconexao PORTA) sobe um servidor proprio nessa
              porta, carrega a pagina, mata o servidor e confere que a faixa
              "conexao perdida" aparece.

Uso, da raiz do projeto com o .venv ativo e o app servido em --url:
    python scripts/verify_browser.py [--url http://localhost:5006/]
                                     [--desconexao 5012] [--saida PASTA]
Sai com codigo 0 se tudo passou. Screenshots dos passos vao para --saida.
"""

import argparse
import asyncio
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from tornado.websocket import websocket_connect

RAIZ = Path(__file__).resolve().parents[1]
PASTA_ISA = RAIZ / "resultados" / "isa"
TABS = ["Instance Space", "Footprint Performance", "Distributions", "Data Explorer"]
LABELS = {
    0: ["Cor dos pontos"],
    1: ["Algoritmo"],
    2: ["Variaveis"],
    3: ["Eixo x", "Eixo y", "Cor", "Filtro (pandas query)"],
}
NAVEGADORES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
]
URL = "http://localhost:5006/"

# ------------------------------------------------------------------ JS lido/executado na pagina
ESTADO_JS = r"""
(() => {
  const q = (s, r) => [...(r || document).querySelectorAll(s)];
  const sel = q('select').find(s => [...s.options].some(o => o.value === 'iris'));
  const h = document.querySelector('#header-items');
  const sb = document.querySelector('#sidebar');
  const p = q('p', sb).find(p => p.innerText.includes('Instancias:'));
  const linhas = p ? p.innerText.split('\n').map(s => s.trim()) : [];
  const main = document.querySelector('#main');
  let pontos = [], titulos = [], eixos = [], glifos = [];
  try {
    for (const m of Bokeh.documents[0]._all_models.values()) {
      if (m.type === 'ColumnDataSource' && m.data && m.data.z_1 !== undefined) pontos.push(m.data.z_1.length);
      if (m.type === 'Title' && m.text) titulos.push(m.text);
      if ((m.type === 'LinearAxis' || m.type === 'CategoricalAxis') && m.axis_label) eixos.push(m.axis_label);
      if (m.type === 'Quad' || m.type === 'VBar' || m.type === 'Patches') glifos.push(m.type);
    }
  } catch (e) { pontos = ['erro: ' + e]; }
  return {
    title: document.title,
    header: h ? h.innerText.trim() : null,
    seletor: sel ? sel.value : null,
    info: linhas[0] || null,
    pontos, titulos, eixos, glifos: [...new Set(glifos)],
    h2: q('h2', sb).map(x => x.innerText.trim()),
    labels: q('label', sb).map(l => l.innerText.trim()).filter(Boolean),
    active: q('.bk-tab').findIndex(t => t.classList.contains('bk-active')),
    h3: q('h3', main).map(x => x.innerText.trim()).filter(Boolean),
    sidebarTexto: sb ? sb.innerText : '',
    faixa: !!document.getElementById('isa-conexao-perdida'),
  };
})()
"""
TAB_JS = "document.querySelectorAll('.bk-tab')[%d].click(); 'ok'"
SELECT_JS = r"""((rotulo, valor) => {
  const lab = [...document.querySelectorAll('label')].find(l => l.innerText.trim() === rotulo);
  const sel = lab.parentElement.querySelector('select');
  sel.value = valor; sel.dispatchEvent(new Event('change', {bubbles: true})); return sel.value;
})(%s, %s)"""
SELECT_OPCOES_JS = r"""(rotulo => {
  const lab = [...document.querySelectorAll('label')].find(l => l.innerText.trim() === rotulo);
  return [...lab.parentElement.querySelector('select').options].map(o => o.value);
})(%s)"""
BOTAO_JS = r"""(texto => {
  // RadioButtonGroup do Bokeh 2.4: os botoes sao <div class="bk bk-btn">, nao <button>
  const b = [...document.querySelectorAll('.bk-btn-group .bk-btn')].find(b => b.innerText.trim() === texto);
  b.click(); return b.innerText.trim();
})(%s)"""
INPUT_JS = r"""((rotulo, valor) => {
  const lab = [...document.querySelectorAll('label')].find(l => l.innerText.trim() === rotulo);
  const inp = lab.parentElement.querySelector('input');
  inp.value = valor; inp.dispatchEvent(new Event('change', {bubbles: true})); return inp.value;
})(%s, %s)"""


def achar_navegador(caminho):
    if caminho:
        return caminho
    for c in NAVEGADORES:
        if Path(c).is_file():
            return c
    for nome in ("msedge", "chrome", "google-chrome", "chromium"):
        w = shutil.which(nome)
        if w:
            return w
    sys.exit("nenhum Edge/Chrome encontrado; passe --navegador CAMINHO")


def porta_livre():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Navegador:
    """Edge/Chrome headless controlado por DevTools Protocol."""

    def __init__(self, exe, pasta):
        self.exe, self.pasta = exe, pasta
        self.port = porta_livre()
        self.proc = self.ws = None
        self._id = 0

    async def __aenter__(self):
        self.proc = subprocess.Popen(
            [self.exe, "--headless=new", "--disable-gpu", "--no-first-run",
             f"--remote-debugging-port={self.port}",
             f"--user-data-dir={self.pasta / 'perfil'}", "--window-size=1500,950",
             "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        alvo = None
        for _ in range(60):
            try:
                alvos = json.load(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json"))
                alvo = next(t for t in alvos if t["type"] == "page")
                break
            except Exception:
                await asyncio.sleep(0.5)
        if alvo is None:
            raise RuntimeError("navegador nao respondeu ao DevTools Protocol")
        self.ws = await websocket_connect(alvo["webSocketDebuggerUrl"],
                                          max_message_size=200 * 1024 * 1024)
        await self.call("Page.enable")
        await self.call("Runtime.enable")
        return self

    async def __aexit__(self, *exc):
        if self.ws:
            self.ws.close()
        if self.proc:
            self.proc.terminate()

    async def call(self, method, **params):
        self._id += 1
        await self.ws.write_message(json.dumps({"id": self._id, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.read_message())
            if msg.get("id") == self._id:
                return msg.get("result", msg)

    async def js(self, expr):
        r = await self.call("Runtime.evaluate", expression=expr, returnByValue=True)
        return r.get("result", {}).get("value")

    async def estado(self):
        return await self.js(ESTADO_JS)

    async def esperar(self, cond, timeout=20.0):
        """Le o estado ate cond(estado) ser verdade (ou estourar o tempo)."""
        fim = time.monotonic() + timeout
        st = await self.estado()
        while not cond(st) and time.monotonic() < fim:
            await asyncio.sleep(0.5)
            st = await self.estado()
        return st

    async def abrir(self, url):
        await self.call("Page.navigate", url=url)
        return await self.esperar(
            lambda st: bool(st and st["seletor"] and st["pontos"]
                            and st["title"].endswith(st["seletor"])),
            timeout=60)

    async def screenshot(self, nome):
        r = await self.call("Page.captureScreenshot", format="png")
        (self.pasta / nome).write_bytes(base64.b64decode(r["data"]))


# ------------------------------------------------------------------ verificacoes
class Relatorio:
    def __init__(self):
        self.linhas, self.falhas = [], 0

    def registra(self, cenario, passo, problemas, st=None):
        ok = not problemas
        self.falhas += not ok
        detalhe = "" if ok else f"  problemas={problemas} estado={_resumo(st)}"
        linha = f"{'OK   ' if ok else 'FALHA'} [{cenario}] {passo}{detalhe}"
        self.linhas.append(linha)
        print(linha, flush=True)


def _resumo(st):
    if not st:
        return st
    chaves = ("title", "header", "seletor", "info", "pontos", "h2", "labels", "active", "h3", "faixa")
    return {k: st.get(k) for k in chaves}


def n_instancias(ds):
    with open(PASTA_ISA / ds / "coordinates.csv", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def checa_dataset(st, ds):
    prob = []
    if not st:
        return ["sem-estado"]
    if st["title"] != f"isa-instance - {ds}":
        prob.append("title")
    if not (st["header"] or "").endswith(ds):
        prob.append("header")
    if st["seletor"] != ds:
        prob.append("seletor")
    if st["info"] != ds:
        prob.append("info")
    if n_instancias(ds) not in st["pontos"]:
        prob.append("pontos")
    if st["faixa"]:
        prob.append("faixa-conexao-perdida")
    return prob


def checa_aba(st, aba, ds):
    """Sidebar e painel correspondem a aba ativa (com os controles no padrao)."""
    prob = []
    if not st:
        return ["sem-estado"]
    n = n_instancias(ds)
    if st["active"] != aba:
        prob.append("aba-ativa")
    if st["h2"] != [TABS[aba]]:
        prob.append("sidebar-h2")
    if any(l not in st["labels"] for l in LABELS[aba]):
        prob.append("sidebar-labels")
    if aba == 0 and not ("Espaco de instancias (PILOT)" in st["titulos"] and n in st["pontos"]):
        prob.append("painel-espaco")
    if aba == 1 and not ("footprint_performance.csv" in st["h3"]
                         and any(t.startswith("Footprints ") for t in st["titulos"])):
        prob.append("painel-footprint")
    if aba == 2 and not any(f"(Todas n={n}" in t for t in st["titulos"]):
        prob.append("painel-distribuicoes")
    if aba == 3 and f"Linhas filtradas ({n})" not in st["h3"]:
        prob.append("painel-explorer")
    return prob


async def cenario_datasets(nav, rel, datasets):
    st = await nav.abrir(URL)
    rel.registra("datasets", "carga inicial", checa_dataset(st, st["seletor"]), st)
    ordem = datasets[1:] + datasets[-2::-1] + datasets[1:2]      # ida, volta, +1
    for ds in ordem:
        await nav.js(SELECT_JS % (json.dumps("Dataset"), json.dumps(ds)))
        st = await nav.esperar(lambda s, d=ds: not checa_dataset(s, d))
        rel.registra("datasets", f"-> {ds}", checa_dataset(st, ds), st)
    await nav.screenshot("datasets_final.png")


async def cenario_abas(nav, rel, datasets):
    await nav.abrir(URL)
    for ds in datasets:
        await nav.js(SELECT_JS % (json.dumps("Dataset"), json.dumps(ds)))
        st = await nav.esperar(lambda s, d=ds: not checa_dataset(s, d))
        rel.registra("abas", f"dataset {ds}", checa_dataset(st, ds), st)
        for aba in (1, 2, 3, 2, 1, 0, 3, 0):
            await nav.js(TAB_JS % aba)
            st = await nav.esperar(
                lambda s, a=aba, d=ds: not checa_aba(s, a, d) and not checa_dataset(s, d))
            rel.registra("abas", f"{ds}: aba {aba} ({TABS[aba]})",
                         checa_aba(st, aba, ds) + checa_dataset(st, ds), st)
        await nav.screenshot(f"abas_{ds}.png")


async def cenario_controles(nav, rel, datasets):
    ds = datasets[0]
    await nav.abrir(URL)
    n = n_instancias(ds)
    # aba 1: algoritmo e good/best mudam o titulo do mapa
    await nav.js(TAB_JS % 1)
    await nav.esperar(lambda s: not checa_aba(s, 1, ds))
    algos = await nav.js(SELECT_OPCOES_JS % json.dumps("Algoritmo"))
    algo = algos[1]
    await nav.js(SELECT_JS % (json.dumps("Algoritmo"), json.dumps(algo)))
    st = await nav.esperar(lambda s: f"Footprints good -- {algo}" in s["titulos"])
    rel.registra("controles", f"aba 1: Algoritmo={algo}",
                 [] if f"Footprints good -- {algo}" in st["titulos"] else ["titulo-mapa"], st)
    await nav.js(BOTAO_JS % json.dumps("best"))
    st = await nav.esperar(lambda s: f"Footprints best -- {algo}" in s["titulos"])
    rel.registra("controles", "aba 1: Tipo=best",
                 [] if f"Footprints best -- {algo}" in st["titulos"] else ["titulo-mapa"], st)
    await nav.js(BOTAO_JS % json.dumps("good"))
    await nav.js(SELECT_JS % (json.dumps("Algoritmo"), json.dumps("todos")))
    st = await nav.esperar(lambda s: "Footprints good -- todos" in s["titulos"])
    rel.registra("controles", "aba 1: volta a good/todos",
                 [] if "Footprints good -- todos" in st["titulos"] else ["titulo-mapa"], st)
    # aba 2: histograma (Quad) x boxplot (VBar)
    await nav.js(TAB_JS % 2)
    st = await nav.esperar(lambda s: not checa_aba(s, 2, ds) and "Quad" in s["glifos"])
    rel.registra("controles", "aba 2: histograma (padrao)",
                 [] if "Quad" in st["glifos"] else ["sem-histograma"], st)
    await nav.js(BOTAO_JS % json.dumps("boxplot"))
    st = await nav.esperar(lambda s: "VBar" in s["glifos"] and "Quad" not in s["glifos"])
    rel.registra("controles", "aba 2: Tipo=boxplot",
                 [] if ("VBar" in st["glifos"] and "Quad" not in st["glifos"]) else ["sem-boxplot"], st)
    await nav.js(BOTAO_JS % json.dumps("histograma"))
    st = await nav.esperar(lambda s: "Quad" in s["glifos"] and "VBar" not in s["glifos"])
    rel.registra("controles", "aba 2: volta a histograma",
                 [] if ("Quad" in st["glifos"] and "VBar" not in st["glifos"]) else ["sem-histograma"], st)
    # aba 3: filtro reduz as linhas; eixo x muda o titulo do scatter
    await nav.js(TAB_JS % 3)
    await nav.esperar(lambda s: not checa_aba(s, 3, ds))
    await nav.js(INPUT_JS % (json.dumps("Filtro (pandas query)"), json.dumps("ih > 0.5")))
    st = await nav.esperar(lambda s: any(h.startswith("Linhas filtradas (")
                                         and h != f"Linhas filtradas ({n})" for h in s["h3"]))
    h = next((h for h in st["h3"] if h.startswith("Linhas filtradas (")), "")
    k = int(h.split("(")[1].rstrip(")")) if "(" in h else -1
    ok = 0 <= k < n and f"{k} de {n} linhas" in st["sidebarTexto"]
    rel.registra("controles", f"aba 3: filtro ih > 0.5 -> {k} de {n} linhas",
                 [] if ok else ["filtro"], st)
    await nav.js(INPUT_JS % (json.dumps("Filtro (pandas query)"), json.dumps("")))
    st = await nav.esperar(lambda s: f"Linhas filtradas ({n})" in s["h3"])
    rel.registra("controles", "aba 3: filtro limpo",
                 [] if f"Linhas filtradas ({n})" in st["h3"] else ["filtro"], st)
    await nav.js(SELECT_JS % (json.dumps("Eixo x"), json.dumps("d_1")))
    st = await nav.esperar(lambda s: "d_1 x z_2 (cor: ih)" in s["titulos"])
    rel.registra("controles", "aba 3: Eixo x=d_1",
                 [] if "d_1 x z_2 (cor: ih)" in st["titulos"] else ["titulo-scatter"], st)
    await nav.screenshot("controles_final.png")


async def cenario_titulo(nav, rel, datasets):
    """Abas primeiro, dataset depois (ida e volta); o titulo tem de seguir."""
    await nav.abrir(URL)
    ds0 = datasets[0]
    for aba in (1, 2, 3, 0):
        await nav.js(TAB_JS % aba)
        await nav.esperar(lambda s, a=aba: not checa_aba(s, a, ds0))
    for ds in datasets[1:] + datasets[-2::-1]:
        await nav.js(SELECT_JS % (json.dumps("Dataset"), json.dumps(ds)))
        st = await nav.esperar(lambda s, d=ds: not checa_dataset(s, d))
        rel.registra("titulo", f"apos abas -> {ds}", checa_dataset(st, ds), st)
        await nav.js(TAB_JS % 2)
        st = await nav.esperar(lambda s, d=ds: not checa_aba(s, 2, d))
        rel.registra("titulo", f"{ds}: aba 2 apos a troca",
                     checa_aba(st, 2, ds) + checa_dataset(st, ds), st)
        # espera a sidebar acompanhar a aba antes do proximo passo: trocar o
        # dataset enquanto o Bokeh ainda re-renderiza a sidebar (algo que
        # nenhum humano faz em ~100 ms) pode deixar o <select> visual atras
        await nav.js(TAB_JS % 0)
        await nav.esperar(lambda s, d=ds: not checa_aba(s, 0, d))
    await nav.screenshot("titulo_final.png")


async def cenario_desconexao(nav, rel, porta):
    """Servidor proprio: carrega, mata o servidor, espera a faixa vermelha."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "isaspace.ui.app", "--port", str(porta), "--no-show"],
        cwd=str(RAIZ), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://localhost:{porta}/", timeout=2)
                break
            except Exception:
                await asyncio.sleep(1)
        await nav.call("Page.navigate", url=f"http://localhost:{porta}/")
        await nav.esperar(lambda s: bool(s and s["seletor"] and s["pontos"]), timeout=60)
        await asyncio.sleep(8)                         # >= 1 batimento com a sessao viva
        st = await nav.estado()
        rel.registra("desconexao", "sessao viva: sem faixa", ["faixa-indevida"] if st["faixa"] else [], st)
        proc.kill()
        proc.wait()
        st = await nav.esperar(lambda s: s["faixa"], timeout=45)
        rel.registra("desconexao", "servidor morto: faixa apareceu", [] if st["faixa"] else ["sem-faixa"], st)
        await nav.screenshot("desconexao.png")
    finally:
        if proc.poll() is None:
            proc.kill()


async def main(args):
    global URL
    URL = args.url
    datasets = sorted(p.parent.name for p in PASTA_ISA.glob("*/coordinates.csv"))
    if len(datasets) < 2:
        sys.exit(f"precisa de ao menos 2 datasets em {PASTA_ISA}")
    pasta = Path(args.saida) if args.saida else Path(tempfile.mkdtemp(prefix="verify_browser_"))
    pasta.mkdir(parents=True, exist_ok=True)
    print(f"datasets: {datasets} | navegador: {args.navegador} | saida: {pasta}")
    rel = Relatorio()
    async with Navegador(args.navegador, pasta) as nav:
        await cenario_datasets(nav, rel, datasets)
        await cenario_abas(nav, rel, datasets)
        await cenario_controles(nav, rel, datasets)
        await cenario_titulo(nav, rel, datasets)
        if args.desconexao:
            await cenario_desconexao(nav, rel, args.desconexao)
    total = len(rel.linhas)
    print(f"\nRESULTADO: {'PASSOU' if not rel.falhas else 'FALHOU'} "
          f"({total - rel.falhas}/{total} passos OK) | screenshots em {pasta}")
    return 0 if not rel.falhas else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=URL)
    ap.add_argument("--navegador", default=None, help="caminho do msedge.exe/chrome.exe")
    ap.add_argument("--saida", default=None, help="pasta dos screenshots")
    ap.add_argument("--desconexao", type=int, default=None, metavar="PORTA",
                    help="tambem testa a faixa de sessao perdida, subindo um servidor nessa porta")
    a = ap.parse_args()
    a.navegador = achar_navegador(a.navegador)
    sys.exit(asyncio.run(main(a)))
