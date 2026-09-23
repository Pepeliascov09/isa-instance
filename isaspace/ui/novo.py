"""Bloco "Novo instance space" da sidebar: envio de um metadata.csv, validacao
imediata, escolha da regra de desempenho e execucao do engine em subprocesso.

- Validacao (isaspace.ui.upload) a cada arquivo enviado; erros e avisos viram
  texto na tela, nunca traceback.
- A direcao do desempenho (maior ou menor e melhor) e o tipo do limiar
  (absoluto ou relativo) comecam SEM valor: o botao Rodar so habilita depois
  da escolha e de um epsilon. Com a regra escolhida, a previa mostra a fracao
  de instancias boas por algoritmo (a mesma regra do PRELIM) e avisa quando
  algum fica abaixo de 5% ou acima de 95%.
- A execucao roda em subprocesso (isaspace.ui.execucao) e grava em
  runs/<nome>_<AAAAMMDD-HHMMSS>/; o progresso por estagio vem da saida do
  subprocesso, lida numa thread que repassa as atualizacoes ao documento da
  sessao (Document.add_next_tick_callback, o unico metodo do Bokeh seguro
  fora da thread do servidor). A interface continua usavel enquanto roda.

Este modulo nao importa o engine nem o instancespace.
"""

from pathlib import Path

import numpy as np
import panel as pn
from panel.io.state import set_curdoc

from isaspace.ui import execucao
from isaspace.ui.upload import (
    FRACAO_MAX, FRACAO_MIN, LIMIAR_AVISO_TEMPO, TEMPOS_ALGOS, TEMPOS_FEATURES, fora_da_faixa,
    fracao_boas, linhas_do_prelim, tempo_estimado, validar_anotacoes, validar_feature_info,
    validar_metadata,
)

ESCOLHA = "— escolha —"
DIRECOES = {ESCOLHA: "", "maior é melhor": "max", "menor é melhor": "min"}
LIMIARES = {ESCOLHA: "", "absoluto": "abs", "relativo ao melhor da instância": "rel"}
K_PADRAO = 6                 # SiftedOptions.k do instancespace 0.3.0
REGRAS = {
    ("max", "abs"): "boa para o algoritmo se algo_* ≥ ε",
    ("min", "abs"): "boa para o algoritmo se algo_* ≤ ε",
    ("max", "rel"): "boa se algo_* está até ε·100% abaixo do melhor da instância: "
                    "(1 − algo_*/melhor) ≤ ε",
    ("min", "rel"): "boa se algo_* está até ε·100% acima do melhor da instância: "
                    "(algo_*/melhor − 1) ≤ ε",
}


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%".replace(".", ",")


def formatar_tempo(s: float) -> str:
    if s < 90:
        return f"{max(5, round(s / 5) * 5):.0f} s"
    return f"{s / 60:.0f} min".replace(".", ",")


class NovoInstanceSpace:
    """Card da sidebar; `ao_concluir(pasta)` roda na sessao quando uma
    execucao termina bem."""

    def __init__(self, runs, largura, ao_concluir):
        self.runs = Path(runs)
        self.ao_concluir = ao_concluir
        self.validacao = None        # upload.Validacao do metadata atual
        self._erros_aux = []         # annotations.json / feature_info.csv
        self.execucao = None         # execucao.Execucao em andamento ou a ultima
        self._relogio = None
        L = dict(width=largura)

        self.w_meta = pn.widgets.FileInput(accept=".csv", css_classes=["novo-metadata"], **L)
        self.w_ann = pn.widgets.FileInput(accept=".json", css_classes=["novo-annotations"], **L)
        self.w_finfo = pn.widgets.FileInput(accept=".csv", css_classes=["novo-feature-info"], **L)
        self.w_limpar = pn.widgets.Button(name="Limpar arquivos", button_type="light", **L)
        self.msg_validacao = pn.Column(**L)
        self.resumo = pn.pane.Markdown("", **L)
        self.w_nome = pn.widgets.TextInput(name="Nome da execução", **L)
        self.w_direcao = pn.widgets.Select(name="Direção do desempenho", options=DIRECOES, **L)
        self.w_limiar = pn.widgets.Select(name="Limiar", options=LIMIARES, **L)
        self.w_eps = pn.widgets.FloatInput(name="ε (epsilon)", value=None, step=0.05, **L)
        self.regra = pn.pane.Markdown("", **L)
        self.previa = pn.pane.Markdown("", **L)
        self.previa_aviso = pn.Column(**L)
        self.w_k = pn.widgets.IntInput(name="SIFTED k (clusters de features)", value=K_PADRAO,
                                       start=2, **L)
        self.w_usesim = pn.widgets.Checkbox(
            name="trace.usesim: footprints das previsões do PYTHIA", value=False)
        self.avancadas = pn.Card(
            self.w_k, pn.pane.Markdown(
                f"_Padrão do instancespace: k = {K_PADRAO}. Com menos features "
                "sobreviventes que k, o SIFTED não clusteriza._", width=largura - 20),
            self.w_usesim, pn.pane.Markdown(
                "_Desligado (padrão desta interface): footprints do desempenho observado._",
                width=largura - 20),
            title="Opções avançadas", collapsed=True, width=largura, margin=(5, 0))
        self.tempo = pn.Column(**L)
        self.w_rodar = pn.widgets.Button(name="Rodar ISA", button_type="primary", disabled=True, **L)
        self.falta = pn.pane.Markdown("", **L)
        self.progresso = pn.indicators.Progress(max=len(execucao.ESTAGIOS), value=0,
                                                visible=False, **L)
        self.status = pn.pane.Markdown("", css_classes=["novo-status"], **L)

        def rotulo(texto):
            return pn.pane.Markdown(texto, margin=(8, 10, 0, 10), **L)

        self.card = pn.Card(
            rotulo("**metadata.csv** (obrigatório)"), self.w_meta,
            rotulo("annotations.json (opcional)"), self.w_ann,
            rotulo("feature_info.csv (opcional)"), self.w_finfo, self.w_limpar,
            self.msg_validacao, self.resumo, self.w_nome,
            pn.pane.Markdown("### Regra de desempenho", margin=(0, 10)),
            self.w_direcao, self.w_limiar, self.w_eps, self.regra, self.previa, self.previa_aviso,
            self.avancadas, self.tempo, self.w_rodar, self.falta, self.progresso, self.status,
            title="Novo instance space", collapsed=True, width=largura + 20, margin=(5, 10),
            css_classes=["novo-card"])

        for w in (self.w_meta, self.w_ann, self.w_finfo):
            w.param.watch(lambda _: self._validar(), "value")
        for w in (self.w_direcao, self.w_limiar, self.w_eps):
            w.param.watch(lambda _: self._atualizar_regra(), "value")
        self.w_nome.param.watch(lambda _: self._atualizar_botao(), "value")
        self.w_limpar.on_click(self._on_limpar)
        self.w_rodar.on_click(self._on_rodar)
        self._validar()

    # ------------------------------------------------------------ validacao
    def _on_limpar(self, _):
        for w in (self.w_meta, self.w_ann, self.w_finfo):
            w.clear()
        self._validar()

    def _validar(self):
        self.validacao, self._erros_aux = None, []
        erros, avisos = [], []
        if self.w_meta.value:
            try:
                v = validar_metadata(self.w_meta.value)
            except Exception as exc:  # noqa: BLE001 -- nunca traceback na tela
                v = None
                erros.append(f"Não foi possível ler o metadata ({type(exc).__name__}: {exc}).")
            if v is not None:
                self.validacao = v
                erros += v.erros
                avisos += v.avisos
                if self.w_ann.value:
                    try:
                        _, e = validar_anotacoes(self.w_ann.value, v)
                    except Exception as exc:  # noqa: BLE001
                        e = [f"annotations.json ilegível ({type(exc).__name__}: {exc})."]
                    self._erros_aux += e
                if self.w_finfo.value:
                    try:
                        e = validar_feature_info(self.w_finfo.value, v)
                    except Exception as exc:  # noqa: BLE001
                        e = [f"feature_info.csv ilegível ({type(exc).__name__}: {exc})."]
                    self._erros_aux += e
                erros += self._erros_aux
            if not self.w_nome.value and self.w_meta.filename:
                self.w_nome.value = execucao.nome_seguro(Path(self.w_meta.filename).stem)
        elif self.w_ann.value or self.w_finfo.value:
            avisos.append("Envie também o metadata.csv.")
        blocos = []
        if erros:
            blocos.append(pn.pane.Alert(
                "**Metadata inválido, corrija e envie de novo:**\n\n"
                + "\n".join(f"- {e}" for e in erros), alert_type="danger",
                css_classes=["novo-erros"], sizing_mode="stretch_width"))
        if avisos:
            blocos.append(pn.pane.Alert("\n".join(f"- {a}" for a in avisos), alert_type="warning",
                                        css_classes=["novo-avisos"], sizing_mode="stretch_width"))
        self.msg_validacao.objects = blocos
        v = self.validacao
        if v is not None and not erros:
            anot = ", ".join(v.anotacoes) if v.anotacoes else "nenhuma"
            self.resumo.object = (
                f"✔ **{v.n} instâncias**, {len(v.features)} features, {len(v.algos)} "
                f"algoritmos. Anotações: {anot}."
                + (f" Source: `{v.source}`." if v.source else ""))
        else:
            self.resumo.object = ""
        self._atualizar_tempo()
        self._atualizar_regra()

    def _ok(self) -> bool:
        return self.validacao is not None and self.validacao.ok and not self._erros_aux

    # ----------------------------------------------------------------- regra
    def regra_escolhida(self):
        """(maior_melhor, absoluto, epsilon) ou None enquanto falta escolher."""
        d, lim, eps = self.w_direcao.value, self.w_limiar.value, self.w_eps.value
        if not d or not lim or eps is None or not np.isfinite(eps):
            return None
        return d == "max", lim == "abs", float(eps)

    def opcoes(self) -> dict:
        maior, absoluto, eps = self.regra_escolhida()
        return {"perf": {"max_perf": maior, "abs_perf": absoluto, "epsilon": eps},
                "sifted": {"k": int(self.w_k.value or K_PADRAO)},
                "trace": {"use_sim": bool(self.w_usesim.value)}}

    def _atualizar_regra(self):
        d, lim = self.w_direcao.value, self.w_limiar.value
        v = self.validacao
        texto = REGRAS.get((d, lim), "")
        if texto and self._ok():
            y = linhas_do_prelim(v).to_numpy(dtype=float)
            if np.isfinite(y).any():
                texto += (f"  \n_algo_* vai de {np.nanmin(y):.4g} a {np.nanmax(y):.4g}._")
            if lim == "rel" and (y < 0).any():
                texto += "  \n⚠ _Há algo_* negativos: a razão ao melhor perde o sentido._"
        self.regra.object = texto
        regra = self.regra_escolhida()
        if regra is None or not self._ok():
            self.previa.object = ""
            self.previa_aviso.objects = []
            self._atualizar_botao()
            return
        fr = fracao_boas(linhas_do_prelim(v), *regra)
        fora = fora_da_faixa(fr)
        linhas = ["| algoritmo | instâncias boas |", "|:--|--:|"]
        linhas += [f"| {a} | {_pct(f)}{' ⚠' if a in fora else ''} |" for a, f in fr.items()]
        self.previa.object = ("**Prévia (regra do PRELIM):** fração de instâncias boas\n\n"
                              + "\n".join(linhas))
        if fora:
            self.previa_aviso.objects = [pn.pane.Alert(
                f"⚠ **Confira a direção e o limiar.** {', '.join(fora)}: menos de "
                f"{_pct(FRACAO_MIN)} ou mais de {_pct(FRACAO_MAX)} das instâncias boas. "
                "Com a direção invertida, quase tudo vira bom ou quase tudo vira ruim.",
                alert_type="warning", css_classes=["novo-aviso-direcao"],
                sizing_mode="stretch_width")]
        else:
            self.previa_aviso.objects = []
        self._atualizar_botao()

    def _atualizar_tempo(self):
        v = self.validacao
        if not self._ok():
            self.tempo.objects = []
            return
        t = tempo_estimado(v.n, len(v.algos))
        if t is None:
            self.tempo.objects = []
            return
        base = (f"medido neste computador em metadata sintético com {TEMPOS_FEATURES} features e "
                f"{TEMPOS_ALGOS} algoritmos; o PYTHIA domina e cresce com o número de algoritmos")
        if v.n >= LIMIAR_AVISO_TEMPO:
            self.tempo.objects = [pn.pane.Alert(
                f"⏱ **Arquivo grande: {v.n} instâncias × {len(v.algos)} algoritmos.** Tempo "
                f"estimado: **~{formatar_tempo(t)}** ({base}). A interface continua usável "
                "enquanto roda.", alert_type="info", css_classes=["novo-tempo"],
                sizing_mode="stretch_width")]
        else:
            self.tempo.objects = [pn.pane.Markdown(
                f"_Tempo estimado: ~{formatar_tempo(t)} ({base})._", css_classes=["novo-tempo"])]

    def _atualizar_botao(self):
        rodando = self.execucao is not None and not self.execucao.terminou
        falta = []
        if not self._ok():
            falta.append("um metadata válido")
        if not self.w_direcao.value:
            falta.append("a direção do desempenho")
        if not self.w_limiar.value:
            falta.append("o tipo de limiar")
        if self.w_eps.value is None:
            falta.append("ε")
        self.w_rodar.disabled = bool(falta) or rodando
        self.falta.object = ("_Para rodar, falta escolher: " + ", ".join(falta) + "._"
                             if falta and not rodando else "")

    # -------------------------------------------------------------- execucao
    def _na_sessao(self, doc, funcao):
        """Envolve `funcao` para ser chamada da thread leitora e rodar na
        thread do servidor, dentro do documento da sessao."""
        def agendar(*args):
            def cb():
                with set_curdoc(doc):
                    funcao(*args)
            try:
                doc.add_next_tick_callback(cb)
            except Exception:  # noqa: BLE001 -- sessao fechada: nada a atualizar
                pass
        return agendar if doc is not None else funcao

    def _on_rodar(self, _):
        if self.w_rodar.disabled or self.regra_escolhida() is None or not self._ok():
            return
        try:
            pasta = execucao.nova_pasta(self.w_nome.value or "metadata", self.runs)
            meta = execucao.gravar_entrada(
                pasta, self.w_meta.value, self.w_ann.value or None, self.w_finfo.value or None)
        except OSError as exc:
            self.status.object = f"**Não foi possível criar a pasta da execução:** {exc}"
            return
        doc = pn.state.curdoc
        self.execucao = execucao.iniciar(
            meta, pasta, self.opcoes(),
            ao_estagio=self._na_sessao(doc, self._on_estagio),
            ao_terminar=self._na_sessao(doc, self._on_fim))
        self.progresso.value, self.progresso.visible = 0, True
        self.status.object = f"Iniciando em `runs/{pasta.name}`…"
        if doc is not None and doc.session_context is not None:
            self._relogio = pn.state.add_periodic_callback(self._tick, period=1000)
        self._atualizar_botao()

    def _texto_estagio(self, exe):
        est = exe.estagio
        if est is None:
            return f"Iniciando… · {exe.duracao:.0f} s"
        i = execucao.ESTAGIOS.index(est) if est in execucao.ESTAGIOS else 0
        return (f"Rodando **{est}** (estágio {i + 1} de {len(execucao.ESTAGIOS)}) · "
                f"{exe.duracao:.0f} s")

    def _tick(self):
        exe = self.execucao
        if exe is not None and not exe.terminou:
            self.status.object = self._texto_estagio(exe)

    def _on_estagio(self, exe, estagio):
        if exe is not self.execucao:
            return
        if estagio in execucao.ESTAGIOS:
            self.progresso.value = execucao.ESTAGIOS.index(estagio)
        self.status.object = self._texto_estagio(exe)

    def _on_fim(self, exe):
        if exe is not self.execucao:
            return
        if self._relogio is not None:
            self._relogio.stop()
            self._relogio = None
        pasta = f"runs/{exe.pasta.name}"
        if exe.ok:
            self.progresso.value = len(execucao.ESTAGIOS)
            self.status.object = f"✔ **Concluído** em {exe.duracao:.0f} s: `{pasta}`."
            self.ao_concluir(exe.pasta)
        else:
            self.progresso.visible = False
            self.status.object = (f"✖ **Falhou** ({exe.estagio or 'antes do primeiro estágio'}): "
                                  f"{exe.erro}  \n_Saída completa em `{pasta}/{execucao.LOG}`._")
        self._atualizar_botao()

