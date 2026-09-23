"""Dispara o engine (isaspace.engine) em subprocesso e acompanha o progresso.

Unico modulo da interface que conhece o engine: monta a linha de comando
``python -m isaspace.engine --metadata ... --outdir ... --options ...`` com o
mesmo interpretador do app (o .venv-isa) e le a saida do subprocesso numa
thread. NAO importa isaspace.engine nem o instancespace: o subprocesso isola a
memoria, e um erro no instancespace termina so ele, nunca o servidor do app.

Protocolo (linhas da saida padrao do engine que comecam com PREFIXO):
    @@isa estagio <NOME>   antes de cada estagio (ESTAGIOS)
    @@isa ok <outdir>      execucao concluida
    @@isa erro <mensagem>  falha (o codigo de saida e 1)
Todo o resto da saida (logs do instancespace) vai para <outdir>/execucao.log.
"""

import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
PASTA_RUNS = RAIZ / "runs"
PREFIXO = "@@isa"
ESTAGIOS = ("PREPROCESSING", "PRELIM", "SIFTED", "PILOT", "PYTHIA", "CLOISTER", "TRACE")
LOG = "execucao.log"
ENTRADA = "entrada"          # subpasta com os arquivos enviados (o engine os copia)


@dataclass
class Execucao:
    """Estado de uma execucao em andamento ou terminada."""

    pasta: Path
    processo: subprocess.Popen
    inicio: float
    estagio: str | None = None
    terminou: bool = False
    ok: bool | None = None
    erro: str | None = None
    fim: float | None = None
    linhas: list = field(default_factory=list)   # linhas de protocolo recebidas

    @property
    def log(self) -> Path:
        return self.pasta / LOG

    @property
    def duracao(self) -> float:
        return (self.fim or time.perf_counter()) - self.inicio


def nome_seguro(nome: str) -> str:
    """So letras, numeros, '-' e '_' (vira nome de pasta)."""
    limpo = re.sub(r"[^A-Za-z0-9_-]+", "_", nome.strip()).strip("_")
    return limpo[:60] or "metadata"


def nova_pasta(nome: str, runs=PASTA_RUNS, agora=None) -> Path:
    """runs/<nome>_<AAAAMMDD-HHMMSS>/ (criada, com a subpasta de entrada)."""
    carimbo = (agora or datetime.now()).strftime("%Y%m%d-%H%M%S")
    pasta = Path(runs) / f"{nome_seguro(nome)}_{carimbo}"
    sufixo = 2
    while pasta.exists():
        pasta = Path(runs) / f"{nome_seguro(nome)}_{carimbo}_{sufixo}"
        sufixo += 1
    (pasta / ENTRADA).mkdir(parents=True)
    return pasta


def gravar_entrada(pasta: Path, metadata: bytes, annotations: bytes | None = None,
                   feature_info: bytes | None = None) -> Path:
    """Grava os arquivos enviados em <pasta>/entrada/; devolve o metadata.csv."""
    entrada = Path(pasta) / ENTRADA
    entrada.mkdir(parents=True, exist_ok=True)
    (entrada / "metadata.csv").write_bytes(metadata)
    if annotations is not None:
        (entrada / "annotations.json").write_bytes(annotations)
    if feature_info is not None:
        (entrada / "feature_info.csv").write_bytes(feature_info)
    return entrada / "metadata.csv"


def comando(metadata_path, outdir, options) -> list:
    return [sys.executable, "-m", "isaspace.engine", "--metadata", str(metadata_path),
            "--outdir", str(outdir), "--options", json.dumps(options)]


def iniciar(metadata_path, outdir, options, ao_estagio=None, ao_terminar=None) -> Execucao:
    """Inicia o engine em subprocesso; os callbacks rodam na thread leitora
    (quem atualiza a interface deve repassa-los ao documento da sessao)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        comando(metadata_path, outdir, options), cwd=RAIZ, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1, encoding="utf-8", errors="replace",
    )
    execucao = Execucao(pasta=outdir, processo=proc, inicio=time.perf_counter())

    def ler():
        with open(execucao.log, "w", encoding="utf-8") as log:
            for linha in proc.stdout:
                log.write(linha)
                if not linha.startswith(PREFIXO + " "):
                    continue
                execucao.linhas.append(linha.rstrip("\n"))
                tipo, _, resto = linha[len(PREFIXO) + 1:].rstrip("\n").partition(" ")
                if tipo == "estagio":
                    execucao.estagio = resto
                    if ao_estagio is not None:
                        ao_estagio(execucao, resto)
                elif tipo == "ok":
                    execucao.ok = True
                elif tipo == "erro":
                    execucao.ok, execucao.erro = False, resto
        codigo = proc.wait()
        if codigo != 0 and execucao.erro is None:
            execucao.ok = False
            execucao.erro = f"o engine terminou com código {codigo} sem mensagem (ver {execucao.log})"
        if execucao.ok is None:
            execucao.ok = codigo == 0
        execucao.fim = time.perf_counter()
        execucao.terminou = True
        if ao_terminar is not None:
            ao_terminar(execucao)

    threading.Thread(target=ler, name=f"engine-{outdir.name}", daemon=True).start()
    return execucao
