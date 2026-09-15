"""Aplica, uma unica vez, o patch de compatibilidade do pyispace com Python 3.11.

Bug: pyispace 0.3.7 declara em pyispace/train.py (dataclass ``Data``, linhas
23 e 24) dois campos com ``field(init=False, default=np.array([]))``. Desde o
Python 3.11 o modulo ``dataclasses`` rejeita qualquer default nao hasheavel
(antes so list/dict/set), e ``import pyispace`` morre com::

    ValueError: mutable default <class 'numpy.ndarray'> for field Yraw is not
    allowed: use default_factory

Correcao: trocar os dois defaults por ``field(init=False,
default_factory=lambda: np.array([]))``. Semantica identica (cada instancia
recebe um array vazio novo), sem efeito no restante do pacote.

O script localiza o train.py do interpretador em uso (sem caminho fixo), faz
backup em train.py.orig, aplica a troca, garante o import de ``field`` e valida
importando o pacote num subprocesso. E idempotente: na segunda execucao apenas
informa que o patch ja esta aplicado. Detalhes em patches/README.md.

Uso: python scripts/apply_pyispace_patch.py
"""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

PADRAO_BUG = re.compile(r"field\(init=False, default=np\.array\(\[\]\)\)")
CORRECAO = "field(init=False, default_factory=lambda: np.array([]))"
OCORRENCIAS_ESPERADAS = 2
PADRAO_IMPORT = re.compile(r"^from dataclasses import (?P<nomes>.+)$", re.MULTILINE)


def localizar_train_py() -> Path:
    """Resolve pyispace/train.py do ambiente ativo sem importar o pacote.

    ``find_spec`` de um pacote de topo so consulta os finders de sys.path e nao
    executa o ``__init__`` (que e justamente o que quebra).
    """
    spec = importlib.util.find_spec("pyispace")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError(
            f"pacote pyispace nao encontrado em {sys.executable}; "
            "ative o .venv correto."
        )
    train_py = Path(next(iter(spec.submodule_search_locations))) / "train.py"
    if not train_py.is_file():
        raise FileNotFoundError(f"{train_py} nao existe")
    return train_py


def garantir_import_field(texto: str) -> str:
    """Garante ``from dataclasses import ..., field`` no arquivo."""
    m = PADRAO_IMPORT.search(texto)
    if m is None:
        # sem import de dataclasses: insere antes da primeira linha 'import'
        pos = texto.find("\nimport ")
        pos = 0 if pos < 0 else pos + 1
        return texto[:pos] + "from dataclasses import dataclass, field\n" + texto[pos:]
    nomes = [n.strip() for n in m.group("nomes").split(",")]
    if "field" in nomes:
        return texto
    nova = f"from dataclasses import {', '.join(nomes + ['field'])}"
    return texto[: m.start()] + nova + texto[m.end():]


def validar() -> str:
    """Importa o pacote num subprocesso limpo e devolve a versao."""
    codigo = (
        "import pyispace; from pyispace import train_is; "
        "print(pyispace.__version__)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", codigo], capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError("validacao falhou apos o patch:\n" + proc.stderr.strip())
    return proc.stdout.strip()


def main() -> int:
    train_py = localizar_train_py()
    texto = train_py.read_bytes().decode("utf-8")

    n_bug = len(PADRAO_BUG.findall(texto))
    if n_bug == 0 and CORRECAO in texto:
        print(f"patch ja aplicado em {train_py}; nada a fazer.")
        return 0
    if n_bug != OCORRENCIAS_ESPERADAS:
        print(
            f"ERRO: esperava {OCORRENCIAS_ESPERADAS} ocorrencias do default "
            f"mutavel em {train_py}, encontrei {n_bug}. Versao do pyispace "
            "diferente da 0.3.7? Nada foi alterado.",
            file=sys.stderr,
        )
        return 1

    backup = train_py.parent / "train.py.orig"
    if backup.exists():
        print(f"backup ja existe, preservado: {backup}")
    else:
        backup.write_bytes(train_py.read_bytes())
        print(f"backup gravado em {backup}")

    novo = PADRAO_BUG.sub(CORRECAO, texto)
    novo = garantir_import_field(novo)
    train_py.write_bytes(novo.encode("utf-8"))
    print(
        f"patch aplicado em {train_py} "
        f"({n_bug} defaults trocados por default_factory)"
    )

    versao = validar()
    print(
        "validado: import pyispace / from pyispace import train_is OK "
        f"(pyispace {versao})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
