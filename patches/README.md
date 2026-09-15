# Patch: pyispace 0.3.7 em Python 3.11

Upstream: <https://gitlab.com/ita-ml/pyispace> (pacote `pyispace`, versão 0.3.7,
instalado no `.venv` como dependência do `pyhard` 2.2.4).

## O bug

`pyispace/train.py` declara o dataclass `Data` com dois campos cujo default é um
array NumPy:

```python
# pyispace/train.py, linhas 23 e 24 (versão 0.3.7)
Yraw: np.ndarray = field(init=False, default=np.array([]))
Xraw: np.ndarray = field(init=False, default=np.array([]))
```

Até o Python 3.10 o módulo `dataclasses` só recusava defaults do tipo `list`,
`dict` e `set`. A partir do 3.11 ele recusa qualquer default não hasheável, e
`np.ndarray` define `__hash__ = None`. Resultado: o simples `import pyispace`
(que importa `pyispace.train` na linha 3 do `__init__.py`) morre com

```
ValueError: mutable default <class 'numpy.ndarray'> for field Yraw is not allowed: use default_factory
```

Como `pyhard.integrator` e `pyhard.cli` importam o `pyispace`, esses dois módulos
do pyhard também deixam de importar. O `.venv` deste projeto usa Python 3.11.9.

## A correção

Trocar os dois defaults por uma fábrica, que é a forma recomendada pela própria
mensagem de erro:

```python
Yraw: np.ndarray = field(init=False, default_factory=lambda: np.array([]))
Xraw: np.ndarray = field(init=False, default_factory=lambda: np.array([]))
```

A semântica é idêntica: cada instância de `Data` recebe um array vazio novo. O
import `from dataclasses import dataclass, field` já existe na linha 2 do
arquivo; o script confere mesmo assim.

Nenhum outro arquivo do pacote é alterado.

## Como aplicar

```
python scripts/apply_pyispace_patch.py
```

O script:

1. localiza `pyispace/train.py` do interpretador ativo via
   `importlib.util.find_spec("pyispace")` (sem caminho fixo e sem executar o
   `__init__`, que é o que quebra);
2. se o patch já está aplicado, apenas informa e sai (idempotente);
3. senão, grava um backup em `pyispace/train.py.orig` (nunca sobrescreve um
   backup existente), aplica a troca e garante o import de `field`;
4. valida com `import pyispace; from pyispace import train_is` num subprocesso
   e imprime a versão.

## Como reverter

Copie `pyispace/train.py.orig` de volta sobre `pyispace/train.py`, ou reinstale
o pacote (`pip install --force-reinstall --no-deps pyispace==0.3.7`). Atenção:
qualquer reinstalação ou atualização do `pyispace` desfaz o patch, e o script
precisa ser executado de novo.
