"""Interface do espaco de instancias.

REGRA ARQUITETURAL: nenhum modulo deste pacote importa instancespace, pyispace,
pyhard ou sklearn. A interface le pastas de saida prontas e nada mais:
- app.py (Panel 1.x, .venv-isa) le resultados/is/<nome>/ via loader_is.py; o
  formato esta em docs/output_format.md e a pasta e gerada por isaspace.engine;
- app_legacy.py (Panel 0.14, .venv) le resultados/isa/<nome>/ via loader.py.
"""
