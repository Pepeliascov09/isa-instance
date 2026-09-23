"""Instance space user interface.

ARCHITECTURAL RULE: no module of this package imports instancespace, pyispace,
pyhard or sklearn. The UI reads ready output folders:
- app.py (Panel 1.x, .venv-isa) reads the engine's output folders
  (resultados/is/<name>/ and runs/<name>_<date>/) via loader_is.py; the format
  is in docs/output_format.md and the folders are written by isaspace.engine,
  which only runner.py launches, in a subprocess;
- app_legacy.py (Panel 0.14, .venv; legacy, in Portuguese) reads
  resultados/isa/<name>/ via loader.py.
"""
