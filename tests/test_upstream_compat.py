"""The engine's --upstream-compat mode against plain instancespace.

scripts/run_upstream_reference.py runs instancespace 0.3.0 through its
documented path (metadata.csv + options.json, build, save_to_csv) without
importing isaspace; ``python -m isaspace.engine --upstream-compat`` runs on the
same metadata.csv with the same options. Every file the library writes must
come out byte-identical; the engine's extra files may exist. Both run in
subprocesses, as a user would run them.

The options are the app's defaults (engine.DEFAULT_OPTIONS) written to an
options.json for both sides; the library's seed (general.seed = 0) makes the
runs deterministic. One more case runs both with no options at all (library
defaults on both sides).

Marked slow: about 1.5 min. Needs the .venv-isa (imports instancespace).
"""

import fnmatch
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

engine = pytest.importorskip("isaspace.engine")
from instancespace import InstanceSpaceOptions  # noqa: E402

from isaspace.ui.loader_is import load_is_output  # noqa: E402

REFERENCE = ROOT / "scripts" / "run_upstream_reference.py"
CASES = {
    "iris": engine.DEFAULT_OPTIONS,
    "diabetes": engine.DEFAULT_OPTIONS,
    "iris-library-defaults": None,
}
LIBRARY_PATTERNS = ("footprint_*_good.csv", "footprint_*_best.csv")


def _library_files(folder):
    return sorted(p.name for p in folder.iterdir()
                  if p.name in engine.SAVE_TO_CSV_FILES
                  or any(fnmatch.fnmatch(p.name, pat) for pat in LIBRARY_PATTERNS))


def _run(cmd):
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = {}
    for case, options in CASES.items():
        dataset = case.removesuffix("-library-defaults")
        metadata = ROOT / "resultados" / "isa" / dataset / "metadata.csv"
        folder = tmp_path_factory.mktemp(f"compat_{case}")
        ref_cmd = [sys.executable, str(REFERENCE), "--metadata", str(metadata),
                   "--outdir", str(folder / "reference")]
        eng_cmd = [sys.executable, "-m", "isaspace.engine", "--metadata", str(metadata),
                   "--outdir", str(folder / "engine"), "--upstream-compat"]
        if options is not None:
            (folder / "options.json").write_text(json.dumps(options))
            ref_cmd += ["--options", str(folder / "options.json")]
            eng_cmd += ["--options", json.dumps(options)]
        _run(ref_cmd)
        _run(eng_cmd)
        out[case] = folder
    return out


def test_compat_options_are_the_library_defaults():
    assert engine.build_options(None, upstream_compat=True) == InstanceSpaceOptions.default()
    # the app's own defaults are untouched
    opts = engine.build_options(None)
    assert (opts.perf.max_perf, opts.perf.epsilon, opts.trace.use_sim) == (True, 0.5, False)


@pytest.mark.slow
@pytest.mark.parametrize("case", list(CASES))
def test_library_files_are_byte_identical(runs, case):
    ref, eng = runs[case] / "reference", runs[case] / "engine"
    names = _library_files(ref)
    assert names == sorted(p.name for p in ref.iterdir())     # the reference writes only these
    assert "coordinates.csv" in names and any(n.startswith("footprint_") for n in names)
    assert _library_files(eng) == names
    different = [n for n in names if (ref / n).read_bytes() != (eng / n).read_bytes()]
    assert different == []


@pytest.mark.slow
@pytest.mark.parametrize("case", list(CASES))
def test_compat_run_records_the_mode_and_opens(runs, case):
    eng = runs[case] / "engine"
    info = json.loads((eng / "run_info.json").read_text())
    assert info["upstream_compat"]["enabled"] is True
    assert info["orientation"]["applied"] is False
    assert info["trace_robustness"]["jitter_applied"] is False
    assert not (eng / "coordinates_trace.csv").exists()
    expected = InstanceSpaceOptions.from_dict(CASES[case] or {})
    assert InstanceSpaceOptions.from_dict(json.loads((eng / "run_options.json").read_text())) == expected
    r = load_is_output(eng)
    assert r.n == info["n_instances"]
