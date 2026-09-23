"""Launch the engine (isaspace.engine) in a subprocess and follow its progress.

The only UI module that knows the engine: it builds the command line
``python -m isaspace.engine --metadata ... --outdir ... --options ...`` with
the app's own interpreter (the .venv-isa) and reads the subprocess output in a
thread. It does NOT import isaspace.engine or instancespace: the subprocess
isolates memory, and an instancespace error ends only the subprocess, never
the app server.

Protocol (lines of the engine's standard output that start with PREFIX):
    @@isa stage <NAME>     before each stage (STAGES)
    @@isa ok <outdir>      run finished
    @@isa error <message>  failure (exit code 1)
Everything else (instancespace logs) goes to <outdir>/run.log.
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

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "runs"
PREFIX = "@@isa"
STAGES = ("PREPROCESSING", "PRELIM", "SIFTED", "PILOT", "PYTHIA", "CLOISTER", "TRACE")
LOG = "run.log"
INPUT_DIR = "input"          # subfolder with the uploaded files (the engine copies them)


@dataclass
class Run:
    """State of a running or finished engine run."""

    path: Path
    process: subprocess.Popen
    started: float
    stage: str | None = None
    finished: bool = False
    ok: bool | None = None
    error: str | None = None
    ended: float | None = None
    lines: list = field(default_factory=list)   # protocol lines received

    @property
    def log(self) -> Path:
        return self.path / LOG

    @property
    def duration(self) -> float:
        return (self.ended or time.perf_counter()) - self.started


def safe_name(name: str) -> str:
    """Only letters, digits, '-' and '_' (it becomes a folder name)."""
    clean = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")
    return clean[:60] or "metadata"


def new_run_dir(name: str, runs=RUNS_DIR, now=None) -> Path:
    """runs/<name>_<YYYYMMDD-HHMMSS>/ (created, with the input subfolder)."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    path = Path(runs) / f"{safe_name(name)}_{stamp}"
    suffix = 2
    while path.exists():
        path = Path(runs) / f"{safe_name(name)}_{stamp}_{suffix}"
        suffix += 1
    (path / INPUT_DIR).mkdir(parents=True)
    return path


def write_inputs(path: Path, metadata: bytes, annotations: bytes | None = None,
                 feature_info: bytes | None = None) -> Path:
    """Write the uploaded files to <path>/input/; return the metadata.csv path."""
    folder = Path(path) / INPUT_DIR
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metadata.csv").write_bytes(metadata)
    if annotations is not None:
        (folder / "annotations.json").write_bytes(annotations)
    if feature_info is not None:
        (folder / "feature_info.csv").write_bytes(feature_info)
    return folder / "metadata.csv"


def command(metadata_path, outdir, options) -> list:
    return [sys.executable, "-m", "isaspace.engine", "--metadata", str(metadata_path),
            "--outdir", str(outdir), "--options", json.dumps(options)]


def launch(metadata_path, outdir, options, on_stage=None, on_finish=None) -> Run:
    """Start the engine in a subprocess; the callbacks run in the reader
    thread (whoever updates the UI must hand them to the session's document)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        command(metadata_path, outdir, options), cwd=ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1, encoding="utf-8", errors="replace",
    )
    run = Run(path=outdir, process=proc, started=time.perf_counter())

    def read():
        with open(run.log, "w", encoding="utf-8") as log:
            for line in proc.stdout:
                log.write(line)
                if not line.startswith(PREFIX + " "):
                    continue
                run.lines.append(line.rstrip("\n"))
                kind, _, rest = line[len(PREFIX) + 1:].rstrip("\n").partition(" ")
                if kind == "stage":
                    run.stage = rest
                    if on_stage is not None:
                        on_stage(run, rest)
                elif kind == "ok":
                    run.ok = True
                elif kind == "error":
                    run.ok, run.error = False, rest
        code = proc.wait()
        if code != 0 and run.error is None:
            run.ok = False
            run.error = f"the engine exited with code {code} and no message (see {run.log})"
        if run.ok is None:
            run.ok = code == 0
        run.ended = time.perf_counter()
        run.finished = True
        if on_finish is not None:
            on_finish(run)

    threading.Thread(target=read, name=f"engine-{outdir.name}", daemon=True).start()
    return run
