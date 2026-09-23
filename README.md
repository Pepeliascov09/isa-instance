# isa-instance

An interactive interface for **Instance Space Analysis (ISA)** on top of the
[`instancespace`](https://github.com/andremun/pyInstanceSpace) package
(Muñoz et al., arXiv:2501.16646). It accepts any metadata in the ISA format,
runs the full ISA pipeline (PRELIM, SIFTED, PILOT, PYTHIA, CLOISTER, TRACE) in
the background, and lets you explore the resulting instance space: where each
algorithm performs well (footprints), what the algorithm selector recommends,
which features were kept and why, and how any subset of instances you select
compares with the rest.

The input is a `metadata.csv` with:

- `instances`: a unique label per instance;
- `source` (optional): the origin of each instance;
- `feature_<name>`: numeric features of the instances (at least 3);
- `algo_<name>`: numeric performance of each algorithm on each instance (at
  least 2);
- any other column: an **annotation**. The ISA pipeline ignores annotations,
  but the interface keeps them, so you can color, group and filter the
  instance space by them (a class label, a difficulty score, an id…).

Optional files next to it: `annotations.json` (the type of each annotation),
`feature_info.csv` (a family per feature) and `degenerate_report.csv`
(features dropped before ISA). The format of everything the engine writes is
in [`docs/output_format.md`](docs/output_format.md).

The repository ships four ready examples in `resultados/is/`, so the app opens
right after cloning. They come from the use case that motivated the tool, an
Iniciação Científica (IC7) project at ITA in which each instance is an
individual observation of a dataset; see [Example: instances as individual
observations](#example-instances-as-individual-observations).

## What the interface shows

Six tabs over a single global state: the selection made with the lasso (or
box) in the Instance Space tab, and the color variable, apply to all of them.

- **Instance Space**: the PILOT projection (z_1 × z_2), colored by any
  annotation, feature, performance column or derived column (number of good
  algorithms, beta-easy, best observed algorithm with ties shown as *tie*,
  number of algorithms tied for the best, algorithm recommended by PYTHIA).
  The PILOT r² of the color variable is shown next to the selector, with a
  warning when the plane explains little of it. Overlays: the footprints of
  one or all algorithms (good or best), the CLOISTER boundary and the hard
  footprint. The space has a **standard orientation**: the instances where
  most algorithms are bad are at the **top left**, as in pyispace, so the hard
  region does not have to be looked for in each dataset; a note says so, with
  a warning when the difficulty gradient is weak (see Known limitations).
- **Footprint Performance**: the footprints on the map with the selection
  highlighted, and `footprint_performance.csv` with each footprint's status
  (*ok*, *suspect* when the purity is below `trace.purity`, drawn dashed, or
  *empty*).
- **Algorithm Selection**: what PYTHIA and CLOISTER produced. A map colored by
  the recommended algorithm (selection0, with *none* as its own category), by
  whether the recommended algorithm is good for the instance (*recommended
  good*, *recommended bad*, *no recommendation*, following
  `algorithm_bin.csv`), or by the out-of-sample probability `pr0_sub` of one
  algorithm, with the CLOISTER boundary. Below it, the `svm_table` (CV
  accuracy, precision and recall per algorithm, plus the Oracle and Selector
  rows) and one cross-validation confusion matrix per algorithm. A warning
  appears when the selector recommends the same algorithm for more than 90%
  of the instances (a nearly trivial selector).
- **Distributions**: histogram, density or violin of up to six numeric
  variables, grouped by a categorical annotation and split into selected and
  not selected instances.
- **Features**: one row per received feature, with what went into PILOT and
  why the rest was dropped (degenerate before ISA, no correlation with the
  performance, or redundant within a SIFTED cluster), the feature × algorithm
  correlation heatmap and the silhouette per k.
- **Data Explorer**: an x-y scatter with the global color, a `pandas.query`
  filter, a "Use filter as selection" button and a table of the filtered rows.

The sidebar has:

- the dataset selector, with `resultados/is/` and `runs/` (runs launched from
  the interface) in separate groups;
- the **New instance space** block, to upload a metadata and run ISA on it
  (see below);
- the numbers of the result (instances, features in PILOT, algorithms, the
  "good" rule, whether TRACE needed jitter) and the annotation types that were
  guessed, which can be changed for the session;
- export buttons: all instances, the current selection, or the labels inside
  a footprint. An exported CSV is itself valid metadata and can be uploaded
  again.

The lower part of the sidebar changes with the active tab.

## Installation

The app and the engine run in a Python **3.12** environment (`instancespace`
0.3.0 requires Python >= 3.12, < 3.13). The versions below are the tested
ones:

```bash
python3.12 -m venv .venv-isa
.venv-isa/bin/python -m pip install instancespace==0.3.0 panel==1.9.4 \
    holoviews==1.23.2 bokeh==3.9.2 param==2.4.2 pytest playwright
.venv-isa/bin/python -m playwright install chromium    # only for the e2e tests
```

A second environment, `.venv` with Python **3.11** (`requirements.txt`), is
needed only to regenerate the IC7 example data with PyHard; see [Generating
the example data](#generating-the-example-data).

## Running the app

From the project root:

```bash
.venv-isa/bin/python -m isaspace.ui.app
```

It opens the browser at <http://localhost:5006>. Options:

- `--port N`;
- `--no-show`: do not open the browser;
- `--root FOLDER`: another folder of engine outputs, in the format of
  `resultados/is/` (default `resultados/is`);
- `--runs FOLDER`: where to write the runs launched from the interface
  (default `runs/`, ignored by git).

The engine can also be used without the interface:

```bash
.venv-isa/bin/python -m isaspace.engine --metadata metadata.csv --outdir out/ \
    --options '{"perf": {"max_perf": true, "abs_perf": true, "epsilon": 0.5}}'
```

`--no-orient` keeps PILOT's own orientation (see below).

## Running ISA on a new metadata

In the **New instance space** block of the sidebar:

1. **Upload** a `metadata.csv` and, optionally, `annotations.json` and
   `feature_info.csv`. The files are validated immediately, and problems are
   shown as text, never as a traceback:
   - the `instances` column exists and has no empty or repeated labels;
   - at least 3 `feature_*` and 2 `algo_*` columns;
   - `feature_*` and `algo_*` values are numeric and finite;
   - NaN are reported per column;
   - the declared annotation types match the columns.
2. **Choose the performance rule.** The direction (*higher is better* or
   *lower is better*), the threshold type (*absolute*, or *relative to the
   instance's best*) and ε have **no default**: the Run ISA button stays
   disabled until they are chosen. Then the block shows:
   - a preview of the fraction of good instances per algorithm, computed with
     the same rule as the instancespace PRELIM;
   - a **degenerate ε threshold** warning when some algorithm has fewer than
     5% or more than 95% good instances: with that ε, "good" is almost
     constant for it, and its footprints and PYTHIA classifier carry little
     information;
   - before the Run button, the **consequence of the direction**: the
     algorithms ranked by the mean of their `algo_*` column under the chosen
     direction, with the best and the worst named, and the sentence "If this
     ranking looks upside down for your problem, the direction is probably
     wrong." Nothing tries to guess the direction for you.
3. **Advanced options** (collapsed): the SIFTED k (default 6),
   `trace.usesim` (off: footprints of the observed performance, not of the
   PYTHIA predictions) and the standard orientation (on).
4. **Run ISA.** The engine runs in a subprocess, with progress per stage, and
   the interface stays usable meanwhile. The result goes to
   `runs/<name>_<YYYYMMDD-HHMMSS>/`, with the uploaded files in `input/` and
   the full subprocess output in `run.log`, and it opens automatically when it
   finishes. An error shows a one-line message; the traceback stays in
   `run.log`.

The block shows an estimated run time, from measurements with
`scripts/measure_engine_time.py` on synthetic metadata with 10 features and 6
algorithms (Apple M5, medians of 3 repetitions):

| instances | time |
|---|---|
| 500 | ~10 s |
| 1000 | ~24 s |
| 2000 | ~77 s |

PYTHIA takes from ~50% (500 instances) to ~87% (2000 instances) of the time
and grows with the number of algorithms. From 1000 instances on, the block
shows a time warning.

## Output format

Every folder the engine writes (`resultados/is/<name>/` or
`runs/<name>_<date>/`) follows [`docs/output_format.md`](docs/output_format.md):
the files instancespace itself saves, plus what the engine adds (`run_info.json`
with provenance, timings and diagnostics; `sifted_report.csv`;
`pythia_proba.csv`; `pythia_confusion.csv`; `pythia_selection.csv`;
`pilot_r2.csv`; the space and hard footprints; the copied metadata and
auxiliary files). The interface reads only these folders, through
`isaspace/ui/loader_is.py`.

## Known limitations

- **Orientation convention.** PILOT's plane has an arbitrary rotation. After
  the whole pipeline, the engine rotates every geometric output so that the
  centroid of the instances where at least half of the algorithms are bad
  sits at 135° (top left), the rule of pyispace's `adjust_rotation`
  (`rotation_adjust=True`), written with the number of bad algorithms so that
  it works for any metadata. It is a proper rotation (no reflection): it fixes
  where the hard region is, not the handedness of the plane, so two datasets
  can still be mirror images of each other around that direction. Nothing
  instancespace computes changes (footprint areas, densities, purities and
  membership, PYTHIA, PILOT r²; checked by `tests/test_orientation.py`). The
  convention is only as meaningful as the difficulty gradient: the R² of the
  number of bad algorithms regressed on (z_1, z_2) is recorded, and below 0.3
  the interface warns that the hard region is spread. In the examples: iris
  0.46, diabetes 0.66, blood-transfusion-service-center 0.69, hill-valley 0.28
  (warning).
- **TRACE jitter.** The legacy TRACE alpha shape in instancespace 0.3.0
  returns an empty polygon when the projection has *distinct* points about
  1e-14 apart (identical points are harmless: TRACE merges them). In the
  hill-valley example this zeroed the good footprints and the area of the
  space. The engine therefore looks for distinct positions closer than 1e-6
  after PILOT and, only for them, adds normal noise with standard deviation
  1e-6 and a fixed seed (identical points get the same shift), and passes that
  z **only to TRACE**. PILOT's z is what the interface draws
  (`coordinates.csv`); the z TRACE used is kept in `coordinates_trace.csv`,
  and everything is recorded in `run_info.json["trace_robustness"]`. Of the
  four examples, only hill-valley needs it (27 points moved, by at most
  2.4e-6).
- **The performance direction is the user's choice.** Whether higher or lower
  `algo_*` is better cannot be read from the data, and the interface does not
  guess it. The <5% / >95% check detects a degenerate ε, not a wrong
  direction: with an absolute threshold, flipping the direction turns each
  fraction f into 1 − f (except for values exactly at ε), so the warning fires
  in both directions or in neither (on iris it fires in both; on the other
  three examples, in neither). The ranking by mean `algo_*` shows the
  consequence of the choice so that a reversed direction is visible.
- **Fixed SIFTED k.** instancespace clusters the features into a fixed number
  of clusters, `sifted.k = 6` by default, and keeps one feature per cluster;
  the k with the highest silhouette is only suggested in its log, not used.
  In the examples the silhouette peaks at k = 3 (diabetes) and k = 4
  (hill-valley) while k = 6 is used. The Features tab shows the silhouette per
  k, and an upload can set k in the advanced options.
- **Ties for the best observed algorithm.** `portfolio.csv` (and so the best
  footprints) breaks ties for the best `algo_*` value at random. Ties are
  common when the performance saturates: iris 123 of 150 instances, diabetes
  217 of 768, blood-transfusion-service-center 215 of 748, hill-valley 54 of
  1212. The interface shows those instances as *tie* in the "best observed
  algorithm" color and offers the number of tied algorithms as a color.
- **Best footprints are often empty or suspect** in the examples (iris: 4 of
  the 6 best footprints empty and 1 suspect; hill-valley: 2 empty and 3
  suspect), while every good footprint is *ok*. This is a result of TRACE with
  these portfolios, not an interface defect; the Footprint Performance tab
  flags those cases.
- **A nearly trivial selector** is possible: on iris PYTHIA recommends
  logistic regression for 143 of the 150 instances, and the Algorithm
  Selection tab warns about it.

## Tests

With the `.venv-isa`:

```bash
.venv-isa/bin/python -m pytest tests/                            # everything, about 5 min
.venv-isa/bin/python -m pytest tests/ -m "not e2e and not slow"  # the fast ones, seconds
```

214 tests: 162 fast, 8 slow (they run the engine again on the four examples)
and 44 end-to-end.

- `tests/test_loader_is.py` and `tests/test_engine.py` test the loader on
  `resultados/is/` and on a synthetic folder (text labels, holes, name
  collisions, ties), and the engine's validation of the auxiliary files.
- `tests/test_orientation.py` tests the standard orientation: the rotation
  itself (135°, determinant +1, the pyispace tie rule), the hard region at the
  top left in the four examples and, marked `slow`, the invariance: the four
  examples run again without the rotation and compared.
- `tests/test_upload.py` tests the upload validation, the fraction of good
  instances (checked against instancespace's own
  `compute_binary_performance` and against the written `algorithm_bin.csv`),
  the mean ranking, the time estimate and the subprocess launcher.
- `tests/test_app_e2e.py` starts the app with a temporary `runs/` folder and
  drives it in a real Chromium (Playwright), with a new page per test, on the
  four examples: lasso selection across tabs, global color, filters, export,
  ties, the Algorithm Selection categories, the direction ranking, and two
  full runs: the example metadata of the instancespace repository (downloaded
  from GitHub, tag v0.3.0, to the pytest cache because of its non-commercial
  license; skipped without network) and the cycle "export a selection → upload
  it as metadata → run → same number of instances".

## Common problems

- **The page stopped responding** (switching tabs works, but selectors,
  sidebar, header and plots do not change): the Bokeh session with the server
  was lost, for instance because the server was restarted. Reload with F5 or
  open a new tab. The red bar at the top of the page says exactly that; without
  it a dead page is indistinguishable from a live one.
- **After changing the app's code**, close the tab and open a new one instead
  of reusing the old one: the page HTML carries a session token that expires
  in 300 s, and a tab restored from cache tries to reconnect with the expired
  token, which leaves only the template frame with nothing inside.
- **A run from another browser session does not show up** in the dataset
  selector: press "Reload dataset", which lists the folders again.

## Example: instances as individual observations

The use case behind the tool (IC7, ITA) is ISA at the level of **individual
observations**: each point of the instance space is one row of a dataset, not
a whole dataset. For four OpenML datasets (iris, diabetes,
blood-transfusion-service-center and hill-valley):

- the **features** are per-instance hardness measures computed with PyHard
  (kDN, N1, CL, DCP, TD_P, Harmfulness, …); measures that the preprocessing
  would make constant are dropped before ISA and listed in
  `degenerate_report.csv`, and `feature_info.csv` tells the model-derived
  measures from the geometric ones;
- the **algorithms** are six classifiers (kNN, decision tree, Naive Bayes,
  logistic regression, RBF SVM and random forest), and `algo_<name>` is the
  out-of-fold probability each one gives to the true class (5 folds, seed
  42); higher is better, and "good" is `algo_* >= 0.5`;
- the **annotations** are `row_original` (the row index in the OpenML dataset,
  an identifier), `class` (the original label), `ih` (instance hardness, 1 −
  the mean probability of the true class) and `n_wrong` (how many classifiers
  got the instance wrong).

The results are in `resultados/is/<name>/`.

### Generating the example data

From the project root:

1. With the `.venv` (Python 3.11): `python run_table.py` downloads the four
   datasets from OpenML (ids 61, 37, 1464 and 1479), computes the measures and
   the out-of-fold performance, and writes `resultados/table_<name>.csv`.
2. With the `.venv`: `python scripts/build_metadata.py [name ...]` converts
   each table into `resultados/isa/<name>/metadata.csv` plus
   `annotations.json`, `degenerate_report.csv` and `feature_info.csv`.
3. With the `.venv-isa`: `.venv-isa/bin/python scripts/run_is_all.py
   [name ...]` runs the engine and writes `resultados/is/<name>/` (iris 2.6 s,
   diabetes 15 s, blood-transfusion 13 s, hill-valley 32 s).

The `.venv` needs Python **3.11**, not 3.12 or 3.13: `pyhard` 2.2.4 pins
`pandas~=1.5.0`, which has no wheels for newer Pythons.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/apply_pyispace_patch.py
```

The last step patches `pyispace` 0.3.7 (a PyHard dependency) for Python 3.11:
its `train.py` declares dataclass fields with a NumPy array as default, which
Python 3.11 refuses (`ValueError: mutable default <class 'numpy.ndarray'> for
field Yraw is not allowed`), so even `import pyispace` fails. The script
switches them to `default_factory`, keeps a backup in `train.py.orig`, checks
the import and is idempotent; reinstalling pyispace undoes it. Details in
`patches/README.md`.

## Project structure

The tool:

- `isaspace/engine.py`: runs instancespace stage by stage and writes the
  output folder (`docs/output_format.md`); also a command line
  (`python -m isaspace.engine`).
- `isaspace/ui/loader_is.py`: reads an output folder into an `IsResult`
  (pandas and numpy only).
- `isaspace/ui/app.py`: the six-tab interface (Panel 1.x).
- `isaspace/ui/new_space.py`: the New instance space block of the sidebar.
- `isaspace/ui/upload.py`: upload validation, fraction of good instances (the
  PRELIM rule), mean ranking and time estimate (pandas and numpy only).
- `isaspace/ui/runner.py`: the only UI module that knows the engine; it runs
  `python -m isaspace.engine` in a subprocess and reads its progress.
- `scripts/run_is_all.py`: the engine on the four examples, into
  `resultados/is/`.
- `scripts/measure_engine_time.py`: engine run time on synthetic metadata (the
  basis of the time estimate).
- `docs/output_format.md`: the contract between the engine and the interface.
- `tests/`: loader, engine, upload and end-to-end tests.
- `runs/` (ignored by git): runs launched from the interface.

`isaspace/ui/` is deliberately independent of the computing backend: the
interface only reads the engine's output folders through `loader_is.py`; it
does not import instancespace, pyispace, pyhard or scikit-learn; only
`runner.py` knows the engine, and runs it in a subprocess; whatever a folder
lacks is produced by the engine or the scripts, never by the interface.

The IC7 data generation:

- `isaspace/intake.py`: OpenML loading and conversion to PyHard's numeric
  frame.
- `isaspace/measures.py`: the per-instance hardness measures (PyHard's
  `ClassificationMeasures`).
- `isaspace/performance.py`: out-of-fold performance of the six-classifier
  portfolio.
- `isaspace/pipeline.py`: the per-instance table (`feature_*`, `algo_*`,
  `proba_*`, `class`, `n_wrong`, `ih`).
- `isaspace/isa.py`: `to_isa_metadata` and `write_metadata` (metadata and
  auxiliary files, dropping degenerate measures) and `run_isa` (the legacy
  pyispace PILOT + TRACE, with a guard against an inverted "good").
- `run_table.py`: writes `resultados/table_<name>.csv`.
- `scripts/build_metadata.py`: writes the metadata and auxiliary files into
  `resultados/isa/<name>/`.
- `scripts/run_isa_all.py`: the legacy pyispace PILOT + TRACE into
  `resultados/isa/<name>/` (for the legacy app).
- `resultados/`: `table_<name>.csv`, `isa/<name>/` (metadata and the legacy
  pyispace outputs), `is/<name>/` (engine outputs) and the PNGs of the first
  experiments. The folder keeps its Portuguese name.

**Legacy, in Portuguese, kept as they are** (earlier phases of the project,
before the move to instancespace; they are not maintained):

- `isaspace/ui/app_legacy.py` (Panel 0.14, pyispace, `.venv`:
  `python -m isaspace.ui.app_legacy`) with its reader `isaspace/ui/loader.py`
  (the MATILDA folders of `resultados/isa/`);
- `isaspace/app.py` and `isaspace/app_v2.py`: the earlier one-tab interface
  (identical files);
- `isaspace/projection.py` and `isaspace/footprint.py`: the earlier PCA
  projection and grid footprints, replaced by PILOT and TRACE;
- `scripts/exploratorio/`: the ten experiment scripts of the first phase
  (run from the root with `python -m scripts.exploratorio.<name>`), whose
  numbers are in `resumo.md`;
- `scripts/verify_browser.py`: the browser regression test of the legacy app;
- `scripts/build_data_space.py`: the PCA data space of the legacy app;
- `scripts/apply_pyispace_patch.py` and `patches/README.md`: the pyispace
  patch for Python 3.11;
- `resumo.md`: technical summary of the first phase.

## Credits

Developed as an Iniciação Científica (IC7) project at ITA (Instituto
Tecnológico de Aeronáutica), advised by Profa. Ana Carolina Lorena. The ISA
pipeline itself is the `instancespace` package by Mario Andrés Muñoz and
collaborators.

## References

- instancespace: Muñoz et al., arXiv:2501.16646. Code:
  <https://github.com/andremun/pyInstanceSpace>.
- Smith-Miles, K.; Muñoz, M. A. *Instance Space Analysis for Algorithm
  Testing: Methodology and Software Tools*. ACM Computing Surveys, 55(12),
  2023.
- PyHard: Paiva, P. Y. A.; Moreno, C. C.; Smith-Miles, K.; Valeriano, M. G.;
  Lorena, A. C. *Relating instance hardness to classification performance in a
  dataset: a visual approach*. Machine Learning, 111, 2022. Code:
  <https://gitlab.com/ita-ml/pyhard>.
- pyispace: a Python implementation of parts of MATILDA (partial PRELIM, PILOT
  and TRACE), used by the legacy app. <https://gitlab.com/ita-ml/pyispace>.
- MATILDA / InstanceSpace (MATLAB reference):
  <https://github.com/andremun/InstanceSpace>.
