# Engine output folder format

Contract between the writer, `isaspace.engine.run_instancespace` (Python 3.12,
`.venv-isa`, instancespace 0.3.0), and the reader,
`isaspace.ui.loader_is.load_is_output`. One folder corresponds to one run on a
`metadata.csv`. The versioned folders are in `resultados/is/<name>/` and are
written by `scripts/run_is_all.py`.

Runs launched from the interface (the "New instance space" block) go to
`runs/<name>_<YYYYMMDD-HHMMSS>/`, outside git. The engine runs in a subprocess
(`python -m isaspace.engine --metadata ... --outdir ... --options '<json>'`,
launched by `isaspace.ui.runner`), and the folder follows this contract with
two extra items that the engine does not touch:

- `input/`: the uploaded files (`metadata.csv` and, if uploaded,
  `annotations.json` and `feature_info.csv`); the engine reads from here;
- `run.log`: the whole subprocess output (instancespace logs and, on error,
  the traceback). The lines that start with `@@isa` are the progress protocol:
  `@@isa stage <NAME>` before each stage, `@@isa ok <folder>` at the end and
  `@@isa error <message>` on failure (exit code 1). Only the one-line error
  message reaches the interface; the traceback stays in the log.

The legacy `resultados/isa/<name>/` (pyispace, read by `isaspace/ui/loader.py`)
has another format and does not follow this contract.

## General conventions

- **Complete folder.** `run_info.json` is written last. A folder without it is
  an incomplete run, and the loader refuses it. On a rerun, the engine deletes
  only the files listed here, before writing; it refuses a folder that has
  files with these names but no `run_info.json` of its own
  (`generated_by` must be `"isaspace.engine.run_instancespace"`).
- **CSV.** Comma separator, header on the first line, UTF-8, decimal point, no
  comment lines. Floats in pandas' full representation, except where rounding
  is stated.
- **`Row` in the per-instance files.** It is the **instance label** (the
  metadata's `instances` column), not a counter. It must be read as text
  (`dtype={"Row": str}`): the label `"1"` is not the integer 1. All
  per-instance files have the same rows, in the same order (the metadata's).
  With the default options, every metadata instance appears; if
  `selvars.small_scale_flag` or `selvars.density_flag` are turned on, it is a
  subset (`run_info.n_instances` < `n_instances_input`).
- **Booleans** are written as `True` / `False`.
- **Empty field** means missing (NaN, or `None` for names).
- **Names** of algorithms and features come without the `algo_` / `feature_`
  prefixes.
- **Algorithm order:** that of the `algorithm_raw.csv` columns, equal to
  `run_info.algorithms`. Portfolio indices refer to this order.
- **Files whose absence is meaningful:** `coordinates_trace.csv` (no jitter)
  and the `footprint_*.csv` (empty footprint).
- **Origin of each file:** *sc* = written by instancespace's
  `Model.save_to_csv`; *eng* = written by the engine.

## Index

| file | origin | one row per | present |
|---|---|---|---|
| `run_info.json` | eng | — | always |
| `run_options.json` | eng | — | always |
| `metadata.csv` | eng (copy) | input instance | always |
| `annotations.json` | eng (copy) | — | if next to the input metadata |
| `degenerate_report.csv` | eng (copy) | feature dropped before the engine | if next to the input metadata |
| `feature_info.csv` | eng (copy) | received feature | if next to the input metadata |
| `coordinates.csv` | eng | instance | always |
| `coordinates_trace.csv` | eng | instance | only with jitter |
| `projection_matrix.csv` | sc | axis (z_1, z_2) | always |
| `pilot_r2.csv` | eng | feature or algorithm | always |
| `feature_raw.csv`, `feature_process.csv` | sc | instance | always |
| `algorithm_raw.csv`, `algorithm_process.csv` | sc | instance | always |
| `algorithm_bin.csv` | sc | instance | always |
| `good_algos.csv`, `beta_easy.csv`, `portfolio.csv` | sc | instance | always |
| `sifted_report.csv` | eng | input feature | always |
| `sifted_correlations.csv` | eng | feature x algorithm pair | always (may have only the header) |
| `sifted_silhouette.csv` | eng | k tried | always (may have only the header) |
| `algorithm_svm.csv`, `portfolio_svm.csv` | sc | instance | always |
| `pythia_proba.csv`, `pythia_selection.csv` | eng | instance | always |
| `pythia_confusion.csv` | eng | algorithm | always |
| `svm_table.csv` | sc | algorithm, plus `Oracle` and `Selector` | always |
| `bounds.csv`, `bounds_prunned.csv` | sc | boundary vertex | always |
| `footprint_<algo>_<good\|best>.csv` | sc | vertex | only if not empty |
| `footprint_space.csv`, `footprint_hard.csv` | eng | vertex | only if not empty |
| `footprint_performance.csv` | sc | algorithm | always |

---

## Run

### `run_info.json` (eng)

JSON object.

| key | type | content |
|---|---|---|
| `generated_by` | str | `"isaspace.engine.run_instancespace"`; marks the folder as the engine's |
| `generated_at` | str | local ISO 8601 date and time, in seconds |
| `instancespace_version`, `python` | str | versions used |
| `input_metadata` | str | absolute path of the metadata read |
| `n_instances_input`, `n_instances` | int | metadata rows and rows in the per-instance files |
| `n_features_input`, `n_features_selected` | int | metadata features and features chosen by SIFTED |
| `algorithms` | list[str] | canonical algorithm order |
| `has_source` | bool | whether the metadata had a `source` column |
| `annotation_types` | dict | `{"file": "annotations.json" or null, "declared": {annotation: type}}` |
| `auxiliary_files` | list[str] | which of `annotations.json`, `degenerate_report.csv` and `feature_info.csv` were copied |
| `good_rule` | str | the rule of `algorithm_bin.csv`, e.g. `"good = algo_* >= 0.5"` |
| `timings_s` | dict | seconds per stage (`PREPROCESSING` … `TRACE`), plus `near_duplicate_check`, `build_total` and `writing` |
| `trace_robustness` | dict | see below |
| `footprint_files` | dict | `{algo: {"good": file or null, "best": file or null}}`; see TRACE |
| `special_footprints` | dict | `{"space": {...}, "hard": {...}}`; see TRACE |
| `pythia` | dict | PYTHIA diagnostic counts; see PYTHIA |
| `files` | list[str] | files of this folder written by the engine |
| `warnings` | list[str] | inconsistencies the engine detected while building the files; empty is expected |
| `instancespace_warnings` | list[str] | WARNING-or-above messages of the instancespace log, without repetition |
| `python_warnings` | list[dict] | `{category, message, n}`: Python warnings captured during the run, with counts |

`good_rule` has one of four forms, following the PRELIM rule
(prelim.py:120-170): `good = algo_* >= ε` (higher is better, absolute),
`good = algo_* <= ε` (lower is better, absolute), `good = 1 - algo_*/best <= ε`
(higher is better, relative) and `good = algo_*/best - 1 <= ε` (lower is
better, relative).

`trace_robustness` records the correction of near-coincident projection points
before TRACE. The legacy TRACE alpha shape returns an empty polygon when there
are **distinct** points ~1e-14 apart.

- Always present:
  - `threshold` (1e-6);
  - `near_duplicate_pairs`: all pairs closer than the threshold;
  - `identical_pairs`: the pairs at distance 0; TRACE already merges them with
    `np.unique`, so they do no harm;
  - `distinct_close_pairs`: pairs between distinct positions;
  - `min_distance_distinct_before`;
  - `rule`;
  - `jitter_applied`.
- When `jitter_applied` is `true`:
  - `jitter_scale` (1e-6), `jitter_seed` (0) and `jitter_distribution`;
  - `perturbed_positions` and `perturbed_points`;
  - `perturbed_labels`: the `Row` labels that were moved;
  - `max_shift`;
  - `min_distance_distinct_after` and `distinct_close_pairs_after`;
  - `applied_to`: TRACE only; `coordinates.csv` keeps the PILOT z and the
    perturbed z goes to `coordinates_trace.csv`; PYTHIA used the PILOT z and
    CLOISTER does not use z.
- `reason_no_jitter` appears when the correction was turned off
  (`fix_near_duplicates=False`).

In the four versioned datasets, only hill-valley needs the jitter (22 pairs of
distinct positions closer than 1e-6, the closest 5.3e-14 apart; 27 points
moved).

### `run_options.json` (eng)

It is `dataclasses.asdict(InstanceSpaceOptions)`: every effective option,
grouped (`parallel`, `perf`, `auto`, `bound`, `norm`, `selvars`, `sifted`,
`pilot`, `cloister`, `pythia`, `trace`, `outputs`, `general`, `prelim`), with
the dataclass field names (`max_perf`, `use_sim`, `purity`…).
`InstanceSpaceOptions.from_dict(json.load(...))` rebuilds the same options.

The engine only changes these library defaults: `perf.max_perf=true`,
`perf.abs_perf=true`, `perf.epsilon=0.5` and `trace.use_sim=false`. Runs from
the interface also set `perf` from the user's choice (there is no default
direction there) and `sifted.k` and `trace.use_sim` from the advanced options.
The loader uses `trace.purity` as the threshold of a "suspect" footprint and
`perf.max_perf` for the tie rule.

### `metadata.csv` (eng)

Byte-for-byte copy of the input metadata. Columns, with prefixes and names
compared case-insensitively:

- `instances`: instance label; becomes the `Row` of the other files;
- `source` (optional): origin of the instance;
- `feature_<f>`: features (numeric);
- `algo_<a>`: performance of each algorithm (numeric);
- any other column is an **annotation**: instancespace ignores it, and the
  loader loads it into `IsResult.instances`, keeping the name or using
  `ann_<name>` if it collides with a derived column.

Annotation types in the loader, by priority:

1. `load_is_output(..., annotation_types={column: type})`: origin `"forced"`;
   this is what the interface uses to change, for the session, the type of an
   inferred annotation;
2. the one declared in `annotations.json`: origin `"declared"`;
3. the heuristic, only without a declaration: origin `"inferred"`. Text or
   bool → `"categorical"`; number → `"numeric"`, except integers with at most 2
   distinct values (a binary code), which become `"categorical"`.

The origin of each type is in `IsResult.annotation_origins`. A CSV does not
store types: the text label `"1"` comes back as a number, quoted or not. That
is why the declaration exists.

### `annotations.json` (eng, optional copy)

JSON object `{annotation: type}`, with `type` ∈:

- `"categorical"`;
- `"numeric"`;
- `"integer"`: numeric with integer values, such as counts;
- `"identifier"`: key or id (in IC7, `row_original`, the row index in the
  OpenML dataset). The value is kept as it came. The column appears in the
  Data Explorer table and in the export, but not in the color and "group by"
  selectors. The heuristic never infers this type: it is only declared or
  chosen in the session.

The keys use the column name in `metadata.csv`. Not every annotation has to be
declared.

The engine validates the file **before** running and refuses three cases:

- unknown type;
- a key that is not an annotation column (it is `instances`, `source`,
  `feature_*` or `algo_*`, or does not exist);
- a numeric type with a non-numeric value, or `integer` with a non-integer
  value.

The declared types also go to `run_info.annotation_types`. The same check
(`loader_is.declared_type_errors`) runs in the loader and in the upload
validation of the interface.

### `degenerate_report.csv` (eng, optional copy)

Measures dropped **before** the engine by the metadata generator (in IC7,
`isaspace.isa.to_isa_metadata`, for zero variance after the outlier clipping
and the z-score). Columns: `feature` (str, without the prefix), `raw_variance`
(float), `iqr` (float) and `reason` (str). One row per dropped measure. Having
only the header means "checked, none was dropped". The absence of the file
means there is no information about drops before the engine.

### `feature_info.csv` (eng, optional copy)

`feature` (str, without the prefix) and `family` (str, free). One row per
received feature, including the degenerate ones. The interface shows the
family as a column in the Features tab, and the row order sets the order of
that table. In IC7, `family` is `model_derived` (CL, CLD, DS, DCP, TD_U, TD_P)
or `geometric`.

The Features tab table (`IsResult.features_table()`) joins
`degenerate_report.csv` (status `dropped_degenerate`) and `sifted_report.csv`,
with `r2_pilot` from `pilot_r2.csv` for the kept features. Its columns are
`feature`, `family` (only with `feature_info.csv`), `status`, `reason`,
`replaced_by`, `max_abs_rho`, `rho_algorithm`, `pval` and `r2_pilot`.

---

## Projection (PILOT)

### `coordinates.csv` (eng)

`Row`, `z_1`, `z_2` (float). The **PILOT z, without correction**. It is what
the interface draws. The engine rewrites this file after `save_to_csv`, which
would write the TRACE z.

### `coordinates_trace.csv` (eng, optional)

Same columns. The z TRACE used, **only when `trace_robustness.jitter_applied`**.
It differs from `coordinates.csv` only in the rows of `perturbed_labels`, and
by at most `max_shift`. The footprints were computed on this z. The loader
requires consistency: the file exists if, and only if, the jitter was applied.
It exposes the content in `IsResult.coordinates_trace` (or `None`), and uses it
to decide which instances lie inside a footprint
(`IsResult.instances_in_footprint`).

### `projection_matrix.csv` (sc)

`Row` ∈ {`Z_{1}`, `Z_{2}`} (the loader renames them to `z_1`, `z_2`), plus one
column per selected feature. It is the PILOT matrix A, **rounded to 4
decimals**. `z ≈ A · x` holds, with `x` the row of `feature_process.csv`. Full
precision is only in the `Model`.

### `pilot_r2.csv` (eng)

| column | type | content |
|---|---|---|
| `variable` | str | feature or algorithm name |
| `kind` | str | `feature` or `algorithm` |
| `r2` | float | squared correlation between the processed variable and its reconstruction from z (`x̂ = z Bᵀ`, pilot.py:478) |

The rows come first with the selected features, in `feature_raw.csv` order, and
then with the algorithms, in canonical order. `r2` says how much of each
variable the 2D plane explains.

---

## Data and PRELIM

All per instance (`Row`), with one column per feature or algorithm.

- **`feature_raw.csv`**: input values of the features **selected by SIFTED**.
  The others remain in `metadata.csv` (the loader takes them from there).
- **`feature_process.csv`**: the same features after PRELIM: outlier clipping
  at median ± `prelim.iqr_multiplier`·IQR (`bound.flag`), shift to positive
  values, Box-Cox and z-score (`norm.flag`). Both require `auto.preproc`. With
  the defaults, each column has mean 0 and standard deviation 1.
- **`algorithm_raw.csv`**: the input `algo_*`.
- **`algorithm_process.csv`**: the performance PILOT and SIFTED use.
  - With `abs_perf` it is the raw value; with relative performance it is
    `1 − algo/best` (or `algo/best − 1` without `max_perf`).
  - It is then shifted to positive values and goes through Box-Cox and z-score
    (`auto.preproc` and `norm.flag`; prelim.py:922).
- **`algorithm_bin.csv`**: bool, "good" according to `run_info.good_rule` (the
  instancespace `y_bin`).
- **`good_algos.csv`**: `NumGoodAlgos` (int), the number of good algorithms in
  the instance.
- **`beta_easy.csv`**: `IsBetaEasy` (bool), equal to
  `NumGoodAlgos > perf.beta_threshold × n_algorithms`.
- **`portfolio.csv`**: `Best_Algorithm` (int), **1-based** index of the best
  algorithm by `algo_*` (argmax with `max_perf`, argmin without). **Ties are
  broken at random** with `general.seed`. Always between 1 and n. The loader
  converts it to a name in `instances["best_algo"]`.

### Ties for the best observed value (loader)

The loader does not trust `portfolio.csv` where several algorithms share the
best value. It counts the ties with the PRELIM rule
(`compute_binary_performance`: NaN is the worst value, and an algorithm is
tied at the top when `np.equal(y_raw, y_best)`, with `y_best` the instance's
max, or min without `max_perf`), in two columns of `IsResult.instances`:

- `n_tied_best` (int): algorithms sharing the best value; 0 when the instance
  has no values;
- `best_algo_or_tie` (str): the best algorithm when `n_tied_best == 1`,
  `"tie"` when it is larger, `None` when it is 0.

Where there is no tie, the loader checks that `portfolio.csv` names the unique
best algorithm and raises an error otherwise. In the four versioned datasets
ties are common: iris 123 of 150 instances, diabetes 217 of 768,
blood-transfusion-service-center 215 of 748, hill-valley 54 of 1212.

---

## SIFTED

### `sifted_report.csv` (eng)

One row per metadata feature, in metadata order.

| column | type | content |
|---|---|---|
| `feature` | str | name |
| `status` | str | `kept`, `dropped_correlation`, `dropped_redundancy`; rarely `dropped_preprocessing` (removed before SIFTED) or `undetermined` (inconsistent reconstruction, also recorded in `run_info.warnings`) |
| `rho` | float | the correlation with the largest absolute value between the feature and the algorithms, with its sign |
| `rho_algo` | str | algorithm of that correlation |
| `pval` | float | p-value of that same correlation |
| `n_algos_sig` | int | algorithms with \|rho\| ≥ `sifted.rho` and p ≤ `sifted.pval` |
| `cluster` | int or empty | cluster (1..k) of the features that passed the correlation filter, when there was clustering |
| `kept_instead` | str or empty | only in `dropped_redundancy`: the feature kept in the same cluster |

Rules, reconstructed from `Model.sifted` (sifted.py:774-808 and 1056-1078):

- A feature **passes the correlation filter** when it:
  - is the one most correlated with some algorithm, **or**
  - has \|rho\| ≥ `sifted.rho` with p ≤ `sifted.pval` for some algorithm.
- Those that do not pass are `dropped_correlation`.
- **Clustering:** if more than 3 features and more than `sifted.k` remain,
  they are grouped into `sifted.k` clusters, and a genetic algorithm keeps
  **exactly one per cluster**. The others are `dropped_redundancy`.
- Without clustering, every feature that passed the correlation filter is
  `kept`.

### `sifted_correlations.csv` (eng)

Long format, with the full matrix: one row per feature x algorithm pair, in the
columns `feature`, `algorithm`, `rho` and `pval` (float). It is the Pearson
correlation between the processed feature and the processed performance
(`algorithm_process.csv`), over every feature that entered SIFTED. It has only
the header if SIFTED computed no correlations. The loader exposes the long
format and the matrices `IsResult.sifted_rho` and `IsResult.sifted_pval`
(feature x algorithm).

### `sifted_silhouette.csv` (eng)

| column | type | content |
|---|---|---|
| `k` | int | number of clusters tried: 3 .. (features that passed the correlation filter) − 1 |
| `silhouette` | float | mean silhouette with correlation distance |
| `used` | bool | `k == sifted.k`; the k actually used, fixed in the options |
| `best` | bool | the k with the highest silhouette; instancespace only suggests it in the log, it does not use it |

It has only the header when there was no clustering. In the versioned
datasets the k used is 6 in all four; the highest silhouette is at k = 3 in
diabetes and k = 4 in hill-valley.

---

## PYTHIA

PYTHIA trains one classifier per algorithm (SVM by default,
`pythia.classifier`) that predicts "good" (`algorithm_bin.csv`) from z. The
hyperparameters are tuned by stratified cross-validation with
`pythia.cv_folds` folds.

- **`algorithm_svm.csv`** (sc): bool, `y_hat`. It is the `predict()` of the
  final classifier, **in sample**. The loader puts it in `algo_<a>_svm`.
- **`portfolio_svm.csv`** (sc): `Best_Algorithm` (int), equal to
  `selection0`, **0-based** index, with **-1 = none**. The loader converts it
  to a name in `instances["best_algo_svm"]`.

### `pythia_proba.csv` (eng)

`Row`, then one `<algo>` column per algorithm, then one `<algo>_hat` per
algorithm, in canonical order. Every value is a float in [0, 1] and means
**P(bad)**: the probability (`predict_proba`, with Platt scaling for the SVM)
that the algorithm is **not** good on the instance. P(good) = 1 − value.

- **`<algo>` = `pr0_sub`, the default:** **out-of-sample** probability, from
  `cross_val_predict` with the tuned classifier. Each instance is evaluated by
  a model that did not see it. It is the honest estimate.
- **`<algo>_hat` = `pr0_hat`:** the final model, trained on all instances and
  evaluated on them, **in sample**. It is more optimistic.
- `y_hat` (`algorithm_svm.csv`) can disagree with `pr0_hat < 0.5`, because the
  SVC's `predict` and `predict_proba` are not equivalent.
  `run_info.pythia.y_hat_disagrees_with_pr0_hat` counts the pairs in
  disagreement.

The engine refuses, before running, algorithms whose name clashes with
`<other>_hat`. The loader exposes `IsResult.pythia_proba` (pr0_sub) and
`IsResult.pythia_proba_hat` (pr0_hat), both with columns = algorithms.

### `pythia_confusion.csv` (eng)

`Algorithm` (str), `tn`, `fp`, `fn`, `tp` (int). The confusion matrix per
algorithm: the truth is "good" (`algorithm_bin.csv`), and the prediction is
the cross-validation one (`y_sub`, out of sample).

- positive = good;
- `tn + fp` = bad instances;
- `fn + tp` = good instances;
- each row sums to `n_instances`.

Accuracy, precision and recall computed from here match the `CV_model_*`
columns of `svm_table.csv`. The engine checks this, because instancespace uses
another column order in its evaluation path.

### `pythia_selection.csv` (eng)

`Row`, `selection0`, `selection1` (algorithm name; empty = none).

- **`selection0`:** among the algorithms with `y_hat` true, the one with the
  highest cross-validation precision. Empty when none is predicted good. It
  is the same as `portfolio_svm.csv`.
- **`selection1`:** equal to `selection0`, but, when there is no
  recommendation, it uses the algorithm with the largest fraction of good
  instances. Never empty.
- `run_info.pythia` has the counts `instance_algorithm_pairs`,
  `y_hat_disagrees_with_pr0_hat`, `selection0_none` and
  `selection1_differs_from_selection0`.

### `svm_table.csv` (sc)

`Row` is an algorithm, `Oracle` or `Selector`. Values rounded to 3 decimals;
percentages to 1 decimal; empty cell where it does not apply.

| column | algorithm | `Oracle` | `Selector` |
|---|---|---|---|
| `Avg_Perf_all_instances`, `Std_Perf_all_instances` | mean and std of `algo_*` over all instances | of the best performance per instance | of the `selection1` algorithm |
| `Probability_of_good` | fraction of good instances | 1 | fraction where the `selection1` algorithm is good |
| `Avg_Perf_selected_instances`, `Std_Perf_selected_instances` | of `algo_*` where `y_hat` is true | empty | of the `selection0` algorithm (without the "none" instances) |
| `CV_model_accuracy` | CV accuracy (%) | empty | empty |
| `CV_model_precision`, `CV_model_recall` | CV precision and recall (%) | empty | precision and recall of the selector, MATLAB definition (pythia.py:2006-2016) |
| `BoxConstraint`, `KernelScale` | SVM hyperparameters | empty | empty |

The Selector recall follows the MATLAB definition, which counts as a miss
every instance with some good algorithm that was not recommended; with
several good algorithms per instance it stays close to 50%.

---

## CLOISTER

### `bounds.csv` and `bounds_prunned.csv` (sc)

`Row` (`bnd_pnt_1` … `bnd_pnt_m`), `z_1`, `z_2`. The vertices, in order and
without repeating the first, of a convex polygon in z: the estimated boundary
of the region where instances can exist.

- The boundary is obtained by projecting with A the min/max combinations of
  each processed feature and taking the convex hull.
- `bounds_prunned.csv` uses only the combinations compatible with the
  significant correlations between features (`cloister.p_val`,
  `cloister.c_thres`). It can be equal to `bounds.csv`.
- With more than `cloister.max_features` features (default 20), CLOISTER does
  not enumerate the combinations: it uses the convex hull of the projected
  instances, and both files are equal (cloister.py:190-203).

The loader exposes `IsResult.bounds` and `IsResult.bounds_pruned`, as
`Polygon`.

---

## TRACE

### Footprint schema

Applies to `footprint_<algo>_good.csv`, `footprint_<algo>_best.csv`,
`footprint_space.csv` and `footprint_hard.csv`.

| column | type | content |
|---|---|---|
| `Row` | int | vertex counter, 1..k; **not** an instance label |
| `Part` | int | part of the (multi)polygon, 1..p |
| `Ring` | str | `exterior` or `hole_<j>` (hole j of the part) |
| `Vertex` | int | order of the vertex in the ring, 1..; the first is not repeated at the end |
| `z_1`, `z_2` | float | coordinates |

- Each `Part` has one `exterior` ring and zero or more holes.
- The loader returns one `Polygon` per `Part` (`exterior` and `holes`), with
  the area by the shoelace formula, grouped in a `Footprint` (`polygons`,
  `status`, `file`, `normalized_area`, `normalized_density`, `purity`).
- Missing file = **empty** footprint. The loader sets the status to `empty`
  (never `ok`); to `suspect` when the purity is below `trace.purity`; and to
  `ok` otherwise.
- The footprints are computed on the z of `coordinates_trace.csv`, when it
  exists.

### Files

- **`footprint_<algo>_good.csv`** (sc): region where the algorithm is good
  (`algorithm_bin.csv`), kept only where the purity is ≥ `trace.purity`.
  - With `trace.use_sim=false` (the engine's default) it uses the observed
    performance; with `true`, the PYTHIA predictions.
  - The file name uses a sanitized version of the algorithm name
    (instancespace's `_portable_stems`). The exact name is in
    `run_info.footprint_files`, with `null` for the empty ones.
- **`footprint_<algo>_best.csv`** (sc): the same, for the region where the
  algorithm is the best (`portfolio.csv`, so ties are broken at random here
  too). With `trace.contra` (default), overlaps between the best footprints of
  different algorithms are resolved and removed (trace.py:646-648). The good
  footprints do not go through that step.
- **`footprint_space.csv`** (eng): region of all instances. Its area and
  density are the denominators of the `*_Normalized` columns.
- **`footprint_hard.csv`** (eng): region of the instances that are **not**
  beta-easy (`IsBetaEasy` false), the TRACE "beta footprint".
- The metrics of the last two are in `run_info.special_footprints.<space|hard>`:
  `file`, `area`, `density`, `purity`, `elements` (covered instances),
  `good_elements` and, only in `hard`, `normalized_area` and
  `normalized_density`.

### `footprint_performance.csv` (sc)

`Row` (algorithm), `Area_Good_Normalized`, `Density_Good_Normalized`,
`Purity_Good`, `Area_Best_Normalized`, `Density_Best_Normalized` and
`Purity_Best` (float, **3 decimals**).

- Area and density are normalized by those of the space (`footprint_space`).
- Purity is the fraction of covered instances that are good (or best).
- It is 0 when the footprint is empty.
- The absolute areas and densities are not written.
