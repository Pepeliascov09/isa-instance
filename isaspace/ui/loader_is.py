"""Reader for the output folder written by isaspace.engine (instancespace).

ARCHITECTURAL RULE: this module does not import instancespace, pyispace, pyhard
or sklearn; only pandas and numpy. The legacy isaspace/ui/loader.py (pyispace
output) still reads resultados/isa/.

What it does:
- Row is the instance label (the metadata's instances column), read as text;
  it does not have to be 1..n;
- footprints use the Row, Part, Ring, Vertex, z_1, z_2 schema: one Polygon per
  Part, holes (Ring hole_k) preserved; a footprint without a file is EMPTY;
- True/False files (algorithm_bin, algorithm_svm, beta_easy) become bool;
- portfolio.csv (PRELIM) is 1-based and portfolio_svm.csv (PYTHIA) is 0-based
  with -1 = none: both become algorithm names, None for none;
- ties for the best observed performance are counted with the PRELIM rule
  (instances["n_tied_best"]); instances["best_algo_or_tie"] is the best
  algorithm, or TIE when several share the best value (portfolio.csv breaks
  those ties at random);
- metadata.csv annotations (every column that is not instances, source,
  feature_* or algo_*) go into instances; their type comes from
  annotations.json when the folder has it (declared) and from the heuristic
  _infer_annotation_type otherwise (inferred), with the origin in
  `annotation_origins`;
- degenerate_report.csv and feature_info.csv when present, and
  features_table() with one row per received feature;
- CLOISTER: bounds.csv and bounds_prunned.csv as Polygon;
- coordinates.csv (the PILOT z) is what gets drawn; coordinates_trace.csv (the
  z TRACE used, only when jitter was applied) is in `coordinates_trace`;
- the space and hard footprints, sifted_report, sifted_correlations,
  sifted_silhouette, pilot_r2, pythia_proba (pr0_sub and pr0_hat),
  pythia_confusion, pythia_selection, svm_table, run_info and run_options.

The format of every file is in docs/output_format.md.

Columns of IsResult.instances (index Row, text labels):
  z_1, z_2                     coordinates.csv
  source                       the metadata's source column, if any
  feature_<f>                  input values of ALL metadata features
                               (the SIFTED ones come from feature_raw.csv, equal to the metadata)
  feature_<f>_z                feature_process.csv (SIFTED features only)
  algo_<a>                     algorithm_raw.csv
  algo_<a>_bin                 algorithm_bin.csv (bool: good in the observed performance)
  algo_<a>_svm                 algorithm_svm.csv (bool: good according to PYTHIA)
  NumGoodAlgos, IsBetaEasy     good_algos.csv, beta_easy.csv (bool)
  n_bad_algos                  number of algorithms minus NumGoodAlgos
  best_algo, best_algo_svm     portfolio.csv, portfolio_svm.csv -> name or None
  n_tied_best                  algorithms sharing the best observed value (0 = no values)
  best_algo_or_tie             best_algo when n_tied_best == 1, TIE when > 1, None when 0
  <annotations>                from the metadata; ann_<name> if the name collides
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROW = "Row"
FOOTPRINT_TYPES = ("good", "best")
NUMERIC = "numeric"
CATEGORICAL = "categorical"
INTEGER = "integer"              # numeric with integer values (counts)
IDENTIFIER = "identifier"        # key/id: in the table and the export, not in plots
ANNOTATION_TYPES = (CATEGORICAL, NUMERIC, INTEGER, IDENTIFIER)
DECLARED, INFERRED, FORCED = "declared", "inferred", "forced"
OK, EMPTY, SUSPECT = "ok", "empty", "suspect"
TIE = "tie"
FOOTPRINT_COLUMNS = ["Row", "Part", "Ring", "Vertex", "z_1", "z_2"]
_BOOL = {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False}
_RE_Z = re.compile(r"^Z_\{(\d+)\}$")
_RE_HOLE = re.compile(r"^hole_(\d+)$")


def _area(ring) -> float:
    """Shoelace area (unsigned); ring (k, 2) without repeating the first vertex."""
    x, y = ring[:, 0], ring[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def is_numeric_type(kind: str) -> bool:
    return kind in (NUMERIC, INTEGER)


def annotation_columns(columns) -> list:
    """Metadata columns that are not instances, source, feature_* or algo_*."""
    return [c for c in columns if str(c).casefold() not in ("instances", "source")
            and not str(c).casefold().startswith(("feature_", "algo_"))]


def declared_type_errors(meta: pd.DataFrame, types) -> list:
    """Problems of an {annotation: type} declaration (annotations.json) for the
    metadata `meta`; empty list if everything is fine. Used by the loader, by
    the engine (before running) and by the upload validation in the UI."""
    if not isinstance(types, dict):
        return ["annotations.json must be an object {annotation: type}"]
    annotations = annotation_columns(meta.columns)
    errors = []
    for col, kind in types.items():
        if kind not in ANNOTATION_TYPES:
            errors.append(f"{col!r}: type {kind!r} is not one of {', '.join(ANNOTATION_TYPES)}")
        elif col not in annotations:
            errors.append(f"{col!r} is not an annotation column of the metadata")
        elif is_numeric_type(kind):
            v = pd.to_numeric(meta[col], errors="coerce")
            if (v.isna() & meta[col].notna()).any():
                errors.append(f"{col!r} declared {kind} has non-numeric values")
            elif kind == INTEGER and (np.mod(v.dropna(), 1) != 0).any():
                errors.append(f"{col!r} declared {kind} has non-integer values")
    return errors


def best_ties(y: np.ndarray, higher_is_better: bool) -> np.ndarray:
    """Number of algorithms sharing the best value of each instance, with the
    PRELIM rule (instancespace prelim.py, compute_binary_performance): NaN
    counts as the worst value, and an algorithm is tied at the top when its raw
    value equals the instance's best. 0 when the instance has no values."""
    y = np.asarray(y, dtype=float)
    aux = np.where(np.isnan(y), -np.inf if higher_is_better else np.inf, y)
    best = aux.max(axis=1) if higher_is_better else aux.min(axis=1)
    return np.equal(y, best[:, None]).sum(axis=1)


def _in_ring(pts, ring, tol):
    """(inside or on the boundary, on the boundary) of each point for a ring."""
    x, y = pts[:, :1], pts[:, 1:2]
    x1, y1 = ring[None, :, 0], ring[None, :, 1]
    x2, y2 = np.roll(ring[:, 0], -1)[None, :], np.roll(ring[:, 1], -1)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        crosses = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
        dx, dy = x2 - x1, y2 - y1
        l2 = dx * dx + dy * dy
        t = np.clip(((x - x1) * dx + (y - y1) * dy) / np.where(l2 == 0, 1, l2), 0, 1)
    boundary = (((x1 + t * dx - x) ** 2 + (y1 + t * dy - y) ** 2) <= tol * tol).any(axis=1)
    return (crosses.sum(axis=1) % 2 == 1) | boundary, boundary


@dataclass
class Polygon:
    """A polygon in z_1 x z_2: exterior ring and holes, without repeating the first vertex."""

    exterior: np.ndarray                       # (k, 2)
    holes: list = field(default_factory=list)  # list of (k, 2)

    @property
    def area(self) -> float:
        return _area(self.exterior) - sum(_area(h) for h in self.holes)

    @property
    def n_vertices(self) -> int:
        return len(self.exterior)

    def contains(self, pts, tol=1e-9) -> np.ndarray:
        """Points (n, 2) inside or on the boundary and outside the holes (the
        hole boundary counts as inside), like TRACE's pointwise_covers."""
        pts = np.asarray(pts, dtype=float)
        inside, _ = _in_ring(pts, self.exterior, tol)
        for hole in self.holes:
            in_hole, boundary = _in_ring(pts, hole, tol)
            inside &= ~(in_hole & ~boundary)
        return inside


@dataclass
class Footprint:
    """Footprint of an algorithm (or of the space / the hard instances, with
    algo None); `polygons` is empty when there is no file."""

    algo: str | None
    kind: str                       # "good" | "best" | "space" | "hard"
    polygons: list                  # list[Polygon], one per Part
    status: str                     # "ok" | "empty" | "suspect" (purity < trace.purity)
    file: str | None
    normalized_area: float          # from footprint_performance.csv
    normalized_density: float
    purity: float

    @property
    def is_empty(self) -> bool:
        return not self.polygons

    @property
    def area(self) -> float:
        return sum(p.area for p in self.polygons)

    def contains(self, pts) -> np.ndarray:
        pts = np.asarray(pts, dtype=float)
        inside = np.zeros(len(pts), dtype=bool)
        for p in self.polygons:
            inside |= p.contains(pts)
        return inside


@dataclass
class IsResult:
    """Content of a folder written by isaspace.engine, ready for the UI."""

    name: str
    path: Path
    instances: pd.DataFrame          # one row per instance, index Row (text)
    algos: list                      # instancespace order (portfolio indices)
    features: list                   # features chosen by SIFTED
    features_input: list             # metadata features, in sifted_report order
    features_all: list               # metadata.csv features, in file order
    annotations: dict                # {column in instances: annotation type}
    annotation_renames: dict         # {name in the metadata: name in instances}
    annotation_origins: dict         # {column in instances: "declared" | "inferred" | "forced"}
    source_column: str | None        # "source" if the metadata had source
    footprints: dict                 # {(algo, type): Footprint}, every algo x type
    footprint_space: Footprint       # TRACE: all instances
    footprint_hard: Footprint        # TRACE: instances that are not beta-easy
    footprint_performance: pd.DataFrame   # index Algorithm
    projection_matrix: pd.DataFrame  # rows z_1, z_2; columns = features
    bounds: Polygon | None           # CLOISTER, bounds.csv
    bounds_pruned: Polygon | None    # CLOISTER, bounds_prunned.csv
    coordinates_trace: pd.DataFrame | None   # z_1, z_2 used by TRACE (only with jitter)
    sifted_report: pd.DataFrame
    sifted_correlations: pd.DataFrame  # feature, algorithm, rho, pval (long format)
    sifted_silhouette: pd.DataFrame  # k, silhouette, used, best
    pilot_r2: pd.DataFrame           # variable, kind, r2
    pythia_proba: pd.DataFrame       # P(bad) out of sample (pr0_sub), instance x algorithm
    pythia_proba_hat: pd.DataFrame   # P(bad) in sample (pr0_hat)
    pythia_confusion: pd.DataFrame   # index Algorithm; tn, fp, fn, tp
    pythia_selection: pd.DataFrame   # index Row; selection0, selection1 (name or None)
    svm_table: pd.DataFrame          # index Algorithm (+ Oracle, Selector)
    run_info: dict
    run_options: dict
    pi: float                        # effective trace.purity
    degenerate_report: pd.DataFrame | None   # feature, raw_variance, iqr, reason (before the engine)
    feature_info: pd.DataFrame | None        # feature, family

    @property
    def n(self) -> int:
        return len(self.instances)

    @property
    def n_features_used(self) -> int:
        return len(self.features)

    @property
    def n_features_dropped(self) -> int:
        return len(self.features_input) - len(self.features)

    @property
    def higher_is_better(self) -> bool:
        return bool(self.run_options.get("perf", {}).get("max_perf", True))

    @property
    def orientation(self) -> dict:
        """run_info.json["orientation"] (empty for folders written before it existed)."""
        return self.run_info.get("orientation", {})

    @property
    def features_outside_pilot(self) -> list:
        """Metadata features that SIFTED did not pass to PILOT."""
        return [f for f in self.features_all if f not in self.features]

    @property
    def inferred_annotations(self) -> list:
        return [c for c, o in self.annotation_origins.items() if o == INFERRED]

    def instances_in_footprint(self, fp) -> list:
        """Labels of the instances inside (or on the boundary of) the footprint,
        in the z TRACE used (coordinates_trace when jitter was applied)."""
        z = self.coordinates_trace if self.coordinates_trace is not None else self.instances[["z_1", "z_2"]]
        return list(z.index[fp.contains(z.to_numpy())])

    def features_table(self) -> pd.DataFrame:
        """One row per received feature: the degenerate ones from
        degenerate_report (dropped before the engine) and those of sifted_report.

        Columns: feature, [family], status, reason, replaced_by, max_abs_rho,
        rho_algorithm, pval, r2_pilot. Order: that of feature_info when it
        exists; otherwise the metadata's, with the degenerate ones at the end.
        """
        rep = self.sifted_report
        sif = self.run_options.get("sifted", {})
        lim_rho, lim_p = sif.get("rho"), sif.get("pval")
        r2 = self.pilot_r2[self.pilot_r2["kind"] == "feature"].set_index("variable")["r2"]
        rows = []
        if self.degenerate_report is not None:
            for d in self.degenerate_report.itertuples(index=False):
                rows.append({"feature": str(d.feature), "status": "dropped_degenerate",
                             "reason": str(d.reason)})
        for x in rep.itertuples(index=False):
            cluster = None if pd.isna(x.cluster) else int(x.cluster)
            if x.status == "kept":
                reason = f"kept (cluster {cluster})" if cluster else "kept"
            elif x.status == "dropped_correlation":
                reason = (f"no algorithm with |rho| ≥ {lim_rho} and p ≤ {lim_p} "
                          f"(largest |rho| {abs(x.rho):.2f}, p = {x.pval:.2g})")
            elif x.status == "dropped_redundancy":
                reason = f"redundant in cluster {cluster}: {x.kept_instead} was kept"
            elif x.status == "dropped_preprocessing":
                reason = "removed by the instancespace PREPROCESSING"
            else:
                reason = "reason not reconstructed (see run_info.warnings)"
            rows.append({
                "feature": x.feature, "status": x.status, "reason": reason,
                "replaced_by": x.kept_instead if x.status == "dropped_redundancy" else None,
                "max_abs_rho": abs(x.rho) if pd.notna(x.rho) else np.nan,
                "rho_algorithm": x.rho_algo, "pval": x.pval,
                "r2_pilot": float(r2[x.feature]) if x.status == "kept" and x.feature in r2 else np.nan,
            })
        columns = ["feature", "status", "reason", "replaced_by", "max_abs_rho",
                   "rho_algorithm", "pval", "r2_pilot"]
        table = pd.DataFrame(rows).reindex(columns=columns)
        if self.feature_info is not None:
            family = dict(zip(self.feature_info["feature"].astype(str), self.feature_info["family"]))
            table.insert(1, "family", table["feature"].map(family))
            order = list(self.feature_info["feature"].astype(str))
        else:
            order = list(self.features_all)
        position = {f: i for i, f in enumerate(order)}
        table["_order"] = table["feature"].map(lambda f: position.get(f, len(order)))
        return table.sort_values("_order", kind="stable").drop(columns="_order").reset_index(drop=True)

    @property
    def empty_footprints(self) -> list:
        return [k for k, f in self.footprints.items() if f.status == EMPTY]

    @property
    def suspect_footprints(self) -> dict:
        return {k: f.purity for k, f in self.footprints.items() if f.status == SUSPECT}

    def _pivot(self, value) -> pd.DataFrame:
        c = self.sifted_correlations
        table = c.pivot(index="feature", columns="algorithm", values=value)
        return table.reindex(index=list(dict.fromkeys(c["feature"])), columns=self.algos)

    @property
    def sifted_rho(self) -> pd.DataFrame:
        """Feature x algorithm correlation (rows = SIFTED input features)."""
        return self._pivot("rho")

    @property
    def sifted_pval(self) -> pd.DataFrame:
        return self._pivot("pval")


def list_available(root) -> list:
    """Names of the subfolders of `root` with run_info.json (complete engine run)."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "run_info.json").is_file())


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{path.name} not found in {path.parent}")
    return path


def _read_by_row(path: Path, dtype=None, skiprows=0) -> pd.DataFrame:
    """CSV indexed by Row, with Row as text (a label, never an integer)."""
    types = {ROW: str} if dtype is None else {ROW: str, **dtype}
    df = pd.read_csv(_require(path), dtype=types, skiprows=skiprows)
    if ROW not in df.columns:
        raise ValueError(f"{path.name} has no {ROW} column")
    df = df.set_index(ROW)
    if df.index.isna().any():
        raise ValueError(f"{path.name}: empty Row")
    return df


def _aligned(base: pd.Index, df: pd.DataFrame, source: str) -> pd.DataFrame:
    """The instancespace outputs follow the instance order; require the same one."""
    if not df.index.equals(base):
        raise ValueError(f"{source}: Row does not match coordinates.csv "
                         f"({len(df)} rows against {len(base)})")
    return df


def _to_bool(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = {}
    for col in df.columns:
        values = df[col].astype(str).str.strip().str.lower().map(_BOOL)
        if values.isna().any():
            bad = sorted(set(df.loc[values.isna(), col].astype(str)))[:5]
            raise ValueError(f"{source}: non-binary values in {col}: {bad}")
        out[col] = values.astype(bool)
    return pd.DataFrame(out, index=df.index)


def _name_from_index(values, algos, base, source):
    """Portfolio index -> algorithm name; out of range = None."""
    n = len(algos)
    names = []
    for v in values:
        if pd.isna(v):
            names.append(None)
            continue
        if int(v) != v:
            raise ValueError(f"{source}: non-integer index {v}")
        i = int(v) - base
        names.append(algos[i] if 0 <= i < n else None)
    return names


def _read_footprint(path: Path) -> list:
    """File in the Row, Part, Ring, Vertex, z_1, z_2 schema -> list[Polygon]."""
    df = pd.read_csv(path)
    if list(df.columns) != FOOTPRINT_COLUMNS:
        raise ValueError(f"{path.name}: columns {list(df.columns)}, expected "
                         f"{FOOTPRINT_COLUMNS} (instancespace format)")
    polygons = []
    for part, block in df.groupby("Part", sort=True):
        rings = {}
        for ring, pts in block.groupby("Ring", sort=False):
            xy = pts.sort_values("Vertex")[["z_1", "z_2"]].to_numpy(dtype=float)
            if len(xy) < 3:
                raise ValueError(f"{path.name}: Part {part} Ring {ring} with {len(xy)} vertices")
            rings[str(ring)] = xy
        if "exterior" not in rings:
            raise ValueError(f"{path.name}: Part {part} without an exterior Ring")
        holes = []
        for name in rings:
            if name == "exterior":
                continue
            m = _RE_HOLE.match(name)
            if m is None:
                raise ValueError(f"{path.name}: unknown Ring {name!r}")
            holes.append((int(m.group(1)), rings[name]))
        polygons.append(Polygon(rings["exterior"], [xy for _, xy in sorted(holes, key=lambda t: t[0])]))
    return polygons


def _read_bounds(path: Path) -> Polygon | None:
    if not path.is_file():
        return None
    df = pd.read_csv(path, dtype={ROW: str})
    xy = df[["z_1", "z_2"]].to_numpy(dtype=float)
    if len(xy) < 3:
        raise ValueError(f"{path.name}: boundary with {len(xy)} points")
    return Polygon(xy)


def _names_or_none(series: pd.Series, algos, source) -> pd.Series:
    """Column of algorithm names; empty = None; unknown name = error."""
    out = series.where(series.notna(), None).astype(object)
    bad = sorted({v for v in out if v is not None and v not in algos})
    if bad:
        raise ValueError(f"{source}: unknown algorithms {bad}")
    return out


def _special_footprint(path: Path, kind: str, record: dict, pi: float) -> Footprint:
    """footprint_space / footprint_hard with the metrics from run_info.json."""
    file = record.get("file")
    if file is not None and not (path / file).is_file():
        raise FileNotFoundError(f"run_info.json cites {file}, which does not exist in {path}")
    polygons = _read_footprint(path / file) if file is not None else []
    purity = float(record.get("purity", np.nan))
    status = EMPTY if not polygons else (SUSPECT if purity < pi else OK)
    return Footprint(
        algo=None, kind=kind, polygons=polygons, status=status, file=file,
        normalized_area=float(record.get("normalized_area", 1.0 if polygons else 0.0)),
        normalized_density=float(record.get("normalized_density", 1.0 if polygons else 0.0)),
        purity=purity,
    )


def _read_optional(path: Path, dtype=None) -> pd.DataFrame | None:
    return pd.read_csv(path, dtype=dtype) if path.is_file() else None


def _infer_annotation_type(series: pd.Series) -> str:
    """categorical: text, bool, or integers with at most 2 values (a binary
    code, such as class 0/1); numeric: everything else."""
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        return CATEGORICAL
    v = series.dropna()
    if len(v) and np.all(np.mod(v.to_numpy(dtype=float), 1) == 0) and v.nunique() <= 2:
        return CATEGORICAL
    return NUMERIC


def _text(v):
    if isinstance(v, (str, bool, np.bool_)):
        return str(v)
    if isinstance(v, (int, float, np.integer, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def _as_category(series: pd.Series) -> pd.Series:
    """Values as text (1 -> "1", 2.0 -> "2"), keeping missing values as NaN."""
    return series.map(_text).where(series.notna(), np.nan).astype(object)


def _value(row, column) -> float:
    return float(row[column]) if row is not None and column in row else np.nan


def _read_metadata(path: Path):
    """(DataFrame indexed by label, source column or None, annotation columns,
    [(column, name) of the features])."""
    cols = list(pd.read_csv(_require(path), nrows=0).columns)
    lower = [str(c).casefold() for c in cols]
    if lower.count("instances") != 1:
        raise ValueError("metadata.csv needs exactly one instances column")
    col_inst = cols[lower.index("instances")]
    col_src = cols[lower.index("source")] if "source" in lower else None
    meta = pd.read_csv(path, dtype={col_inst: str})
    meta = meta.set_index(col_inst)
    if meta.index.isna().any():
        raise ValueError("metadata.csv: instances column with an empty label")
    ann = [c for c, b in zip(cols, lower)
           if c not in (col_inst, col_src) and not b.startswith(("feature_", "algo_"))]
    feats = [(c, c[len("feature_"):]) for c, b in zip(cols, lower) if b.startswith("feature_")]
    return meta, col_src, ann, feats


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def load_is_output(dirpath, annotation_types=None) -> IsResult:
    """Read the folder `dirpath` written by isaspace.engine.run_instancespace.

    `annotation_types` ({column: annotation type}) overrides the automatic
    detection of annotation types (uses the column name in the metadata).
    """
    path = Path(dirpath)
    if not path.is_dir():
        raise NotADirectoryError(f"invalid folder: {path}")
    run_info = json.loads(_require(path / "run_info.json").read_text())
    run_options = json.loads(_require(path / "run_options.json").read_text())

    coords = _read_by_row(path / "coordinates.csv")
    if not coords.index.is_unique:
        raise ValueError("coordinates.csv: repeated instance labels")
    base = coords.index

    algo_raw = _aligned(base, _read_by_row(path / "algorithm_raw.csv"), "algorithm_raw.csv")
    algos = list(algo_raw.columns)
    if run_info.get("algorithms") not in (None, algos):
        raise ValueError(f"run_info.json lists {run_info['algorithms']}, algorithm_raw.csv has {algos}")
    all_str = {a: str for a in algos}
    algo_bin = _to_bool(_aligned(base, _read_by_row(path / "algorithm_bin.csv", all_str),
                                 "algorithm_bin.csv"), "algorithm_bin.csv")
    algo_svm = _to_bool(_aligned(base, _read_by_row(path / "algorithm_svm.csv", all_str),
                                 "algorithm_svm.csv"), "algorithm_svm.csv")
    feat_raw = _aligned(base, _read_by_row(path / "feature_raw.csv"), "feature_raw.csv")
    feat_proc = _aligned(base, _read_by_row(path / "feature_process.csv"), "feature_process.csv")
    features = list(feat_raw.columns)
    good = _aligned(base, _read_by_row(path / "good_algos.csv"), "good_algos.csv")
    beta = _to_bool(_aligned(base, _read_by_row(path / "beta_easy.csv", {"IsBetaEasy": str}),
                             "beta_easy.csv"), "beta_easy.csv")
    port = _aligned(base, _read_by_row(path / "portfolio.csv"), "portfolio.csv")
    port_svm = _aligned(base, _read_by_row(path / "portfolio_svm.csv"), "portfolio_svm.csv")

    parts = [
        coords[["z_1", "z_2"]],
        feat_raw.add_prefix("feature_"),
        feat_proc.add_prefix("feature_").add_suffix("_z"),
        algo_raw.add_prefix("algo_"),
        algo_bin.add_prefix("algo_").add_suffix("_bin"),
        algo_svm.add_prefix("algo_").add_suffix("_svm"),
        good[["NumGoodAlgos"]],
        beta[["IsBetaEasy"]],
    ]
    instances = pd.concat(parts, axis=1)
    instances["n_bad_algos"] = len(algos) - instances["NumGoodAlgos"]
    instances["best_algo"] = _name_from_index(port["Best_Algorithm"], algos, 1, "portfolio.csv")
    instances["best_algo_svm"] = _name_from_index(port_svm["Best_Algorithm"], algos, 0,
                                                  "portfolio_svm.csv")
    # ties for the best observed value: portfolio.csv picks one at random
    higher = bool(run_options.get("perf", {}).get("max_perf", True))
    y = algo_raw.to_numpy(dtype=float)
    n_tied = best_ties(y, higher)
    aux = np.where(np.isnan(y), -np.inf if higher else np.inf, y)
    unique_best = np.asarray(algos, dtype=object)[aux.argmax(axis=1) if higher else aux.argmin(axis=1)]
    unique = n_tied == 1
    if (instances["best_algo"].to_numpy(dtype=object)[unique] != unique_best[unique]).any():
        raise ValueError("portfolio.csv: best algorithm differs from the unique best of algorithm_raw.csv")
    instances["n_tied_best"] = n_tied
    instances["best_algo_or_tie"] = np.where(
        n_tied > 1, TIE, np.where(n_tied == 1, instances["best_algo"].to_numpy(dtype=object), None)
    ).astype(object)

    # --- metadata: source and annotations, by label
    meta, col_src, col_ann, col_feats = _read_metadata(path / "metadata.csv")
    if meta.index.equals(base):
        meta_al = meta
    elif meta.index.is_unique and base.isin(meta.index).all():
        meta_al = meta.loc[base]         # PRELIM/SIFTED used a subset
    else:
        raise ValueError(f"metadata.csv: labels do not match coordinates.csv "
                         f"({len(meta)} rows against {len(base)})")
    # features SIFTED did not pass to PILOT: input values, from the metadata
    for col, name in col_feats:
        if name not in features:
            instances[f"feature_{name}"] = pd.to_numeric(meta_al[col], errors="coerce")
    source_column = None
    if col_src is not None:
        instances["source"] = _as_category(meta_al[col_src])
        source_column = "source"
    forced = dict(annotation_types or {})
    declared = {}
    if (path / "annotations.json").is_file():
        declared = json.loads((path / "annotations.json").read_text())
        errors = declared_type_errors(meta.reset_index(), declared)
        if errors:
            raise ValueError("annotations.json: " + "; ".join(errors))
    annotations, renames, origins = {}, {}, {}
    for col in col_ann:
        target = col
        while target in instances.columns or target == ROW:
            target = f"ann_{target}"
        if target != col:
            renames[col] = target
        series = meta_al[col]
        if col in forced:
            kind, origins[target] = forced[col], FORCED
        elif col in declared:
            kind, origins[target] = declared[col], DECLARED
        else:
            kind, origins[target] = _infer_annotation_type(series), INFERRED
        if kind not in ANNOTATION_TYPES:
            raise ValueError(f"invalid annotation type for {col}: {kind!r}")
        if kind == CATEGORICAL:
            instances[target] = _as_category(series)
        elif kind == IDENTIFIER:
            instances[target] = series          # as it came: it only identifies the instance
        else:
            numbers = pd.to_numeric(series, errors="coerce")
            if (numbers.isna() & series.notna()).any():
                raise ValueError(f"annotation {col} ({origins[target]} {kind}) has non-numeric values")
            instances[target] = numbers
        annotations[target] = kind

    # --- TRACE
    perf = _read_by_row(path / "footprint_performance.csv")
    perf.index.name = "Algorithm"
    pi = float(run_options["trace"]["purity"])
    files = run_info.get("footprint_files", {})
    footprints = {}
    for algo in algos:
        for kind in FOOTPRINT_TYPES:
            if algo in files:
                file = files[algo].get(kind)
            else:
                pattern = f"footprint_{algo}_{kind}.csv"
                file = pattern if (path / pattern).is_file() else None
            if file is not None and not (path / file).is_file():
                raise FileNotFoundError(f"run_info.json cites {file}, which does not exist in {path}")
            polygons = _read_footprint(path / file) if file is not None else []
            suffix = "Good" if kind == "good" else "Best"
            row = perf.loc[algo] if algo in perf.index else None
            purity = _value(row, f"Purity_{suffix}")
            status = EMPTY if not polygons else (SUSPECT if purity < pi else OK)
            footprints[(algo, kind)] = Footprint(
                algo=algo, kind=kind, polygons=polygons, status=status, file=file,
                normalized_area=_value(row, f"Area_{suffix}_Normalized"),
                normalized_density=_value(row, f"Density_{suffix}_Normalized"),
                purity=purity,
            )

    proj = _read_by_row(path / "projection_matrix.csv")
    proj.index = [(f"z_{m.group(1)}" if (m := _RE_Z.match(i)) else i) for i in proj.index]

    svm_table = _read_by_row(path / "svm_table.csv")
    svm_table.index.name = "Algorithm"

    sifted = pd.read_csv(_require(path / "sifted_report.csv"), dtype={"feature": str})
    for col in ("cluster", "n_algos_sig"):
        if col in sifted.columns:
            sifted[col] = sifted[col].astype("Int64")

    proba = _aligned(base, _read_by_row(path / "pythia_proba.csv"), "pythia_proba.csv")
    hats = [f"{a}_hat" for a in algos]
    if list(proba.columns) != algos + hats:
        raise ValueError(f"pythia_proba.csv: columns {list(proba.columns)}, expected {algos + hats}")
    proba_hat = proba[hats].rename(columns=dict(zip(hats, algos)))
    proba = proba[algos]

    conf = pd.read_csv(_require(path / "pythia_confusion.csv"), index_col="Algorithm",
                       dtype={"Algorithm": str})
    if list(conf.index) != algos or list(conf.columns) != ["tn", "fp", "fn", "tp"]:
        raise ValueError("pythia_confusion.csv: expected one row per algorithm and tn, fp, fn, tp")

    sel = _aligned(base, _read_by_row(path / "pythia_selection.csv", {"selection0": str, "selection1": str}),
                   "pythia_selection.csv")
    for col in ("selection0", "selection1"):
        sel[col] = _names_or_none(sel[col], algos, f"pythia_selection.csv:{col}")

    r2 = pd.read_csv(_require(path / "pilot_r2.csv"), dtype={"variable": str, "kind": str})
    correl = pd.read_csv(_require(path / "sifted_correlations.csv"),
                         dtype={"feature": str, "algorithm": str})
    silhouette = pd.read_csv(_require(path / "sifted_silhouette.csv"))
    for col in ("used", "best"):
        silhouette[col] = silhouette[col].astype(str).str.lower().map(_BOOL).astype(bool)

    coords_trace = None
    if (path / "coordinates_trace.csv").is_file():
        coords_trace = _aligned(base, _read_by_row(path / "coordinates_trace.csv"),
                                "coordinates_trace.csv")[["z_1", "z_2"]]
    jitter = bool(run_info.get("trace_robustness", {}).get("jitter_applied", False))
    if jitter != (coords_trace is not None):
        raise ValueError("coordinates_trace.csv must exist if, and only if, "
                         "run_info.json records jitter_applied")

    special = run_info.get("special_footprints", {})
    fp_space = _special_footprint(path, "space", special.get("space", {}), pi)
    fp_hard = _special_footprint(path, "hard", special.get("hard", {}), pi)

    return IsResult(
        name=path.name,
        path=path,
        instances=instances,
        algos=algos,
        features=features,
        features_input=list(sifted["feature"]),
        features_all=[name for _, name in col_feats],
        annotations=annotations,
        annotation_renames=renames,
        annotation_origins=origins,
        source_column=source_column,
        footprints=footprints,
        footprint_space=fp_space,
        footprint_hard=fp_hard,
        footprint_performance=perf,
        projection_matrix=proj,
        bounds=_read_bounds(path / "bounds.csv"),
        bounds_pruned=_read_bounds(path / "bounds_prunned.csv"),
        coordinates_trace=coords_trace,
        sifted_report=sifted,
        sifted_correlations=correl,
        sifted_silhouette=silhouette,
        pilot_r2=r2,
        pythia_proba=proba,
        pythia_proba_hat=proba_hat,
        pythia_confusion=conf,
        pythia_selection=sel,
        svm_table=svm_table,
        run_info=run_info,
        run_options=run_options,
        pi=pi,
        degenerate_report=_read_optional(path / "degenerate_report.csv", {"feature": str}),
        feature_info=_read_optional(path / "feature_info.csv", {"feature": str}),
    )
