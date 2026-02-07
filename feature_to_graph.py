"""
Inputs (in-memory)
    - df_records_feat (pandas.DataFrame)
        Required columns:
            activity_id (str)
            t_s (float/int)                 elapsed seconds from activity start
            pace_s_per_mile (float)         pace in seconds per mile
            heart_rate_bpm (float/int)      heart rate in bpm
            cadence_spm (float/int)         cadence in steps per minute
            elevation_ft (float/int)        elevation in feet (optional; included if present)

    - df_summary (pandas.DataFrame)
        One row per activity with activity-level metrics (mean/max pace, HR, cadence, elevation change)
        Used for daily/weekly trend charts.

Outputs (in-memory)
    - GraphBundle
        activity_graphs: activity_id -> GraphSpec (dict)
        summary_graphs:  graph_id -> GraphSpec (dict), includes "daily" and "weekly"

Optional disk outputs (debug / validation chokepoints only)
    This module can be executed as a CLI tool to load a FeatureBundle export directory
    (from fit_to_feature.py --bundle-out) and write GraphBundle JSON files to a user-
    specified output directory.

Usage
    python feature_to_graph.py --input <feature_bundle_dir> --out <graphs_dir> [--force]

Notes
    - The pipeline is in-memory by default; JSON exports are optional and intended only
      for human-readable inspection while downstream rendering/web stages are built.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
import pandas as pd


# Constants / defaults
HR_COL = "heart_rate_bpm"

# From feature tables
ACTIVITY_X_COL = "t_s"
ACTIVITY_SERIES_DEFAULT = {
    "pace_s_per_mile": {"label": "Pace", "unit": "s/mi"},
    HR_COL: {"label": "Heart Rate", "unit": "bpm"},
    "cadence_spm": {"label": "Cadence", "unit": "spm"},
    "elevation_ft": {"label": "Elevation", "unit": "ft"},
}

SUMMARY_DATE_COLS = ["start_time_utc", "start_time", "date_yyyymmdd"]
SUMMARY_SERIES_ALIASES = {
    # pace
    "pace_mean_s_per_mile": ["pace_mean_s_per_mile", "mean_pace_s_per_mile", "pace_mean"],
    "pace_max_s_per_mile": ["pace_max_s_per_mile", "max_pace_s_per_mile", "pace_max"],
    # hr
    "hr_mean_bpm": ["hr_mean_bpm", "mean_hr_bpm", "mean_hr"],
    "hr_max_bpm": ["hr_max_bpm", "max_hr_bpm", "max_hr"],
    # cadence
    "cadence_mean_spm": ["cadence_mean_spm", "mean_cadence_spm", "mean_cadence"],
    "cadence_max_spm": ["cadence_max_spm", "max_cadence_spm", "max_cadence"],
    # elevation change
    "elev_change_ft": ["elev_change_ft", "elevation_change_ft", "elevation_change", "elev_change"],
}


# Public data container
@dataclass(frozen=True)
class GraphBundle:
    """In-memory graph outputs (JSON-serializable) for visualization stages."""
    activity_graphs: Dict[str, Dict[str, Any]]   # activity_id -> GraphSpec
    summary_graphs: Dict[str, Dict[str, Any]]    # graph_id -> GraphSpec


# Small helpers
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_jsonable_scalar(x: Any) -> Any:
    """
    Convert scalars to JSON-safe equivalents (NaN -> None; Timestamp -> ISO string)
    """
    if x is None:
        return None
    if isinstance(x, (pd.Timestamp, datetime)):
        try:
            if isinstance(x, pd.Timestamp):
                if x.tzinfo is None:
                    x = x.tz_localize("UTC")
                x = x.to_pydatetime()
            if x.tzinfo is None:
                x = x.replace(tzinfo=timezone.utc)
            return x.isoformat().replace("+00:00", "Z")
        except Exception:
            return str(x)
    if isinstance(x, float) and (pd.isna(x) or math.isnan(x)):
        return None
    return x


def _series_to_list(s: pd.Series) -> List[Any]:
    return [safe_jsonable_scalar(v) for v in s.tolist()]


def require_columns(df: pd.DataFrame, cols: Sequence[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for {context}: {missing}")


def _pick_first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _coerce_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def _iso_week_key(dt: pd.Timestamp) -> str:
    """Return ISO week key like '2026-W05'."""
    iso = dt.isocalendar()
    return f"{int(iso.year):04d}-W{int(iso.week):02d}"


# GraphSpec builders
def make_graph_spec(
    graph_id: str,
    category: str,
    title: str,
    x_label: str,
    x_unit: str,
    x_values: List[Any],
    series: List[Dict[str, Any]],
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a JSON-serializable graph spec dict
    """
    return {
        "graph_id": graph_id,
        "category": category,
        "title": title,
        "x": {"label": x_label, "unit": x_unit, "values": x_values},
        "series": series,
        "meta": meta or {},
    }


# Activity graphs (record-level)
def make_activity_graph(
    df_records_feat: pd.DataFrame,
    activity_id: str,
    *,
    graph_id: str = "series",
    title: Optional[str] = None,
    x_col: str = ACTIVITY_X_COL,
    series_map: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Build a record-level GraphSpec for a single activity_id.
    Required columns (minimum)
        - activity_id
        - t_s
        - pace_s_per_mile
        - heart_rate_bpm
        - cadence_spm
        - elevation_ft
    Returns
        GraphSpec dict (JSON-serializable)
    """
    require_columns(df_records_feat, ["activity_id", x_col], context="activity graph")
    df_one = df_records_feat[df_records_feat["activity_id"] == activity_id].copy()
    if df_one.empty:
        raise ValueError(f"No records for activity_id={activity_id}")

    # Sort by elapsed time
    df_one[x_col] = pd.to_numeric(df_one[x_col], errors="coerce")
    df_one = df_one.sort_values(by=x_col, kind="mergesort")

    series_map = series_map or ACTIVITY_SERIES_DEFAULT
    # Only include series that actually exist in the DataFrame
    series_specs: List[Dict[str, Any]] = []
    for col, info in series_map.items():
        if col not in df_one.columns:
            continue
        vals = _series_to_list(_coerce_numeric_series(df_one, col))
        series_specs.append(
            {"name": info["label"], "unit": info["unit"], "values": vals, "source_col": col}
        )

    if title is None:
        title = f"Activity {activity_id}"

    x_vals = _series_to_list(_coerce_numeric_series(df_one, x_col))

    return make_graph_spec(
        graph_id=f"{activity_id}:{graph_id}",
        category="activity",
        title=title,
        x_label="Elapsed Time",
        x_unit="s",
        x_values=x_vals,
        series=series_specs,
        meta={
            "activity_id": activity_id,
            "created_utc": _utc_now_iso(),
            "x_col": x_col,
        },
    )


def make_activity_graphs(
    df_records_feat: pd.DataFrame,
    activity_ids: Sequence[str],
    *,
    graph_id: str = "series",
) -> Dict[str, Dict[str, Any]]:
    """
    Build activity graphs for many activity_ids. Returns mapping activity_id -> GraphSpec
    """
    out: Dict[str, Dict[str, Any]] = {}
    for aid in activity_ids:
        out[aid] = make_activity_graph(df_records_feat, aid, graph_id=graph_id)
    return out


# Summary graphs (activity-level)
def _get_summary_time_axis(df_summary: pd.DataFrame) -> Tuple[pd.Series, str, str]:
    """
    Return (datetime_series_utc, x_label, x_unit)
    """
    col = _pick_first_existing(df_summary, SUMMARY_DATE_COLS)
    if col is None:
        raise KeyError(f"df_summary missing any supported date columns: {SUMMARY_DATE_COLS}")

    if col in ("start_time_utc", "start_time"):
        dt = pd.to_datetime(df_summary[col], errors="coerce", utc=True)
        return dt, "Date", "iso"

    dt = pd.to_datetime(df_summary[col].astype(str), format="%Y%m%d", errors="coerce", utc=True)
    return dt, "Date", "iso"


def _resolve_summary_col(df_summary: pd.DataFrame, canonical: str) -> Optional[str]:
    candidates = SUMMARY_SERIES_ALIASES.get(canonical, [])
    return _pick_first_existing(df_summary, candidates)


def make_summary_daily_graph(
    df_summary: pd.DataFrame,
    *,
    graph_id: str = "summary_daily",
    title: str = "Summary Metrics (Daily)",
) -> Dict[str, Any]:
    """
    Build a daily (per-activity point) summary GraphSpec
    """
    dt, x_label, x_unit = _get_summary_time_axis(df_summary)
    tmp = df_summary.copy()
    tmp["_dt"] = dt
    tmp = tmp.sort_values("_dt", kind="mergesort")

    x_vals = [safe_jsonable_scalar(v) for v in tmp["_dt"].dt.strftime("%Y-%m-%d").tolist()]

    canonical_order = [
        "pace_mean_s_per_mile",
        "pace_max_s_per_mile",
        "hr_mean_bpm",
        "hr_max_bpm",
        "cadence_mean_spm",
        "cadence_max_spm",
        "elev_change_ft",
    ]

    series_specs: List[Dict[str, Any]] = []
    for canonical in canonical_order:
        col = _resolve_summary_col(tmp, canonical)
        if col is None:
            continue
        vals = _series_to_list(_coerce_numeric_series(tmp, col))

        if "pace_" in canonical:
            name = "Mean Pace" if "mean" in canonical else "Max Pace"
            unit = "s/mi"
        elif "hr_" in canonical:
            name = "Mean HR" if "mean" in canonical else "Max HR"
            unit = "bpm"
        elif "cadence_" in canonical:
            name = "Mean Cadence" if "mean" in canonical else "Max Cadence"
            unit = "spm"
        else:
            name = "Elevation Change"
            unit = "ft"

        series_specs.append({"name": name, "unit": unit, "values": vals, "source_col": col})

    return make_graph_spec(
        graph_id=graph_id,
        category="summary",
        title=title,
        x_label=x_label,
        x_unit=x_unit,
        x_values=x_vals,
        series=series_specs,
        meta={
            "created_utc": _utc_now_iso(),
            "time_grain": "daily",
        },
    )


def make_summary_weekly_graph(
    df_summary: pd.DataFrame,
    *,
    graph_id: str = "summary_weekly",
    title: str = "Summary Metrics (Weekly)",
    week_mode: str = "iso",
) -> Dict[str, Any]:
    """
    Build a weekly-aggregated summary GraphSpec
    """
    dt, _, _ = _get_summary_time_axis(df_summary)
    tmp = df_summary.copy()
    tmp["_dt"] = dt

    if week_mode != "iso":
        raise ValueError(f"Unsupported week_mode: {week_mode} (supported: 'iso')")

    tmp["_week"] = tmp["_dt"].apply(lambda x: _iso_week_key(x) if pd.notna(x) else None)
    tmp = tmp[tmp["_week"].notna()].copy()

    canonical_order = [
        "pace_mean_s_per_mile",
        "pace_max_s_per_mile",
        "hr_mean_bpm",
        "hr_max_bpm",
        "cadence_mean_spm",
        "cadence_max_spm",
        "elev_change_ft",
    ]

    resolved: List[Tuple[str, str]] = []
    for canonical in canonical_order:
        col = _resolve_summary_col(tmp, canonical)
        if col is not None:
            resolved.append((canonical, col))

    weeks = sorted(tmp["_week"].unique().tolist())
    if not resolved:
        return make_graph_spec(
            graph_id=graph_id,
            category="summary",
            title=title,
            x_label="ISO Week",
            x_unit="week",
            x_values=weeks,
            series=[],
            meta={"created_utc": _utc_now_iso(), "time_grain": "weekly", "week_mode": week_mode},
        )

    agg: Dict[str, Any] = {}
    for canonical, col in resolved:
        if "mean" in canonical:
            agg[col] = "mean"
        elif "max" in canonical:
            agg[col] = "max"
        else:
            agg[col] = "mean"

    grouped = tmp.groupby("_week", sort=True).agg(agg).reset_index()

    series_specs: List[Dict[str, Any]] = []
    for canonical, col in resolved:
        vals = _series_to_list(_coerce_numeric_series(grouped, col))

        if "pace_" in canonical:
            name = "Mean Pace" if "mean" in canonical else "Max Pace"
            unit = "s/mi"
        elif "hr_" in canonical:
            name = "Mean HR" if "mean" in canonical else "Max HR"
            unit = "bpm"
        elif "cadence_" in canonical:
            name = "Mean Cadence" if "mean" in canonical else "Max Cadence"
            unit = "spm"
        else:
            name = "Elevation Change"
            unit = "ft"

        series_specs.append({"name": name, "unit": unit, "values": vals, "source_col": col})

    return make_graph_spec(
        graph_id=graph_id,
        category="summary",
        title=title,
        x_label="ISO Week",
        x_unit="week",
        x_values=grouped["_week"].tolist(),
        series=series_specs,
        meta={
            "created_utc": _utc_now_iso(),
            "time_grain": "weekly",
            "week_mode": week_mode,
            "transforms": [{"type": "groupby_week", "mode": week_mode}],
        },
    )


# In-memory stage runner
def run_build_2(
    *,
    df_records_feat: pd.DataFrame,
    df_summary: pd.DataFrame,
    activity_ids: Optional[Sequence[str]] = None,
) -> GraphBundle:
    """
    Run the graph generation stage in-memory.
    Inputs
        df_records_feat: record-level feature table from earlier stage
        df_summary:      per-activity summary table from earlier stage
        activity_ids:    optional subset; if omitted, uses all activity_ids present in df_summary
    Output
        GraphBundle (activity_graphs + summary_graphs), JSON-serializable dicts.
    """
    require_columns(df_records_feat, ["activity_id", ACTIVITY_X_COL], context="df_records_feat")
    require_columns(df_records_feat, [HR_COL], context="df_records_feat")

    if activity_ids is None:
        require_columns(df_summary, ["activity_id"], context="df_summary")
        activity_ids = df_summary["activity_id"].dropna().astype(str).tolist()

    activity_graphs = make_activity_graphs(df_records_feat, activity_ids)

    summary_graphs = {
        "daily": make_summary_daily_graph(df_summary),
        "weekly": make_summary_weekly_graph(df_summary),
    }

    return GraphBundle(activity_graphs=activity_graphs, summary_graphs=summary_graphs)


# Optional disk I/O (debug / validation chokepoints)
def _read_manifest(in_dir: Path) -> Optional[dict]:
    p = in_dir / "feature_bundle_manifest.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _load_table_from_bundle_dir(in_dir: Path, table_name: str) -> pd.DataFrame:
    """
    Load a table from a FeatureBundle export directory.

    Resolution order:
        1) If a feature_bundle_manifest.json exists, prefer parquet if present, else csv.
        2) Otherwise, try <table_name>.parquet then <table_name>.csv.
    """
    manifest = _read_manifest(in_dir)
    candidates: List[Path] = []

    if manifest and "tables" in manifest and table_name in manifest["tables"]:
        files = (manifest["tables"][table_name] or {}).get("files", {}) or {}
        parquet_path = files.get("parquet")
        csv_path = files.get("csv")
        if parquet_path:
            candidates.append(Path(parquet_path))
        if csv_path:
            candidates.append(Path(csv_path))

    # Fallbacks if manifest missing / incomplete
    candidates.extend([
        in_dir / f"{table_name}.parquet",
        in_dir / f"{table_name}.csv",
    ])

    for p in candidates:
        if p.exists():
            if p.suffix.lower() == ".parquet":
                return pd.read_parquet(p)
            if p.suffix.lower() == ".csv":
                return pd.read_csv(p)
    raise FileNotFoundError(
        f"Could not find table '{table_name}' in bundle directory: {in_dir}"
    )


def read_feature_bundle_minimum(in_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the minimum required inputs to build a GraphBundle:
        - df_records_feat
        - df_summary
    """
    df_records_feat = _load_table_from_bundle_dir(in_dir, "df_records_feat")
    df_summary = _load_table_from_bundle_dir(in_dir, "df_summary")
    return df_records_feat, df_summary


def _ensure_writable(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Output exists: {path} (use --force to overwrite)")


def write_graph_bundle(out_dir: Path, bundle: GraphBundle, force: bool) -> None:
    """
    Write GraphBundle to JSON files (human-readable debug chokepoint).

    Layout:
        out_dir/
            graph_bundle_manifest.json
            graph_bundle.json
            activity_graphs/<activity_id>.json
            summary_graphs/daily.json
            summary_graphs/weekly.json
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    act_dir = out_dir / "activity_graphs"
    sum_dir = out_dir / "summary_graphs"
    act_dir.mkdir(parents=True, exist_ok=True)
    sum_dir.mkdir(parents=True, exist_ok=True)

    # Individual graphs
    written_activity: Dict[str, str] = {}
    for activity_id, spec in (bundle.activity_graphs or {}).items():
        p = act_dir / f"{activity_id}.json"
        _ensure_writable(p, force)
        p.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        written_activity[str(activity_id)] = str(p)

    written_summary: Dict[str, str] = {}
    for graph_id, spec in (bundle.summary_graphs or {}).items():
        p = sum_dir / f"{graph_id}.json"
        _ensure_writable(p, force)
        p.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        written_summary[str(graph_id)] = str(p)

    # Combined file (convenient for inspection / future site assembly)
    combined = {
        "created_utc": _utc_now_iso(),
        "activity_graphs": bundle.activity_graphs,
        "summary_graphs": bundle.summary_graphs,
    }
    combined_path = out_dir / "graph_bundle.json"
    _ensure_writable(combined_path, force)
    combined_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")

    manifest = {
        "created_utc": _utc_now_iso(),
        "counts": {
            "activity_graphs": int(len(bundle.activity_graphs or {})),
            "summary_graphs": int(len(bundle.summary_graphs or {})),
        },
        "files": {
            "combined": str(combined_path),
            "activity_graphs": written_activity,
            "summary_graphs": written_summary,
        },
    }
    manifest_path = out_dir / "graph_bundle_manifest.json"
    _ensure_writable(manifest_path, force)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# Public runner alias (stage naming)
def run_graph_stage(
    df_records_feat: pd.DataFrame,
    df_summary: pd.DataFrame,
    activity_ids: Optional[Sequence[str]] = None,
) -> GraphBundle:
    """
    Stage B runner (preferred name): FeatureBundle -> GraphBundle.
    """
    return run_build_2(df_records_feat=df_records_feat, df_summary=df_summary, activity_ids=activity_ids)


# CLI
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Feature -> Graph stage: FeatureBundle tables -> GraphBundle JSON specs"
    )
    p.add_argument(
        "--input",
        help="Path to a FeatureBundle export directory (from fit_to_feature.py --bundle-out).",
        required=False,
    )
    p.add_argument(
        "--out",
        help="Directory to write GraphBundle JSON outputs (debug chokepoint).",
        required=False,
    )
    p.add_argument(
        "--activity-ids",
        help="Optional comma-separated list of activity_id values to build (default: all).",
        required=False,
    )
    p.add_argument("--force", action="store_true", help="Overwrite outputs if they exist.")
    return p.parse_args(argv)


def _resolve_input(args: argparse.Namespace) -> Path:
    if args.input:
        return Path(args.input).expanduser().resolve()
    s = input("Enter full path to a FeatureBundle export directory: ").strip().strip('"')
    return Path(s).expanduser().resolve()


def _resolve_out(args: argparse.Namespace) -> Optional[Path]:
    if args.out:
        return Path(args.out).expanduser().resolve()
    return None


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        in_dir = _resolve_input(args)
        if not in_dir.exists():
            raise FileNotFoundError(f"Input path does not exist: {in_dir}")
        if not in_dir.is_dir():
            raise ValueError(f"--input must be a directory: {in_dir}")

        out_dir = _resolve_out(args)
        if out_dir is None:
            s = input("Enter output directory for GraphBundle JSON (or leave blank for no write): ").strip().strip('"')
            out_dir = Path(s).expanduser().resolve() if s else None

        activity_ids: Optional[List[str]] = None
        if args.activity_ids:
            activity_ids = [x.strip() for x in args.activity_ids.split(",") if x.strip()]

        df_records_feat, df_summary = read_feature_bundle_minimum(in_dir)
        graph_bundle = run_graph_stage(df_records_feat=df_records_feat, df_summary=df_summary, activity_ids=activity_ids)

        if out_dir is not None:
            write_graph_bundle(out_dir, graph_bundle, force=args.force)

        # Always print a small in-console confirmation for quick validation
        print(f"Built GraphBundle: {len(graph_bundle.activity_graphs)} activity graph(s), {len(graph_bundle.summary_graphs)} summary graph(s)")
        if out_dir is not None:
            print(f"Wrote JSON outputs to: {out_dir}")
        return 0

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
