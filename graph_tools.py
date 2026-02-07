"""
Tools: Build graph-ready (JSON-serializable) specs from Feature tables.

Graph Categories
    1) Activity (record-level; one activity_id at a time)
        Pace, Heart Rate, Cadence, Elevation vs elapsed time (t_s)

    2) Summary (activity-level; trends over time)
        Max/Mean Pace, Max/Mean HR, Max/Mean Cadence, Elevation Change vs day/week

Notes
    - The output "GraphSpec" objects are plain dicts to keep tooling simple and portable.
    - This module is intentionally renderer-agnostic: it does not create plots or webpages.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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
