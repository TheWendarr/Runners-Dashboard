"""
GraphBundle -> SVG (Matplotlib renderer)

Purpose
    Renders GraphSpecs from a GraphBundle (output of feature_to_graph.py)
    into static SVG plots

Inputs (on disk)
    - Path to graph_bundle.json OR directory containing graph_bundle.json.

Outputs (on disk)
    - SVG plots + render_manifest.json

Activity plots (4 SVGs per activity)
    - <activity_id>__heart_rate.svg   y: 60..200 bpm
    - <activity_id>__cadence.svg      y: 120..200 spm
    - <activity_id>__pace.svg         y: formatted MM:SS (pace)
    - <activity_id>__elevation.svg    y: autoscale (ft)

Summary plots
    - daily_hr.svg, weekly_hr.svg           
    - daily_cadence.svg, weekly_cadence.svg
    - daily_efficiency.svg, weekly_efficiency.svg

Usage
    python graph_to_svg.py --input <graph_bundle_dir_or_json> --out <plots_dir> [--force]
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt  # noqa: E402

VERSION = "1e_summary_fix"


# JSON helpers
def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


# Numeric helpers
def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (
        isinstance(x, float) and (math.isnan(x) or math.isinf(x))
    )


def _pairwise_filter(x_vals: Sequence[Any], y_vals: Sequence[Any]) -> Tuple[List[float], List[float]]:
    n = min(len(x_vals), len(y_vals))
    xs: List[float] = []
    ys: List[float] = []
    for i in range(n):
        x = x_vals[i]
        y = y_vals[i]
        if _is_number(x) and _is_number(y):
            xs.append(float(x))
            ys.append(float(y))
    return xs, ys


def _pairwise_filter_prepared(xs: Sequence[float], y_vals: Sequence[Any]) -> Tuple[List[float], List[float]]:
    n = min(len(xs), len(y_vals))
    out_x: List[float] = []
    out_y: List[float] = []
    for i in range(n):
        x = xs[i]
        y = y_vals[i]
        if _is_number(x) and _is_number(y):
            out_x.append(float(x))
            out_y.append(float(y))
    return out_x, out_y


def _downsample_stride(xs: List[float], ys: List[float], max_points: int) -> Tuple[List[float], List[float]]:
    if max_points <= 0 or len(xs) <= max_points:
        return xs, ys
    step = max(1, len(xs) // max_points)
    return xs[::step], ys[::step]


# Time axis formatting (elapsed)
def _format_elapsed_mmss(v: float) -> str:
    """
    Format elapsed seconds as M:SS (minutes can exceed 59).
    """
    if not _is_number(v) or v < 0:
        return ""
    total_s = int(round(v))
    m = total_s // 60
    s = total_s % 60
    return f"{m}:{s:02d}"


def _choose_elapsed_tick_step(max_seconds: float) -> int:
    """
    Choose a major tick step (seconds) based on total duration.
    """
    s = float(max_seconds or 0.0)
    if s <= 6 * 60:
        return 15
    if s <= 12 * 60:
        return 30
    if s <= 30 * 60:
        return 60
    if s <= 60 * 60:
        return 120
    if s <= 2 * 60 * 60:
        return 300
    if s <= 4 * 60 * 60:
        return 600
    return 900


def _apply_elapsed_time_axis(ax, max_seconds: float) -> None:
    """
    Apply M:SS formatting and adaptive tick spacing for elapsed-time x axes.
    """
    import matplotlib.ticker as mticker

    max_seconds = float(max_seconds or 0.0)
    if max_seconds > 0:
        ax.set_xlim(0.0, max_seconds)

    step = _choose_elapsed_tick_step(max_seconds)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(step))
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, pos: _format_elapsed_mmss(float(v)))
    )


# Summary X parsing (strings -> datetime/categorical)
def _try_parse_iso_datetime(s: str):
    if not s:
        return None
    ss = str(s).strip()
    if ss.endswith("Z"):
        ss = ss[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(ss)
    except Exception:
        return None


def _prepare_x_values(x_vals: Sequence[Any]):
    """
    Prepare X values for plotting.

    Returns:
      xs: list[float]
      mode: "numeric" | "datetime" | "categorical"
      labels: optional list[str] (only for categorical)
    """
    import matplotlib.dates as mdates

    n = len(x_vals)
    if n == 0:
        return [], "numeric", None

    # numeric
    all_num = True
    xs_num: List[float] = []
    for v in x_vals:
        if _is_number(v):
            xs_num.append(float(v))
        else:
            all_num = False
            break
    if all_num:
        return xs_num, "numeric", None

    # datetime
    parsed = []
    ok = 0
    for v in x_vals:
        dt = _try_parse_iso_datetime(v)
        if dt is None:
            parsed.append(None)
        else:
            parsed.append(dt)
            ok += 1

    if ok >= max(3, int(0.8 * n)):
        last = None
        xs_dt = []
        for dt in parsed:
            if dt is None:
                dt = last
            else:
                last = dt
            xs_dt.append(dt)

        xs = []
        for dt in xs_dt:
            if dt is None:
                xs.append(float("nan"))
            else:
                xs.append(mdates.date2num(dt))
        return xs, "datetime", None

    # categorical
    xs = [float(i) for i in range(n)]
    labels = ["" if v is None else str(v) for v in x_vals]
    return xs, "categorical", labels


# Axis formatters (pace, ints)
def _format_seconds_per_mile(s: float) -> str:
    if s <= 0 or not _is_number(s):
        return ""
    m = int(s // 60)
    sec = int(round(s - 60 * m))
    if sec >= 60:
        m += 1
        sec -= 60
    return f"{m}:{sec:02d}"


def _apply_pace_axis_format(ax) -> None:
    import matplotlib.ticker as mticker

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, pos: _format_seconds_per_mile(float(v)))
    )


def _apply_integer_ticks(ax, step: int = 10) -> None:
    import matplotlib.ticker as mticker

    ax.yaxis.set_major_locator(mticker.MultipleLocator(step))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))


# Series selection for activity plots
def _series_key(series: Dict[str, Any]) -> str:
    name = (series.get("name") or "").lower().strip()
    src = (series.get("source_col") or "").lower().strip()
    return src or name


def _select_series(gs: Dict[str, Any], want: Sequence[str]) -> Optional[Dict[str, Any]]:
    series_list = gs.get("series") or []
    want_set = {w.lower().strip() for w in want}

    for s in series_list:
        if _series_key(s) in want_set:
            return s

    for s in series_list:
        k = _series_key(s)
        for w in want_set:
            if w and w in k:
                return s
    return None


# Rendering primitives
def _render_single_series(*, gs: Dict[str, Any], series: Dict[str, Any], ax, max_points: int = 0) -> None:
    x_raw = (gs.get("x") or {}).get("values") or []
    x_label = (gs.get("x") or {}).get("label") or "time"
    x_unit = (gs.get("x") or {}).get("unit") or ""

    y = series.get("values") or []
    name = series.get("name") or "series"
    unit = series.get("unit") or ""

    xs, ys = _pairwise_filter(x_raw, y)
    if max_points and len(xs) > max_points:
        xs, ys = _downsample_stride(xs, ys, max_points=max_points)

    if xs:
        ax.plot(xs, ys, linewidth=1.25)

    ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
    ax.set_xlabel(f"{x_label}{(' (' + x_unit + ')') if x_unit else ''}")
    ax.set_ylabel(f"{name}{(' (' + unit + ')') if unit else ''}")

    # If elapsed seconds axis, format as M:SS and scale to activity duration
    xu = (x_unit or "").lower()
    xl = (x_label or "").lower()
    if xu in {"s", "sec", "secs", "second", "seconds"} or "elapsed" in xl or "t_s" in xl:
        max_x = max(xs) if xs else 0.0
        _apply_elapsed_time_axis(ax, max_x)


def _render_graphspec_multi(gs: Dict[str, Any], ax, max_points: int = 0) -> None:
    x_raw = (gs.get("x") or {}).get("values") or []
    x_label = (gs.get("x") or {}).get("label") or "x"
    x_unit = (gs.get("x") or {}).get("unit") or ""

    x, x_mode, x_labels = _prepare_x_values(x_raw)

    for s in (gs.get("series") or []):
        y = s.get("values") or []
        name = s.get("name") or "series"
        unit = s.get("unit") or ""

        xs, ys = _pairwise_filter_prepared(x, y)
        if max_points and len(xs) > max_points:
            xs, ys = _downsample_stride(xs, ys, max_points=max_points)
        if not xs:
            continue

        # Use markers for summary plots so individual runs are visible
        ax.plot(xs, ys, linewidth=1.25, marker="o", markersize=3, label=name)

        # If any plotted series is pace in s/mi, format y-axis as M:SS
        if "pace" in name.lower() or "s/mi" in (unit or "").lower():
            _apply_pace_axis_format(ax)

    ax.grid(True, which="major", linewidth=0.6, alpha=0.35)

    # Only apply elapsed axis formatting if x really is elapsed seconds (numeric)
    xu = (x_unit or "").lower()
    xl = (x_label or "").lower()
    if (xu in {"s", "sec", "secs", "second", "seconds"} or "elapsed" in xl or "t_s" in xl) and x_mode == "numeric":
        max_x = 0.0
        for s in (gs.get("series") or []):
            xs2, _ys2 = _pairwise_filter_prepared(x, s.get("values") or [])
            if xs2:
                max_x = max(max_x, max(xs2))
        _apply_elapsed_time_axis(ax, max_x)

    if x_mode == "categorical" and x_labels is not None:
        n = len(x_labels)
        if n > 0:
            step = max(1, n // 10)
            ticks = list(range(0, n, step))
            ax.set_xticks([float(t) for t in ticks])
            ax.set_xticklabels([x_labels[t] for t in ticks], rotation=30, ha="right", fontsize=8)
    elif x_mode == "datetime":
        import matplotlib.dates as mdates

        locator = mdates.AutoDateLocator(minticks=3, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")

    # Add legend if there are multiple series (e.g. Mean + Max)
    if len(gs.get("series") or []) > 1:
        ax.legend(loc="best", fontsize=8, frameon=False)


# Activity rendering (4 plots)
def render_activity_svgs(
    activity_id: str,
    gs: Dict[str, Any],
    out_dir: Path,
    *,
    max_points: int = 0,
    width_in: float = 12.0,
    height_in: float = 6.0,
) -> Dict[str, str]:
    title_base = gs.get("title") or f"Activity {activity_id}"
    outputs: Dict[str, str] = {}

    def _save(fig, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, format="svg")
        plt.close(fig)

    # Heart rate
    hr_series = _select_series(gs, want=["heart_rate_bpm", "heart_rate", "hr", "bpm"])
    if hr_series is not None:
        fig = plt.figure(figsize=(width_in, height_in), dpi=96)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(f"{title_base} — Heart Rate")
        _render_single_series(gs=gs, series=hr_series, ax=ax, max_points=max_points)
        ax.set_ylim(60, 200)
        _apply_integer_ticks(ax, step=10)
        p = out_dir / f"{activity_id}__heart_rate.svg"
        _save(fig, p)
        outputs["heart_rate"] = p.as_posix()

    # Cadence
    cad_series = _select_series(gs, want=["cadence_spm", "cadence", "spm"])
    if cad_series is not None:
        fig = plt.figure(figsize=(width_in, height_in), dpi=96)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(f"{title_base} — Cadence")
        _render_single_series(gs=gs, series=cad_series, ax=ax, max_points=max_points)
        ax.set_ylim(120, 200)
        _apply_integer_ticks(ax, step=10)
        p = out_dir / f"{activity_id}__cadence.svg"
        _save(fig, p)
        outputs["cadence"] = p.as_posix()

    # Pace
    pace_series = _select_series(gs, want=["pace_s_per_mile", "pace", "s_per_mile", "sec_per_mile"])
    if pace_series is not None:
        fig = plt.figure(figsize=(width_in, height_in), dpi=96)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(f"{title_base} — Pace")
        _render_single_series(gs=gs, series=pace_series, ax=ax, max_points=max_points)
        _apply_pace_axis_format(ax)
        p = out_dir / f"{activity_id}__pace.svg"
        _save(fig, p)
        outputs["pace"] = p.as_posix()

    # Elevation
    elev_series = _select_series(gs, want=["elevation_ft", "elevation", "altitude_ft", "altitude"])
    if elev_series is not None:
        fig = plt.figure(figsize=(width_in, height_in), dpi=96)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(f"{title_base} — Elevation")
        _render_single_series(gs=gs, series=elev_series, ax=ax, max_points=max_points)
        p = out_dir / f"{activity_id}__elevation.svg"
        _save(fig, p)
        outputs["elevation"] = p.as_posix()

    return outputs


def render_summary_svg(
    graph_id: str,
    gs: Dict[str, Any],
    out_path: Path,
    *,
    max_points: int = 0,
    width_in: float = 12.0,
    height_in: float = 6.5,
) -> None:
    title = gs.get("title") or graph_id
    fig = plt.figure(figsize=(width_in, height_in), dpi=96)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)

    # Generic render first
    _render_graphspec_multi(gs, ax=ax, max_points=max_points)

    #  SUMMARY SPECIFIC STYLING OVERRIDES
    gid_lower = graph_id.lower()

    # 1. Heart Rate Summaries
    if "hr" in gid_lower or "heart" in gid_lower:
        ax.set_ylim(60, 200)
        _apply_integer_ticks(ax, step=10)
        ax.set_ylabel("Heart Rate (bpm)")

    # 2. Cadence Summaries
    elif "cadence" in gid_lower:
        ax.set_ylim(120, 200)
        _apply_integer_ticks(ax, step=10)
        ax.set_ylabel("Cadence (spm)")

    # 3. Efficiency Summaries
    elif "efficiency" in gid_lower:
        # Autoscale is usually fine, but ensure grid is visible
        ax.set_ylabel("Efficiency Factor")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def run_render_stage(
    graph_bundle: Dict[str, Any],
    out_dir: Optional[Path] = None,
    *,
    max_points: int = 0,
    width_in: float = 12.0,
    height_in: float = 6.0,
    activity_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    activity_graphs: Dict[str, Any] = graph_bundle.get("activity_graphs") or {}
    summary_graphs: Dict[str, Any] = graph_bundle.get("summary_graphs") or {}

    if activity_ids is not None:
        keep = set(activity_ids)
        activity_graphs = {k: v for k, v in activity_graphs.items() if k in keep}

    manifest: Dict[str, Any] = {
        "activity_count": len(activity_graphs),
        "summary_count": len(summary_graphs),
        "activity_outputs": {},  # activity_id -> {plot_name: path}
        "summary_outputs": {},  # graph_id -> path
        "params": {"version": VERSION, "max_points": max_points, "width_in": width_in, "height_in": height_in},
    }

    if out_dir is None:
        return manifest

    out_dir = Path(out_dir)
    act_dir = out_dir / "activity"
    sum_dir = out_dir / "summary"

    for activity_id, gs in sorted(activity_graphs.items(), key=lambda kv: kv[0]):
        outputs = render_activity_svgs(
            activity_id=activity_id,
            gs=gs,
            out_dir=act_dir,
            max_points=max_points,
            width_in=width_in,
            height_in=height_in,
        )
        manifest["activity_outputs"][activity_id] = outputs

    for graph_id, gs in sorted(summary_graphs.items(), key=lambda kv: kv[0]):
        out_path = sum_dir / f"{graph_id}.svg"
        render_summary_svg(
            graph_id=graph_id,
            gs=gs,
            out_path=out_path,
            max_points=max_points,
            width_in=width_in,
            height_in=max(height_in, 6.5),
        )
        manifest["summary_outputs"][graph_id] = str(out_path.as_posix())

    _write_json(out_dir / "render_manifest.json", manifest)
    return manifest


# CLI
def load_graph_bundle_from_path(input_path: Path) -> Dict[str, Any]:
    p = Path(input_path)
    if p.is_file() and p.name.lower() == "graph_bundle.json":
        return _read_json(p)
    if p.is_dir():
        gb = p / "graph_bundle.json"
        if gb.exists():
            return _read_json(gb)
    raise FileNotFoundError(
        f"Could not locate graph_bundle.json at: {p}. Provide a path to graph_bundle.json or its parent directory."
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage C renderer: GraphBundle -> SVG plots (matplotlib)")
    p.add_argument("--input", required=True, help="Path to graph_bundle.json OR directory containing it.")
    p.add_argument("--out", required=True, help="Output directory to write SVG plots + render_manifest.json")
    p.add_argument("--force", action="store_true", help="Allow writing into a non-empty output directory.")
    p.add_argument("--max-points", type=int, default=0, help="Optional downsampling cap (stride). 0 disables.")
    p.add_argument("--width-in", type=float, default=12.0, help="Figure width in inches.")
    p.add_argument("--height-in", type=float, default=6.0, help="Figure height in inches.")
    p.add_argument("--activity-ids", default="", help="Optional comma-separated activity_ids to render; default all.")
    return p.parse_args(argv)


def _ensure_empty_or_force(out_dir: Path, force: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if force:
        return
    if any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_dir}. Use --force to overwrite/add files.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    _ensure_empty_or_force(out_dir, force=bool(args.force))

    activity_ids: Optional[List[str]] = None
    if args.activity_ids.strip():
        activity_ids = [s.strip() for s in args.activity_ids.split(",") if s.strip()]

    graph_bundle = load_graph_bundle_from_path(input_path)

    run_render_stage(
        graph_bundle=graph_bundle,
        out_dir=out_dir,
        max_points=int(args.max_points),
        width_in=float(args.width_in),
        height_in=float(args.height_in),
        activity_ids=activity_ids,
    )

    print(f"[OK] graph_to_svg v{VERSION} wrote SVG plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())