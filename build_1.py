"""
Build 1: FIT -> Unified schema tables -> Per-activity summaries (+ feature tables)

Orchestrates the end-to-end FIT analysis pipeline to convert Garmin .fit files into
per-activity summary records.

This script wires together two stages:
1) fit_to_schema: discovers and parses .fit files into standardized schema DataFrames
    df_activity (1 row per activity)
    df_records  (many rows per activity; record stream)
    df_skipped  (non-running or parse failures)

2) schema_to_analysis: aggregates those schema tables into one summary row per activity
    df_summary (date/time, distance, pace/efficiency, HR stats,
                cadence stats, elevation-change metric)

3) Build-1 feature tables (lightweight, in this file)
    Add unit conversions + commonly used derived fields to support Build 2
    (graph JSON generation) without changing the upstream tools.

Outputs (in-memory)
    df_activity:       schema table (from fit_to_schema)
    df_records:        schema table (from fit_to_schema)
    df_skipped:        schema table (from fit_to_schema)
    df_summary:        analysis table (from schema_to_analysis)
    df_activity_feat:  schema + converted units/derived fields
    df_records_feat:   schema + converted units/derived fields

Optional Outputs (via --bundle-out)
    Writes the above tables plus a small build manifest JSON.

Notes
    Later builds (build_2, build_3, ...) consume these tables/artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

import fit_to_schema
from schema_to_analysis import FitDatasetAnalyzer


# Constants / simple converters
M_PER_MILE = 1609.344
FT_PER_M = 3.280839895
MPH_PER_MPS = 2.2369362920544


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_float(v) -> Optional[float]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return float(v)
    except Exception:
        return None


def _meters_to_miles(m: Optional[float]) -> Optional[float]:
    if m is None:
        return None
    try:
        return float(m) / M_PER_MILE
    except Exception:
        return None


def _meters_to_feet(m: Optional[float]) -> Optional[float]:
    if m is None:
        return None
    try:
        return float(m) * FT_PER_M
    except Exception:
        return None


def _mps_to_mph(mps: Optional[float]) -> Optional[float]:
    if mps is None:
        return None
    try:
        return float(mps) * MPH_PER_MPS
    except Exception:
        return None


def _pace_seconds_per_mile_from_mps(mps: Optional[float]) -> Optional[float]:
    """
    Convert speed (m/s) -> pace seconds per mile
    """
    if mps is None:
        return None
    try:
        s = float(mps)
    except Exception:
        return None
    if s <= 0:
        return None
    return M_PER_MILE / s


def _pick_best_series(df: pd.DataFrame, primary: str, fallback: str) -> pd.Series:
    """
    Prefer primary column when it exists and has any non-null values
    """
    if primary in df.columns and df[primary].notna().any():
        return df[primary]
    return df.get(fallback, pd.Series([pd.NA] * len(df), index=df.index))


# Build 1 bundle (in-memory)
@dataclass
class Build1Bundle:
    """
    Container for Build 1 outputs to support downstream build stages
    """

    df_activity: pd.DataFrame
    df_records: pd.DataFrame
    df_skipped: pd.DataFrame
    df_summary: pd.DataFrame
    df_activity_feat: pd.DataFrame
    df_records_feat: pd.DataFrame


# CLI
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parses CLI arguments to achieve configurable Build 1 execution.
    """
    p = argparse.ArgumentParser(
        description=(
            "Build 1 pipeline: .fit -> schema tables -> per-activity summaries "
            "+ feature tables"
        )
    )
    p.add_argument(
        "--input",
        help="Path to a .fit file or a directory containing .fit files.",
        required=False,
    )
    p.add_argument(
        "--csv-out",
        help=(
            "Optional directory to write schema CSVs "
            "(runs_activity.csv / runs_records.csv / skipped.csv)."
        ),
        required=False,
    )
    p.add_argument(
        "--summary-out",
        help="Optional path to write the summary CSV (analysis stage).",
        required=False,
    )
    p.add_argument(
        "--bundle-out",
        help=(
            "Optional directory to write the Build 1 bundle tables (schema + summary + feature tables) "
            "and a small manifest JSON."
        ),
        required=False,
    )
    p.add_argument(
        "--bundle-format",
        choices=["csv", "parquet", "both"],
        default="csv",
        help="File format for --bundle-out tables.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite outputs if they exist (applies to --csv-out, --summary-out, --bundle-out).",
    )
    return p.parse_args(argv)


def resolve_input_path(args: argparse.Namespace) -> Path:
    """
    Resolves input path to achieve consistent CLI + prompt behavior.
    """
    if args.input:
        return Path(args.input)
    raw_in = input("Enter full path to a .fit file OR a directory of .fit files: ").strip().strip('"')
    if not raw_in:
        raise SystemExit("[ERROR] No input path provided.")
    return Path(raw_in)


# Stage 1: Schema build
def build_schema_from_fit_paths(fit_paths: List[Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Builds schema DataFrames from a list of .fit paths
    """
    activity_rows = []
    record_rows = []
    skipped_rows = []

    print(f"Processing {len(fit_paths)} files...")

    for fp in fit_paths:
        extracted, skipped = fit_to_schema.extract_activity_and_records(fp)
        if skipped is not None:
            skipped_rows.append({c: skipped.get(c) for c in fit_to_schema.SKIPPED_COLUMNS})
        else:
            assert extracted is not None
            activity_rows.append(extracted.activity_row)
            record_rows.extend(extracted.record_rows)

    df_activity = pd.DataFrame(activity_rows, columns=fit_to_schema.ACTIVITY_COLUMNS)
    df_records = pd.DataFrame(record_rows, columns=fit_to_schema.RECORD_COLUMNS)
    df_skipped = pd.DataFrame(skipped_rows, columns=fit_to_schema.SKIPPED_COLUMNS)

    # Match fit_to_schema.fit_to_dataframes dtype normalization
    if not df_activity.empty and "start_time" in df_activity.columns:
        df_activity["start_time"] = pd.to_datetime(df_activity["start_time"], errors="coerce", utc=True)
    if not df_records.empty and "timestamp" in df_records.columns:
        df_records["timestamp"] = pd.to_datetime(df_records["timestamp"], errors="coerce", utc=True)

    return df_activity, df_records, df_skipped


def maybe_write_schema_csvs(
    csv_out: Optional[str],
    df_activity: pd.DataFrame,
    df_records: pd.DataFrame,
    df_skipped: pd.DataFrame,
    force: bool,
) -> None:
    """
    Optionally writes schema CSVs using fit_to_schema.write_csvs
    """
    if not csv_out:
        return
    fit_to_schema.write_csvs(Path(csv_out), df_activity, df_records, df_skipped, force=bool(force))


# Stage 2: Analysis build
def build_summary(df_activity: pd.DataFrame, df_records: pd.DataFrame) -> pd.DataFrame:
    """
    Builds per-activity summaries from schema tables
    """
    analyzer = FitDatasetAnalyzer(df_activity=df_activity, df_records=df_records)
    return analyzer.summarize_df()


def maybe_write_summary_csv(summary_out: Optional[str], df_summary: pd.DataFrame, force: bool) -> None:
    """
    Optionally writes per-activity summary CSV
    """
    if not summary_out:
        return
    out_path = Path(summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        raise FileExistsError(f"Output already exists: {out_path} (use --force)")
    df_summary.to_csv(out_path, index=False)


# Build 1 feature tables (derived + converted)
def build_feature_tables(
    df_activity: pd.DataFrame,
    df_records: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adds unit conversions + lightweight derived fields for downstream visualization.
    This does not change upstream tools; it simply produces convenient feature tables.
    """
    df_a = df_activity.copy()
    df_r = df_records.copy()

    # Activity feature table
    if "start_time" in df_a.columns:
        # Keep start_time as UTC datetime and add basic date/time strings for quick grouping
        st = pd.to_datetime(df_a["start_time"], errors="coerce", utc=True)
        df_a["date_yyyymmdd"] = st.dt.strftime("%Y%m%d")
        df_a["time_hhmmss"] = st.dt.strftime("%H:%M:%S")

    if "total_distance_m" in df_a.columns:
        df_a["total_distance_miles"] = df_a["total_distance_m"].apply(lambda x: _meters_to_miles(_safe_float(x)))

    if "avg_speed_mps" in df_a.columns:
        df_a["avg_speed_mph"] = df_a["avg_speed_mps"].apply(lambda x: _mps_to_mph(_safe_float(x)))
    if "max_speed_mps" in df_a.columns:
        df_a["max_speed_mph"] = df_a["max_speed_mps"].apply(lambda x: _mps_to_mph(_safe_float(x)))

    # Cadence: upstream schema uses "raw" cadence; convert to steps-per-minute if appropriate.
    # Many Garmin running cadence values are per-leg; doubling yields total steps/min.
    if "avg_cadence_raw" in df_a.columns:
        df_a["avg_cadence_spm"] = df_a["avg_cadence_raw"].apply(lambda x: None if _safe_float(x) is None else float(x) * 2.0)
    if "max_cadence_raw" in df_a.columns:
        df_a["max_cadence_spm"] = df_a["max_cadence_raw"].apply(lambda x: None if _safe_float(x) is None else float(x) * 2.0)

    # Record feature table
    # elapsed seconds per activity (t_s)
    if not df_r.empty and "activity_id" in df_r.columns and "timestamp" in df_r.columns:
        ts = pd.to_datetime(df_r["timestamp"], errors="coerce", utc=True)
        df_r["timestamp"] = ts
        t0 = ts.groupby(df_r["activity_id"]).transform("min")
        df_r["t_s"] = (ts - t0).dt.total_seconds()
    else:
        df_r["t_s"] = pd.NA

    if "distance_m" in df_r.columns:
        df_r["distance_miles"] = df_r["distance_m"].apply(lambda x: _meters_to_miles(_safe_float(x)))

    # Prefer enhanced_speed_mps when present
    speed_mps = _pick_best_series(df_r, "enhanced_speed_mps", "speed_mps")
    df_r["speed_mps_best"] = pd.to_numeric(speed_mps, errors="coerce")
    df_r["speed_mph"] = df_r["speed_mps_best"].apply(lambda x: _mps_to_mph(_safe_float(x)))
    df_r["pace_s_per_mile"] = df_r["speed_mps_best"].apply(lambda x: _pace_seconds_per_mile_from_mps(_safe_float(x)))

    # Cadence spm (from cadence_raw)
    if "cadence_raw" in df_r.columns:
        df_r["cadence_spm"] = pd.to_numeric(df_r["cadence_raw"], errors="coerce") * 2.0
    else:
        df_r["cadence_spm"] = pd.NA

    # Elevation: prefer enhanced_altitude_m when present
    alt_m = _pick_best_series(df_r, "enhanced_altitude_m", "altitude_m")
    df_r["altitude_m_best"] = pd.to_numeric(alt_m, errors="coerce")
    df_r["elevation_ft"] = df_r["altitude_m_best"].apply(lambda x: _meters_to_feet(_safe_float(x)))

    return df_a, df_r


# Bundle export
def _ensure_writable(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Output already exists: {path} (use --force)")


def _write_df_csv(df: pd.DataFrame, path: Path, force: bool) -> None:
    _ensure_writable(path, force)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_df_parquet(df: pd.DataFrame, path: Path, force: bool) -> None:
    _ensure_writable(path, force)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Parquet export requested but 'pyarrow' is not installed. "
            "Install it (pip install pyarrow) or use --bundle-format csv."
        ) from e
    df.to_parquet(path, index=False)


def write_bundle(bundle_out: str, bundle: Build1Bundle, fmt: str, force: bool) -> Path:
    """
    Writes bundle tables + a manifest JSON. Returns the output directory
    """
    out_dir = Path(bundle_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic filenames
    tables: Dict[str, pd.DataFrame] = {
        "df_activity": bundle.df_activity,
        "df_records": bundle.df_records,
        "df_skipped": bundle.df_skipped,
        "df_summary": bundle.df_summary,
        "df_activity_feat": bundle.df_activity_feat,
        "df_records_feat": bundle.df_records_feat,
    }

    written: Dict[str, Dict[str, str]] = {}

    def write_one(name: str, df: pd.DataFrame) -> None:
        written[name] = {}
        if fmt in ("csv", "both"):
            p = out_dir / f"{name}.csv"
            _write_df_csv(df, p, force)
            written[name]["csv"] = str(p)
        if fmt in ("parquet", "both"):
            p = out_dir / f"{name}.parquet"
            _write_df_parquet(df, p, force)
            written[name]["parquet"] = str(p)

    for name, df in tables.items():
        write_one(name, df)

    manifest = {
        "build": "build_1",
        "created_utc": _utc_now_iso(),
        "tables": {
            name: {
                "rows": int(0 if df is None else len(df)),
                "columns": [] if df is None else list(df.columns),
                "files": written.get(name, {}),
            }
            for name, df in tables.items()
        },
    }

    manifest_path = out_dir / "build_1_manifest.json"
    _ensure_writable(manifest_path, force)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


# Main
def run_build_1(input_path: Path) -> Build1Bundle:
    """
    Runs Build 1 end-to-end and returns all in-memory outputs
    """
    fit_paths = fit_to_schema.iter_fit_paths(input_path)
    if not fit_paths:
        raise FileNotFoundError(f"No .fit files found at: {input_path}")

    df_activity, df_records, df_skipped = build_schema_from_fit_paths(fit_paths)
    df_summary = build_summary(df_activity, df_records)
    df_activity_feat, df_records_feat = build_feature_tables(df_activity, df_records)

    return Build1Bundle(
        df_activity=df_activity,
        df_records=df_records,
        df_skipped=df_skipped,
        df_summary=df_summary,
        df_activity_feat=df_activity_feat,
        df_records_feat=df_records_feat,
    )


def main(argv: Optional[List[str]] = None) -> int:
    # CLI entrypoint for Build 1
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)

    try:
        input_path = resolve_input_path(args)
        fit_paths = fit_to_schema.iter_fit_paths(input_path)
        if not fit_paths:
            print(f"[ERROR] No .fit files found at: {input_path}")
            return 2

        # Stage 1: schema tables
        df_activity, df_records, df_skipped = build_schema_from_fit_paths(fit_paths)

        # Optional: write schema CSVs
        maybe_write_schema_csvs(args.csv_out, df_activity, df_records, df_skipped, force=bool(args.force))

        # Stage 2: summaries
        df_summary = build_summary(df_activity, df_records)
        maybe_write_summary_csv(args.summary_out, df_summary, force=bool(args.force))

        # Feature tables (converted units + derived fields)
        df_activity_feat, df_records_feat = build_feature_tables(df_activity, df_records)

        # Optional: write full bundle
        if args.bundle_out:
            bundle = Build1Bundle(
                df_activity=df_activity,
                df_records=df_records,
                df_skipped=df_skipped,
                df_summary=df_summary,
                df_activity_feat=df_activity_feat,
                df_records_feat=df_records_feat,
            )
            out_dir = write_bundle(args.bundle_out, bundle, fmt=args.bundle_format, force=bool(args.force))
        else:
            out_dir = None

        # Print quick stats and a preview
        print("[DONE]")
        print(f"  fit_files:   {len(fit_paths)}")
        print(f"  activities:  {len(df_activity)}")
        print(f"  records:     {len(df_records)}")
        print(f"  skipped:     {len(df_skipped)}")
        print(f"  summaries:   {len(df_summary)}")
        print(f"  activity_feat:     {len(df_activity_feat)}")
        print(f"  records_feat:      {len(df_records_feat)}")
        if out_dir is not None:
            print(f"  bundle_out:        {out_dir}")

        print("\nSummary preview:")
        print(df_summary.head(10).to_string(index=False))

        # Return code OK
        return 0

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    """
    Sample command to run:
        python .\build_1.py `
        --input "C:\TEST\ACTIVITIES" `
        --csv-out "C:\TEST\EXPORTS\schema" `
        --summary-out "C:\TEST\EXPORTS\runs_summary.csv" `
        --bundle-out "C:\TEST\EXPORTS\build_1_bundle" `
        --bundle-format csv `
        --force
    """
    raise SystemExit(main())
