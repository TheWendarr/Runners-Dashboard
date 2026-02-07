"""
Pipeline
    1) Parse .fit files into a unified schema:
        - df_activity: one row per running activity
        - df_records:  record stream (many rows per activity)
        - df_skipped:  non-running or parse failures

    2) Analyze schema tables into per-activity summaries:
        - df_summary: date/time, distance, pace/efficiency, HR stats,
                      cadence stats, elevation-change metric

    3) Produce feature tables for downstream visualization work:
        - df_activity_feat: schema + selected unit conversions / convenience fields
        - df_records_feat:  schema + derived fields used by graph builders
                            (t_s, pace_s_per_mile, cadence_spm, elevation_ft, etc.)

Outputs (in-memory)
    FeatureBundle:
        df_activity
        df_records
        df_skipped
        df_summary
        df_activity_feat
        df_records_feat

Optional Outputs (debug / validation chokepoints)
    --csv-out:    writes schema CSVs (runs_activity.csv / runs_records.csv / skipped.csv)
    --summary-out writes summary CSV
    --bundle-out  writes all tables plus a small manifest JSON

Notes
    - This module intentionally runs in-memory; disk outputs are optional and exist
      only to validate dataflow at chokepoints during development.
    - Downstream stages (e.g., graph_tools.py) should ingest FeatureBundle directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fitparse import FitFile


# Unified schema (activity / record / skipped)
ACTIVITY_COLUMNS = [
    "activity_id",
    "source_file",
    "manufacturer",
    "product",
    "serial_number",
    "sport",
    "sub_sport",
    "start_time",
    "total_elapsed_time_s",
    "total_timer_time_s",
    "total_distance_m",
    "avg_speed_mps",
    "max_speed_mps",
    "avg_heart_rate_bpm",
    "max_heart_rate_bpm",
    "avg_cadence_raw",
    "max_cadence_raw",
    "total_calories_kcal",
    "avg_temperature_c",
    "max_temperature_c",
    "record_count",
    "parse_warnings",
]

RECORD_COLUMNS = [
    "activity_id",
    "seq",
    "timestamp",
    "distance_m",
    "speed_mps",
    "heart_rate_bpm",
    "cadence_raw",
    "altitude_m",
    "temperature_c",
    "position_lat_deg",
    "position_long_deg",
    "enhanced_speed_mps",
    "enhanced_altitude_m",
    "fractional_cadence",
]

SKIPPED_COLUMNS = ["source_file", "reason", "sport", "sub_sport"]


@dataclass
class ActivityExtract:
    """
    Bundles extracted activity metadata and record series to achieve a single return object.
    """
    activity_row: Dict[str, Any]
    record_rows: List[Dict[str, Any]]


# Utilities: discovery + safe casts + timestamp/geo normalization
def iter_fit_paths(input_path: Path) -> List[Path]:
    """
    Discovers .fit files to achieve a stable list of inputs for parsing.
    Returns the file if the input is a single .fit, or recursively searches
    a directory for *.fit files and returns them sorted.
    """
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() == ".fit" else []
    if input_path.is_dir():
        return sorted([p for p in input_path.rglob("*.fit") if p.is_file()])
    return []


def to_iso_z(dt: Any) -> Optional[str]:
    """
    Normalizes timestamps to achieve a consistent UTC ISO-8601 representation.
    """
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    return None


def semicircles_to_degrees(v: Any) -> Optional[float]:
    """
    Converts FIT GPS semicircles to degrees to achieve usable latitude/longitude values.
    """
    if v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    return x * (180.0 / 2**31)


def safe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def normalize_sport(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower()
    return s or None


def make_activity_id(source_file: Path, start_time_iso: Optional[str]) -> str:
    """
    Creates a reproducible activity identifier to achieve stable joins across outputs.
    """
    s = f"{source_file.name}|{start_time_iso or ''}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def get_message_dict(msg) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for f in msg:
        out[f.name] = f.value
    return out


def first_message(fit: FitFile, name: str):
    for msg in fit.get_messages(name):
        return msg
    return None


def first_value(fit: FitFile, message_names: List[str], field_name: str) -> Any:
    for mn in message_names:
        for msg in fit.get_messages(mn):
            d = get_message_dict(msg)
            if field_name in d:
                return d[field_name]
    return None


# Stage 1: FIT -> unified schema tables
def extract_activity_and_records(fit_path: Path) -> Tuple[Optional[ActivityExtract], Optional[Dict[str, Any]]]:
    """
    Parses one FIT file to achieve a normalized activity row and many record rows for running activities.
    Returns (ActivityExtract, None) on success or (None, skipped_reason_dict) on skip/failure.
    """
    warnings: List[str] = []

    try:
        fit = FitFile(str(fit_path))
        fit.parse()
    except Exception as e:
        return None, {"source_file": str(fit_path), "reason": f"parse_error: {e}", "sport": None, "sub_sport": None}

    session_msg = first_message(fit, "session")
    if session_msg is None:
        return None, {"source_file": str(fit_path), "reason": "missing_session_message", "sport": None, "sub_sport": None}

    session = get_message_dict(session_msg)
    sport = normalize_sport(session.get("sport"))
    sub_sport = normalize_sport(session.get("sub_sport"))

    if sport != "running":
        return None, {"source_file": str(fit_path), "reason": "not_running", "sport": sport, "sub_sport": sub_sport}

    manufacturer = first_value(fit, ["file_id", "device_info"], "manufacturer")
    product = first_value(fit, ["file_id", "device_info"], "product")
    serial_number = first_value(fit, ["file_id", "device_info"], "serial_number")

    start_time_iso = to_iso_z(session.get("start_time"))
    activity_id = make_activity_id(fit_path, start_time_iso)

    activity_row: Dict[str, Any] = {c: None for c in ACTIVITY_COLUMNS}
    activity_row.update(
        {
            "activity_id": activity_id,
            "source_file": str(fit_path),
            "manufacturer": str(manufacturer) if manufacturer else None,
            "product": str(product) if product else None,
            "serial_number": str(serial_number) if serial_number else None,
            "sport": sport,
            "sub_sport": sub_sport,
            "start_time": start_time_iso,
            "total_elapsed_time_s": safe_float(session.get("total_elapsed_time")),
            "total_timer_time_s": safe_float(session.get("total_timer_time")),
            "total_distance_m": safe_float(session.get("total_distance")),
            "avg_speed_mps": safe_float(session.get("avg_speed")),
            "max_speed_mps": safe_float(session.get("max_speed")),
            "avg_heart_rate_bpm": safe_int(session.get("avg_heart_rate")),
            "max_heart_rate_bpm": safe_int(session.get("max_heart_rate")),
            "avg_cadence_raw": safe_int(session.get("avg_running_cadence")),
            "max_cadence_raw": safe_int(session.get("max_running_cadence")),
            "total_calories_kcal": safe_int(session.get("total_calories")),
            "avg_temperature_c": safe_float(session.get("avg_temperature")),
            "max_temperature_c": safe_float(session.get("max_temperature")),
        }
    )

    record_rows: List[Dict[str, Any]] = []
    seq = 0
    for rec_msg in fit.get_messages("record"):
        rec = get_message_dict(rec_msg)
        ts_iso = to_iso_z(rec.get("timestamp"))
        if ts_iso is None:
            continue

        row: Dict[str, Any] = {c: None for c in RECORD_COLUMNS}
        row.update(
            {
                "activity_id": activity_id,
                "seq": seq,
                "timestamp": ts_iso,
                "distance_m": safe_float(rec.get("distance")),
                "speed_mps": safe_float(rec.get("speed")),
                "heart_rate_bpm": safe_int(rec.get("heart_rate")),
                "cadence_raw": safe_int(rec.get("cadence")),
                "altitude_m": safe_float(rec.get("altitude")),
                "temperature_c": safe_float(rec.get("temperature")),
                "position_lat_deg": semicircles_to_degrees(rec.get("position_lat")),
                "position_long_deg": semicircles_to_degrees(rec.get("position_long")),
                "enhanced_speed_mps": safe_float(rec.get("enhanced_speed")),
                "enhanced_altitude_m": safe_float(rec.get("enhanced_altitude")),
                "fractional_cadence": safe_float(rec.get("fractional_cadence")),
            }
        )
        record_rows.append(row)
        seq += 1

    activity_row["record_count"] = len(record_rows)

    if len(record_rows) == 0:
        return None, {
            "source_file": str(fit_path),
            "reason": "no_timestamped_record_messages",
            "sport": sport,
            "sub_sport": sub_sport,
        }

    activity_row["parse_warnings"] = "; ".join(warnings) if warnings else None
    return ActivityExtract(activity_row=activity_row, record_rows=record_rows), None


def fit_to_dataframes(input_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Converts FIT files to DataFrames to achieve analysis-ready tabular outputs.
    """
    fit_paths = iter_fit_paths(input_path)
    if not fit_paths:
        raise FileNotFoundError(f"No .fit files found at: {input_path}")

    activity_rows: List[Dict[str, Any]] = []
    record_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []

    print(f"Processing {len(fit_paths)} files...")

    for fp in fit_paths:
        extracted, skipped = extract_activity_and_records(fp)
        if skipped is not None:
            skipped_rows.append({c: skipped.get(c) for c in SKIPPED_COLUMNS})
        else:
            assert extracted is not None
            activity_rows.append(extracted.activity_row)
            record_rows.extend(extracted.record_rows)

    df_activity = pd.DataFrame(activity_rows, columns=ACTIVITY_COLUMNS)
    df_records = pd.DataFrame(record_rows, columns=RECORD_COLUMNS)
    df_skipped = pd.DataFrame(skipped_rows, columns=SKIPPED_COLUMNS)

    if not df_activity.empty and "start_time" in df_activity.columns:
        df_activity["start_time"] = pd.to_datetime(df_activity["start_time"], errors="coerce", utc=True)
    if not df_records.empty and "timestamp" in df_records.columns:
        df_records["timestamp"] = pd.to_datetime(df_records["timestamp"], errors="coerce", utc=True)

    return df_activity, df_records, df_skipped


def write_schema_csvs(out_dir: Path, df_activity: pd.DataFrame, df_records: pd.DataFrame, df_skipped: pd.DataFrame, force: bool) -> None:
    """
    Writes schema tables to CSV to achieve inspectable outputs for verification.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = out_dir / "runs_activity.csv"
    p2 = out_dir / "runs_records.csv"
    p3 = out_dir / "skipped.csv"

    for p in (p1, p2, p3):
        if p.exists() and not force:
            raise FileExistsError(f"Output already exists: {p} (use --force)")

    df_activity.to_csv(p1, index=False)
    df_records.to_csv(p2, index=False)
    df_skipped.to_csv(p3, index=False)


# Stage 2: schema tables -> per-activity analysis
@dataclass(frozen=True)
class ActivitySummary:
    """
    Summarizes one activity to achieve a per-activity analysis record.
    """
    date_yyyymmdd: str
    time_hhmmss: str
    distance_miles: float
    pace_mmss: Optional[str]
    efficiency: Optional[float]
    mean_hr_bpm: Optional[float]
    max_hr_bpm: Optional[int]
    mean_cadence_spm: Optional[float]
    max_cadence_spm: Optional[int]
    elevation_change: Optional[float]


class FitDatasetAnalyzer:
    """
    Analyzes FIT-derived DataFrames to achieve per-activity summaries.
    """

    def __init__(self, df_activity: pd.DataFrame, df_records: pd.DataFrame):
        self.df_activity = df_activity.copy()
        self.df_records = df_records.copy()

        if "start_time" in self.df_activity.columns:
            self.df_activity["start_time"] = pd.to_datetime(self.df_activity["start_time"], errors="coerce", utc=True)
        if "timestamp" in self.df_records.columns:
            self.df_records["timestamp"] = pd.to_datetime(self.df_records["timestamp"], errors="coerce", utc=True)

    @staticmethod
    def _meters_to_miles(m: Optional[float]) -> Optional[float]:
        if m is None:
            return None
        try:
            return float(m) / 1609.344
        except Exception:
            return None

    @staticmethod
    def _meters_to_feet(m: Optional[float]) -> Optional[float]:
        if m is None:
            return None
        try:
            return float(m) * 3.280839895
        except Exception:
            return None

    @staticmethod
    def _format_mmss(total_seconds: Optional[float]) -> Optional[str]:
        if total_seconds is None:
            return None
        try:
            s = int(round(float(total_seconds)))
        except Exception:
            return None
        if s < 0:
            s = 0
        minutes = s // 60
        seconds = s % 60
        return f"{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _compute_pace_seconds_per_mile(time_seconds: Optional[float], distance_miles: float) -> Optional[float]:
        if time_seconds is None:
            return None
        if distance_miles <= 0:
            return None
        try:
            return float(time_seconds) / float(distance_miles)
        except Exception:
            return None

    @staticmethod
    def _compute_efficiency(pace_seconds_per_mile: Optional[float], mean_hr_bpm: Optional[float]) -> Optional[float]:
        """
        Efficiency = 0.8547 x Pace as Decimal Minutes x (Mean HR / 100)
        """
        if pace_seconds_per_mile is None or mean_hr_bpm is None:
            return None
        try:
            pace_decimal_minutes = float(pace_seconds_per_mile) / 60.0
            return 0.8547 * pace_decimal_minutes * (float(mean_hr_bpm) / 100.0)
        except Exception:
            return None

    @staticmethod
    def _pick_altitude_series(df: pd.DataFrame) -> pd.Series:
        if "enhanced_altitude_m" in df.columns and df["enhanced_altitude_m"].notna().any():
            return df["enhanced_altitude_m"]
        return df.get("altitude_m", pd.Series([pd.NA] * len(df), index=df.index))

    @staticmethod
    def _compute_up_down_m(alt_m: pd.Series) -> Tuple[Optional[float], Optional[float]]:
        if alt_m is None or alt_m.empty:
            return None, None
        alt = pd.to_numeric(alt_m, errors="coerce").dropna()
        if alt.size < 2:
            return None, None
        d = alt.diff()
        up = d[d > 0].sum()
        down = (-d[d < 0]).sum()
        return float(up), float(down)

    def summarize(self) -> List[ActivitySummary]:
        out: List[ActivitySummary] = []
        if self.df_activity.empty:
            return out

        act = self.df_activity.set_index("activity_id", drop=False)
        grouped = self.df_records.groupby("activity_id") if not self.df_records.empty else {}

        for activity_id, arow in act.iterrows():
            start_time = arow.get("start_time")
            if pd.isna(start_time):
                continue

            dt: datetime = pd.Timestamp(start_time).to_pydatetime()
            date_yyyymmdd = dt.strftime("%Y%m%d")
            time_hhmmss = dt.strftime("%H:%M:%S")

            # Distance (meters): prefer activity total; otherwise max record distance
            dist_m: Optional[float] = None
            if pd.notna(arow.get("total_distance_m")):
                dist_m = float(arow.get("total_distance_m"))
            else:
                if activity_id in grouped:
                    r = grouped.get_group(activity_id)
                    if "distance_m" in r.columns:
                        dmax = pd.to_numeric(r["distance_m"], errors="coerce").max()
                        if pd.notna(dmax):
                            dist_m = float(dmax)

            distance_miles = self._meters_to_miles(dist_m) or 0.0

            # Time seconds: prefer timer time; fallback to elapsed
            time_s: Optional[float] = None
            if pd.notna(arow.get("total_timer_time_s")):
                time_s = float(arow.get("total_timer_time_s"))
            elif pd.notna(arow.get("total_elapsed_time_s")):
                time_s = float(arow.get("total_elapsed_time_s"))

            pace_sec_per_mile = self._compute_pace_seconds_per_mile(time_s, distance_miles)
            pace_mmss = self._format_mmss(pace_sec_per_mile) if pace_sec_per_mile is not None else None

            r = grouped.get_group(activity_id) if activity_id in grouped else pd.DataFrame()

            # HR
            mean_hr: Optional[float] = None
            max_hr: Optional[int] = None
            if not r.empty and "heart_rate_bpm" in r.columns and r["heart_rate_bpm"].notna().any():
                hr = pd.to_numeric(r["heart_rate_bpm"], errors="coerce").dropna()
                if not hr.empty:
                    mean_hr = float(hr.mean())
                    max_hr = int(hr.max())
            else:
                if pd.notna(arow.get("avg_heart_rate_bpm")):
                    mean_hr = float(arow.get("avg_heart_rate_bpm"))
                if pd.notna(arow.get("max_heart_rate_bpm")):
                    max_hr = int(arow.get("max_heart_rate_bpm"))

            # Cadence raw -> spm (double)
            mean_cad_raw: Optional[float] = None
            max_cad_raw: Optional[int] = None
            if not r.empty and "cadence_raw" in r.columns and r["cadence_raw"].notna().any():
                cad = pd.to_numeric(r["cadence_raw"], errors="coerce").dropna()
                if not cad.empty:
                    mean_cad_raw = float(cad.mean())
                    max_cad_raw = int(cad.max())
            else:
                if pd.notna(arow.get("avg_cadence_raw")):
                    mean_cad_raw = float(arow.get("avg_cadence_raw"))
                if pd.notna(arow.get("max_cadence_raw")):
                    max_cad_raw = int(arow.get("max_cadence_raw"))

            efficiency = self._compute_efficiency(pace_sec_per_mile, mean_hr)

            elevation_change_ft: Optional[float] = None
            if not r.empty:
                alt_series = self._pick_altitude_series(r)
                up_m, down_m = self._compute_up_down_m(alt_series)
                if up_m is not None and down_m is not None:
                    elevation_change_ft = self._meters_to_feet(abs(up_m - down_m))

            out.append(
                ActivitySummary(
                    date_yyyymmdd=date_yyyymmdd,
                    time_hhmmss=time_hhmmss,
                    distance_miles=round(distance_miles, 3),
                    pace_mmss=pace_mmss,
                    efficiency=None if efficiency is None else round(efficiency, 3),
                    mean_hr_bpm=None if mean_hr is None else round(mean_hr, 1),
                    max_hr_bpm=max_hr,
                    mean_cadence_spm=None if mean_cad_raw is None else round(mean_cad_raw * 2.0, 1),
                    max_cadence_spm=None if max_cad_raw is None else int(max_cad_raw * 2),
                    elevation_change=None if elevation_change_ft is None else round(elevation_change_ft, 1),
                )
            )

        return out

    def summarize_df(self) -> pd.DataFrame:
        rows = [s.__dict__ for s in self.summarize()]
        return pd.DataFrame(
            rows,
            columns=[
                "activity_id",  # added below if available
                "date_yyyymmdd",
                "time_hhmmss",
                "distance_miles",
                "pace_mmss",
                "efficiency",
                "mean_hr_bpm",
                "max_hr_bpm",
                "mean_cadence_spm",
                "max_cadence_spm",
                "elevation_change",
            ],
        )


def build_summary(df_activity: pd.DataFrame, df_records: pd.DataFrame) -> pd.DataFrame:
    """
    Builds per-activity summaries from schema tables.
    """
    analyzer = FitDatasetAnalyzer(df_activity=df_activity, df_records=df_records)
    df = analyzer.summarize_df()

    # Preserve activity_id in df_summary when possible to support joins/selection downstream.
    # schema_to_analysis.py historically returned no activity_id; this adds it deterministically.
    if "activity_id" not in df.columns:
        df.insert(0, "activity_id", pd.NA)

    # Attempt to map activity_id by aligning df_activity start_time -> date/time keys.
    # If this is too indirect for your preference, build_1 can pass activity_id through directly later.
    if not df_activity.empty and "activity_id" in df_activity.columns and "start_time" in df_activity.columns:
        a = df_activity[["activity_id", "start_time"]].copy()
        a["start_time"] = pd.to_datetime(a["start_time"], errors="coerce", utc=True)
        a["date_yyyymmdd"] = a["start_time"].dt.strftime("%Y%m%d")
        a["time_hhmmss"] = a["start_time"].dt.strftime("%H:%M:%S")
        key = a[["activity_id", "date_yyyymmdd", "time_hhmmss"]]
        df = df.drop(columns=["activity_id"], errors="ignore").merge(
            key, on=["date_yyyymmdd", "time_hhmmss"], how="left"
        )
        # Put activity_id first
        cols = ["activity_id"] + [c for c in df.columns if c != "activity_id"]
        df = df[cols]

    return df


# Stage 3: feature tables (converted / derived fields)
M_PER_MILE = 1609.344
FT_PER_M = 3.280839895
MPH_PER_MPS = 2.2369362920544


def _safe_float2(v: Any) -> Optional[float]:
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
    if primary in df.columns and df[primary].notna().any():
        return df[primary]
    return df.get(fallback, pd.Series([pd.NA] * len(df), index=df.index))


def build_feature_tables(df_activity: pd.DataFrame, df_records: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adds unit conversions + lightweight derived fields for downstream visualization.
    """
    df_a = df_activity.copy()
    df_r = df_records.copy()

    # Activity feature table
    if "start_time" in df_a.columns:
        st = pd.to_datetime(df_a["start_time"], errors="coerce", utc=True)
        df_a["date_yyyymmdd"] = st.dt.strftime("%Y%m%d")
        df_a["time_hhmmss"] = st.dt.strftime("%H:%M:%S")

    if "total_distance_m" in df_a.columns:
        df_a["total_distance_miles"] = df_a["total_distance_m"].apply(lambda x: _meters_to_miles(_safe_float2(x)))

    if "avg_speed_mps" in df_a.columns:
        df_a["avg_speed_mph"] = df_a["avg_speed_mps"].apply(lambda x: _mps_to_mph(_safe_float2(x)))
    if "max_speed_mps" in df_a.columns:
        df_a["max_speed_mph"] = df_a["max_speed_mps"].apply(lambda x: _mps_to_mph(_safe_float2(x)))

    if "avg_cadence_raw" in df_a.columns:
        df_a["avg_cadence_spm"] = df_a["avg_cadence_raw"].apply(lambda x: None if _safe_float2(x) is None else float(x) * 2.0)
    if "max_cadence_raw" in df_a.columns:
        df_a["max_cadence_spm"] = df_a["max_cadence_raw"].apply(lambda x: None if _safe_float2(x) is None else float(x) * 2.0)

    # Record feature table: elapsed seconds (t_s)
    if not df_r.empty and "activity_id" in df_r.columns and "timestamp" in df_r.columns:
        ts = pd.to_datetime(df_r["timestamp"], errors="coerce", utc=True)
        df_r["timestamp"] = ts
        t0 = ts.groupby(df_r["activity_id"]).transform("min")
        df_r["t_s"] = (ts - t0).dt.total_seconds()
    else:
        df_r["t_s"] = pd.NA

    if "distance_m" in df_r.columns:
        df_r["distance_miles"] = df_r["distance_m"].apply(lambda x: _meters_to_miles(_safe_float2(x)))

    speed_mps = _pick_best_series(df_r, "enhanced_speed_mps", "speed_mps")
    df_r["speed_mps_best"] = pd.to_numeric(speed_mps, errors="coerce")
    df_r["speed_mph"] = df_r["speed_mps_best"].apply(lambda x: _mps_to_mph(_safe_float2(x)))
    df_r["pace_s_per_mile"] = df_r["speed_mps_best"].apply(lambda x: _pace_seconds_per_mile_from_mps(_safe_float2(x)))

    if "cadence_raw" in df_r.columns:
        df_r["cadence_spm"] = pd.to_numeric(df_r["cadence_raw"], errors="coerce") * 2.0
    else:
        df_r["cadence_spm"] = pd.NA

    alt_m = _pick_best_series(df_r, "enhanced_altitude_m", "altitude_m")
    df_r["altitude_m_best"] = pd.to_numeric(alt_m, errors="coerce")
    df_r["elevation_ft"] = df_r["altitude_m_best"].apply(lambda x: _meters_to_feet(_safe_float2(x)))

    return df_a, df_r


# FeatureBundle + optional debug export
@dataclass(frozen=True)
class FeatureBundle:
    """
    Container for feature-stage outputs to support downstream visualization stages.
    """
    df_activity: pd.DataFrame
    df_records: pd.DataFrame
    df_skipped: pd.DataFrame
    df_summary: pd.DataFrame
    df_activity_feat: pd.DataFrame
    df_records_feat: pd.DataFrame


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def write_bundle(out_dir: Path, bundle: FeatureBundle, fmt: str, force: bool) -> None:
    """
    Writes all tables plus a small manifest JSON (debug / validation chokepoint).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    tables: Dict[str, pd.DataFrame] = {
        "df_activity": bundle.df_activity,
        "df_records": bundle.df_records,
        "df_skipped": bundle.df_skipped,
        "df_summary": bundle.df_summary,
        "df_activity_feat": bundle.df_activity_feat,
        "df_records_feat": bundle.df_records_feat,
    }

    written: Dict[str, Dict[str, str]] = {}

    for name, df in tables.items():
        written[name] = {}
        if fmt in ("csv", "both"):
            p = out_dir / f"{name}.csv"
            _write_df_csv(df, p, force)
            written[name]["csv"] = str(p)
        if fmt in ("parquet", "both"):
            p = out_dir / f"{name}.parquet"
            _write_df_parquet(df, p, force)
            written[name]["parquet"] = str(p)

    manifest = {
        "created_utc": _utc_now_iso(),
        "tables": {
            name: {
                "rows": int(len(df)),
                "columns": list(df.columns),
                "files": written.get(name, {}),
            }
            for name, df in tables.items()
        },
    }

    manifest_path = out_dir / "feature_bundle_manifest.json"
    _ensure_writable(manifest_path, force)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# Public runner + CLI
def run_feature_stage(input_path: Path) -> FeatureBundle:
    """
    Runs the consolidated feature stage end-to-end and returns FeatureBundle.
    """
    df_activity, df_records, df_skipped = fit_to_dataframes(input_path)
    df_summary = build_summary(df_activity, df_records)
    df_activity_feat, df_records_feat = build_feature_tables(df_activity, df_records)

    return FeatureBundle(
        df_activity=df_activity,
        df_records=df_records,
        df_skipped=df_skipped,
        df_summary=df_summary,
        df_activity_feat=df_activity_feat,
        df_records_feat=df_records_feat,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Consolidated feature stage: .fit -> schema -> summary -> feature tables"
    )
    p.add_argument("--input", help="Path to a .fit file or a directory containing .fit files.", required=False)
    p.add_argument("--csv-out", help="Optional directory to write schema CSVs (runs_activity/records/skipped).", required=False)
    p.add_argument("--summary-out", help="Optional path to write df_summary CSV.", required=False)
    p.add_argument("--bundle-out", help="Optional directory to write the full FeatureBundle tables + manifest.", required=False)
    p.add_argument("--bundle-format", choices=["csv", "parquet", "both"], default="csv", help="File format for --bundle-out tables.")
    p.add_argument("--force", action="store_true", help="Overwrite outputs if they exist.")
    return p.parse_args(argv)


def _resolve_input(args: argparse.Namespace) -> Path:
    if args.input:
        return Path(args.input)
    raw_in = input("Enter full path to a .fit file OR a directory of .fit files: ").strip().strip('"')
    if not raw_in:
        raise SystemExit("[ERROR] No input path provided.")
    return Path(raw_in)


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    try:
        input_path = _resolve_input(args)
        bundle = run_feature_stage(input_path)

        if args.csv_out:
            write_schema_csvs(Path(args.csv_out), bundle.df_activity, bundle.df_records, bundle.df_skipped, force=bool(args.force))

        if args.summary_out:
            out_path = Path(args.summary_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and not args.force:
                raise FileExistsError(f"Output already exists: {out_path} (use --force)")
            bundle.df_summary.to_csv(out_path, index=False)

        if args.bundle_out:
            write_bundle(Path(args.bundle_out), bundle, fmt=args.bundle_format, force=bool(args.force))

        print("[DONE]")
        print(f"  activities:    {len(bundle.df_activity)}")
        print(f"  records:       {len(bundle.df_records)}")
        print(f"  skipped:       {len(bundle.df_skipped)}")
        print(f"  summaries:     {len(bundle.df_summary)}")
        print(f"  activity_feat: {len(bundle.df_activity_feat)}")
        print(f"  records_feat:  {len(bundle.df_records_feat)}")
        print("\nSummary preview:")
        print(bundle.df_summary.head(10).to_string(index=False))
        return 0

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
