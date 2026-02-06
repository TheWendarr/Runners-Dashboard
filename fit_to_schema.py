"""
FIT -> pandas DataFrames (Running activities only)
Version: v2 (direct translation; no derived metrics)

Outputs:
- df_activity: 1 row per running activity
- df_records:  many rows per activity (record stream)
- df_skipped:  non-running or parse failures

Lat/Long stored as degrees.

Dependencies:
    pip install fitparse pandas
"""

from __future__ import annotations

import argparse
import sys
import hashlib
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fitparse import FitFile

# UTILS
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
    Checks for datetime inputs, assigns/ converts timezone to UTC, then returns an ISO
    string using 'Z' for +00:00.
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
    Safely casts the value to float and applies the FIT semicircle-to-degree scale factor
    (180 / 2**31).
    """
    if v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    # Corrected conversion logic
    return x * (180.0 / 2**31)

def safe_int(v: Any) -> Optional[int]:
    """
    Safely converts values to integers to parse across inconsistent FIT fields.
    Returns None for missing values and wraps int() in a try/except.
    """
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None

def safe_float(v: Any) -> Optional[float]:
    """
    Converts values to floats to achieve robust numeric extraction from FIT fields.
    Returns None for missing values and wraps float() in a try/except.
    """
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None

def make_activity_id(source_file: Path, start_time_iso: Optional[str]) -> str:
    """
    Creates a reproducible activity identifier to achieve stable joins across outputs. Hashes
    a string composed of source filename and start time using SHA-1 to produce a reproducible ID.
    """
    # create reproducible unique ID for each record
    s = f"{source_file.name}|{start_time_iso or ''}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def get_message_dict(msg) -> Dict[str, Any]:
    """
    Converts a FIT message object to a dictionary to achieve simple field access by name.
    Iterates message fields and stores each field's name/value into a Python dict.
    """
    out: Dict[str, Any] = {}
    for f in msg:
        out[f.name] = f.value
    return out

def first_message(fit, name: str):
    """
    Returns the first message of a given type to achieve quick access to primary metadata.
    Iterates messages by name and returns the first one found, otherwise returns None.
    """
    for msg in fit.get_messages(name):
        return msg
    return None

def first_value(fit, message_names: List[str], field_name: str) -> Any:
    """
    Extracts the first available value for a field across message types to achieve resilient metadata lookup.
    Scans messages in priority order, converts each to a dict, and returns the first occurrence of the requested field.
    """
    for mn in message_names:
        for msg in fit.get_messages(mn):
            d = get_message_dict(msg)
            if field_name in d:
                return d[field_name]
    return None

def normalize_sport(value: Any) -> Optional[str]:
    """
    Normalizes sport/sub_sport values to achieve consistent comparisons and filtering.
    Stringifies the value, lowercases it, trims whitespace, and returns None for empty results.
    """
    if value is None:
        return None
    s = str(value).strip().lower()
    return s or None

# SCHEMA
ACTIVITY_COLUMNS = [
    "activity_id", "source_file", "manufacturer", "product", "serial_number",
    "sport", "sub_sport", "start_time", "total_elapsed_time_s", "total_timer_time_s",
    "total_distance_m", "avg_speed_mps", "max_speed_mps", "avg_heart_rate_bpm",
    "max_heart_rate_bpm", "avg_cadence_raw", "max_cadence_raw", "total_calories_kcal",
    "avg_temperature_c", "max_temperature_c", "record_count", "parse_warnings",
]

RECORD_COLUMNS = [
    "activity_id", "seq", "timestamp", "distance_m", "speed_mps", "heart_rate_bpm",
    "cadence_raw", "altitude_m", "temperature_c", "position_lat_deg",
    "position_long_deg", "enhanced_speed_mps", "enhanced_altitude_m", "fractional_cadence",
]

SKIPPED_COLUMNS = ["source_file", "reason", "sport", "sub_sport"]

# EXTRACTION
@dataclass
class ActivityExtract:
    """
    This dataclass bundles extracted activity metadata and record series to achieve a single return object.
    Stores one activity_row dict plus a list of per-record dict rows so the caller can build DataFrames efficiently.
    """
    activity_row: Dict[str, Any]
    record_rows: List[Dict[str, Any]]

def extract_activity_and_records(fit_path: Path) -> Tuple[Optional[ActivityExtract], Optional[Dict[str, Any]]]:
    """
    Parses one FIT file to achieve a normalized activity row and many record rows for running activities.
    Loads/parses the FIT, validates a session message, filters to running, extracts device/session fields,
    then iterates record messages to produce timestamped rows or returns a skipped reason.
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
    activity_row.update({
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
    })

    record_rows: List[Dict[str, Any]] = []
    seq = 0
    for rec_msg in fit.get_messages("record"):
        rec = get_message_dict(rec_msg)
        ts_iso = to_iso_z(rec.get("timestamp"))
        if ts_iso is None:
            continue

        row: Dict[str, Any] = {c: None for c in RECORD_COLUMNS}
        row.update({
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
        })
        record_rows.append(row)
        seq += 1

    activity_row["record_count"] = len(record_rows)
    if len(record_rows) == 0:
        return None, {"source_file": str(fit_path), "reason": "no_timestamped_record_messages", "sport": sport, "sub_sport": sub_sport}

    activity_row["parse_warnings"] = "; ".join(warnings) if warnings else None
    return ActivityExtract(activity_row=activity_row, record_rows=record_rows), None

def fit_to_dataframes(input_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Converts FIT files to DataFrames to achieve analysis-ready tabular outputs.
    Discovers FIT paths, extracts activity/record rows for each, accumulates skipped
    reasons, builds three DataFrames, and parses timestamp columns as UTC datetimes.
    """
    fit_paths = iter_fit_paths(input_path)
    if not fit_paths:
        raise FileNotFoundError(f"No .fit files found at: {input_path}")

    activity_rows = []
    record_rows = []
    skipped_rows = []

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

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parses CLI arguments to achieve configurable script behavior at runtime. Defines expected
    flags (input/out/csv/force) and returns an argparse.Namespace populated from argv.
    """
    p = argparse.ArgumentParser(description="Translate running .fit files to pandas DataFrames.")
    p.add_argument("--input", help="Path to a .fit file or a directory containing .fit files.")
    p.add_argument("--out", help="Output directory (only used with --csv).")
    p.add_argument("--csv", action="store_true", help="Also write CSVs (debug/verification).")
    p.add_argument("--force", action="store_true", help="Overwrite CSV outputs if they exist.")
    return p.parse_args(argv)

def write_csvs(out_dir: Path, df_activity: pd.DataFrame, df_records: pd.DataFrame, df_skipped: pd.DataFrame, force: bool) -> None:
    """
    Writes DataFrames to CSV to achieve inspectable, portable outputs for verification and downstream use.
    Ensures the output directory exists, checks overwrite rules, and writes activity/records/skipped tables
    to fixed filenames.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = out_dir / "runs_activity.csv"
    p2 = out_dir / "runs_records.csv"
    p3 = out_dir / "skipped.csv"

    for p in [p1, p2, p3]:
        if p.exists() and not force:
            raise FileExistsError(f"Output already exists: {p} (use --force)")

    df_activity.to_csv(p1, index=False)
    df_records.to_csv(p2, index=False)
    df_skipped.to_csv(p3, index=False)

def main(argv: Optional[List[str]] = None) -> int:
    """
    Orchestrates the CLI workflow to achieve a complete parse-and-report run.
    Reads args (or prompts for input), runs extraction into DataFrames,
    prints a summary, optionally writes CSV outputs, and returns a process exit code.
    """
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    if not args.input:
        raw_in = input("Enter full path to a .fit file OR a directory of .fit files: ").strip().strip('"')
        if not raw_in:
            print("[ERROR] No input path provided.")
            return 2
        input_path = Path(raw_in)
    else:
        input_path = Path(args.input)

    try:
        df_activity, df_records, df_skipped = fit_to_dataframes(input_path)
    except Exception as e:
        print(f"[ERROR] {e}")
        # Helpful traceback for debugging
        import traceback
        traceback.print_exc()
        return 2

    print("[DONE]")
    print(f"  activities: {len(df_activity)}")
    print(f"  records:    {len(df_records)}")
    print(f"  skipped:    {len(df_skipped)}")

    if args.csv:
        if not args.out:
            print("[ERROR] --out is required when using --csv")
            return 2
        try:
            write_csvs(Path(args.out), df_activity, df_records, df_skipped, force=bool(args.force))
            print(f"  csv_out:    {Path(args.out)}")
        except Exception as e:
            print(f"[ERROR] {e}")
            return 2

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
