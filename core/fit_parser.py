"""
FIT file -> Activity parser.

Distilled and refactored from the original `fit_to_feature.py` in the
Runners-Dashboard repo. Key changes:

- Produces a single Activity object instead of split DataFrames (df_activity,
  df_records, df_summary, df_activity_feat, df_records_feat).
- All disk I/O, CSV/Parquet/manifest scaffolding, and CLI machinery removed.
- Running-only filter is enforced here (per app requirements); non-running
  files raise RunningFilterError so the route layer can report them cleanly.
- Cadence doubling (FIT raw stores half-cadence) happens at parse time,
  not in a later feature stage.
- Elevation gain/loss are computed once at parse time so they're available
  on the summary dict without holding a separate analysis stage.
- Pace (s/mi) is precomputed on the record stream for chart use.
"""

import hashlib
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import pandas as pd
from fitparse import FitFile

from .models import RECORD_COLUMNS, Activity


METERS_PER_MILE = 1609.344


class FitParseError(Exception):
    """Raised when a FIT file cannot be parsed at all."""


class RunningFilterError(Exception):
    """Raised when a FIT file parses but isn't a running activity."""
    def __init__(self, message: str, sport: Optional[str] = None, sub_sport: Optional[str] = None):
        super().__init__(message)
        self.sport = sport
        self.sub_sport = sub_sport


# small primitives

def _to_iso_z(dt: Any) -> Optional[str]:
    """Normalize a datetime-ish to ISO-8601 with trailing Z."""
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    return None


def _semicircles_to_degrees(v: Any) -> Optional[float]:
    """FIT stores lat/long as int32 semicircles; convert to decimal degrees."""
    if v is None:
        return None
    try:
        return float(v) * (180.0 / 2**31)
    except (TypeError, ValueError):
        return None


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _norm_sport(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().lower()
    return s or None


def _msg_to_dict(msg) -> Dict[str, Any]:
    return {f.name: f.value for f in msg}


def _first_value(fit: FitFile, message_names: List[str], field_name: str) -> Any:
    """Walk multiple message types looking for the first non-None value of a field."""
    for mn in message_names:
        for msg in fit.get_messages(mn):
            d = _msg_to_dict(msg)
            if field_name in d and d[field_name] is not None:
                return d[field_name]
    return None


def _make_activity_id(source_filename: str, start_time_iso: Optional[str]) -> str:
    """Reproducible ID: SHA1 of filename + start time. Same file -> same ID."""
    s = f"{source_filename}|{start_time_iso or ''}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


# elevation gain / loss (computed once on the records DataFrame)

def _elev_gain_loss_m(alt: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    """Sum of positive / negative deltas, in meters. Returns (gain, loss)."""
    a = pd.to_numeric(alt, errors="coerce").dropna()
    if a.size < 2:
        return None, None
    d = a.diff()
    gain = float(d[d > 0].sum())
    loss = float((-d[d < 0]).sum())
    return gain, loss


# main entry point

def parse_fit(source: Union[bytes, BinaryIO], source_filename: str) -> Activity:
    """
    Parse a .fit file (bytes or open file-like) into an Activity.

    Raises:
        FitParseError: file unreadable / no session message / no records
        RunningFilterError: sport is not running
    """
    # fitparse accepts file-like; wrap raw bytes
    if isinstance(source, (bytes, bytearray)):
        source = BytesIO(source)

    try:
        fit = FitFile(source)
        fit.parse()
    except Exception as e:
        raise FitParseError(f"could not parse FIT: {e}") from e

    # The 'session' message has the rolled-up summary for the activity
    session_msg = next(fit.get_messages("session"), None)
    if session_msg is None:
        raise FitParseError("FIT file has no session message")

    session = _msg_to_dict(session_msg)
    sport = _norm_sport(session.get("sport"))
    sub_sport = _norm_sport(session.get("sub_sport"))

    if sport != "running":
        raise RunningFilterError(
            f"activity is sport={sport!r}, not running",
            sport=sport,
            sub_sport=sub_sport,
        )

    # Identity / device
    start_dt = session.get("start_time")
    start_time_iso = _to_iso_z(start_dt)
    activity_id = _make_activity_id(source_filename, start_time_iso)

    manufacturer = _first_value(fit, ["file_id", "device_info"], "manufacturer")
    product = _first_value(fit, ["file_id", "device_info"], "product")

    # ----- record stream -----
    records_df = _build_records_df(fit, start_dt)
    if records_df.empty:
        raise FitParseError("FIT file has no timestamped record messages")

    # Elevation gain/loss from records (more reliable than session totals,
    # which Garmin sometimes rounds heavily on watch-side)
    gain_m, loss_m = _elev_gain_loss_m(records_df["altitude_m"])

    # Convert datetime fields
    start_time_dt: Optional[datetime] = None
    if isinstance(start_dt, datetime):
        if start_dt.tzinfo is None:
            start_time_dt = start_dt.replace(tzinfo=timezone.utc)
        else:
            start_time_dt = start_dt.astimezone(timezone.utc)

    # Cadence doubling: FIT stores running_cadence as "single-foot strides/min",
    # convention is to report total steps/min (both feet).
    avg_cad_raw = _safe_float(session.get("avg_running_cadence"))
    max_cad_raw = _safe_int(session.get("max_running_cadence"))
    avg_cad_spm = avg_cad_raw * 2.0 if avg_cad_raw is not None else None
    max_cad_spm = max_cad_raw * 2 if max_cad_raw is not None else None

    activity = Activity(
        activity_id=activity_id,
        source_filename=source_filename,
        manufacturer=str(manufacturer) if manufacturer is not None else None,
        product=str(product) if product is not None else None,
        sub_sport=sub_sport,
        start_time=start_time_dt,
        total_elapsed_time_s=_safe_float(session.get("total_elapsed_time")),
        total_timer_time_s=_safe_float(session.get("total_timer_time")),
        total_distance_m=_safe_float(session.get("total_distance")),
        avg_speed_mps=_safe_float(session.get("avg_speed")),
        max_speed_mps=_safe_float(session.get("max_speed")),
        avg_heart_rate_bpm=_safe_int(session.get("avg_heart_rate")),
        max_heart_rate_bpm=_safe_int(session.get("max_heart_rate")),
        avg_cadence_spm=avg_cad_spm,
        max_cadence_spm=max_cad_spm,
        total_calories_kcal=_safe_int(session.get("total_calories")),
        elev_gain_m=gain_m,
        elev_loss_m=loss_m,
        records=records_df,
    )
    return activity


def _build_records_df(fit: FitFile, start_dt: Any) -> pd.DataFrame:
    """
    Walk record messages and assemble a clean per-sample DataFrame.

    Derived columns:
        t_s: elapsed seconds from start (float)
        cadence_spm: doubled from FIT raw cadence (steps/min, both feet)
        pace_s_per_mile: derived from instantaneous speed
        altitude_m: prefers enhanced_altitude when present
        speed_mps: prefers enhanced_speed when present
    """
    rows: List[Dict[str, Any]] = []

    # Normalize start to UTC datetime for elapsed calculation
    if isinstance(start_dt, datetime):
        if start_dt.tzinfo is None:
            start_utc = start_dt.replace(tzinfo=timezone.utc)
        else:
            start_utc = start_dt.astimezone(timezone.utc)
    else:
        start_utc = None

    for msg in fit.get_messages("record"):
        d = _msg_to_dict(msg)
        ts = d.get("timestamp")
        if not isinstance(ts, datetime):
            continue

        ts_utc = ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)
        t_s = (ts_utc - start_utc).total_seconds() if start_utc is not None else None

        # Prefer enhanced_* variants when present
        speed_mps = _safe_float(d.get("enhanced_speed"))
        if speed_mps is None:
            speed_mps = _safe_float(d.get("speed"))

        altitude_m = _safe_float(d.get("enhanced_altitude"))
        if altitude_m is None:
            altitude_m = _safe_float(d.get("altitude"))

        # Cadence doubling at record level
        cadence_raw = _safe_int(d.get("cadence"))
        cadence_spm = cadence_raw * 2 if cadence_raw is not None else None

        # Pace from speed
        pace_s_per_mile: Optional[float] = None
        if speed_mps is not None and speed_mps > 0:
            pace_s_per_mile = METERS_PER_MILE / speed_mps

        rows.append({
            "t_s": t_s,
            "distance_m": _safe_float(d.get("distance")),
            "speed_mps": speed_mps,
            "pace_s_per_mile": pace_s_per_mile,
            "heart_rate_bpm": _safe_int(d.get("heart_rate")),
            "cadence_spm": cadence_spm,
            "altitude_m": altitude_m,
            "temperature_c": _safe_float(d.get("temperature")),
            "position_lat_deg": _semicircles_to_degrees(d.get("position_lat")),
            "position_long_deg": _semicircles_to_degrees(d.get("position_long")),
        })

    df = pd.DataFrame(rows, columns=RECORD_COLUMNS)
    return df
