"""
In-memory data model for a parsed running activity.

Design notes:
- One Activity object per FIT file. Replaces the old df_activity / df_records /
  df_summary / df_activity_feat / df_records_feat split from the original repo.
- All numeric fields stored in SI internally (meters, m/s, seconds, bpm, spm, deg C).
  Unit conversion to imperial/metric happens client-side based on user preference.
- The record stream stays as a pandas DataFrame because (a) the FIT parser already
  produces one cheaply, (b) downsampling via stride slicing is trivial, and
  (c) NaN handling is consistent with the parsing layer.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd


# Columns expected in Activity.records (all SI units).
# Any subset may be present depending on the device that recorded the file.
RECORD_COLUMNS = [
    "t_s",                # elapsed seconds from activity start
    "distance_m",         # cumulative distance in meters
    "speed_mps",          # instantaneous speed, m/s
    "pace_s_per_mile",    # derived: seconds per mile at that instant
    "heart_rate_bpm",     # int bpm
    "cadence_spm",        # already doubled from FIT raw (steps/min, both feet)
    "altitude_m",         # meters
    "temperature_c",      # celsius
    "position_lat_deg",   # decimal degrees
    "position_long_deg",  # decimal degrees
]


@dataclass
class Activity:
    """One parsed running activity, summary + record stream."""

    # Identity
    activity_id: str          # SHA1(source_filename + start_time_iso)
    source_filename: str      # original .fit filename (no path)

    # Device / context
    manufacturer: Optional[str] = None
    product: Optional[str] = None
    sub_sport: Optional[str] = None

    # Time bounds
    start_time: Optional[datetime] = None     # UTC, tz-aware
    total_elapsed_time_s: Optional[float] = None
    total_timer_time_s: Optional[float] = None

    # Summary stats (SI)
    total_distance_m: Optional[float] = None
    avg_speed_mps: Optional[float] = None
    max_speed_mps: Optional[float] = None
    avg_heart_rate_bpm: Optional[int] = None
    max_heart_rate_bpm: Optional[int] = None
    avg_cadence_spm: Optional[float] = None   # doubled
    max_cadence_spm: Optional[int] = None     # doubled
    total_calories_kcal: Optional[int] = None
    elev_gain_m: Optional[float] = None
    elev_loss_m: Optional[float] = None

    # Record stream (per-sample timeseries)
    records: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=RECORD_COLUMNS))

    # Bookkeeping
    parse_warnings: list = field(default_factory=list)

    # ---- derived convenience ----

    @property
    def record_count(self) -> int:
        return len(self.records)

    @property
    def has_gps(self) -> bool:
        if self.records.empty:
            return False
        if "position_lat_deg" not in self.records.columns:
            return False
        # Wrap in bool() — pandas returns numpy.bool which fails `is True`
        # and can cause subtle JSON serialization issues
        return bool(self.records["position_lat_deg"].notna().any())

    # ---- serialization ----

    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Compact JSON-safe summary for activity-list endpoints.
        Excludes the record stream.
        """
        return {
            "activity_id": self.activity_id,
            "source_filename": self.source_filename,
            "manufacturer": self.manufacturer,
            "product": self.product,
            "sub_sport": self.sub_sport,
            "start_time": _iso_z(self.start_time),
            "total_elapsed_time_s": _f(self.total_elapsed_time_s),
            "total_timer_time_s": _f(self.total_timer_time_s),
            "total_distance_m": _f(self.total_distance_m),
            "avg_speed_mps": _f(self.avg_speed_mps),
            "max_speed_mps": _f(self.max_speed_mps),
            "avg_heart_rate_bpm": _i(self.avg_heart_rate_bpm),
            "max_heart_rate_bpm": _i(self.max_heart_rate_bpm),
            "avg_cadence_spm": _f(self.avg_cadence_spm),
            "max_cadence_spm": _i(self.max_cadence_spm),
            "total_calories_kcal": _i(self.total_calories_kcal),
            "elev_gain_m": _f(self.elev_gain_m),
            "elev_loss_m": _f(self.elev_loss_m),
            "record_count": self.record_count,
            "has_gps": self.has_gps,
        }

    def to_full_dict(self, max_points: int = 1000) -> Dict[str, Any]:
        """
        Full JSON-safe payload including the (optionally downsampled) record stream.
        Records are returned as parallel arrays (column-oriented), which Chart.js
        consumes more efficiently than row-oriented JSON.
        """
        out = self.to_summary_dict()
        out["records"] = _records_to_columns(self.records, max_points=max_points)
        return out


# helpers (kept private to this module so models.py stays self-contained)

def _iso_z(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.isoformat().replace("+00:00", "Z")


def _f(v: Any) -> Optional[float]:
    """JSON-safe float: None for NaN/None, plain float otherwise."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def _records_to_columns(df: pd.DataFrame, max_points: int) -> Dict[str, list]:
    """
    Convert records DataFrame to a dict of column arrays, applying uniform
    stride downsampling if needed. NaN -> null in the output.
    """
    if df.empty:
        return {col: [] for col in RECORD_COLUMNS if col in df.columns}

    sampled = df
    n = len(df)
    if max_points and max_points > 0 and n > max_points:
        # Ceiling division so result count is always <= max_points
        step = (n + max_points - 1) // max_points
        sampled = df.iloc[::step].reset_index(drop=True)

    out: Dict[str, list] = {}
    for col in RECORD_COLUMNS:
        if col not in sampled.columns:
            continue
        # Replace NaN with None; emit everything as float (JS handles display formatting)
        out[col] = [None if pd.isna(v) else float(v) for v in sampled[col].tolist()]
    return out
