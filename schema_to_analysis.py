"""
Inputs:
- df_activity (pandas.DataFrame)
    One row per activity, produced by fit_to_schema.py
    Required columns:
        activity_id (hashable key used to join with df_records)
        start_time (ISO datetime; parsed as UTC when possible)
    Optional columns (used as fallbacks when record-level data is missing):
        total_distance_m
        total_timer_time_s, total_elapsed_time_s
        avg_heart_rate_bpm, max_heart_rate_bpm
        avg_cadence_raw, max_cadence_raw
- df_records (pandas.DataFrame)
    Many rows per activity (record messages), produced by fit_to_schema.py.
    Required columns:
        activity_id (same key as df_activity)
        timestamp (ISO datetime; parsed as UTC when possible)
    Optional columns (used for per-record computations):
        distance_m
        heart_rate_bpm
        cadence_raw
        enhanced_altitude_m and/or altitude_m

Outputs:
- List[ActivitySummary] via FitDatasetAnalyzer.summarize()
    Per-activity summary fields:
        date_yyyymmdd (YYYYMMDD)
        time_hhmmss (HH:MM:SS, UTC by default)
        distance_miles
        pace_mmss (MM:SS, minutes per mile)
        efficiency (0.8547 * pace_decimal_minutes * (mean_hr_bpm / 100))
        mean_hr_bpm, max_hr_bpm
        mean_cadence_spm, max_cadence_spm
        elevation_change (ft): |(total_ascent - total_descent)|
- pandas.DataFrame via FitDatasetAnalyzer.summarize_df()
    Table form of the ActivitySummary list with the same columns
- Convenience constructor FitDatasetAnalyzer.from_csvs(activity_csv, records_csv)
    Loads schema CSV exports (e.g., runs_activity.csv and runs_records.csv) and returns a ready analyzer
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd


# Output model
@dataclass(frozen=True)
class ActivitySummary:
    """
    Summarizes one activity to achieve a per-activity analysis record.
    Computes requested fields from df_activity + df_records for a given activity_id.
    """
    date_yyyymmdd: str              # YYYYMMDD
    time_hhmmss: str                # HH:MM:SS (UTC by default)
    distance_miles: float
    pace_mmss: Optional[str]        # MM:SS (minutes per mile)
    efficiency: Optional[float]     # lower is better
    mean_hr_bpm: Optional[float]
    max_hr_bpm: Optional[int]
    mean_cadence_spm: Optional[float]
    max_cadence_spm: Optional[int]
    elevation_change: Optional[float]  # |(Up - Down)| ft


# Analyzer
class FitDatasetAnalyzer:
    """
    Analyzes FIT-derived DataFrames to achieve per-activity summaries.
    Groups records by activity_id and derives distance/HR/cadence/elevation-change fields.
    """

    def __init__(self, df_activity: pd.DataFrame, df_records: pd.DataFrame):
        self.df_activity = df_activity.copy()
        self.df_records = df_records.copy()

        # Normalize dtypes expected from fit_to_schema.py outputs
        if "start_time" in self.df_activity.columns:
            self.df_activity["start_time"] = pd.to_datetime(self.df_activity["start_time"], errors="coerce", utc=True)
        if "timestamp" in self.df_records.columns:
            self.df_records["timestamp"] = pd.to_datetime(self.df_records["timestamp"], errors="coerce", utc=True)

    @staticmethod
    def _meters_to_miles(m: Optional[float]) -> Optional[float]:
        """
        Converts meters to miles (1 mile = 1609.344 m)
        """
        if m is None:
            return None
        try:
            return float(m) / 1609.344
        except Exception:
            return None

    @staticmethod
    def _meters_to_feet(m: Optional[float]) -> Optional[float]:
        """
        Converts meters to feet (1 m = 3.280839895 ft)
        """
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

        minutes = s // 60          # can exceed 59, that's fine for pace
        seconds = s % 60
        return f"{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _compute_pace_mmss(time_seconds: Optional[float], distance_miles: float) -> Optional[str]:
        if time_seconds is None:
            return None
        if distance_miles <= 0:
            return None
        pace_sec_per_mile = float(time_seconds) / float(distance_miles)
        return FitDatasetAnalyzer._format_mmss(pace_sec_per_mile)
    
    @staticmethod
    def _pace_seconds_per_mile(time_seconds: Optional[float], distance_miles: float) -> Optional[float]:
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
        This is a custom efficiency formula used to gauge ability to perform... efficiently
        The lower the number, the better. 10 is the ability to run a 6 minute mile with 195bpm average HR
        """
        if pace_seconds_per_mile is None:
            return None
        if mean_hr_bpm is None:
            return None
        try:
            pace_decimal_minutes = float(pace_seconds_per_mile) / 60.0
            return 0.8547 * pace_decimal_minutes * (float(mean_hr_bpm) / 100.0)
        except Exception:
            return None

    @staticmethod
    def _pick_altitude_series(df: pd.DataFrame) -> pd.Series:
        """
        Selects the best altitude column to achieve consistent elevation computations.
        Prefers enhanced_altitude_m if present; otherwise falls back to altitude_m
        """
        if "enhanced_altitude_m" in df.columns and df["enhanced_altitude_m"].notna().any():
            return df["enhanced_altitude_m"]
        return df.get("altitude_m", pd.Series([pd.NA] * len(df), index=df.index))

    @staticmethod
    def _compute_up_down_m(alt_m: pd.Series) -> Tuple[Optional[float], Optional[float]]:
        """
        Computes total ascent and descent to achieve a robust elevation-change basis.
        Uses successive differences: sum of positive deltas = up, sum of absolute negative deltas = down
        """
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
        """
        Produces ActivitySummary objects to achieve per-activity reporting.
        Uses df_activity for start_time + total distance when present, and
        df_records for mean/max HR/cadence and elevation change.
        """
        out: List[ActivitySummary] = []

        if self.df_activity.empty:
            return out

        # Index activity rows by activity_id for quick lookup
        act = self.df_activity.set_index("activity_id", drop=False)

        # Group records by activity_id (records drive HR/cadence/elevation)
        grouped = self.df_records.groupby("activity_id") if not self.df_records.empty else {}

        for activity_id, arow in act.iterrows():
            start_time = arow.get("start_time")
            if pd.isna(start_time):
                # If start_time is missing, skip (or set placeholders)
                continue

            # Date/Time (UTC, because fit_to_schema normalizes to Z/UTC)
            dt: datetime = pd.Timestamp(start_time).to_pydatetime()
            date_yyyymmdd = dt.strftime("%Y%m%d")
            time_hhmmss = dt.strftime("%H:%M:%S")

            # Distance miles: prefer activity total_distance_m; otherwise use max record distance_m
            total_distance_m = arow.get("total_distance_m")
            dist_m = None
            if pd.notna(total_distance_m):
                dist_m = float(total_distance_m)
            else:
                if activity_id in grouped:
                    r = grouped.get_group(activity_id)
                    if "distance_m" in r.columns:
                        dmax = pd.to_numeric(r["distance_m"], errors="coerce").max()
                        if pd.notna(dmax):
                            dist_m = float(dmax)

            distance_miles = self._meters_to_miles(dist_m) or 0.0

            # Pace (minutes per mile): prefer timer time; fallback to elapsed time
            time_s = None
            if pd.notna(arow.get("total_timer_time_s")):
                time_s = float(arow.get("total_timer_time_s"))
            elif pd.notna(arow.get("total_elapsed_time_s")):
                time_s = float(arow.get("total_elapsed_time_s"))

            pace_mmss = self._compute_pace_mmss(time_s, distance_miles)

            # Records slice for this activity
            r = grouped.get_group(activity_id) if activity_id in grouped else pd.DataFrame()

            # HR: mean/max from records if present, else fall back to df_activity avg/max fields
            mean_hr = None
            max_hr = None
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

            # Cadence (spm): mean/max from records if present, else fall back to df_activity avg/max fields
            mean_cad = None
            max_cad = None
            if not r.empty and "cadence_raw" in r.columns and r["cadence_raw"].notna().any():
                cad = pd.to_numeric(r["cadence_raw"], errors="coerce").dropna()
                if not cad.empty:
                    mean_cad = float(cad.mean())
                    max_cad = int(cad.max())
            else:
                if pd.notna(arow.get("avg_cadence_raw")):
                    mean_cad = float(arow.get("avg_cadence_raw"))
                if pd.notna(arow.get("max_cadence_raw")):
                    max_cad = int(arow.get("max_cadence_raw"))

            # Efficiency: Derived from HR and Pace calculated with constants
            pace_sec_per_mile = self._pace_seconds_per_mile(time_s, distance_miles)
            pace_mmss = self._format_mmss(pace_sec_per_mile) if pace_sec_per_mile is not None else None
            efficiency = self._compute_efficiency(pace_sec_per_mile, mean_hr)

            # Elevation Change:
            elevation_change_ft = None
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
                    mean_cadence_spm=None if mean_cad is None else (round(mean_cad, 1) * 2),
                    max_cadence_spm=max_cad * 2,
                    elevation_change=None if elevation_change_ft is None else round(elevation_change_ft, 1),
                )
            )

        return out

    def summarize_df(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame to achieve easy viewing/export of summaries.
        Converts the list of ActivitySummary objects into a table
        """
        rows = [s.__dict__ for s in self.summarize()]
        return pd.DataFrame(
            rows,
            columns=[
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

    @staticmethod
    def from_csvs(activity_csv: Path, records_csv: Path) -> "FitDatasetAnalyzer":
        """
        Loads CSV outputs to achieve analysis without re-parsing FIT files.
        Reads runs_activity.csv and runs_records.csv and constructs an analyzer instance.
        """
        df_activity = pd.read_csv(activity_csv)
        df_records = pd.read_csv(records_csv)
        return FitDatasetAnalyzer(df_activity=df_activity, df_records=df_records)
