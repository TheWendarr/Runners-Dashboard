"""
Orchestrates the end-to-end FIT analysis pipeline to convert Garmin .fit files into
per-activity summary records.

This script wires together two stages:
1) fit_to_schema: discovers and parses .fit files into standardized schema DataFrames
   (activity-level + record-level tables, plus a skipped/errors table).
2) schema_to_analysis: aggregates those schema tables into one summary row per activity
   (date/time, distance, HR stats, cadence stats, elevation-change metric).

Usage:
Run with CLI flags to specify an input file/folder and optional CSV exports, or omit
    input to be prompted interactively.

Produces a summary DataFrame in-memory and optionally writes it to CSV via --summary-out
"""


from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Import your two pipeline stages
import fit_to_schema
from schema_to_analysis import FitDatasetAnalyzer


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parses CLI arguments to achieve configurable end-to-end execution.
    Accepts an input path (file/dir), optional CSV export, and optional summary export path
    """
    p = argparse.ArgumentParser(
        description="End-to-end pipeline: .fit -> schema DataFrames -> per-activity summaries"
    )
    p.add_argument(
        "--input",
        help="Path to a .fit file or a directory containing .fit files.",
        required=False,
    )
    p.add_argument(
        "--csv-out",
        help="Optional directory to write runs_activity.csv / runs_records.csv / skipped.csv (schema stage).",
        required=False,
    )
    p.add_argument(
        "--summary-out",
        help="Optional path to write the summary CSV (analysis stage).",
        required=False,
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite CSV outputs if they exist (applies to --csv-out and --summary-out).",
    )
    return p.parse_args(argv)


def resolve_input_path(args: argparse.Namespace) -> Path:
    """
    Resolves an input path to achieve consistent CLI + prompt behavior.
    Prompts the user if --input is not provided.
    """
    if args.input:
        return Path(args.input)

    raw_in = input("Enter full path to a .fit file OR a directory of .fit files: ").strip().strip('"')
    if not raw_in:
        raise SystemExit("[ERROR] No input path provided.")
    return Path(raw_in)


def iter_fit_paths(input_path: Path) -> List[Path]:
    """
    Discovers .fit files to achieve a stable list of inputs for parsing.
    Returns a single file if input_path is a .fit file, otherwise recursively finds *.fit.
    """
    return fit_to_schema.iter_fit_paths(input_path)


def build_schema_from_fit_paths(fit_paths: List[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Builds schema DataFrames from a list of .fit paths for a list-driven dataflow.
    Calls the fit_to_schema extraction on each file and returns activity/records/skipped DataFrames.
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


def maybe_write_schema_csvs(csv_out: Optional[str], df_activity: pd.DataFrame, df_records: pd.DataFrame, df_skipped: pd.DataFrame, force: bool) -> None:
    """
    Optionally writes schema CSVs
    Uses fit_to_schema.write_csvs for consistent filenames and overwrite rules.
    """
    if not csv_out:
        return
    fit_to_schema.write_csvs(Path(csv_out), df_activity, df_records, df_skipped, force=bool(force))


def maybe_write_summary_csv(summary_out: Optional[str], df_summary: pd.DataFrame, force: bool) -> None:
    """
    Optionally writes summary
    Enforces overwrite behavior unless --force is provided.
    """
    if not summary_out:
        return
    out_path = Path(summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        raise FileExistsError(f"Output already exists: {out_path} (use --force)")
    df_summary.to_csv(out_path, index=False)


def main(argv: Optional[List[str]] = None) -> int:
    """
    Orchestrates schema + analysis stages to achieve end-to-end .fit -> analyzed records.
    Produces a df where each row corresponds to one .fit file's activity summary.
    """
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)

    try:
        input_path = resolve_input_path(args)
        fit_paths = iter_fit_paths(input_path)
        if not fit_paths:
            print(f"[ERROR] No .fit files found at: {input_path}")
            return 2

        # Stage 1: .fit -> schema DataFrames (activity/records/skipped)
        df_activity, df_records, df_skipped = build_schema_from_fit_paths(fit_paths)

        # Optional intermediate CSVs
        maybe_write_schema_csvs(args.csv_out, df_activity, df_records, df_skipped, force=bool(args.force))

        # Stage 2: schema DataFrames -> per-activity summary df
        analyzer = FitDatasetAnalyzer(df_activity=df_activity, df_records=df_records)
        df_summary = analyzer.summarize_df()

        """
        Ensure "each record is one .fit file's data" (i.e., one per activity)
        If multiple activities somehow share a source_file in df_activity, this still yields one per activity_id.
        (Optional) attach source_file for traceability (not requested in output schema)
        df_summary = df_summary.merge(df_activity[["activity_id", "source_file"]], left_index=False, right_index=False)
        """

        # Optional analysis CSV
        maybe_write_summary_csv(args.summary_out, df_summary, force=bool(args.force))

        # Print quick stats and a preview
        print("[DONE]")
        print(f"  fit_files:   {len(fit_paths)}")
        print(f"  activities:  {len(df_activity)}")
        print(f"  records:     {len(df_records)}")
        print(f"  skipped:     {len(df_skipped)}")
        print(f"  summaries:   {len(df_summary)}")
        print(df_summary.head(10).to_string(index=False))

        # Return code OK
        return 0

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
