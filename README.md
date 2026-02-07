# Runners-Dashboard
Convert a series of selected .fit file running activities to analyze performance trends over time. All processed on your local machine to no longer rely on web portals or subscription services to analyze training data.

## Schema
### 1. Activity Table (df_activity)
One row per activity (.fit file)
Required Fields:
activity_id – STR – Deterministic unique activity identifier (join key)
start_time – DATETIME (UTC) – Activity start timestamp
Optional / Common Fields:
total_distance_m – FLOAT – Total activity distance (meters)
avg_heart_rate_bpm – FLOAT – Average heart rate (bpm)
max_heart_rate_bpm – INT – Maximum heart rate (bpm)
avg_cadence_raw – FLOAT – Average cadence (steps/min)
max_cadence_raw – INT – Maximum cadence (steps/min)
source_file – STR – Original .fit filename (traceability)
Notes:
Primary authority for activity date/time.
Distance, HR, and cadence act as fallbacks if record data is missing.
Typically one row per .fit file.
### 2. Records Table (df_records)
Many rows per activity (time-series records)
Required Fields:
activity_id – STR – Foreign key to df_activity.activity_id
timestamp – DATETIME (UTC) – Record timestamp
Optional / Common Fields:
distance_m – FLOAT – Cumulative distance at record (meters)
heart_rate_bpm – INT – Heart rate at record (bpm)
cadence_raw – INT – Cadence at record (steps/min)
enhanced_altitude_m – FLOAT – Enhanced altitude (meters, preferred)
altitude_m – FLOAT – Standard altitude (meters, fallback)
Notes:
Primary source for statistical calculations (mean/max HR & cadence).
Drives elevation change computation.
enhanced_altitude_m is preferred when available.
### 3. Skipped Table (df_skipped)
One row per skipped or failed .fit file
Fields:
source_file – STR – File that failed parsing
reason – STR – Skip or error reason
details – STR – Optional diagnostic detail
Notes:
Informational only.
Not used in analysis.
### 4. Activity Summary Table (df_summary)
One row per activity (final output)
Fields:
date_yyyymmdd – STR – Activity date (YYYYMMDD, UTC)
time_hhmmss – STR – Activity start time (HH:MM:SS, UTC)
distance_miles – FLOAT – Total distance (miles)
mean_hr_bpm – FLOAT | NULL – Mean heart rate (bpm)
max_hr_bpm – INT | NULL – Maximum heart rate (bpm)
mean_cadence_spm – FLOAT | NULL – Mean cadence (steps/min)
max_cadence_spm – INT | NULL – Maximum cadence (steps/min)
elevation_change – FLOAT | NULL – |total_ascent − total_descent| (feet)
### Derivation Rules:
Distance: df_activity.total_distance_m → fallback to max(df_records.distance_m)
HR/Cadence: record-level stats → fallback to activity-level fields
Elevation: successive altitude deltas, meters → feet
One summary row per activity_id