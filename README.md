# Runners Dashboard

Self-hosted, in-memory analyzer for `.fit` running activity files. Drop your runs
in, get interactive charts, overlay multiple runs to compare. No accounts, no
cloud, no persistence — restart the server and state is gone (by design).

Successor to the original 3-stage pipeline; rewritten as a Flask web app with
client-side Chart.js so visualizations are interactive rather than static SVG.

* Adapted from original Repo using Claude Opus 4.7 model to behave as a webapp with interactive graphs

## Architecture

```
runners-dashboard/
├── app.py               # Flask app + routes
├── core/
│   ├── fit_parser.py    # .fit → Activity dataclass (refactored from old repo)
│   ├── models.py        # Activity dataclass + JSON serialization
│   └── store.py         # In-memory singleton, dict[id] → Activity
├── static/
│   ├── css/app.css      # All styles
│   └── js/
│       ├── chart.umd.min.js   # vendored Chart.js 4.5.1
│       ├── units.js     # unit toggle + conversions + formatters
│       ├── charts.js    # Chart.js theme + chart factories
│       ├── upload.js    # drag-drop upload (index)
│       ├── library.js   # activity table (index)
│       ├── activity.js  # per-activity detail page
│       └── compare.js   # overlay comparison page
└── templates/
    ├── base.html        # shared chrome (topbar, scripts)
    ├── index.html       # library + uploader
    ├── activity.html    # detail page
    ├── compare.html     # comparison page
    └── not_found.html   # soft 404
```

## Key design decisions

- **In-memory only.** No database, no disk cache. Server restart = clean slate.
  This is single-user homelab software — the simplicity is the feature.
- **Server speaks SI.** All JSON responses are meters, m/s, seconds, °C. Unit
  conversion to imperial/metric happens in the browser based on the toggle.
  Server is stateless w.r.t. user preference.
- **Running-only.** Non-running activities (cycling, walking, swim) are rejected
  at parse time. Easy to relax later by dropping the filter in `fit_parser.py`.
- **Server-side downsampling.** Long runs get uniform-stride downsampling to
  ~1000 points before serialization. Configurable via `?points=N` on the
  detail and compare endpoints (clamped to 10000).

## Setup

```bash
# Recommended: virtualenv
python -m venv .venv
source .venv/bin/activate           # Linux/macOS
# .venv\Scripts\activate            # Windows

pip install -r requirements.txt

# Run dev server (localhost:5000 only)
python app.py
```

Then open <http://127.0.0.1:5000/> and drag `.fit` files into the dropzone.

## Making it LAN-accessible

Edit `app.py`, change `host="127.0.0.1"` to `host="0.0.0.0"`. There's no
authentication — if it's exposed beyond loopback, anyone on your network can
upload files and read everyone else's data. For homelab use this is usually
fine; for anything more, put it behind a reverse proxy with auth.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Library + upload page |
| `GET`  | `/activity/<id>` | Detail page |
| `GET`  | `/compare?ids=a,b,c` | Comparison page |
| `POST` | `/api/upload` | Multipart `.fit` upload; returns per-file results |
| `GET`  | `/api/activities` | Summary list (no record streams) |
| `GET`  | `/api/activities/<id>` | Full activity (summary + downsampled records) |
| `DELETE` | `/api/activities/<id>` | Remove from store |
| `GET`  | `/api/compare?ids=a,b,c&points=N` | Bulk full payload for compare view |
| `POST` | `/api/clear` | Wipe all activities |

## What's not here yet

These were intentionally cut for the v0.1 scope:
- Trend dashboards (daily/weekly summaries over time)
- GPS map view of the route
- Activity tagging / notes
- Persistence between server restarts
- Multi-user support

All of these are additive and can be layered on without changing the parsing
core or the API shape.

## Notes on the original repo

The parsing logic in `core/fit_parser.py` is distilled from the original
`fit_to_feature.py` — same FIT-format handling (enhanced-vs-basic series
selection, cadence doubling, semicircle→degree conversion, elevation
gain/loss). The 3-stage pipeline scaffolding (`feature_to_graph.py`,
`graph_to_svg.py`, all the CSV/Parquet/manifest I/O) was removed since the
web app shape doesn't need it.

The "efficiency" metric from the original was deliberately dropped from the
UI — it computed `0.8547 × pace_min × HR/100`, which is actually cardiac
cost per mile (not efficiency in the aerobic sense). Easy to add back as a
trend line later if useful.
