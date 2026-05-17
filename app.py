"""
Runners Dashboard — Flask app entry point.

Routes are split into:
- Page routes (HTML): /, /activity/<id>, /compare
- API routes (JSON): /api/upload, /api/activities, /api/activities/<id>,
                     /api/activities/<id> (DELETE), /api/clear

Server returns all numeric values in SI units. Unit conversion to imperial
or metric display happens client-side based on the user's toggle.
"""

from typing import List

from flask import Flask, jsonify, render_template, request

from core.fit_parser import parse_fit, FitParseError, RunningFilterError
from core.store import store


# Cap individual uploads. A typical .fit file for an hour-long run is well
# under 1 MB; 16 MB leaves plenty of headroom for ultra-long ultras while
# protecting against accidental huge uploads.
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

# Default downsampling target for chart-bound record streams. Chart.js handles
# ~1k points smoothly with hover/zoom; raw FIT files often have 3-10k.
DEFAULT_CHART_POINTS = 1000


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    # ---------- page routes ----------

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/activity/<activity_id>")
    def activity_page(activity_id: str):
        activity = store.get(activity_id)
        if activity is None:
            # Soft 404: redirect-like page so the user sees the library again.
            # A real 404 would be jarring on a tool with in-memory state where
            # restarting the server wipes everything.
            return render_template("not_found.html", activity_id=activity_id), 404
        return render_template("activity.html", activity_id=activity_id)

    @app.route("/compare")
    def compare_page():
        # ids come from the query string: ?ids=abc,def,ghi
        return render_template("compare.html")

    # ---------- api routes ----------

    @app.route("/api/upload", methods=["POST"])
    def api_upload():
        """
        Accept one or more .fit files via multipart upload.
        Returns a per-file result list so the client can show parse errors
        next to specific filenames.
        """
        files = request.files.getlist("files")
        if not files:
            return jsonify({"error": "no files uploaded"}), 400

        results = []
        for f in files:
            filename = f.filename or "unknown.fit"
            if not filename.lower().endswith(".fit"):
                results.append({
                    "filename": filename,
                    "status": "rejected",
                    "reason": "not a .fit file",
                })
                continue

            try:
                data = f.read()
                activity = parse_fit(data, source_filename=filename)
            except RunningFilterError as e:
                results.append({
                    "filename": filename,
                    "status": "skipped",
                    "reason": f"sport={e.sport!r}, not running",
                })
                continue
            except FitParseError as e:
                results.append({
                    "filename": filename,
                    "status": "error",
                    "reason": str(e),
                })
                continue
            except Exception as e:
                # Defensive: don't let one bad file kill the whole batch
                results.append({
                    "filename": filename,
                    "status": "error",
                    "reason": f"unexpected: {type(e).__name__}: {e}",
                })
                continue

            added = store.add(activity)
            results.append({
                "filename": filename,
                "status": "added" if added else "duplicate",
                "activity_id": activity.activity_id,
                "start_time": activity.to_summary_dict()["start_time"],
                "total_distance_m": activity.total_distance_m,
            })

        return jsonify({"results": results, "total_in_store": len(store)})

    @app.route("/api/activities")
    def api_list():
        """Summary list of all activities in the store (no record streams)."""
        return jsonify({
            "activities": [a.to_summary_dict() for a in store.list_all()],
            "count": len(store),
        })

    @app.route("/api/activities/<activity_id>")
    def api_get(activity_id: str):
        """Full activity including downsampled record stream."""
        activity = store.get(activity_id)
        if activity is None:
            return jsonify({"error": "not found"}), 404

        try:
            max_points = int(request.args.get("points", DEFAULT_CHART_POINTS))
        except ValueError:
            max_points = DEFAULT_CHART_POINTS
        max_points = max(0, min(max_points, 10000))  # clamp to a sane range

        return jsonify(activity.to_full_dict(max_points=max_points))

    @app.route("/api/activities/<activity_id>", methods=["DELETE"])
    def api_delete(activity_id: str):
        ok = store.delete(activity_id)
        if not ok:
            return jsonify({"error": "not found"}), 404
        return jsonify({"deleted": activity_id, "total_in_store": len(store)})

    @app.route("/api/compare")
    def api_compare():
        """
        Bulk-fetch multiple activities for the comparison view.
        Query: /api/compare?ids=abc,def,ghi&points=500
        Order in the response matches the order requested.
        """
        ids_param = request.args.get("ids", "").strip()
        if not ids_param:
            return jsonify({"error": "ids query param required"}), 400

        ids: List[str] = [i.strip() for i in ids_param.split(",") if i.strip()]
        if not ids:
            return jsonify({"error": "no valid ids in query"}), 400

        try:
            max_points = int(request.args.get("points", DEFAULT_CHART_POINTS))
        except ValueError:
            max_points = DEFAULT_CHART_POINTS
        max_points = max(0, min(max_points, 10000))

        activities = store.get_many(ids)
        return jsonify({
            "activities": [a.to_full_dict(max_points=max_points) for a in activities],
            "requested": ids,
            "found": [a.activity_id for a in activities],
        })

    @app.route("/api/clear", methods=["POST"])
    def api_clear():
        n = store.clear()
        return jsonify({"cleared": n})

    # ---------- error handlers ----------

    @app.errorhandler(413)
    def too_large(e):
        return jsonify({"error": f"upload exceeds {MAX_CONTENT_LENGTH // (1024 * 1024)} MB limit"}), 413

    return app


# Module-level app instance for `flask run` and gunicorn-style runners
app = create_app()


if __name__ == "__main__":
    # Default: localhost-only on port 5000. Per the design discussion,
    # this is a single-user homelab tool, no auth, no exposure beyond loopback.
    # To make it LAN-accessible later: change host to "0.0.0.0".
    app.run(host="127.0.0.1", port=5000, debug=True)
