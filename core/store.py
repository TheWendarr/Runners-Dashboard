"""
In-memory activity store.

Process-singleton. State is lost when the Flask process restarts — this is
intentional, per the design discussion. If persistence is ever wanted later,
this is the single seam to swap.

Thread-safety: Flask's default dev server is single-threaded, but production
servers (gunicorn with threads, waitress) can hit this concurrently. The
lock keeps the dict operations atomic. The actual Activity objects are
immutable in practice (we never mutate after add()), so reads outside the
lock are fine.
"""

import threading
from typing import Dict, List, Optional

from .models import Activity


class ActivityStore:
    """Dict-backed store keyed by activity_id."""

    def __init__(self):
        self._activities: Dict[str, Activity] = {}
        self._lock = threading.Lock()

    def add(self, activity: Activity) -> bool:
        """
        Add an activity. Returns True if added, False if duplicate (same activity_id).
        Duplicates are silently ignored — same FIT file re-uploaded shouldn't fail.
        """
        with self._lock:
            if activity.activity_id in self._activities:
                return False
            self._activities[activity.activity_id] = activity
            return True

    def get(self, activity_id: str) -> Optional[Activity]:
        return self._activities.get(activity_id)

    def get_many(self, activity_ids: List[str]) -> List[Activity]:
        """Return activities in the order requested. Missing ids are skipped."""
        out: List[Activity] = []
        for aid in activity_ids:
            a = self._activities.get(aid)
            if a is not None:
                out.append(a)
        return out

    def list_all(self) -> List[Activity]:
        """All activities, sorted by start_time descending (newest first).
        Activities with no start_time sort last."""
        items = list(self._activities.values())
        # Use a sentinel for None so comparisons never hit None < None
        from datetime import datetime, timezone
        sentinel = datetime.min.replace(tzinfo=timezone.utc)
        items.sort(
            key=lambda a: a.start_time if a.start_time is not None else sentinel,
            reverse=True,
        )
        return items

    def delete(self, activity_id: str) -> bool:
        with self._lock:
            return self._activities.pop(activity_id, None) is not None

    def clear(self) -> int:
        with self._lock:
            n = len(self._activities)
            self._activities.clear()
            return n

    def __len__(self) -> int:
        return len(self._activities)


# Module-level singleton. Flask routes import this directly.
store = ActivityStore()
