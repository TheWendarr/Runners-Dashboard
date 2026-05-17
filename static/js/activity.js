/* ============================================================
   activity.js — per-activity detail page

   Fetches /api/activities/<id>, fills the stat grid, builds four charts.
   Re-renders stats and charts on unit toggle.
   ============================================================ */

(function() {
    const activityId = window.ACTIVITY_ID;
    if (!activityId) return;

    const titleEl = document.getElementById('activity-title');
    const subtitleEl = document.getElementById('activity-subtitle');
    const statGrid = document.getElementById('stat-grid');

    let activity = null;     // most-recent payload
    const charts = {};       // chart instances keyed by name

    // -------- fetch --------

    async function load() {
        try {
            const res = await fetch(`/api/activities/${activityId}`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            activity = await res.json();
            renderAll();
        } catch (err) {
            subtitleEl.textContent = `Failed to load: ${err}`;
        }
    }

    // -------- rendering --------

    function renderAll() {
        renderHeader();
        renderStats();
        renderCharts();
    }

    function renderHeader() {
        titleEl.textContent = activity.source_filename || 'Activity';
        const date = activity.start_time ? new Date(activity.start_time) : null;
        const dateStr = date ? date.toLocaleString(undefined, {
            weekday: 'short', year: 'numeric', month: 'short', day: 'numeric',
            hour: '2-digit', minute: '2-digit'
        }) : '—';
        const device = [activity.manufacturer, activity.product].filter(Boolean).join(' · ') || '—';
        subtitleEl.innerHTML = `<code>${dateStr}</code> · ${device} · <code>${activity.record_count} samples</code>`;
    }

    function renderStats() {
        // Derive pace from totals
        let paceSI = null;
        if (activity.total_distance_m && activity.total_timer_time_s && activity.total_distance_m > 0) {
            paceSI = activity.total_timer_time_s / (activity.total_distance_m / 1609.344);
        }
        const paceConverted = Units.convert('pace', paceSI);

        const stats = [
            ['Distance',  Units.fmt('distance', activity.total_distance_m), Units.label('distance')],
            ['Duration',  Units.fmtDuration(activity.total_timer_time_s || activity.total_elapsed_time_s), ''],
            ['Avg Pace',  Units.fmtPace(paceConverted), Units.label('pace')],
            ['Avg HR',    activity.avg_heart_rate_bpm != null ? String(activity.avg_heart_rate_bpm) : '—', 'bpm'],
            ['Max HR',    activity.max_heart_rate_bpm != null ? String(activity.max_heart_rate_bpm) : '—', 'bpm'],
            ['Avg Cadence', activity.avg_cadence_spm != null ? Math.round(activity.avg_cadence_spm).toString() : '—', 'spm'],
            ['Elev ↑',   Units.fmt('elev_change', activity.elev_gain_m), Units.label('elev_change')],
            ['Elev ↓',   Units.fmt('elev_change', activity.elev_loss_m), Units.label('elev_change')],
            ['Calories', activity.total_calories_kcal != null ? String(activity.total_calories_kcal) : '—', 'kcal'],
        ];

        statGrid.innerHTML = stats.map(([label, value, unit]) => `
            <div class="stat">
                <div class="stat-label">${label}</div>
                <div class="stat-value">${value}${unit ? `<span class="unit">${unit}</span>` : ''}</div>
            </div>
        `).join('');
    }

    function renderCharts() {
        // Destroy any existing chart instances (we rebuild on unit toggle)
        Object.values(charts).forEach(c => c && c.destroy());

        const records = activity.records || {};
        const t = records.t_s || [];

        // Build a points array against elapsed time (always)
        const ctx = { xKind: 'time' };

        // HR
        const hrData = makePoints(t, records.heart_rate_bpm);
        if (hrData.length > 0) {
            charts.hr = Charts.makeHR(
                document.getElementById('chart-hr'),
                [{ label: 'HR', data: hrData }],
                ctx
            );
            setMeta('hr-meta', `${countNonNull(records.heart_rate_bpm)} samples`);
        } else {
            setMeta('hr-meta', 'no data');
        }

        // Pace — already in s/mile from server, convert per unit
        const paceData = makePoints(t, mapMaybe(records.pace_s_per_mile, v => Units.convert('pace', v)));
        if (paceData.length > 0) {
            charts.pace = Charts.makePace(
                document.getElementById('chart-pace'),
                [{ label: 'Pace', data: paceData }],
                ctx
            );
            setMeta('pace-meta', '');
        } else {
            setMeta('pace-meta', 'no data');
        }

        // Cadence
        const cadData = makePoints(t, records.cadence_spm);
        if (cadData.length > 0) {
            charts.cadence = Charts.makeCadence(
                document.getElementById('chart-cadence'),
                [{ label: 'Cadence', data: cadData }],
                ctx
            );
            setMeta('cadence-meta', '');
        } else {
            setMeta('cadence-meta', 'no data');
        }

        // Elevation
        const elevData = makePoints(t, mapMaybe(records.altitude_m, v => Units.convert('elev_change', v)));
        if (elevData.length > 0) {
            charts.elevation = Charts.makeElevation(
                document.getElementById('chart-elevation'),
                [{ label: 'Elevation', data: elevData }],
                ctx
            );
            setMeta('elevation-meta', '');
        } else {
            setMeta('elevation-meta', 'no data');
        }
    }

    // -------- helpers --------

    function makePoints(xs, ys) {
        if (!xs || !ys) return [];
        const n = Math.min(xs.length, ys.length);
        const out = [];
        for (let i = 0; i < n; i++) {
            if (xs[i] == null || ys[i] == null) continue;
            out.push({ x: xs[i], y: ys[i] });
        }
        return out;
    }

    function mapMaybe(arr, fn) {
        if (!arr) return null;
        return arr.map(v => (v == null ? null : fn(v)));
    }

    function countNonNull(arr) {
        if (!arr) return 0;
        let n = 0;
        for (const v of arr) if (v != null) n++;
        return n;
    }

    function setMeta(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }

    // -------- events --------

    window.addEventListener('unitschange', () => {
        if (activity) renderAll();
    });

    load();
})();
