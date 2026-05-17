/* ============================================================
   compare.js — overlay 2+ activities on the same 4 charts

   Reads ?ids=a,b,c from the URL. Fetches /api/compare?ids=...
   X-axis is user-selectable between elapsed time, distance, and
   percent of distance — each maps the same records differently.
   ============================================================ */

(function() {
    const subtitle = document.getElementById('compare-subtitle');
    const legendEl = document.getElementById('legend');
    const xToggle = document.getElementById('x-axis-toggle');

    let activities = [];          // array of full-payload activities
    let xKind = 'time';           // 'time' | 'distance' | 'percent'
    const charts = {};

    // -------- read ids from query --------

    function getIds() {
        const params = new URLSearchParams(window.location.search);
        const raw = (params.get('ids') || '').trim();
        if (!raw) return [];
        return raw.split(',').map(s => s.trim()).filter(Boolean);
    }

    // -------- fetch --------

    async function load() {
        const ids = getIds();
        if (ids.length < 2) {
            subtitle.textContent = 'Need at least 2 activities to compare. Select activities from the library and click Compare.';
            return;
        }

        try {
            const res = await fetch(`/api/compare?ids=${encodeURIComponent(ids.join(','))}`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            activities = data.activities || [];
            if (activities.length < 2) {
                subtitle.textContent = `Only ${activities.length} of ${ids.length} activities found in memory. Server restart may have wiped some.`;
                if (activities.length === 0) return;
            } else {
                subtitle.textContent = `Comparing ${activities.length} activities`;
            }
            renderLegend();
            renderCharts();
        } catch (err) {
            subtitle.textContent = `Failed to load: ${err}`;
        }
    }

    // -------- legend --------

    function renderLegend() {
        legendEl.innerHTML = activities.map((a, i) => {
            const color = Charts.seriesColor(i);
            const date = a.start_time ? new Date(a.start_time) : null;
            const dateStr = date ? date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) : '?';
            const dist = Units.fmt('distance', a.total_distance_m) + ' ' + Units.label('distance');
            return `
                <span class="legend-item">
                    <span class="legend-swatch" style="background:${color}"></span>
                    <span>${escapeHtml(dateStr)} · ${escapeHtml(dist)} · <span style="color: var(--text-faint)">${escapeHtml(a.source_filename)}</span></span>
                </span>
            `;
        }).join('');
    }

    // -------- charts --------

    function renderCharts() {
        Object.values(charts).forEach(c => c && c.destroy());

        const ctx = { xKind };

        // Build datasets per chart kind
        charts.hr = buildChart('chart-hr', Charts.makeHR, ctx, (r) => r.heart_rate_bpm);
        charts.pace = buildChart('chart-pace', Charts.makePace, ctx, (r) => {
            if (!r.pace_s_per_mile) return null;
            return r.pace_s_per_mile.map(v => v == null ? null : Units.convert('pace', v));
        });
        charts.cadence = buildChart('chart-cadence', Charts.makeCadence, ctx, (r) => r.cadence_spm);
        charts.elevation = buildChart('chart-elevation', Charts.makeElevation, ctx, (r) => {
            if (!r.altitude_m) return null;
            return r.altitude_m.map(v => v == null ? null : Units.convert('elev_change', v));
        });
    }

    function buildChart(canvasId, factory, ctx, ySelector) {
        const canvas = document.getElementById(canvasId);
        const datasets = activities.map((a, i) => {
            const records = a.records || {};
            const ys = ySelector(records);
            if (!ys) return null;
            const xs = xValuesFor(records, a);
            const data = makePoints(xs, ys);
            if (data.length === 0) return null;
            const date = a.start_time ? new Date(a.start_time).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) : '?';
            return {
                label: `${date} · ${a.source_filename}`,
                data,
                color: Charts.seriesColor(i),
            };
        }).filter(Boolean);

        if (datasets.length === 0) return null;
        return factory(canvas, datasets, ctx);
    }

    function xValuesFor(records, activity) {
        // Choose the X-array based on user selection.
        // Convert SI on the fly to keep server payloads small.
        if (xKind === 'time') {
            return records.t_s;
        }
        if (xKind === 'distance') {
            const dist = records.distance_m;
            if (!dist) return null;
            return dist.map(v => v == null ? null : Units.convert('distance', v));
        }
        if (xKind === 'percent') {
            const dist = records.distance_m;
            if (!dist) return null;
            // Use final non-null distance as the run total (more reliable than session)
            let total = 0;
            for (let i = dist.length - 1; i >= 0; i--) {
                if (dist[i] != null) { total = dist[i]; break; }
            }
            if (total <= 0) return null;
            return dist.map(v => v == null ? null : (v / total) * 100);
        }
        return records.t_s;
    }

    // -------- x-axis toggle --------

    xToggle.querySelectorAll('button').forEach(btn => {
        btn.addEventListener('click', () => {
            xKind = btn.dataset.axis;
            xToggle.querySelectorAll('button').forEach(b => b.classList.toggle('active', b === btn));
            renderCharts();
        });
    });

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

    function escapeHtml(s) {
        return String(s).replace(/[&<>"']/g, c => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[c]));
    }

    window.addEventListener('unitschange', () => {
        if (activities.length > 0) {
            renderLegend();
            renderCharts();
        }
    });

    load();
})();
