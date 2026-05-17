/* ============================================================
   library.js — activity list table for index.html

   Behaviors:
   - Fetches /api/activities on load and after library-changed events
   - Checkbox multi-select drives the floating compare bar
   - Click-row navigates to detail
   - Delete button per row
   - Re-renders on unitschange so distances/paces update
   ============================================================ */

(function() {
    const bodyEl = document.getElementById('library-body');
    const countEl = document.getElementById('activity-count');
    const clearBtn = document.getElementById('clear-btn');
    const compareBar = document.getElementById('compare-bar');
    const selectedCountEl = document.getElementById('selected-count');
    const compareGoBtn = document.getElementById('compare-go');
    const compareClearBtn = document.getElementById('compare-clear');

    let activities = [];
    let selectedIds = new Set();

    // -------- fetch + render --------

    async function load() {
        try {
            const res = await fetch('/api/activities');
            const data = await res.json();
            activities = data.activities || [];
            render();
        } catch (err) {
            bodyEl.innerHTML = `<div class="empty-state">Failed to load: ${err}</div>`;
        }
    }

    function render() {
        countEl.textContent = `${activities.length} activit${activities.length === 1 ? 'y' : 'ies'}`;

        if (activities.length === 0) {
            bodyEl.innerHTML = `
                <div class="empty-state">
                    No activities yet.
                    <div class="hint">drop .fit files above to get started</div>
                </div>`;
            updateCompareBar();
            return;
        }

        // Drop any selected ids that no longer exist (deleted, cleared)
        const existing = new Set(activities.map(a => a.activity_id));
        for (const id of selectedIds) if (!existing.has(id)) selectedIds.delete(id);

        const rows = activities.map(rowHTML).join('');
        bodyEl.innerHTML = `
            <table class="runs">
                <thead>
                    <tr>
                        <th class="checkbox-cell"></th>
                        <th>Date</th>
                        <th>File</th>
                        <th class="num">Distance</th>
                        <th class="num">Duration</th>
                        <th class="num">Pace</th>
                        <th class="num">Avg HR</th>
                        <th class="num">Elev ↑</th>
                        <th></th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>`;
        attachRowHandlers();
        updateCompareBar();
    }

    function rowHTML(a) {
        const date = a.start_time ? new Date(a.start_time) : null;
        const dateStr = date
            ? date.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
              + ' ' + date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
            : '—';

        // Compute pace from total distance + timer time
        let paceSI = null;
        if (a.total_distance_m && a.total_timer_time_s && a.total_distance_m > 0) {
            // s per mile (matches what server emits for record-level paces)
            paceSI = a.total_timer_time_s / (a.total_distance_m / 1609.344);
        }

        const checked = selectedIds.has(a.activity_id) ? 'checked' : '';

        return `
            <tr data-id="${a.activity_id}">
                <td class="checkbox-cell"><input type="checkbox" class="select-cb" ${checked}/></td>
                <td class="date">${dateStr}</td>
                <td class="title"><a href="/activity/${a.activity_id}">${escapeHtml(a.source_filename)}</a></td>
                <td class="num">${Units.fmt('distance', a.total_distance_m)} <span class="unit-suffix">${Units.label('distance')}</span></td>
                <td class="num">${Units.fmtDuration(a.total_timer_time_s || a.total_elapsed_time_s)}</td>
                <td class="num">${Units.fmtPace(Units.convert('pace', paceSI))}${paceSI != null ? '/' + Units.label('distance') : ''}</td>
                <td class="num">${a.avg_heart_rate_bpm != null ? a.avg_heart_rate_bpm : '—'}</td>
                <td class="num">${Units.fmt('elev_change', a.elev_gain_m)} <span class="unit-suffix">${Units.label('elev_change')}</span></td>
                <td class="actions">
                    <button class="btn btn-small btn-danger delete-btn" data-id="${a.activity_id}" title="Delete">×</button>
                </td>
            </tr>`;
    }

    function attachRowHandlers() {
        bodyEl.querySelectorAll('tr[data-id]').forEach(tr => {
            const id = tr.dataset.id;
            const cb = tr.querySelector('.select-cb');
            cb.addEventListener('change', () => {
                if (cb.checked) selectedIds.add(id);
                else selectedIds.delete(id);
                updateCompareBar();
            });
            // Don't capture clicks on inputs/buttons; only the link navigates
        });
        bodyEl.querySelectorAll('.delete-btn').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                e.stopPropagation();
                const id = btn.dataset.id;
                if (!confirm('Delete this activity?')) return;
                try {
                    const res = await fetch(`/api/activities/${id}`, { method: 'DELETE' });
                    if (res.ok) {
                        selectedIds.delete(id);
                        load();
                    }
                } catch (err) {
                    alert(`Delete failed: ${err}`);
                }
            });
        });
    }

    // -------- compare bar --------

    function updateCompareBar() {
        const n = selectedIds.size;
        selectedCountEl.textContent = String(n);
        compareGoBtn.disabled = n < 2;
        compareBar.classList.toggle('visible', n > 0);
    }

    compareGoBtn.addEventListener('click', () => {
        if (selectedIds.size < 2) return;
        const ids = Array.from(selectedIds).join(',');
        window.location.href = `/compare?ids=${encodeURIComponent(ids)}`;
    });

    compareClearBtn.addEventListener('click', () => {
        selectedIds.clear();
        bodyEl.querySelectorAll('.select-cb').forEach(cb => cb.checked = false);
        updateCompareBar();
    });

    // -------- clear all --------

    clearBtn.addEventListener('click', async () => {
        if (activities.length === 0) return;
        if (!confirm(`Delete all ${activities.length} activities?`)) return;
        try {
            await fetch('/api/clear', { method: 'POST' });
            selectedIds.clear();
            load();
        } catch (err) {
            alert(`Clear failed: ${err}`);
        }
    });

    // -------- events --------

    window.addEventListener('library-changed', load);
    window.addEventListener('unitschange', render);

    function escapeHtml(s) {
        return String(s).replace(/[&<>"']/g, c => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[c]));
    }

    // initial load
    load();
})();
