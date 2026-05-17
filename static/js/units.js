/* ============================================================
   units.js — unit system, conversions, and value formatters

   The server sends everything in SI. This module owns the conversion
   to imperial / metric display and the formatting of those values.
   Preference is persisted in localStorage.

   Public API:
       Units.get()                       -> 'imperial' | 'metric'
       Units.set(system)                 -> persists + dispatches 'unitschange' on window
       Units.fmt(kind, valueSI)          -> formatted string (e.g. "5.23 mi")
       Units.convert(kind, valueSI)      -> numeric value in display units
       Units.label(kind)                 -> display unit label (e.g. "mi", "bpm")

   `kind` values: distance, speed, pace, hr, cadence, elev_change, temp, duration
   ============================================================ */

const Units = (function() {
    const STORAGE_KEY = 'rd_units_pref';

    // -------- conversion constants --------
    const M_PER_MILE = 1609.344;
    const M_PER_KM = 1000;
    const FT_PER_M = 3.280839895;

    // -------- preference handling --------
    let current = 'imperial';
    try {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored === 'imperial' || stored === 'metric') current = stored;
    } catch (e) { /* localStorage might be blocked */ }

    function set(system) {
        if (system !== 'imperial' && system !== 'metric') return;
        current = system;
        try { localStorage.setItem(STORAGE_KEY, system); } catch (e) {}
        window.dispatchEvent(new CustomEvent('unitschange', { detail: { system } }));
    }

    function get() { return current; }

    // -------- conversion + formatters --------

    function _isNum(v) { return typeof v === 'number' && isFinite(v); }

    function convert(kind, valSI) {
        if (!_isNum(valSI)) return null;
        switch (kind) {
            case 'distance':  // meters -> mi or km
                return current === 'imperial' ? valSI / M_PER_MILE : valSI / M_PER_KM;
            case 'speed':     // m/s -> mph or km/h
                return current === 'imperial' ? valSI * 2.2369362921 : valSI * 3.6;
            case 'pace':      // s/m -> s/mi or s/km
                // NOTE: server sends pace as seconds per *mile* on the records,
                // and as raw m/s for summary speeds. We treat 'pace' input as s/mile
                // by convention since that's what the parser computes.
                return current === 'imperial' ? valSI : valSI / (M_PER_MILE / M_PER_KM);
            case 'elev_change':  // meters -> ft or m
                return current === 'imperial' ? valSI * FT_PER_M : valSI;
            case 'temp':     // celsius -> F or C
                return current === 'imperial' ? (valSI * 9/5) + 32 : valSI;
            case 'hr':
            case 'cadence':
            case 'duration':
                return valSI; // unit-agnostic
            default:
                return valSI;
        }
    }

    function label(kind) {
        const imp = current === 'imperial';
        switch (kind) {
            case 'distance':    return imp ? 'mi'   : 'km';
            case 'speed':       return imp ? 'mph'  : 'km/h';
            case 'pace':        return imp ? '/mi'  : '/km';
            case 'elev_change': return imp ? 'ft'   : 'm';
            case 'temp':        return imp ? '°F'   : '°C';
            case 'hr':          return 'bpm';
            case 'cadence':     return 'spm';
            case 'duration':    return '';
            default:            return '';
        }
    }

    // -------- value formatters --------

    function fmtDuration(seconds) {
        if (!_isNum(seconds) || seconds < 0) return '—';
        const total = Math.round(seconds);
        const h = Math.floor(total / 3600);
        const m = Math.floor((total % 3600) / 60);
        const s = total % 60;
        if (h > 0) return `${h}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
        return `${m}:${String(s).padStart(2,'0')}`;
    }

    function fmtPace(secondsPerMileOrKm) {
        // Already in s/mi or s/km depending on current unit system
        if (!_isNum(secondsPerMileOrKm) || secondsPerMileOrKm <= 0) return '—';
        const s = Math.round(secondsPerMileOrKm);
        const m = Math.floor(s / 60);
        const sec = s % 60;
        return `${m}:${String(sec).padStart(2,'0')}`;
    }

    function fmtNumber(v, digits = 1) {
        if (!_isNum(v)) return '—';
        return v.toFixed(digits);
    }

    function fmt(kind, valSI) {
        const v = convert(kind, valSI);
        if (v === null) return '—';
        switch (kind) {
            case 'distance':    return fmtNumber(v, 2);
            case 'speed':       return fmtNumber(v, 1);
            case 'pace':        return fmtPace(v);
            case 'elev_change': return fmtNumber(v, 0);
            case 'temp':        return fmtNumber(v, 0);
            case 'hr':          return Math.round(v).toString();
            case 'cadence':     return Math.round(v).toString();
            case 'duration':    return fmtDuration(v);
            default:            return fmtNumber(v, 1);
        }
    }

    // -------- topbar toggle wiring --------

    function _wireToggle() {
        const toggle = document.getElementById('unit-toggle');
        if (!toggle) return;
        const buttons = toggle.querySelectorAll('button');
        function syncActive() {
            buttons.forEach(b => {
                b.classList.toggle('active', b.dataset.units === current);
            });
        }
        buttons.forEach(b => {
            b.addEventListener('click', () => set(b.dataset.units));
        });
        window.addEventListener('unitschange', syncActive);
        syncActive();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', _wireToggle);
    } else {
        _wireToggle();
    }

    return { get, set, fmt, convert, label, fmtDuration, fmtPace, fmtNumber };
})();
