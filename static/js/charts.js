/* ============================================================
   charts.js — Chart.js theme + chart factories

   Provides factories that produce Chart.js configs for the four
   chart types used in both the per-activity and compare views.

   Public API:
       Charts.theme()                        -> common Chart.js options object
       Charts.seriesColor(i)                 -> i'th palette color
       Charts.makeHR(canvas, datasets, ctx)
       Charts.makePace(canvas, datasets, ctx)
       Charts.makeCadence(canvas, datasets, ctx)
       Charts.makeElevation(canvas, datasets, ctx)

   `datasets` is an array of {label, data: [{x, y}, ...], color}.
   `ctx` carries x-axis info: { xKind: 'time'|'distance'|'percent', xUnitLabel: 'mi'|'s'|'%' }.
   ============================================================ */

const Charts = (function() {

    // Pull design tokens from CSS variables so JS stays in sync with the theme
    function _cssVar(name) {
        return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    }

    const palette = [
        '--series-1', '--series-2', '--series-3', '--series-4',
        '--series-5', '--series-6', '--series-7', '--series-8',
    ].map(_cssVar);

    function seriesColor(i) { return palette[i % palette.length]; }

    // ---- shared chart theme ----
    function theme() {
        const text     = _cssVar('--text');
        const textDim  = _cssVar('--text-dim');
        const border   = _cssVar('--border');

        return {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 200 },
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: {
                    type: 'linear',
                    grid:   { color: border, drawTicks: false },
                    border: { color: border },
                    ticks:  {
                        color: textDim,
                        font: { family: 'IBM Plex Mono', size: 10 },
                        maxRotation: 0, autoSkip: true, maxTicksLimit: 8,
                    },
                },
                y: {
                    grid:   { color: border, drawTicks: false },
                    border: { color: border },
                    ticks:  {
                        color: textDim,
                        font: { family: 'IBM Plex Mono', size: 10 },
                    },
                },
            },
            plugins: {
                legend: { display: false }, // we draw our own legend
                tooltip: {
                    backgroundColor: _cssVar('--bg-elev-2'),
                    borderColor: _cssVar('--border-hot'),
                    borderWidth: 1,
                    titleColor: text,
                    bodyColor: text,
                    titleFont: { family: 'IBM Plex Mono', size: 11, weight: 600 },
                    bodyFont:  { family: 'IBM Plex Mono', size: 11 },
                    padding: 10,
                    cornerRadius: 0,
                    displayColors: true,
                    boxWidth: 8, boxHeight: 2,
                },
            },
            elements: {
                point: { radius: 0, hitRadius: 8 },
                line:  { borderWidth: 1.5, tension: 0.1 },
            },
        };
    }

    // ---- x-axis formatting based on axis kind ----
    function _xAxisOptions(xKind) {
        const base = {
            type: 'linear',
            grid: { color: _cssVar('--border'), drawTicks: false },
            border: { color: _cssVar('--border') },
            ticks: {
                color: _cssVar('--text-dim'),
                font: { family: 'IBM Plex Mono', size: 10 },
                maxRotation: 0, autoSkip: true, maxTicksLimit: 8,
            },
        };
        if (xKind === 'time') {
            base.ticks.callback = (v) => Units.fmtDuration(v);
            base.title = { display: true, text: 'elapsed', color: _cssVar('--text-faint'),
                          font: { family: 'IBM Plex Mono', size: 10 } };
        } else if (xKind === 'distance') {
            base.ticks.callback = (v) => Units.fmtNumber(v, 1);
            base.title = { display: true, text: `distance (${Units.label('distance')})`,
                          color: _cssVar('--text-faint'),
                          font: { family: 'IBM Plex Mono', size: 10 } };
        } else if (xKind === 'percent') {
            base.ticks.callback = (v) => `${Math.round(v)}%`;
            base.min = 0; base.max = 100;
            base.title = { display: true, text: '% of distance', color: _cssVar('--text-faint'),
                          font: { family: 'IBM Plex Mono', size: 10 } };
        }
        return base;
    }

    // ---- chart factories ----

    function _build(canvas, datasets, ctx, yOptions, yTickFormat, tooltipYFormat) {
        const opts = theme();
        opts.scales.x = _xAxisOptions(ctx.xKind);
        opts.scales.y = { ...opts.scales.y, ...yOptions };
        if (yTickFormat) {
            opts.scales.y.ticks = { ...opts.scales.y.ticks, callback: yTickFormat };
        }
        if (tooltipYFormat) {
            opts.plugins.tooltip.callbacks = {
                label: (item) => `${item.dataset.label}: ${tooltipYFormat(item.parsed.y)}`,
            };
        }

        // Color the datasets if not already colored
        datasets.forEach((ds, i) => {
            const c = ds.color || seriesColor(i);
            ds.borderColor = c;
            ds.backgroundColor = c;
            ds.pointBackgroundColor = c;
        });

        return new Chart(canvas, {
            type: 'line',
            data: { datasets },
            options: opts,
        });
    }

    function makeHR(canvas, datasets, ctx) {
        return _build(canvas, datasets, ctx,
            { min: 60, max: 200, title: { display: true, text: `bpm`,
                color: _cssVar('--text-faint'), font: { family: 'IBM Plex Mono', size: 10 } } },
            (v) => Math.round(v),
            (v) => `${Math.round(v)} bpm`
        );
    }

    function makePace(canvas, datasets, ctx) {
        // Pace in seconds per current unit. Lower = faster, so invert.
        const opts = {
            reverse: true,  // faster at top
            title: { display: true, text: `pace (${Units.label('pace')})`,
                    color: _cssVar('--text-faint'), font: { family: 'IBM Plex Mono', size: 10 } },
        };
        return _build(canvas, datasets, ctx, opts,
            (v) => Units.fmtPace(v),
            (v) => Units.fmtPace(v) + Units.label('pace')
        );
    }

    function makeCadence(canvas, datasets, ctx) {
        return _build(canvas, datasets, ctx,
            { min: 120, max: 200, title: { display: true, text: 'spm',
                color: _cssVar('--text-faint'), font: { family: 'IBM Plex Mono', size: 10 } } },
            (v) => Math.round(v),
            (v) => `${Math.round(v)} spm`
        );
    }

    function makeElevation(canvas, datasets, ctx) {
        const unitLabel = Units.label('elev_change');
        return _build(canvas, datasets, ctx,
            { title: { display: true, text: unitLabel,
                color: _cssVar('--text-faint'), font: { family: 'IBM Plex Mono', size: 10 } } },
            (v) => Math.round(v),
            (v) => `${Math.round(v)} ${unitLabel}`
        );
    }

    return { theme, seriesColor, makeHR, makePace, makeCadence, makeElevation };
})();
