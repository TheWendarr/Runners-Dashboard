/* ============================================================
   upload.js — drag-drop / browse upload UI for the index page
   ============================================================ */

(function() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const resultsEl = document.getElementById('upload-results');

    if (!dropzone || !fileInput || !resultsEl) return;

    // -------- drag visual state --------
    ['dragenter', 'dragover'].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });
    ['dragleave', 'drop'].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    // -------- drop handler --------
    dropzone.addEventListener('drop', (e) => {
        const files = Array.from(e.dataTransfer.files);
        if (files.length) upload(files);
    });

    fileInput.addEventListener('change', () => {
        const files = Array.from(fileInput.files);
        if (files.length) upload(files);
        fileInput.value = ''; // allow re-uploading the same file
    });

    // -------- upload --------
    async function upload(files) {
        // Show pending state for each file
        const pendingRows = files.map(f => {
            const row = document.createElement('div');
            row.className = 'upload-result';
            row.innerHTML = `
                <span class="status">uploading…</span>
                <span class="filename">${escapeHtml(f.name)}</span>
                <span class="detail">${formatBytes(f.size)}</span>
            `;
            resultsEl.prepend(row);
            return row;
        });

        const form = new FormData();
        files.forEach(f => form.append('files', f));

        try {
            const res = await fetch('/api/upload', { method: 'POST', body: form });
            if (!res.ok) {
                const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
                pendingRows.forEach(row => {
                    row.className = 'upload-result error';
                    row.querySelector('.status').textContent = 'error';
                    row.querySelector('.detail').textContent = err.error || 'unknown error';
                });
                return;
            }

            const data = await res.json();
            // Remove pending rows; render the actual results
            pendingRows.forEach(r => r.remove());
            data.results.forEach(r => addResultRow(r));

            // Tell the library to refresh
            window.dispatchEvent(new CustomEvent('library-changed'));
        } catch (err) {
            pendingRows.forEach(row => {
                row.className = 'upload-result error';
                row.querySelector('.status').textContent = 'error';
                row.querySelector('.detail').textContent = String(err);
            });
        }
    }

    function addResultRow(r) {
        const row = document.createElement('div');
        row.className = `upload-result ${r.status}`;
        const detail = r.reason || (r.activity_id ? r.activity_id.slice(0, 8) : '');
        row.innerHTML = `
            <span class="status">${r.status}</span>
            <span class="filename">${escapeHtml(r.filename)}</span>
            <span class="detail">${escapeHtml(detail)}</span>
        `;
        resultsEl.prepend(row);

        // Auto-fade older rows after 5s of new activity
        setTimeout(() => row.style.opacity = '0.55', 8000);
    }

    function escapeHtml(s) {
        return String(s).replace(/[&<>"']/g, c => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[c]));
    }

    function formatBytes(b) {
        if (b < 1024) return `${b} B`;
        if (b < 1024 * 1024) return `${(b/1024).toFixed(1)} KB`;
        return `${(b/1024/1024).toFixed(1)} MB`;
    }
})();
