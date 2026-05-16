"""
ClaimSetu — FastAPI app + demo UI
Gemma 4 Good Hackathon (Kaggle)

Run:
    uv run uvicorn app:app --reload

Endpoints:
    GET  /           — demo UI (HTML)
    GET  /health     — service health + model status
    POST /process    — upload claim files + package_code → run pipeline → return results
"""
from __future__ import annotations

import json
import tempfile
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(
    title="ClaimSetu",
    description="Explainable claim-review assistant for public health insurance",
    version="0.1.0",
)

# ── Lazy import of pipeline (heavy OCR/model deps load once) ─────────────────
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        import claimsAssistant as ca
        _pipeline = ca
    return _pipeline


# ── Health endpoint ───────────────────────────────────────────────────────────
@app.get("/health")
def health():
    try:
        ca = get_pipeline()
        return {
            "status": "ok",
            "mode": "live" if ca.OLLAMA_AVAILABLE else "mock",
            "edge_model": ca.MODEL_EDGE,
            "reasoning_model": ca.MODEL_REASON,
            "ocr": {
                "paddle": ca.PADDLE_AVAILABLE,
                "tesseract": ca.PYTESSERACT_AVAILABLE,
            },
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})


# ── Process endpoint ──────────────────────────────────────────────────────────
@app.post("/process")
async def process_claim(
    files: List[UploadFile] = File(...),
    package_code: str = Form(...),
):
    """
    Upload claim documents + specify package code → returns structured review findings.

    package_code: one of MG064A (Severe Anemia), SG039C (Cholecystectomy),
                          MG006A (Enteric Fever), SB039A (Total Knee Replacement)
    """
    ca = get_pipeline()

    if package_code not in ca.PACKAGE_CODES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown package_code '{package_code}'. "
                   f"Valid codes: {ca.PACKAGE_CODES}",
        )

    # Save uploaded files to a temp directory
    tmp_dir = Path(tempfile.mkdtemp(prefix="claimsetu_"))
    saved_paths: list[Path] = []
    try:
        for upload in files:
            suffix = Path(upload.filename or "doc").suffix.lower()
            if suffix not in ca.SUPPORTED_EXT:
                continue
            dest = tmp_dir / (upload.filename or f"file{suffix}")
            with dest.open("wb") as f:
                shutil.copyfileobj(upload.file, f)
            saved_paths.append(dest)

        if not saved_paths:
            raise HTTPException(status_code=400, detail="No supported files uploaded.")

        claim = {
            "claim_id":     tmp_dir.name,
            "package_code": package_code,
            "files":        sorted(saved_paths),
        }

        result = ca.process_claim(claim, ca.STG_CONFIGS)
        summary = ca.generate_reviewer_summary(result)

        # Build a clean JSON-serialisable response
        response = {
            "claim_id":      result["claim_id"],
            "package_code":  package_code,
            "decision":      summary["decision"],
            "confidence":    summary["confidence"],
            "flags":         summary["flags"],
            "timeline":      result.get("timeline_df", []),
            "classification": [
                {"doc_type": ld[0], "file": ld[1], "pages": ld[2]}
                for ld in result.get("logical_docs", [])
            ],
            "evidence_coverage": {
                slot: {
                    "filled":     info["filled"],
                    "confidence": round(info["confidence"], 2),
                    "doc_type":   info.get("doc_type"),
                    "page":       info.get("best_page"),
                }
                for slot, info in result.get("evidence_coverage", {}).items()
            },
            "reviewer_note": summary.get("reviewer_utility", {}).get("decision_summary", ""),
            "mode": "live" if ca.OLLAMA_AVAILABLE else "mock",
        }

        # Serialize timeline_df (pandas DataFrame → list of dicts)
        if hasattr(result.get("timeline_df"), "to_dict"):
            response["timeline"] = result["timeline_df"].to_dict(orient="records")

        return response

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)



# ── Demo UI ───────────────────────────────────────────────────────────────────
UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ClaimSetu — Claim Review Assistant</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #f0f4f8; color: #1a202c; }
  header { background: #1e3a8a; color: white; padding: 1.25rem 2rem;
           display: flex; align-items: center; gap: 1rem; }
  header h1 { font-size: 1.4rem; font-weight: 700; }
  header p  { font-size: 0.85rem; opacity: 0.8; margin-top: 0.15rem; }
  .badge { background: #f59e0b; color: #1a202c; font-size: 0.7rem;
           font-weight: 700; padding: 0.2rem 0.5rem; border-radius: 4px; }
  main  { max-width: 900px; margin: 2rem auto; padding: 0 1.5rem; }
  .card { background: white; border-radius: 10px; padding: 1.5rem;
          box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 1.5rem; }
  h2 { font-size: 1.05rem; font-weight: 600; margin-bottom: 1rem; color: #1e3a8a; }
  label { display: block; font-size: 0.85rem; font-weight: 500;
          margin-bottom: 0.4rem; color: #374151; }
  select { width: 100%; padding: 0.55rem 0.75rem; border: 1px solid #d1d5db;
           border-radius: 6px; font-size: 0.9rem; margin-bottom: 1rem; }
  .drop-zone { border: 2px dashed #93c5fd; border-radius: 8px; padding: 2rem;
               text-align: center; color: #6b7280; cursor: pointer;
               transition: background 0.2s; }
  .drop-zone.over, .drop-zone:hover { background: #eff6ff; }
  .drop-zone input { display: none; }
  .file-list { margin-top: 0.75rem; font-size: 0.8rem; color: #374151; }
  .file-list span { display: inline-block; background: #e0f2fe;
                    padding: 0.2rem 0.5rem; border-radius: 4px; margin: 0.2rem; }
  button { background: #1e3a8a; color: white; border: none; border-radius: 6px;
           padding: 0.6rem 1.5rem; font-size: 0.9rem; cursor: pointer;
           font-weight: 600; transition: background 0.2s; }
  button:hover { background: #1d4ed8; }
  button:disabled { background: #9ca3af; cursor: not-allowed; }
  #status { font-size: 0.85rem; color: #6b7280; margin-top: 0.75rem; }
  #results { display: none; }
  .decision-badge { display: inline-block; font-size: 1.1rem; font-weight: 700;
                    padding: 0.4rem 1.2rem; border-radius: 6px; margin-bottom: 1rem; }
  .PASS { background: #d1fae5; color: #065f46; }
  .CONDITIONAL { background: #fef3c7; color: #92400e; }
  .REVIEW { background: #fee2e2; color: #991b1b; }
  table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  th { background: #f1f5f9; text-align: left; padding: 0.5rem 0.75rem;
       border-bottom: 2px solid #e2e8f0; color: #475569; }
  td { padding: 0.45rem 0.75rem; border-bottom: 1px solid #f1f5f9; }
  tr:last-child td { border-bottom: none; }
  .filled-y { color: #16a34a; font-weight: 600; }
  .filled-n { color: #dc2626; }
  .flag { background: #fff7ed; border-left: 3px solid #f59e0b;
          padding: 0.5rem 0.75rem; border-radius: 0 6px 6px 0;
          font-size: 0.82rem; margin-bottom: 0.4rem; }
  .flag.mandatory { border-color: #ef4444; background: #fef2f2; }
</style>
</head>
<body>
<header>
  <div>
    <h1>ClaimSetu</h1>
    <p>Explainable claim-review assistant &nbsp;·&nbsp; OCR reads · E4B structures · 26B reasons · humans decide</p>
  </div>
  <span class="badge">Gemma 4 Good</span>
</header>
<main>
  <div class="card">
    <h2>Upload Claim Documents</h2>
    <label for="pkg">Package / Scheme</label>
    <select id="pkg">
      <option value="MG064A">MG064A — Severe Anemia</option>
      <option value="SG039C">SG039C — Cholecystectomy</option>
      <option value="MG006A">MG006A — Enteric Fever</option>
      <option value="SB039A">SB039A — Total Knee Replacement</option>
    </select>
    <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
      <input type="file" id="fileInput" multiple accept=".pdf,.jpg,.jpeg,.png,.tif,.tiff,.bmp">
      <div>&#128196; Drop claim files here or click to browse</div>
      <div style="font-size:0.78rem; margin-top:0.4rem;">PDF, JPG, PNG, TIFF supported</div>
      <div class="file-list" id="fileList"></div>
    </div>
    <br>
    <button id="submitBtn" onclick="submitClaim()">Analyse Claim</button>
    <div id="status"></div>
  </div>

  <div class="card" id="results">
    <h2>Review Findings</h2>
    <div id="decisionBadge"></div>
    <div id="reviewerNote" style="font-size:0.85rem; color:#374151; margin-bottom:1rem;"></div>

    <h2 style="margin-top:1rem;">Evidence Coverage</h2>
    <table id="coverageTable"><thead><tr>
      <th>Evidence Slot</th><th>Filled</th><th>Document Type</th><th>Page</th><th>Confidence</th>
    </tr></thead><tbody></tbody></table>

    <h2 style="margin-top:1.5rem;">Episode Timeline</h2>
    <table id="timelineTable"><thead><tr>
      <th>#</th><th>Event</th><th>Date</th><th>Source</th><th>Status</th>
    </tr></thead><tbody></tbody></table>

    <h2 style="margin-top:1.5rem;">Document Classification</h2>
    <table id="classTable"><thead><tr>
      <th>Document Type</th><th>File</th><th>Pages</th>
    </tr></thead><tbody></tbody></table>

    <div id="flagsSection" style="margin-top:1.5rem;">
      <h2>Flags</h2>
      <div id="flagsList"></div>
    </div>

    <div style="font-size:0.75rem; color:#9ca3af; margin-top:1.5rem;" id="modeNote"></div>
  </div>
</main>
<script>
const dz = document.getElementById('dropZone');
const fi = document.getElementById('fileInput');
let selectedFiles = [];

dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('over'); });
dz.addEventListener('dragleave', () => dz.classList.remove('over'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('over');
  selectedFiles = [...e.dataTransfer.files];
  renderFileList();
});
fi.addEventListener('change', () => { selectedFiles = [...fi.files]; renderFileList(); });

function renderFileList() {
  const el = document.getElementById('fileList');
  el.innerHTML = selectedFiles.map(f => `<span>${f.name}</span>`).join('');
}

async function submitClaim() {
  if (!selectedFiles.length) { alert('Please select claim files first.'); return; }
  const btn = document.getElementById('submitBtn');
  const status = document.getElementById('status');
  btn.disabled = true;
  status.textContent = 'Processing claim… this may take a minute.';
  document.getElementById('results').style.display = 'none';

  const fd = new FormData();
  selectedFiles.forEach(f => fd.append('files', f));
  fd.append('package_code', document.getElementById('pkg').value);

  try {
    const res = await fetch('/process', { method: 'POST', body: fd });
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail || res.statusText); }
    const data = await res.json();
    renderResults(data);
    status.textContent = '';
  } catch (e) {
    status.textContent = 'Error: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}

function renderResults(d) {
  document.getElementById('results').style.display = 'block';
  // Decision badge
  const dec = d.decision || 'REVIEW';
  document.getElementById('decisionBadge').innerHTML =
    `<span class="decision-badge ${dec}">${dec}</span>
     <span style="font-size:0.85rem; color:#374151; margin-left:0.5rem;">confidence: ${(d.confidence||0).toFixed(2)}</span>`;
  document.getElementById('reviewerNote').textContent = d.reviewer_note || '';

  // Evidence coverage
  const cTb = document.querySelector('#coverageTable tbody');
  cTb.innerHTML = '';
  for (const [slot, info] of Object.entries(d.evidence_coverage || {})) {
    const row = cTb.insertRow();
    row.innerHTML = `<td>${slot}</td>
      <td class="${info.filled ? 'filled-y' : 'filled-n'}">${info.filled ? '✓' : '✗'}</td>
      <td>${info.doc_type || '—'}</td>
      <td>${info.page != null ? 'p.' + info.page : '—'}</td>
      <td>${info.filled ? info.confidence.toFixed(2) : '—'}</td>`;
  }

  // Timeline
  const tTb = document.querySelector('#timelineTable tbody');
  tTb.innerHTML = '';
  (d.timeline || []).forEach((row, i) => {
    const tr = tTb.insertRow();
    tr.innerHTML = `<td>${i+1}</td><td>${row.event_type||row.Event||'—'}</td>
      <td>${row.date||row.Date||'—'}</td>
      <td>${row.source_doc||row['Source Document']||'—'}</td>
      <td>${row.validity||row.Status||'—'}</td>`;
  });

  // Classification
  const clTb = document.querySelector('#classTable tbody');
  clTb.innerHTML = '';
  (d.classification || []).forEach(ld => {
    const tr = clTb.insertRow();
    tr.innerHTML = `<td>${ld.doc_type}</td><td>${ld.file}</td><td>${ld.pages}</td>`;
  });

  // Flags
  const fl = document.getElementById('flagsList');
  fl.innerHTML = '';
  (d.flags || []).slice(0,10).forEach(f => {
    const cls = f.severity === 'mandatory' ? 'flag mandatory' : 'flag';
    fl.innerHTML += `<div class="${cls}"><strong>[${f.severity||''}]</strong> ${f.flag||''}</div>`;
  });
  document.getElementById('flagsSection').style.display = d.flags?.length ? 'block' : 'none';
  document.getElementById('modeNote').textContent =
    d.mode === 'mock' ? 'Running in OCR-only mock mode (Ollama not detected). Start Ollama with gemma4:e4b and gemma4:26b for full inference.' : '';
}
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def ui():
    return UI_HTML


def main():
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
