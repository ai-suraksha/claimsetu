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

# Load .env before anything else — so OLLAMA_HOST etc. are set before claimsAssistant imports
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            import os as _os
            _os.environ.setdefault(_k.strip(), _v.strip())

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="ClaimSetu",
    description="Explainable claim-review assistant for public health insurance",
    version="0.1.0",
)

_demo_index = Path(__file__).parent / "demo" / "index.html"

# CORS — allow demo UI (localhost or file://) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
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
            "ollama_host": ca.OLLAMA_HOST,
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
            # webkitdirectory sends filename as "FolderName/file.pdf" — flatten to just the file
            raw_name = upload.filename or "doc"
            flat_name = Path(raw_name).name          # strip any subfolder prefix
            suffix = Path(flat_name).suffix.lower()
            if suffix not in ca.SUPPORTED_EXT:
                continue
            dest = tmp_dir / flat_name
            dest.parent.mkdir(parents=True, exist_ok=True)
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

        # ── Timeline: normalise DataFrame columns to snake_case ───────────────
        timeline_rows = []
        tdf = result.get("timeline_df")
        if hasattr(tdf, "to_dict"):
            for row in tdf.to_dict(orient="records"):
                def _val(*keys):
                    for k in keys:
                        v = row.get(k)
                        if v is not None and str(v).strip() not in ("", "nan"):
                            return str(v)
                    return "—"
                timeline_rows.append({
                    "event_type": _val("Event Type", "event_type"),
                    "date":       _val("Date", "date"),
                    "source_doc": _val("Source Document", "source_doc"),
                    "page":       _val("Best Page", "page"),
                    "validity":   _val("Temporal Validity", "validity"),
                })

        # ── Classification: group per logical doc (doc_type, file, page_count) ─
        # logical_docs is list of (doc_type, file_name, page_count) tuples.
        # Mask patient-name fragments from filenames for safe public demo.
        def _safe_filename(name: str) -> str:
            import re
            # Keep only the numeric prefix and extension e.g. "000585__...SUDHAN.pdf" → "000585.pdf"
            m = re.match(r'^(\d+)', name)
            prefix = m.group(1) if m else "doc"
            suffix = Path(name).suffix
            return f"{prefix}{suffix}"

        classification_rows = [
            {"doc_type": ld[0], "file": _safe_filename(str(ld[1])), "pages": ld[2]}
            for ld in result.get("logical_docs", [])
        ]

        # Build a clean JSON-serialisable response
        response = {
            "claim_id":      result["claim_id"],
            "package_code":  package_code,
            "decision":      summary["decision"],
            "confidence":    summary["confidence"],
            "flags":         summary["flags"],
            "timeline":      timeline_rows,
            "classification": classification_rows,
            "evidence_coverage": {
                slot: {
                    "filled":     info["filled"],
                    "confidence": round(info["confidence"], 2),
                    "doc_type":   info.get("doc_type"),
                    "page":       info.get("best_page") or info.get("page"),
                }
                for slot, info in result.get("evidence_coverage", {}).items()
            },
            "reviewer_note": summary.get("reviewer_utility", {}).get("decision_summary", ""),
            "reasoning_summary": result.get("reasoning_summary", {}),
            "mode": "live" if ca.OLLAMA_AVAILABLE else "mock",
        }

        return response

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)



# ── Demo UI ───────────────────────────────────────────────────────────────────
@app.get("/")
def ui():
    if _demo_index.exists():
        return FileResponse(_demo_index, media_type="text/html")
    return HTMLResponse("<p>Demo UI not found at demo/index.html</p>", status_code=404)


@app.get("/demo")
@app.get("/demo/")
def demo_redirect():
    return RedirectResponse(url="/", status_code=307)

def main():
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
