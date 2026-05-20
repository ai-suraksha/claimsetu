# ClaimSetu

ClaimSetu is a local-first, evidence-backed claim-review assistant for public health insurance workflows.

OCR captures traceable evidence, Gemma 4 structures and explains it, deterministic validators enforce safety, and human reviewers make the final decision.

It reads mixed-quality healthcare claim documents, extracts structured evidence, builds an episode timeline, checks package-level rules, and generates a reviewer-ready recommendation with provenance.

ClaimSetu is designed for human-in-the-loop review. It does not make autonomous medical, diagnostic, or payment decisions.

**Architecture in one line:** ClaimSetu is a hybrid evidence pipeline — **OCR reads, E4B structures, 26B reasons, humans decide.**

---

## Problem

Public health insurance schemes process millions of hospital claims. Many arrive as mixed-quality PDFs, scans, photographs, discharge summaries, lab reports, bills, and clinical notes.

For each claim packet, ClaimSetu helps reviewers answer:

1. **Is the packet complete?** — required document types and visual evidence surfaced with gaps flagged  
2. **Does the timeline make sense?** — admission → investigation → procedure → monitoring → discharge, with temporal checks  
3. **Is there enough evidence for a safe recommendation?** — rule-level findings linked to source pages and fields  

Outputs include a **PASS / CONDITIONAL / REVIEW** recommendation, prioritised reasons, missing-evidence list, document classification table, episode timeline, and provenance links—not a final payment or clinical ruling.

---

## Demo assets

Static materials for walkthroughs and presentations (also listed in the repository table below):

| Asset | Purpose |
|-------|---------|
| [demo/ClaimSetu_Design.pdf](demo/ClaimSetu_Design.pdf) | Five-slide overview: problem, solution, architecture, demo framing |
| [demo/ClaimSetu_Architecture.png](demo/ClaimSetu_Architecture.png) | Architecture diagram: OCR → E4B → validators → 26B → reviewer |
| [demo/claimsetu_thumbnail.png](demo/claimsetu_thumbnail.png) | Project thumbnail / media asset |
| [demo/index.html](demo/index.html) | Interactive demo UI (served at `/demo` when the app is running) |

**Live demo:** after setup, run `uv run uvicorn app:app --reload` and open [http://localhost:8000/demo](http://localhost:8000/demo). Upload your own claim packet (PDF/images) under `Data/claims-data/` or via the UI.

---

## Setup

**Requires**: Python 3.10+, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com), Tesseract OCR binary.

```bash
# Install dependencies
uv sync

# Pull Gemma 4 models via Ollama
ollama pull gemma4:e4b
ollama pull gemma4:26b

# (Optional) Install Tesseract for OCR fallback
# macOS:  brew install tesseract
# Ubuntu: apt-get install tesseract-ocr

# Start the demo UI
uv run uvicorn app:app --reload
# → Open http://localhost:8000/demo
```

**Run pipeline directly (no UI):**

```bash
uv run python claimsAssistant.py
```

Place claim documents under `Data/claims-data/<PACKAGE_CODE>/<CLAIM_ID>/` (see [Data](#data)).

---

## Architecture

ClaimSetu does **not** use a language model as a black-box OCR engine.

Healthcare claim review requires traceability: source document, page number, extracted text, bounding boxes, confidence, and evidence provenance.

| Layer | Role | Tool / model |
|-------|------|----------------|
| **Document reading** | Raw text, lines, bounding boxes, confidence | PaddleOCR + PyTesseract |
| **Edge understanding** | OCR cleanup, page triage, lightweight classification, structured extraction from noisy text | **Gemma 4 E4B** |
| **Claim reasoning** | Timeline interpretation, package/STG rules, contradictions, reviewer recommendation | **Gemma 4 26B** |

**OCR engines read the document. Gemma understands the claim.**

Gemma 4 E4B is **not** a replacement for the OCR pipeline. It is an edge-friendly layer on top of OCR for cleanup and structuring. The default stack uses **E4B** and **26B** for predictable local execution; Gemma 4 31B is not part of the default configuration.

```
Input claim packet
    ↓
PDF / image ingestion
    ↓
PaddleOCR + PyTesseract  →  OCR lines, text, bounding boxes
    ↓
Gemma 4 E4B  →  cleanup, triage, classification fallback, field extraction
    ↓
Deterministic validators  →  dates, confidence gates, timeline, required docs
    ↓
Gemma 4 26B  →  claim-level reasoning, rules, contradictions, recommendation
    ↓
PASS / CONDITIONAL / REVIEW  +  evidence provenance
    ↓
Human reviewer decides
```

Full pipeline detail: [solutionFlow.md](solutionFlow.md) · Problem and safety framing: [problemStatement.md](problemStatement.md) · Implementation: [`claimsAssistant.py`](claimsAssistant.py).

**Escalation:** when E4B page confidence is below threshold or a page needs deep review, that section is escalated to **26B** — keeping the edge model fast while reserving the reasoning model for harder cases.

**Safety gates:** date acceptance, source-text checks, temporal validity, and missing-document rules stay in **deterministic code**. Gemma explains and reasons over evidence; validators enforce what may enter the timeline and final recommendation.

---

## Safety and human oversight

ClaimSetu is a **reviewer co-pilot**, not an autonomous adjudicator.

- Recommendations are evidence-backed and intended for human verification.  
- Weak, missing, or contradictory evidence escalates to **CONDITIONAL** or **REVIEW** rather than forcing a PASS.  
- The system does not diagnose patients, interpret imaging for clinical conclusions, or issue final payment decisions.

---

## Privacy

- This repository does **not** include real patient, hospital, or claim documents.  
- `Data/` is gitignored; use only user-provided or synthetic inputs locally.  
- Private evaluation data and raw claim documents are not redistributed.

---

## Limitations

- **Prototype scope** — single-file pipeline (`claimsAssistant.py`) optimised for clarity and local iteration, not production scale-out.  
- **Package-specific rules** — STG configs in `knowledgeBase/` cover a fixed set of packages; other schemes need new rule definitions.  
- **OCR-dependent quality** — scanned, handwritten, or low-contrast pages may yield unverifiable slots and more **CONDITIONAL** outcomes.  
- **Not adjudication** — outputs are decision-support only; reviewers and scheme governance remain accountable for every claim decision.

---

## Repository

| File / Folder | Description |
|---|---|
| [problemStatement.md](problemStatement.md) | Problem, gap, objective, and safety framing |
| [solutionFlow.md](solutionFlow.md) | Hybrid OCR + Gemma architecture and pipeline stages |
| [claimsAssistant.py](claimsAssistant.py) | Core claim processing pipeline (OCR → E4B → validators → 26B → reviewer pack) |
| [app.py](app.py) | FastAPI web app + demo UI (`/demo`) |
| [pyproject.toml](pyproject.toml) | uv project config and dependencies |
| [demo/index.html](demo/index.html) | Standalone demo UI for recording (served at `/demo`) |
| [demo/ClaimSetu_Design.pdf](demo/ClaimSetu_Design.pdf) | Five-slide overview deck covering problem, solution, architecture, and demo framing |
| [demo/ClaimSetu_Architecture.png](demo/ClaimSetu_Architecture.png) | Architecture image showing OCR → E4B → validators → 26B → reviewer workflow |
| [demo/claimsetu_thumbnail.png](demo/claimsetu_thumbnail.png) | Project thumbnail / media asset |
| [knowledgeBase/](knowledgeBase/) | Standard Treatment Guidelines (4 STG PDFs — MG064A, SG039C, MG006A, SB039A) |

---

## Evaluation

The pipeline was validated on a private set of 40 real-world health insurance claims across 4 packages (MG064A, SG039C, MG006A, SB039A) with manually verified ground truth.

| Metric | Notes |
|--------|-------|
| Document classification | Tier 1/2 (filename + keyword) correct on the majority of pages; Tier 3 (E4B) handles ambiguous pages |
| Mandatory slot fill rate | 4 of 5 slots filled on the sample claim; pre-treatment evidence the most common gap on scanned packets |
| Date extraction | DOA/DOD extracted where digital text present; correctly marked unverifiable on scanned/handwritten pages |
| Hallucinated-date rejection | Dates contradicting source documents rejected by the 4-condition acceptance gate |

---

## Prototype structure

For prototype reproducibility, the core pipeline is kept in a single file (`claimsAssistant.py`), organised internally by pipeline stage. A production version would split this into modules: ocr, classification, timeline, rules, reasoning.

---

## Data

Claim inputs are expected under:

```
Data/claims-data/<PACKAGE_CODE>/<CLAIM_ID>/*.pdf
```

`Data/` is not committed. Demo inputs should be user-provided or synthetic.

---

## License

This project is released under the [Apache 2.0 License](LICENSE). Model use subject to [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
