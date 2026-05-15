# Solution Flow

ClaimSetu converts a claim packet into structured, evidence-backed review findings.

**ClaimSetu is a hybrid evidence pipeline: OCR reads, E4B structures, 26B reasons, humans decide.**

The pipeline runs **locally**, prioritising explainability, reproducibility, and reviewer control over end-to-end automation.

---

## High-level architecture

```
Input claim packet
        │
        ▼
┌─────────────────────────┐
│ PDF / image ingestion   │  → page-level records + source references
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ PaddleOCR + PyTesseract │  → OCR lines, text, bounding boxes, confidence
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ Gemma 4 E4B (edge layer)│  → cleanup, triage, classify, extract from noisy text
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ Deterministic validators│  → dates, gates, timeline consistency, required docs
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ Gemma 4 26B (reasoning) │  → rules, contradictions, recommendation, explanation
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ PASS / CONDITIONAL /     │  → reviewer pack + provenance
│ REVIEW recommendation    │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ Human reviewer          │  → final adjudication
└─────────────────────────┘
```

### Pipeline stages (code-oriented names)

| Stage | Function | Primary tool |
|-------|----------|--------------|
| Ingestion | `ocr_extract_pages()` | PDF/image intake + PaddleOCR + PyTesseract |
| Edge parse | `edge_parse_page_with_e4b()` | Gemma 4 E4B |
| Validate | `validate_extracted_evidence()` | Deterministic gates |
| Timeline | `build_episode_timeline()` | Validators + structured evidence |
| Reason | `reason_claim_with_26b()` | Gemma 4 26B |
| Output | `generate_reviewer_summary()` | Reviewer pack + provenance |

### Planned configuration

```python
MODEL_CONFIG = {
    "edge_model": "gemma4:e4b",
    "reasoning_model": "gemma4:26b",
    "use_ocr_engines": True,
    "use_31b_audit": False,
}
```

---

## Model-assisted evidence understanding

ClaimSetu **separates document reading from claim reasoning**.

Traditional OCR engines extract text and layout evidence from claim documents. Gemma models then convert this noisy evidence into structured review findings.

This design avoids black-box document reading and keeps every recommendation grounded in inspectable evidence: source document, page number, extracted text, bounding box, confidence, and extraction method.

**Gemma 4 E4B does not replace OCR.** A VLM-style “read the whole page” pass may recover some text but typically provides weaker line-level provenance and is harder to audit. PaddleOCR and PyTesseract supply the auditable substrate; E4B cleans and structures it.

### Tiered model strategy

ClaimSetu uses a **two-model strategy**:

1. **Edge model (Gemma 4 E4B)** — page-level cleanup, document triage, lightweight classification, and structured extraction from noisy OCR. Keeps the pipeline lightweight and deployable on consumer hardware.
2. **Reasoning model (Gemma 4 26B)** — claim-level timeline interpretation, package/STG rule checks, contradiction analysis, and final recommendation with human-readable explanation. Invoked when deeper clinical or administrative reasoning is required.

**Why not Gemma 4 31B?** For this submission, 31B adds complexity without enough scoring upside. The stack optimises for a stable demo, clean repo, repeatable outputs, and fast enough local execution. **26B** is the practical “main brain” — stronger than edge models, more reproducible than 31B.

### Escalation rule

When edge confidence is insufficient, escalate to the reasoning model:

```python
if page_result["confidence"] < 0.75 or page_result["needs_deep_review"]:
    page_result = reason_with_26b(page_context)
```

Classification fallback: **E4B first** when filename/OCR rules are insufficient; **26B** when still uncertain.

---

## Model split by stage

| Stage | Use | Model / tool |
|-------|-----|----------------|
| Raw text extraction | OCR + bbox provenance | **PaddleOCR + PyTesseract** |
| OCR cleanup | Fix noisy text into structured fields | **Gemma 4 E4B** |
| Page triage | Relevant, extra, missing, low-quality? | **Gemma 4 E4B** |
| Document classification fallback | When filename/OCR rules insufficient | **E4B** first → **26B** if uncertain |
| Field extraction | Patient, diagnosis, dates, procedure, labs | **E4B** page-level; **26B** for claim-level conflict resolution |
| Timeline reasoning | Admission → investigation → treatment → discharge | **Deterministic code** + **26B** |
| Rule checks | Package/STG reasoning and explanation | **Gemma 4 26B** |
| Final recommendation | PASS / CONDITIONAL / REVIEW | **Gemma 4 26B** |
| Final audit | Optional deep audit | **Skip 31B** |

---

## 1. Document intake

**Input** may include PDFs, scanned images, photographs, discharge summaries, lab reports, bills, clinical notes, and procedure documents.

Each file becomes **page-level records** with file name, page number, image metadata, and stable source references. Logical documents are grouped across multi-page files while retaining page-level granularity for provenance.

---

## 2. OCR and layout extraction

**PaddleOCR** (primary) and **PyTesseract** (fallback) perform document reading — not Gemma.

Layered strategy:

1. Digital PDF text extraction where native text exists  
2. PaddleOCR for scanned pages and images  
3. PyTesseract when PaddleOCR is unavailable or low-confidence  
4. Page-level lines with bounding boxes and per-line confidence  
5. Confidence-aware text promotion — weak OCR retained for search, not promoted as canonical without corroboration  

Outputs feed the edge model as **OCR lines + text + geometry**, preserving auditability.

---

## 3. Gemma 4 E4B — edge page layer

After OCR, **Gemma 4 E4B** runs on page context (noisy text + layout signals). It does **not** re-OCR the document.

| Task | Description |
|------|-------------|
| **OCR cleanup** | Normalise garbled lines into cleaner field candidates |
| **Page triage** | Flag relevant, extra, missing, or low-quality pages |
| **Lightweight classification** | Fallback when Tier 1–2 deterministic signals are insufficient |
| **Structured extraction** | Patient, diagnosis, dates, procedure, lab values from noisy text |

Deterministic tiers still run **before** E4B where possible:

| Tier | Signal | When |
|------|--------|------|
| **1** | Filename / metadata hints | High-confidence hospital naming patterns |
| **2** | OCR keyword signals | Text available, filename ambiguous |
| **3** | **E4B** (then **26B** if uncertain) | Insufficient deterministic confidence |

---

## 4. Field extraction and EvidenceAtom provenance

Extracted fields include patient name, admission/discharge dates, diagnosis, procedure, lab values, billed amount, package evidence, and identifiers where visible.

Every field is an **evidence atom** with source document, page, source text, bounding box (from OCR), extraction method, and confidence.

**Date acceptance gate** (deterministic — not delegated to Gemma):

- labelled date field only  
- grounding in source text or trusted method  
- minimum confidence threshold  
- reject weak, contradictory, or unanchored values  

Rejected gates → **unverifiable** slots, supporting safe **CONDITIONAL** outcomes.

---

## 5. Visual evidence detection

Computer-vision detection (e.g. stamps, signatures, QR/barcodes, stickers) runs alongside OCR. Visual hits are **supporting signals** with page/bbox provenance — not standalone proof.

---

## 6. Deterministic validators

Before claim-level reasoning, **deterministic code** enforces safety:

- date normalisation  
- confidence gates and source-text checks  
- timeline consistency (e.g. discharge before admission → invalid)  
- required-document presence  
- rule severity mapping  

**Gemma explains and reasons over evidence; validators enforce what may proceed.**

---

## 7. Episode timeline construction

Chronological episode: admission → investigation → procedure/treatment → post-treatment monitoring → discharge.

- Dates accepted only when passing the acceptance gate  
- Priority-ordered document types per event  
- OCR anchor search on labelled lines before model-assisted fields  
- No “any date on page” fallback  

Timeline rows carry validity states: **Valid**, **Invalid**, **Unverifiable**, **Conditional**, **Advisory**.

Claim-level timeline interpretation and contradiction surfacing use **Gemma 4 26B** on top of validator output.

---

## 8. Gemma 4 26B — claim reasoning layer

**Gemma 4 26B** handles work that needs broader context:

- package / STG rule interpretation  
- cross-document contradiction analysis  
- timeline reasoning across the full packet  
- **PASS / CONDITIONAL / REVIEW** recommendation  
- human-readable explanation for reviewers  

Each rule finding includes status (pass / not_met / conditional / advisory), reason, evidence links, and confidence.

---

## 9. Decision generation

| Decision | When used |
|----------|-----------|
| **PASS** | Evidence sufficient; no major contradictions |
| **CONDITIONAL** | May be valid but needs clarification or missing evidence |
| **REVIEW** | Major mismatch, contradiction, or unsupported claim |

**Reviewer pack:** summary decision, top reasons, missing evidence, episode timeline, document classification table, reviewer note, provenance links.

---

## 10. Human-in-the-loop design

ClaimSetu does **not** replace adjudicators. Reviewers see what was submitted, what was extracted, what rules ran, why the claim was flagged, and what clarification is needed.

**Demo narrative:** ClaimSetu uses OCR for traceable document reading, an edge model for fast page-level understanding, and a stronger reasoning model for claim-level review. The system never makes a black-box decision — every recommendation links back to extracted evidence.

---

## Design principles

| Principle | Implementation |
|-----------|----------------|
| **OCR reads, Gemma understands** | PaddleOCR + PyTesseract for provenance; E4B/26B for structure and reasoning |
| **Deterministic safety** | Gates and validators before and alongside model outputs |
| **Tiered cost** | Filename/keywords → E4B → 26B escalation only when needed |
| **No 31B** | Skipped for reproducibility and demo stability |
| **Provenance by default** | EvidenceAtom on every material field |
| **Reviewer sovereignty** | PASS / CONDITIONAL / REVIEW as recommendations only |
