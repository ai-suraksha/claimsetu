# Solution Flow

ClaimSetu converts a claim packet into structured, evidence-backed review findings. The pipeline is designed to run **locally** with **Gemma** as the primary model family, prioritising explainability and reviewer control over end-to-end automation.

---

## High-level architecture

```
Claim packet (PDFs, scans, images)
        │
        ▼
┌───────────────────┐
│ 1. Document intake │  → page-level records + source references
└─────────┬─────────┘
          ▼
┌───────────────────────────┐
│ 2. OCR & layout extraction │  → text, lines, bounding boxes, confidence
└─────────┬─────────────────┘
          ▼
┌────────────────────────┐
│ 3. Document classification │  → tiered doc-type labels per page
└─────────┬──────────────┘
          ▼
┌────────────────────┐
│ 4. Field extraction   │  → structured fields + EvidenceAtom provenance
└─────────┬──────────┘
          ▼
┌─────────────────────────────┐
│ 5. Visual evidence detection │  → stamps, signatures, QR, stickers
└─────────┬───────────────────┘
          ▼
┌──────────────────────────────┐
│ 6. Episode timeline construction │  → ordered events + temporal checks
└─────────┬────────────────────┘
          ▼
┌────────────────────────┐
│ 7. Rule & package checks │  → pass / fail / conditional / advisory
└─────────┬──────────────┘
          ▼
┌────────────────────┐
│ 8. Decision generation │  → Pass / Conditional / Fail + reviewer pack
└─────────┬──────────┘
          ▼
┌─────────────────────────┐
│ 9. Human-in-the-loop UI  │  → reviewer inspects, verifies, decides
└─────────────────────────┘
```

---

## 1. Document intake

**Input** may include:

- PDFs
- scanned images
- photographed documents
- discharge summaries
- lab reports
- bills
- clinical notes
- procedure documents

Each file is normalised into **page-level records** with:

| Field | Purpose |
|-------|---------|
| file name | Traceability to original upload |
| page number | Pinpoint evidence within multi-page files |
| extracted text | Digital text or promoted OCR text |
| OCR lines | Line-level text with geometry for anchoring |
| image metadata | Dimensions, scan quality hints |
| source references | Stable IDs for provenance links |

Logical documents are grouped when multiple pages belong to the same file or clinical artifact, so downstream steps operate on coherent units while retaining page-level granularity.

---

## 2. OCR and layout extraction

The system uses a **layered extraction strategy**:

1. **Digital PDF text extraction** where native text is available
2. **OCR** for scanned pages and camera photographs
3. **Page-level line extraction** with bounding boxes and per-line confidence
4. **Confidence-aware text promotion** — low-confidence OCR is retained for search but not promoted as canonical without corroboration

This allows the pipeline to handle both clean PDFs and low-quality scans without a single extraction path for every input type.

---

## 3. Document classification

Each page is classified into claim-document categories such as:

- discharge summary
- admission note / indoor case sheet
- investigation report
- lab report
- clinical note
- bill or administrative document
- procedure note
- extra / non-required document

### Tiered classification

To balance accuracy and efficiency, classification runs in tiers:

| Tier | Signal source | When used |
|------|---------------|-----------|
| **1** | Filename and metadata hints | High-confidence patterns (e.g. discharge, lab, bill tokens) |
| **2** | OCR / digital-text keyword signals | Text available but filename ambiguous |
| **3** | Model-assisted classification | Deterministic tiers insufficient; Gemma multimodal classify+extract |

Lower tiers run first to **reduce model calls** while preserving accuracy on typical hospital naming conventions. Image-only pages classified at Tier 1 still receive **model-assisted field extraction** when text is not available from the PDF layer.

---

## 4. Field extraction

For each document type, the system extracts relevant fields, for example:

- patient name
- admission date
- discharge date
- diagnosis
- procedure
- lab values (e.g. haemoglobin where visible)
- billed amount
- package-related evidence
- doctor / hospital identifiers where visible

### EvidenceAtom provenance

Every extracted field is stored as an **evidence atom** with:

- source document
- page number
- source text snippet
- bounding box (when available)
- extraction method (digital text, OCR anchor, model-assisted, etc.)
- confidence score

### Date acceptance gate

Dates are high-risk fields. Before a date enters the timeline or rules engine, it must pass an **acceptance gate**:

- labelled date field (not free-floating “any date on page”)
- grounding in source text or trusted extraction method
- minimum confidence threshold
- rejection of weak, contradictory, or unanchored values

Failed gates yield **unverifiable** slots rather than invented dates—supporting safe **Conditional** outcomes when required dates are missing.

---

## 5. Visual evidence detection

The system detects claim-supporting **visual elements**:

- hospital stamp
- doctor signature
- QR / barcode
- implant or device sticker (where applicable)
- presence of expected report types as visual artifacts

These signals are **supporting evidence**, not standalone proof. They are attached to the claim review pack with the same provenance model as text fields so reviewers can verify what was detected and where.

---

## 6. Episode timeline construction

The system builds a **chronological treatment timeline**:

1. admission  
2. investigation  
3. procedure / treatment  
4. post-treatment monitoring  
5. discharge  

### Date sourcing

For each event type, dates are drawn from **priority-ordered document types** (e.g. discharge summary for admission/discharge; procedure report for intervention date). Multiple extraction paths may be attempted in order:

1. OCR **anchor search** on labelled lines (exact → substring → fuzzy, with context window)
2. Model-assisted **labelled field** extraction with source text
3. Regex on reliable digital text layers
4. Cross-field facts when already validated elsewhere

Free-floating “any date visible on the page” is **not** used as a timeline fallback, to avoid report dates or chart noise polluting the episode.

### Temporal validity

Each timeline row carries a validity state, for example:

| State | Meaning |
|-------|---------|
| **Valid** | Accepted date, no contradiction |
| **Invalid** | Logical contradiction (e.g. discharge before admission) |
| **Unverifiable** | No acceptable date extracted |
| **Conditional** | Required event date missing for package |
| **Advisory** | Suspicious pattern (e.g. implausible range) flagged for review |

Weak or ungrounded dates are **rejected or marked unverifiable** rather than forced into the timeline.

---

## 7. Rule and package checks

A **rules engine** evaluates the claim against **package-specific** (or scheme-specific) requirements.

Examples of checks:

- required documents present or missing
- length of stay within expected range
- diagnosis / procedure consistency
- required investigation evidence available
- post-treatment evidence present where needed
- extra documents identified but not over-weighted in the decision

Each rule produces:

| Output | Description |
|--------|-------------|
| status | pass / fail / conditional / advisory |
| reason | Human-readable explanation |
| evidence source | Linked atoms and documents |
| confidence | Strength of supporting evidence |

Rules consume structured fields and timeline states—not raw OCR alone—so findings remain auditable.

---

## 8. Decision generation

The system produces a **reviewer-ready** outcome:

| Decision | When used |
|----------|-----------|
| **PASS** | Evidence sufficient; no major contradictions |
| **CONDITIONAL** | Claim may be valid but needs clarification or missing evidence |
| **FAIL / REVIEW** | Major rule mismatch, contradiction, or unsupported claim |

### Reviewer pack

The output includes:

- summary decision
- top reasons (prioritised flags)
- missing evidence list
- episode timeline table
- document classification table
- reviewer note (suggested queries to hospital or patient)
- provenance links back to source pages and regions

Internal rule statuses (e.g. advisory, missing slot) roll up into these three reviewer-facing decisions without hiding intermediate detail.

---

## 9. Human-in-the-loop design

ClaimSetu does **not** replace adjudicators.

It helps reviewers quickly see:

- what documents were submitted
- what evidence was extracted
- what rules were checked
- why a claim was flagged
- what clarification is needed

The final decision remains with the **human reviewer**. The assistant’s role is to compress reading time, surface contradictions early, and attach defensible provenance—not to auto-approve or auto-deny payment.

---

## Design principles

| Principle | Implementation |
|-----------|----------------|
| **Tiered cost** | Deterministic classification and OCR before model calls |
| **Provenance by default** | EvidenceAtom on every material field and date |
| **No silent invention** | Acceptance gate and unverifiable states for weak dates |
| **Timeline discipline** | Priority sources + temporal validity, not “any date on page” |
| **Reviewer sovereignty** | Pass / Conditional / Fail as recommendations only |
| **Local-first** | Core pipeline runnable on consumer hardware with Gemma |
