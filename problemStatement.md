# ClaimSetu — Problem Statement

**Competition**: [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon) on Kaggle  
**Domain**: Public health insurance & equitable healthcare access  
**Focus**: Explainable claim-review assistance for messy, real-world hospital documentation

---

## Problem

Public health insurance schemes process millions of hospital claims. Many arrive as mixed-quality PDFs, scans, photographs, discharge summaries, lab reports, bills, and clinical notes.

Review teams must manually answer three questions:

1. **Is the claim packet complete?**
2. **Does the treatment timeline make clinical and administrative sense?**
3. **Is there enough evidence to support a safe claim decision?**

This creates delays, inconsistent reviews, repeated document queries, and avoidable friction between patients, hospitals, and payers—especially in settings where timely reimbursement affects whether care remains accessible.

---

## Why this matters

In low-resource or high-volume settings, claim reviewers often work with:

- blurry scanned documents
- inconsistent document formats and layouts
- missing signatures, stamps, or supporting reports
- unclear admission, procedure, and discharge timelines
- package- or scheme-specific clinical documentation requirements

A single missing report, ambiguous date, or weak visual evidence can delay reimbursement, trigger back-and-forth with hospitals, and increase administrative burden on both sides of the claim—while patients wait for resolution.

---

## Current gap

Most document AI systems extract text, but **claim review requires more than OCR**.

A useful system must:

- **classify** documents in a claim packet (discharge summary, lab report, invoice, imaging note, etc.)
- **extract** key clinical and administrative fields (patient identifiers, diagnosis, procedures, dates, amounts)
- **detect** visual evidence such as stamps, signatures, stickers, and QR/barcodes
- **build** an episode timeline from admission through investigation, procedure, monitoring, and discharge
- **check** package- or scheme-specific rules (eligibility, required diagnostics, length of stay, temporal plausibility)
- **show provenance** for every finding (source document, page, region, confidence)
- **keep the final reviewer in control** of adjudication

---

## Objective

Build a **local, explainable claim-review assistant**—hybrid OCR plus Gemma 4—that converts messy healthcare claim packets into:

- structured evidence (classified documents, extracted fields, detected visual cues)
- timeline checks (ordered events with temporal validity flags)
- rule-level findings (pass, gap, or contradiction against configurable package rules)
- a reviewer-ready **Pass / Conditional / Fail** recommendation with prioritised flags

The system does **not** make autonomous medical or payment decisions. It supports human reviewers by making evidence easier to inspect, verify, and act on.

### Expected capabilities

| Capability | Description |
|------------|-------------|
| **Document intake** | PDFs and images across quality levels; multilingual where relevant |
| **Document reading** | PaddleOCR + PyTesseract for text, lines, and bounding boxes with provenance |
| **Evidence understanding** | Gemma 4 E4B for page cleanup, triage, and extraction; Gemma 4 26B for claim-level reasoning |
| **Structuring** | Normalised schema: patient, diagnosis, procedures, dates, costs |
| **Timeline construction** | Admission → investigation → procedure → monitoring → discharge |
| **Rules & checks** | Configurable package/scheme logic; flag missing or contradictory evidence |
| **Explainability** | Confidence scores and provenance for every output |
| **Decision support** | Pass / Conditional (needs more info) / Fail with explainable reasons |

### Illustrative outputs

**Document classification (per claim)**

| Claim ID | File | Page | Document type | Review notes |
|----------|------|------|---------------|--------------|
| CLM-001 | D1.pdf | 1 | Discharge summary | Admission date appears after procedure date |
| CLM-001 | D1.pdf | 2 | Clinical note | — |
| CLM-001 | D1.pdf | 3 | Procedure report | — |
| CLM-001 | D2.pdf | 1 | Imaging report | — |
| CLM-001 | D3.png | 1 | Patient photograph | Not required for claim validation |

**Episode timeline**

| Sequence | Event type | Date | Source document | Temporal check |
|----------|------------|------|-----------------|----------------|
| 1 | Admission | 02-Feb-26 | Discharge summary | Valid |
| 2 | Diagnostic investigation | 02-Feb-26 | Imaging report | Before procedure |
| 3 | Procedure (package) | 03-Feb-26 | Procedure report | Valid |
| 4 | Post-procedure monitoring | 04–05-Feb-26 | Clinical notes | Valid |
| 5 | Discharge | 06-Feb-26 | Discharge summary | After treatment |

---

## Safety and human oversight

The system is designed as a **reviewer co-pilot**, not an autonomous adjudicator.

- It does **not** diagnose patients, interpret radiology images for clinical conclusions, or issue final payment decisions.
- All outputs are **evidence-backed recommendations** intended for human review.
- When evidence is missing, weak, contradictory, or low-confidence, the system **escalates** the claim as **Conditional** or **Needs Review** instead of forcing a Pass or Fail.

Reviewers remain accountable for every claim decision. The assistant’s role is to surface structured evidence, highlight gaps, and reduce manual drudgery—not to replace professional judgment or scheme governance.

---

## Design constraints (Gemma 4 Good)

- Run **locally** on consumer hardware where possible; avoid reliance on external APIs for core inference.
- Use a **hybrid evidence pipeline**: **PaddleOCR + PyTesseract** for traceable document reading (text, lines, bounding boxes, confidence); **Gemma 4 E4B** for edge-friendly page cleanup, triage, and structured extraction; **Gemma 4 26B** for claim-level reasoning, package rules, and reviewer recommendations.
- Do **not** use a language model as a black-box OCR replacement — claim review requires auditable provenance at the line level.
- Do **not** rely on **Gemma 4 31B** for this submission — prioritise stable demo, reproducible outputs, and practical local execution.
- Prioritise **transparency**: every recommendation traceable to source documents, OCR evidence, and rule checks.
- Optimise for **public good**: faster, fairer, more consistent claim review that reduces friction for patients and providers in publicly funded insurance programmes.

**Architecture summary:** OCR reads → E4B structures → 26B reasons → humans decide.
