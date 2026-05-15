# ClaimSetu

ClaimSetu is an offline, explainable claim-review assistant for public health insurance workflows.

It reads mixed-quality healthcare claim documents, extracts structured evidence, builds an episode timeline, checks package-level rules, and generates a reviewer-ready recommendation with provenance.

ClaimSetu is designed for human-in-the-loop review. It does not make autonomous medical, diagnostic, or payment decisions.

**Architecture in one line:** ClaimSetu is a hybrid evidence pipeline — **OCR reads, E4B structures, 26B reasons, humans decide.**

---

## What it does

For each claim packet, ClaimSetu helps reviewers answer:

1. **Is the packet complete?** — required document types and visual evidence surfaced with gaps flagged  
2. **Does the timeline make sense?** — admission → investigation → procedure → monitoring → discharge, with temporal checks  
3. **Is there enough evidence for a safe recommendation?** — rule-level findings linked to source pages and fields  

Outputs include a **PASS / CONDITIONAL / REVIEW** recommendation, prioritised reasons, missing-evidence list, document classification table, episode timeline, and provenance links—not a final payment or clinical ruling.

---

## Why hybrid OCR + Gemma?

ClaimSetu does **not** use a language model as a black-box OCR engine.

Healthcare claim review requires traceability: source document, page number, extracted text, bounding boxes, confidence, and evidence provenance.

Therefore, ClaimSetu uses **PaddleOCR** and **PyTesseract** for document reading, then uses **Gemma 4** models for understanding and reasoning:

| Layer | Role | Tool / model |
|-------|------|----------------|
| **Document reading** | Raw text, lines, bounding boxes, confidence | PaddleOCR + PyTesseract |
| **Edge understanding** | OCR cleanup, page triage, lightweight classification, structured extraction from noisy text | **Gemma 4 E4B** |
| **Claim reasoning** | Timeline interpretation, package/STG rules, contradictions, reviewer recommendation | **Gemma 4 26B** |

**OCR engines read the document. Gemma understands the claim.**

Gemma 4 E4B is **not** a replacement for the OCR pipeline. It is an edge-friendly layer on top of OCR for cleanup and structuring. Gemma 4 31B is intentionally **not** used — it adds complexity without enough scoring upside for a stable, reproducible demo.

---

## How it works

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

## Gemma 4 Good Hackathon

Built for the [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon) on Kaggle—focused on local, transparent AI that supports equitable access to publicly funded healthcare.

**Design goals:** stable demo · clean repo · repeatable outputs · fast enough local execution · clear explanation of why Gemma adds value on top of traceable OCR.

**Model stack (locked):**

| Use | Choice |
|-----|--------|
| OCR + bbox provenance | PaddleOCR + PyTesseract |
| Edge page layer | Gemma 4 E4B |
| Claim reasoning layer | Gemma 4 26B |
| Final audit | Skip 31B |

---

## Repository

| File | Description |
|------|-------------|
| [problemStatement.md](problemStatement.md) | Problem, gap, objective, and safety framing |
| [solutionFlow.md](solutionFlow.md) | Hybrid OCR + Gemma architecture and pipeline stages |
| [claimsAssistant.py](claimsAssistant.py) | Claim processing pipeline (implementation) |

---

## Data

This repository does **not** include real patient, hospital, or claim documents. Demo inputs should be user-provided or synthetic. Sample outputs may be included only after removing any personally identifiable or protected health information.

---

## License

See repository license file when published. Competition and model use are subject to [Kaggle](https://www.kaggle.com/competitions/gemma-4-good-hackathon) and [Gemma](https://ai.google.dev/gemma) terms.
