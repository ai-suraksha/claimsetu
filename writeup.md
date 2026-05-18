# ClaimSetu — Kaggle Writeup (Gemma 4 Good Hackathon)

Submission guide and paste-ready copy for [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon).

---

## Submission metadata

| Field | Value |
|-------|--------|
| **Title** | ClaimSetu: Evidence-Backed Claim Review for Public Health Insurance |
| **Title character count** | 68 / 80 |
| **Subtitle** | A local, human-in-the-loop assistant that turns messy hospital claim packets into transparent review recommendations. |
| **Subtitle character count** | 113 / 140 |
| **Writeup URL slug** | `claimsetu-evidence-backed-claim-review` |
| **Full writeup URL** | `https://www.kaggle.com/competitions/gemma-4-good-hackathon/writeups/claimsetu-evidence-backed-claim-review` |
| **Submission tracks** | Main Track, Impact Track, Special Technology Track |
| **Impact Track category** | Health & Sciences |
| **Safety & Trust alignment** | Mention in description, but do not select as the primary Impact category unless Kaggle forces one prize category only and Health & Sciences is unavailable |
| **Special Technology category** | Ollama — the demo UI and `/health` endpoint explicitly surface the local Gemma 4 model stack |

**Final selection on the form:** Main Track · Impact Track → **Health & Sciences** · Special Technology Track → **Ollama**

Safety & Trust is woven into the project description as a credibility layer but is **not** selected as the primary Impact category — Health & Sciences is the right primary impact story.

---

## Card and thumbnail (560 × 280)

**Suggested thumbnail text**

> **ClaimSetu**  
> Evidence-backed claim review for public health insurance

**Suggested layout**

| Left | Middle | Right |
|------|--------|-------|
| Messy Claim Packet | OCR + Gemma Reasoning | **PASS** / **CONDITIONAL** / **REVIEW** |
| PDFs • Scans • Bills • Reports | | with Evidence |

**Footer:** Human-in-the-loop • Local-first • Privacy-aware

Keep it clean. Avoid patient images or real document screenshots.

---

## Media gallery (upload order)

### 1. YouTube demo video

- **Title:** ClaimSetu Demo: Evidence-Backed Claim Review with Gemma  
- **Length:** max 3 minutes  
- **URL:** `https://www.youtube.com/watch?v=<your-video-id>`

### 2. Architecture screenshot

- **Image title:** ClaimSetu Architecture  
- **Caption:** OCR reads the documents, Gemma structures and reasons over evidence, deterministic validators enforce safety, and the final recommendation stays human-reviewed.

### 3. Output screenshot

- **Image title:** Reviewer Recommendation  
- **Caption:** ClaimSetu produces a PASS / CONDITIONAL / REVIEW recommendation with timeline, rule checks, missing evidence, and provenance.

### 4. Timeline screenshot

- **Image title:** Episode Timeline  
- **Caption:** Admission, investigation, procedure, post-treatment, and discharge events are extracted and checked for temporal plausibility.

---

## Project description

*Paste the section below into the Kaggle **Project Description** box.*

---

### ClaimSetu: Evidence-Backed Claim Review for Public Health Insurance

Public health insurance claim teams often review messy hospital claim packets: scanned discharge summaries, lab reports, bills, clinical notes, procedure documents, and photographed records. Reviewers must determine whether the claim packet is complete, whether the treatment timeline is plausible, and whether the submitted evidence supports a safe review decision.

This process is often manual, slow, inconsistent, and difficult to audit. Claim delays can happen not because treatment is invalid, but because evidence is incomplete, dates are unclear, or documents are hard to verify.

**ClaimSetu** is a local, human-in-the-loop claim-review assistant that converts messy healthcare claim packets into structured, evidence-backed review findings.

It produces:

- document classification
- key field extraction
- episode timeline
- missing evidence checks
- package/STG-style rule findings
- contradiction flags
- reviewer-ready **PASS / CONDITIONAL / REVIEW** recommendation
- evidence provenance for every major finding

ClaimSetu is **not** an autonomous adjudicator. It does not diagnose patients, interpret radiology images for clinical verdicts, or approve or reject claims by itself. It helps human reviewers inspect evidence faster and more consistently.

#### How it works

ClaimSetu uses a hybrid deterministic + Gemma pipeline.

1. **Document intake** — Claim packets are ingested as PDFs, scans, images, or text-like files.

2. **OCR and layout extraction** — PaddleOCR and PyTesseract extract text, lines, and layout evidence. This preserves traceability: source document, page number, extracted text, and bounding boxes where available.

3. **Gemma 4 E4B page understanding** — The edge model performs OCR cleanup, page triage, lightweight document classification, and structured field extraction from noisy document text.

4. **Deterministic validation** — Rule-based validators normalize dates, reject weak or hallucinated extractions, check timeline consistency, and detect missing mandatory evidence.

5. **Gemma 4 26B claim reasoning** — The reasoning model performs claim-level interpretation: timeline reasoning, package-rule analysis, contradiction detection, and reviewer-ready explanation generation.

6. **Human-in-the-loop recommendation** — The final output is **PASS**, **CONDITIONAL**, or **REVIEW**, with reasons and provenance. Weak, missing, contradictory, or low-confidence evidence is escalated instead of forcing a decision.

#### Why Gemma

ClaimSetu does **not** use Gemma as a black-box OCR replacement. Healthcare claim review requires traceability and auditability. Traditional OCR engines read documents and preserve evidence provenance. Gemma is used where reasoning matters most: structuring noisy evidence, interpreting timelines, checking package-style rules, and generating reviewer-ready explanations.

| Model | Role |
|-------|------|
| **Gemma 4 E4B** | Edge-friendly page triage, OCR cleanup, classification fallback, structured extraction |
| **Gemma 4 26B** | Claim-level reasoning, timeline interpretation, rule checks, contradiction analysis, final explanation |

This keeps the system practical for privacy-sensitive and low-connectivity environments while using a stronger reasoning model for complex claim review.

**OCR reads the document. Gemma understands the claim. Humans decide.**

#### Safety and trust

ClaimSetu is designed for responsible human-in-the-loop review.

It does **not**:

- make autonomous payment decisions
- diagnose patients
- replace medical officers or adjudicators
- issue final clinical verdicts
- interpret radiology images for medical conclusions

Instead, it makes evidence easier to inspect. Every recommendation is tied back to source evidence: document type, page number, source text, extraction method, and confidence.

When evidence is incomplete, weak, contradictory, or low-confidence, ClaimSetu marks the case as **CONDITIONAL** or **REVIEW**.

#### Data and privacy

This repository does **not** include real patient, hospital, or claim documents.

The system was developed with privacy in mind. Any real or redacted healthcare claim documents used during private development are not redistributed. The public repository includes synthetic examples, schemas, sample outputs, and evaluation scaffolding so the pipeline can be inspected safely.

#### Track alignment

**Main Track:** ClaimSetu demonstrates an end-to-end, real-world AI system for public health insurance claim review, combining document intelligence, local model inference, deterministic validation, and human-in-the-loop recommendations.

**Impact Track — Health & Sciences:** ClaimSetu helps bridge the gap between healthcare data and human reviewers by converting messy hospital claim packets into structured, evidence-backed findings. It can reduce manual review burden, improve consistency, and support faster access to reimbursement.

**Safety & Trust alignment:** The system is designed to stay grounded in source evidence. It does not make autonomous medical or payment decisions. Weak, missing, contradictory, or low-confidence evidence is escalated to **CONDITIONAL** or **REVIEW**.

**Special Technology Track — Ollama:** ClaimSetu runs Gemma 4 locally via Ollama, supporting privacy-aware and low-connectivity deployment patterns for sensitive healthcare workflows.

#### Impact

In high-volume public health insurance workflows, claim reviewers spend significant time verifying document completeness, dates, treatment timelines, and rule evidence. ClaimSetu can reduce manual review burden by turning unstructured claim packets into structured, evidence-backed review notes.

Potential benefits include:

- faster first-pass claim review
- fewer repeated document queries
- more consistent reviewer decisions
- stronger audit trails
- safer escalation of uncertain cases
- privacy-aware local deployment

---

## Shorter project description (fallback)

*Use if the Kaggle box has a tight character limit.*

ClaimSetu is a local, human-in-the-loop claim-review assistant for public health insurance workflows.

It converts messy hospital claim packets into structured, evidence-backed review findings: document classification, key field extraction, episode timeline, missing evidence checks, package/STG-style rule findings, contradiction flags, and a reviewer-ready **PASS / CONDITIONAL / REVIEW** recommendation.

The system uses a hybrid deterministic + Gemma pipeline. PaddleOCR and PyTesseract extract traceable text and layout evidence. Gemma 4 E4B performs edge-friendly page triage, OCR cleanup, document classification fallback, and structured extraction. Gemma 4 26B performs claim-level reasoning, timeline interpretation, rule checks, contradiction analysis, and final explanation generation.

ClaimSetu intentionally does not use Gemma as a black-box OCR replacement. Healthcare claim review requires auditability, provenance, and human oversight. Every major recommendation links back to source evidence: document type, page number, source text, extraction method, and confidence.

ClaimSetu does not diagnose patients, interpret radiology images for clinical verdicts, or autonomously approve or reject claims. When evidence is weak, missing, contradictory, or low-confidence, it escalates the case to **CONDITIONAL** or **REVIEW**.

The public repository does not include real patient, hospital, or claim documents. It includes synthetic examples, schemas, sample outputs, and evaluation scaffolding so the pipeline can be inspected safely.

**OCR reads the document. Gemma understands the claim. Humans decide.**

---

## Attachments and project links

| # | Title | URL | Description (if prompted) |
|---|-------|-----|---------------------------|
| 1 | GitHub Repository | `https://github.com/<your-username>/claimsetu` | Public code repository with implementation, synthetic examples, sample outputs, and documentation. |
| 2 | Demo Video | `https://www.youtube.com/watch?v=<your-video-id>` | 3-minute walkthrough showing the ClaimSetu pipeline and reviewer output. |
| 3 | ClaimSetu Demo Notebook *(optional)* | `https://www.kaggle.com/code/<your-username>/claimsetu-demo` | Notebook walkthrough using synthetic examples. |
| 4 | Live Demo *(optional)* | — | Interactive demo using synthetic claim examples. Only add if available. |

---

## Files to upload

**Recommended (lightweight, public-safe):**

- `architecture.png`
- `sample_output_syn_001.json`
- `claimsetu_screenshots.zip`

**Do not upload:**

- real claim packets
- private evaluation files
- raw hospital PDFs or images
- patient-like records

---

## Final submission checklist

- [ ] **Title:** ClaimSetu: Evidence-Backed Claim Review for Public Health Insurance  
- [ ] **Subtitle:** A local, human-in-the-loop assistant that turns messy hospital claim packets into transparent review recommendations.  
- [ ] **Slug:** `claimsetu-evidence-backed-claim-review`  
- [ ] **Tracks selected:** Main Track + Impact Track + Special Technology Track  
- [ ] **Impact category:** Health & Sciences  
- [ ] **Special Technology category:** Ollama  
- [ ] **Repo:** `https://github.com/<your-username>/claimsetu` (replace placeholder)
- [ ] **Video:** YouTube 3-minute demo (replace `<your-video-id>` placeholder)
- [ ] **Demo UI:** `uv run uvicorn app:app --reload` → record at http://localhost:8000  
- [ ] **Files:** `architecture.png`, sample output JSON, screenshots only  
- [ ] Replace all `<your-username>` and `<your-video-id>` placeholders before submitting
