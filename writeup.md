# ClaimSetu: Evidence-Backed Claim Review for Public Health Insurance

Every delayed public health insurance claim is more than paperwork.

It can mean a hospital waiting for reimbursement, a patient facing friction, and a reviewer trying to make a fair decision from incomplete, messy documents.

India's PM-JAY is one of the world's largest public health assurance schemes, covering approximately 55 crore beneficiaries across more than 12 crore families. PM-JAY is also widely reported to process 50,000+ claims per day. At that scale, even small improvements in first-pass claim review can reduce repeated document queries, improve consistency, and speed up reimbursement.

But healthcare claim packets are messy. A single packet may include scanned discharge summaries, handwritten notes, lab reports, bills, procedure documents, photos, stamps, signatures, and missing pages.

Reviewers must answer three questions quickly:

1. Is the claim packet complete?
2. Does the treatment timeline make sense?
3. Is the evidence strong enough for a safe review decision?

Today, much of this work is manual, inconsistent, and difficult to audit.

**ClaimSetu** is a human-in-the-loop claim-review assistant powered by Gemma 4. It turns messy hospital claim packets into a reviewer-ready evidence pack: document classification, evidence coverage, episode timeline, missing-document checks, rule findings, contradiction flags, and a **PASS / CONDITIONAL / REVIEW** recommendation with provenance.

ClaimSetu is not a final verdict system. It does not approve or reject claims. It helps human reviewers inspect evidence faster, more consistently, and with a clearer audit trail.

---

## Demo Outcome

The demo reviews a privacy-safe **Severe Anemia** claim packet against a package-specific treatment guideline.

ClaimSetu returns a **CONDITIONAL** recommendation with **56% confidence**.

It finds:

- admission evidence
- diagnostic evidence
- clinical notes

But it cannot verify:

- treatment details
- post-treatment evidence
- discharge summary

Instead of guessing, ClaimSetu escalates uncertainty. It tells the reviewer exactly what is missing, which rule fields are affected, and which evidence slots were filled.

This is the core behavior: **when evidence is weak, missing, or unverifiable, ClaimSetu does not hallucinate. It asks for review.**

---

## How It Works

ClaimSetu uses a hybrid deterministic + Gemma pipeline.

1. **Document intake**  
   Claim packets are ingested as PDFs, scans, images, or text-like files.

2. **OCR and layout extraction**  
   PaddleOCR and PyTesseract extract text, lines, layout evidence, confidence scores, and bounding boxes where available.

3. **Gemma 4 E4B page understanding**  
   Gemma 4 E4B performs page-level structuring: OCR cleanup, document triage, classification fallback, and field extraction from noisy document text.

4. **Deterministic validation**  
   Rule-based validators normalize dates, reject weak or hallucinated extractions, check timeline consistency, and identify missing mandatory evidence.

5. **Gemma 4 26B reviewer reasoning**  
   Gemma 4 26B generates a concise reviewer-facing explanation grounded in the extracted evidence and deterministic rule findings.

6. **Human-in-the-loop recommendation**  
   The final output is **PASS**, **CONDITIONAL**, or **REVIEW**, with reasons and provenance. The human reviewer remains in control.

---

## Why Gemma 4

ClaimSetu does **not** use Gemma as a black-box OCR replacement.

Healthcare claim review requires auditability: source document, page number, extracted text, confidence, method, and provenance. Traditional OCR engines capture the traceable document evidence. Gemma is used where it adds the most value: structuring noisy evidence, reasoning over claim context, and producing a clear reviewer explanation.

| Model | Role |
|---|---|
| **Gemma 4 E4B** | Page-level cleanup, triage, classification fallback, structured extraction |
| **Gemma 4 26B** | Claim-level reviewer explanation grounded in evidence and rule checks |

The system runs through **Ollama** on a local-network workstation, avoiding external cloud APIs for sensitive claim documents while keeping the demo practical on available hardware.

**OCR captures the evidence. Gemma understands the claim. Humans decide.**

---

## Safety and Trust

ClaimSetu is designed as a reviewer co-pilot, not an autonomous adjudicator.

It does not:

- diagnose patients
- interpret radiology images for clinical conclusions
- replace medical officers or claim adjudicators
- approve or reject claims by itself
- force a PASS when evidence is weak

When evidence is missing, contradictory, low-confidence, or unverifiable, ClaimSetu marks the case as **CONDITIONAL** or **REVIEW**.

Every major finding is tied back to source evidence: document type, page number, extracted field, method, confidence, and rule result.

---

## Data and Privacy

This repository does not include real patient, hospital, or raw claim documents.

The public version includes anonymized sample outputs, schemas, Standard Treatment Guideline references, and a functional demo UI so the pipeline can be inspected safely.

Any real or redacted healthcare claim documents used during private development are not redistributed.

---

## Track Alignment

**Main Track:** ClaimSetu demonstrates a working end-to-end AI system for a real public-health workflow: document ingestion, OCR, Gemma-based structuring, deterministic validation, and reviewer-facing recommendations.

**Impact Track - Health & Sciences:** ClaimSetu targets a high-volume healthcare administration problem where better first-pass review can reduce delays, improve consistency, and support faster reimbursement.

**Special Technology Track - Ollama:** ClaimSetu runs Gemma 4 through Ollama, supporting privacy-aware local or local-network inference for sensitive healthcare documents.

---

## Impact

At public-health scale, claim reviewers spend significant time verifying document completeness, dates, treatment timelines, and rule evidence. ClaimSetu can reduce that burden by converting unstructured claim packets into structured, evidence-backed review notes.

Potential benefits include:

- faster first-pass claim review
- fewer repeated document queries
- more consistent reviewer decisions
- stronger audit trails
- safer escalation of uncertain cases
- privacy-aware local deployment
- better human oversight in high-volume public insurance workflows

ClaimSetu is built for environments where speed matters, but trust matters more.

**Local-first. Evidence-backed. Human-controlled.**

---

## References

- [National Health Authority: About PM-JAY](https://nha.gov.in/PM-JAY.php)
