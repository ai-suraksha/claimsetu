# NHA Hackathon — Problem Statement 01

**Source**: https://nha.gov.in/hackathon | https://nha.gov.in/problemStatement1
**Domain**: PM-JAY Claim Adjudication
**Submission Format**: Python Notebook (NHA Sandbox Environment)

---

## Title

Automatically read mixed-quality healthcare documents, extract key data, detect mandatory visual elements (stamps/signatures), check compliance with Standard Treatment Guidelines (STG) provided by NHA per package code, and produce an explainable **Pass / Conditional / Fail** decision with reasons.

---

## Core Problem Statement

Hospitals submit many documents (scans, photos, PDFs) for claims. These vary in language and quality. The system must:

- **Read** the documents — even if blurry or in different languages
- **Extract** important fields: patient name, diagnosis, procedures, amounts
- **Detect** visual cues: hospital stamp, doctor signature, implant stickers, QR/barcodes
- **Check** the claim against STG rules: eligibility, diagnostics, length of stay, package logic
- **Explain** the decision with confidence scores and evidence provenance (page number, bounding box, source doc ID)
- **Construct an Episode Timeline**: extract admission, investigation, procedure, and discharge dates; order chronologically; check temporal plausibility
- **Identify** extra or non-required documents submitted with the claim

---

## Expected Solution

| Component | Description |
|-----------|-------------|
| **Document Intake** | PDFs/images (low to high quality), multilingual |
| **OCR & Layout Understanding** | Extract text + structure (sections, tables, line items) |
| **Visual Element Detection** | Stamps, signatures, implant stickers, QR/barcodes, invoice line items |
| **Data Structuring** | Standard schema: patient, diagnosis, procedures, dates, costs |
| **Rules Engine** | Encode STG/policy checks: eligibility, contraindications, length of stay, package logic |
| **Explainability** | Confidence scores + provenance: page number, bounding box, source doc ID |
| **Decisioning** | Pass / Conditional (needs more info) / Fail with prioritised flags |

---

## Sample Output — Document Classification Table

| Claim ID | File | Page | Document Classification | Clinical Rules Checks |
|----------|------|------|------------------------|-----------------------|
| CN001 | D1.pdf | 1 | Discharge Summary | Admission date is after surgery date \|\| Clinical rules as per STG are not matching |
| CN001 | D1.pdf | 2 | Clinical Note | — |
| CN001 | D1.pdf | 3 | Angioplasty Report | — |
| CN001 | D2.pdf | 1 | X-Ray | — |
| CN001 | D3.pdf | 1 | CT Scan | — |
| CN001 | d4.png | 1 | Patient Photo | — |
| CN001 | d4.png | 1 | Angioplasty | Not required for clinical or claim validation |

---

## Sample Output — Episode Timeline

| Sequence | Event Type | Date | Source Document | Temporal Validity |
|----------|-----------|------|----------------|-------------------|
| 1 | Admission | 02-Feb-26 | Discharge Summary | Valid |
| 2 | Diagnostic Investigation | 02-Feb-26 | CT Scan / X-Ray | Before procedure |
| 3 | Procedure (Package) | 03-Feb-26 | Angioplasty Report | Valid |
| 4 | Post-Procedure Monitoring | 04–05-Feb-26 | Clinical Notes | Valid |
| 5 | Discharge | 06-Feb-26 | Discharge Summary | After treatment |

---

## Scoring

Minimum qualifying score: **≥ 70%**. Awards for top 3 teams.

| Category | Weightage | Rank 1 | Rank 2 | Rank 3 |
|----------|-----------|--------|--------|--------|
| Document Classification | 40% | F1 ≥ 0.95 | 0.90 ≤ F1 < 0.95 | 0.85 ≤ F1 < 0.90 |
| Rule Creation Logic & Provenance Detection | 40% | F1 ≥ 0.96 | 0.90 ≤ F1 < 0.96 | 0.85 ≤ F1 < 0.90 |
| Solution Design | 20% | F1 ≥ 0.93 | 0.88 ≤ F1 < 0.93 | 0.80 ≤ F1 < 0.88 |
