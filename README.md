# ClaimSetu

ClaimSetu is an offline, explainable claim-review assistant for public health insurance workflows.

It reads mixed-quality healthcare claim documents, extracts structured evidence, builds an episode timeline, checks package-level rules, and generates a reviewer-ready recommendation with provenance.

ClaimSetu is designed for human-in-the-loop review. It does not make autonomous medical, diagnostic, or payment decisions.

---

## What it does

For each claim packet, ClaimSetu helps reviewers answer:

1. **Is the packet complete?** — required document types and visual evidence surfaced with gaps flagged  
2. **Does the timeline make sense?** — admission → investigation → procedure → monitoring → discharge, with temporal checks  
3. **Is there enough evidence for a safe recommendation?** — rule-level findings linked to source pages and fields  

Outputs include a **Pass / Conditional / Fail** recommendation, prioritised reasons, missing-evidence list, document classification table, episode timeline, and provenance links—not a final payment or clinical ruling.

---

## How it works

The pipeline runs locally and is documented in [solutionFlow.md](solutionFlow.md):

| Stage | Purpose |
|-------|---------|
| Document intake | Normalise PDFs, scans, and images into page-level records |
| OCR & layout | Digital text, OCR, line boxes, confidence-aware promotion |
| Classification | Tiered doc-type labelling (metadata → keywords → model) |
| Field extraction | Structured fields with **EvidenceAtom** provenance |
| Visual evidence | Stamps, signatures, QR/barcodes, stickers (supporting signals) |
| Timeline | Chronological episode with date acceptance gates |
| Rules | Package/scheme checks with pass / fail / conditional / advisory |
| Decision | Reviewer pack for human adjudication |

Primary implementation: [`claimsAssistant.py`](claimsAssistant.py).

Problem framing and safety context: [problemStatement.md](problemStatement.md).

---

## Safety and human oversight

ClaimSetu is a **reviewer co-pilot**, not an autonomous adjudicator.

- Recommendations are evidence-backed and intended for human verification.  
- Weak, missing, or contradictory evidence escalates to **Conditional** or **Needs Review** rather than forcing Pass or Fail.  
- The system does not diagnose patients, interpret imaging for clinical conclusions, or issue final payment decisions.

---

## Gemma 4 Good Hackathon

Built for the [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon) on Kaggle—focused on local, transparent AI that supports equitable access to publicly funded healthcare.

**Design goals:** run on consumer hardware where possible · use **Gemma** (including multimodal variants) for classification and extraction · expose provenance on every material finding · optimise for public-good claim review, not black-box automation.

---

## Repository

| File | Description |
|------|-------------|
| [problemStatement.md](problemStatement.md) | Problem, gap, objective, and safety framing |
| [solutionFlow.md](solutionFlow.md) | Product architecture and pipeline stages |
| [claimsAssistant.py](claimsAssistant.py) | Claim processing pipeline |

---

## License

See repository license file when published. Competition and model use are subject to [Kaggle](https://www.kaggle.com/competitions/gemma-4-good-hackathon) and [Gemma](https://ai.google.dev/gemma) terms.
