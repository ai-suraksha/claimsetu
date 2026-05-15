# PS-1 Solution Flow

**Version**: 5.0 (end of thread 1)
**Constraint**: Python Notebook · NHA Sandbox · Gemma-3-12b-it · CPU-only
**NOTE: Nova Lite is NOT an allowed model in the NHA Sandbox.**

---

## 0. What Actually Works (Thread 1 Summary)

After extensive iteration, here is the honest state:

| Component | Status | Notes |
|---|---|---|
| Document classification (Tier 1/2/3) | ✅ Working | Tier 1 covers ~70% of files via filename |
| VLM field extraction on image pages | ✅ Fixed | Was silently skipped for Tier 1 pages |
| PaddleOCR line extraction | ✅ Wired | Preferred over Tesseract; both optional |
| Date anchor search (OCR lines) | ✅ Working | exact→substring→fuzzy, widened window |
| EvidenceAtom provenance | ✅ Working | Every date carries method/bbox/source_text |
| DATE_SOURCE_PRIORITY per event | ✅ Working | Event-specific doc_type ordering |
| Acceptance gate (4 conditions) | ✅ Working | Prevents hallucinated/garbage dates |
| Prompt leakage guard | ✅ Fixed | `15/07/2023` example removed from prompt |
| Temporal validity states | ✅ Working | Valid/Invalid/Unverifiable/CONDITIONAL |
| Rules engine + decision | ✅ Working | PASS/FAIL/ADVISORY/CONDITIONAL/MISSING |
| Date extraction from DIS.jpg | ✅ First working extraction | Confirmed end of thread 1 |
| CV anchor pipeline (Tesseract only) | ❌ Removed | Failed — wiring was the real bug |
| Nova Lite | ❌ Not allowed | NHA Sandbox restriction |
| Full Hb extraction from scanned CBC | ⚠️ Partial | VLM reads printed; handwritten unreliable |

---

## 1. Architecture (v5.0)

```
Page image
    │
    ├─► extract_pages()
    │     ├─ fitz.get_text()         (digital PDFs → text)
    │     └─ ocr_page_lines()        (images + scanned PDFs)
    │           ├─ PaddleOCR PP-OCRv4  (primary, if installed)
    │           └─ Tesseract           (fallback)
    │     → page["_ocr_lines"] = [{text, bbox, confidence, method}]
    │     → page["text"] = OCR text (promotes to Tier 2 for images)
    │
    ├─► classify_and_extract_page()
    │     ├─ Tier 1: filename → doc_type (free)
    │     │    └─ if image/scan: VLM call for extracted_fields  ← KEY FIX
    │     ├─ Tier 2: OCR/fitz keywords (free)
    │     └─ Tier 3: Gemma-3-12b-it full classify+extract (metered)
    │
    ├─► run_visual_detection()  (cv2: stamp, signature, QR, implant)
    │
    ├─► build_output_row()
    │     └─ _ocr_lines from page["_ocr_lines"]  ← KEY FIX
    │
    ├─► group_into_logical_docs()  (collapse multi-page files)
    │
    ├─► rank_rows()
    │     sort: anchor_score → inv_confidence → doc_rank → page_num
    │
    ├─► derive_temporal_facts()
    │     DOA, DOD, LOS, pre/post dates, Hb values, post-tx evidence
    │
    ├─► build_timeline()  [v5.0]
    │     DATE_SOURCE_PRIORITY per event_type
    │     For each priority doc_type:
    │       Path 1: OCR anchor → EvidenceAtom (widened window)
    │       Path 2: VLM labeled field → EvidenceAtom
    │       Path 3: Fitz regex → EvidenceAtom
    │       Path 4: Facts dict → EvidenceAtom
    │       ✗ dates_found: disabled (debug only)
    │     EvidenceAtom.is_accepted() gate (4 conditions)
    │     REQUIRED_TIMELINE_FIELDS → CONDITIONAL if missing
    │
    ├─► score_evidence_coverage()
    │
    ├─► run_rules_engine()
    │
    └─► make_decision()  → APPROVE / QUERY / REJECT / CONDITIONAL
```

---

## 2. The Root Cause That Was Fixed

**Why dates were missing from DIS.jpg and MS_DIS.jpg:**

1. `classify_and_extract_page()` Tier 1 matched `DIS` → `discharge_summary` at conf 0.78
2. Tier 1 returned immediately with `extracted_fields = extract_fields_from_text('')`
3. `text=''` for image files — fitz returns nothing, OCR wasn't wired yet
4. Gemma-3-12b-it was **never called** for these pages
5. `_ocr_lines=[]` — so anchor search also found nothing

**Fix:** For Tier 1 classified pages that are not text-based, the VLM is now called specifically for field extraction. The Tier 1 doc_type is preserved; only `extracted_fields` and `dates_found` are taken from VLM. `method = 'filename+vlm_extract'`.

---

## 3. Date Extraction Detail

### VLM Prompt Schema (structured, no example values)

```json
"date_of_admission": {
  "value": "<date as visible, or null>",
  "source_text": "<nearby text containing label and date>",
  "confidence": 0.92
}
```

No concrete example dates in the prompt — previously `15/07/2023` appeared as an example and was being echoed verbatim by Gemma.

### EvidenceAtom Gate (4 conditions, all must pass)

```
1. field in _LABELED_DATE_FIELDS  (dates_found excluded)
2. source_text exists              (or fitz/OCR method — bypass)
3. raw date appears in source_text (VLM fields only — proves not hallucinated)
4. confidence >= 0.70
```

Below threshold → `'—'` in timeline. Required field blank → `CONDITIONAL`.

### Temporal Validity States

```
Valid (conf=X, anchor: Y)              — accepted, no contradictions
Invalid — DOD before DOA               — logical contradiction
Invalid — post before pre              — logical contradiction
Unverifiable — date not extracted      — nothing found
Unverifiable — weak/ungrounded         — gate rejected
CONDITIONAL — missing required date    — REQUIRED_TIMELINE_FIELDS not met
SUSPICIOUS — <reason>                  — future/pre-2020/date-reuse
MULTIPLE_SOURCES (N pages) — ...       — multi-page packet, canonical noted
```

---

## 4. Classification Tier Detail

### Tier 1 FILENAME_HINTS (key patterns)

| Token | Doc Type |
|---|---|
| `bht`, `admission`, `adm`, `adit` | `indoor_case` |
| `dis`, `dc`, `ms_dis`, `sum`, `summary`, `discharge` | `discharge_summary` |
| `fb`, `n4`, `fbc`, `ptinr`, `cbc`, `hb`, `report` | `cbc_hb_report` |
| `bedside` | `vitals_treatment` |
| `urine` | `investigation_pre` |
| `notes`, `cn` | `clinical_notes` |
| `bill`, `adhar`, `feedback`, `card` | `extra_document` (KNOWN_EXTRAS) |

`_11zon` and `_compressed` suffixes stripped before matching.

### Tier 1 + VLM Extract (image pages only)

After Tier 1 classifies, if `is_text_based=False`:
- VLM called with full `VLM_CLASSIFY_PROMPT`
- `doc_type` from Tier 1 preserved (not overridden)
- `extracted_fields`, `dates_found`, visual signals taken from VLM
- `method = 'filename+vlm_extract'`

---

## 5. Models

| Use | Model | Notes |
|---|---|---|
| Classification + extraction | `Gemma-3-12b-it` | Only vision-capable model allowed |
| Text-only (local Ollama) | `qwen2.5` or similar | Dev/testing only |
| Nova Lite | **NOT ALLOWED** | NHA Sandbox restriction |

---

## 6. What to Refine Next (Thread 2 Priorities)

1. **Hb value extraction from scanned CBC reports** — Gemma reads printed lab values reasonably; handwritten Hb is the gap. Verify on N.jpg, N4.jpg, FB.jpg from the gold standard claim.

2. **Post-treatment evidence slot (MG064A CR-003)** — needs `post_hb_report` page or `hb_post_value` from VLM. Currently inferred from `blood_transfusion` flag.

3. **Model selection experiment** — test Gemma-3-4b-it for focused crops (cheaper, same family). Test Mistral for rules justification text.

4. **Submission eval loop** — run `run_batch()` on full dataset, check NHA scorer F1 on classification and rule provenance. Classification F1 is 40% of score.

5. **Surya OCR** — if PaddleOCR installation fails on sandbox, Surya is a strong alternative (pure Python, no C deps, good on Indian documents).

6. **ALOS validation (CR-005)** — currently marked CONDITIONAL for most claims. When DOA/DOD are now being extracted, LOS will compute and ALOS comparison becomes possible.

---

## 7. Experiments That Failed (do not retry)

| Experiment | Why it failed | Verdict |
|---|---|---|
| CV Tesseract anchor pipeline | OCR text was never flowing — wiring bug, not OCR quality | Removed. The underlying approach (anchor search) works — it just needed PaddleOCR + proper wiring |
| `dates_found` as timeline fallback | Any visible date on page entered timeline — report dates, drug chart dates, etc. | Permanently disabled |
| Full-page VLM as primary date reader | Hallucinated `15/07/2023` from prompt example | Fixed by prompt + acceptance gate |
| `nearby_text` from `dates_found` as anchor | Treated any VLM-returned nearby_text as a real anchor | Removed |
| `normalize_date` returning raw string on failure | Garbage OCR strings like `1/24/44` entered timeline | Fixed: returns `None` |
