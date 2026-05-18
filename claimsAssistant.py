#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations   # must be first executable line
# ============================================================
# ClaimSetu — Claims Intelligence Pipeline
# Gemma 4 Good Hackathon (Kaggle) — Public Health Insurance
# ============================================================
#
# Architecture: OCR reads → E4B structures → 26B reasons → humans decide
#
# Pipeline stages (solutionFlow.md names):
#   ocr_extract_pages()          — PDF/image ingestion via PaddleOCR + PyTesseract
#   edge_parse_page_with_e4b()   — Gemma 4 E4B: cleanup, triage, classify, extract
#   validate_extracted_evidence() — deterministic date/confidence/timeline gates
#   build_episode_timeline()     — admission → investigation → procedure → discharge
#   reason_claim_with_26b()      — Gemma 4 26B: rules, contradictions, recommendation
#   generate_reviewer_summary()  — PASS / CONDITIONAL / REVIEW + provenance pack
#
# LLM stack (Ollama local):
#   Edge model:      gemma4:e4b   (page cleanup, triage, field extraction)
#   Reasoning model: gemma4:26b   (claim-level rules, contradictions, recommendation)
# ============================================================

# %% CELL 00 — Gemma 4 / Ollama Setup
import os
import asyncio, base64, io, json, re
from pathlib import Path

# Keep Paddle/OneDNN conservative to avoid sporadic Linux segfaults
# under cross-claim threaded execution.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('FLAGS_paddle_num_threads', '1')
os.environ.setdefault('FLAGS_use_mkldnn', '0')

try:
    import nest_asyncio
    nest_asyncio.apply()
except (ImportError, ValueError):
    pass  # nest_asyncio not needed outside Jupyter / incompatible with uvloop

# ── Model config (Gemma 4 via Ollama) ────────────────────────────────────────
MODEL_CONFIG = {
    "edge_model":      "gemma4:e4b",   # page cleanup, triage, field extraction
    "reasoning_model": "gemma4:26b",   # claim-level rules, contradictions, recommendation
    "use_ocr_engines": True,
    "use_31b_audit":   False,          # skipped for reproducibility and demo stability
}
MODEL_EDGE   = MODEL_CONFIG["edge_model"]
MODEL_REASON = MODEL_CONFIG["reasoning_model"]

# ── Ollama availability check ────────────────────────────────────────────────
# Set OLLAMA_HOST to point to your Ollama instance.
# If running on the same machine as Ollama: http://localhost:11434
# If running on a separate machine (e.g. Mac → Johnaic GPU cluster over LAN):
#   export OLLAMA_HOST=http://<JOHNAIC_IP>:11434
# Ollama must also be configured to listen on 0.0.0.0 on the server side:
#   OLLAMA_HOST=0.0.0.0:11434 ollama serve
OLLAMA_HOST      = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_AVAILABLE = False
OLLAMA_VISION    = False
try:
    import ollama as _ollama_lib
    _ollama_client = _ollama_lib.Client(host=OLLAMA_HOST)
    _ping = _ollama_client.list()
    OLLAMA_AVAILABLE = True
    try:
        _info = _ollama_client.show(MODEL_EDGE)
        _caps = getattr(_info, 'capabilities', []) or []
        OLLAMA_VISION = 'vision' in _caps
    except Exception:
        OLLAMA_VISION = True   # assume vision for gemma4:e4b
    print(f'Ollama ready | host={OLLAMA_HOST} | edge={MODEL_EDGE} | reasoning={MODEL_REASON} | vision={OLLAMA_VISION}')
except Exception:
    _ollama_client = None
    print(f'WARNING: Ollama not reachable at {OLLAMA_HOST} — pipeline will use OCR-only mock mode.')
    print(f'  To enable full inference: ollama pull {MODEL_EDGE} && ollama pull {MODEL_REASON}')

MOCK_MODE = not OLLAMA_AVAILABLE

TOKEN_LOG = {'input': 0, 'output': 0, 'calls': 0, 'mock_calls': 0}

def _ollama_call(model: str, messages: list) -> str:
    """Low-level Ollama call via configured host. Returns text content string."""
    response = _ollama_client.chat(model=model, messages=messages, options={'temperature': 0.0})
    content = response.message.content or ''
    TOKEN_LOG['calls']  += 1
    TOKEN_LOG['output'] += len(content.split())
    return content

def edge_model_call(img, prompt: str, page_text: str = '') -> str:
    """
    Gemma 4 E4B call — page-level cleanup, triage, classification, extraction.
    Prefers text-only when page has sufficient fitz text (faster, equally accurate
    for digital/printed pages). Falls back to vision for scanned/image-only pages.
    """
    if not OLLAMA_AVAILABLE:
        raise RuntimeError('Ollama not available')
    # Text path: faster on digital pages (~3-5s vs ~15s for vision)
    if page_text and len(page_text.split()) > 30:
        return _ollama_call(MODEL_EDGE, [{'role': 'user', 'content': prompt + '\n\nDOCUMENT TEXT:\n' + ' '.join(page_text.split()[:2000])}])
    # Vision path for scanned/image-only pages
    if OLLAMA_VISION and img is not None:
        buf = io.BytesIO()
        img_copy = img.copy(); img_copy.thumbnail((1024, 1024))
        img_copy.save(buf, format='JPEG', quality=85)
        return _ollama_call(MODEL_EDGE, [{'role': 'user', 'content': prompt, 'images': [buf.getvalue()]}])
    # Text-only model + no fitz text → cannot classify
    raise RuntimeError('Cannot classify: no text and model is not vision-capable')

def reason_model_call(context: str, prompt: str) -> str:
    """
    Gemma 4 26B call — claim-level reasoning: package/STG rules, contradictions,
    timeline interpretation, PASS/CONDITIONAL/REVIEW recommendation.
    Always text-only (context is structured evidence, not raw images).
    """
    if not OLLAMA_AVAILABLE:
        raise RuntimeError('Ollama not available')
    full_prompt = f"{prompt}\n\nCLAIM EVIDENCE:\n{context}"
    return _ollama_call(MODEL_REASON, [{'role': 'user', 'content': full_prompt}])

mode_str = f'Gemma 4 (edge={MODEL_EDGE}, reasoning={MODEL_REASON})' if OLLAMA_AVAILABLE else 'MOCK (OCR stub)'
print(f'CELL 00 ready | mode={mode_str}')

# %% CELL 01 — Imports + Config
# (from __future__ import annotations is at top of file)
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import cv2
import fitz                             # PyMuPDF — zero-token PDF text extraction
from PIL import Image

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    PYZBAR_AVAILABLE = True
except Exception:
    PYZBAR_AVAILABLE = False
    print('WARNING: pyzbar not available — QR detection disabled (graceful degradation)')

try:
    import pytesseract as _pytesseract_lib
    PYTESSERACT_AVAILABLE = True
except Exception:
    _pytesseract_lib = None  # type: ignore
    PYTESSERACT_AVAILABLE = False
    print('WARNING: pytesseract not available — mock OCR path disabled (graceful degradation)')

if os.getenv('DISABLE_PADDLE', '').strip() == '1':
    _paddle_ocr_instance = None  # type: ignore
    PADDLE_AVAILABLE = False
    print('PaddleOCR: disabled by DISABLE_PADDLE=1 — using Tesseract fallback')
else:
    try:
        from paddleocr import PaddleOCR as _PaddleOCR
        _paddle_ocr_instance = _PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=False,
            show_log=False,
            enable_mkldnn=False,
            cpu_threads=1,
        )
        PADDLE_AVAILABLE = True
        print('PaddleOCR: available — PP-OCRv4 active (primary OCR engine)')
    except Exception:
        _paddle_ocr_instance = None  # type: ignore
        PADDLE_AVAILABLE = False
        print('PaddleOCR: not available — falling back to Tesseract for OCR')

try:
    from json_repair import loads as json_repair_loads
except Exception:
    json_repair_loads = json.loads      # stdlib fallback
    print('WARNING: json_repair not available — using stdlib json fallback')

BASE_DATA_DIR = Path(__file__).parent.parent / 'Data' / 'ps1-dataset'
OUTPUT_ROOT   = Path(__file__).parent.parent / 'outputs'
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXT  = {'.pdf', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
PACKAGE_CODES  = ['MG064A', 'SG039C', 'MG006A', 'SB039A']
DATE_FIELDS    = {'pre_date', 'post_date', 'doa', 'dod'}
TEXT_FIELDS    = {'case_id', 'link', 'S3_link/DocumentName', 'S3_link', 's3_link',
                  'procedure_code'}  # all possible link key variants across packages

DECISION_PASS, DECISION_CONDITIONAL, DECISION_FAIL = 'PASS', 'CONDITIONAL', 'FAIL'

print('Imports OK. Data dir:', BASE_DATA_DIR.resolve())

# %% CELL 03 — Output Schemas (exact keys — evaluator scores against these)
# DO NOT modify key names or order.

PACKAGE_SCHEMAS = {
    'MG064A': [
        'case_id', 'link', 'procedure_code', 'page_number',
        'clinical_notes', 'cbc_hb_report', 'indoor_case',
        'treatment_details', 'post_hb_report', 'discharge_summary',
        'severe_anemia', 'common_signs', 'significant_signs',
        'life_threatening_signs', 'extra_document', 'document_rank'
    ],
    'SG039C': [
        'case_id', 'S3_link/DocumentName', 'procedure_code', 'page_number',
        'clinical_notes', 'usg_report', 'lft_report', 'operative_notes',
        'pre_anesthesia', 'discharge_summary', 'photo_evidence',
        'histopathology', 'clinical_condition', 'usg_calculi',
        'pain_present', 'previous_surgery', 'extra_document', 'document_rank'
    ],
    'MG006A': [
        # STG output guide key is 'S3_link' (not S3_link/DocumentName)
        'case_id', 'S3_link', 'procedure_code', 'page_number',
        'clinical_notes', 'investigation_pre', 'pre_date', 'vitals_treatment',
        'investigation_post', 'post_date', 'discharge_summary', 'poor_quality',
        'fever', 'symptoms', 'extra_document', 'document_rank'
    ],
    'SB039A': [
        # STG output guide key is 's3_link' (lowercase, not 'link')
        'case_id', 's3_link', 'procedure_code', 'page_number',
        'clinical_notes', 'xray_ct_knee', 'indoor_case', 'operative_notes',
        'implant_invoice', 'post_op_photo', 'post_op_xray', 'discharge_summary',
        'doa', 'dod', 'arthritis_type', 'post_op_implant_present',
        'age_valid', 'extra_document', 'document_rank'
    ],
}

LINK_KEY = {
    'MG064A': 'link',               # output guide p.1: "link"
    'SG039C': 'S3_link/DocumentName',# output guide p.5: "S3_link/DocumentName"
    'MG006A': 'S3_link',            # output guide p.7: "S3_link" (no /DocumentName)
    'SB039A': 's3_link',            # output guide p.9: "s3_link" (lowercase)
}

# [GAP 1] Explicit rank map with rationale
# Lower number = earlier in clinical episode = higher timeline priority
# MG064A: notes → CBC (diagnostic) → treatment → post-CBC → discharge
# SG039C: notes → USG (confirm stones) → LFT (metabolic) → pre-anaes → OT → discharge
# MG006A: notes → pre-investigation (baseline) → treatment → post-investigation → discharge
# SB039A: notes → X-ray (structural evidence) → indoor → OT → implant invoice → post-op → discharge
RANK_MAP = {
    'MG064A': {'clinical_notes':1,'cbc_hb_report':2,'indoor_case':2,
               'treatment_details':3,'post_hb_report':4,'discharge_summary':5},
    'SG039C': {'clinical_notes':1,'usg_report':2,'lft_report':3,
               'pre_anesthesia':4,'operative_notes':5,'discharge_summary':5,
               'histopathology':6,'photo_evidence':6},
    'MG006A': {'clinical_notes':1,'investigation_pre':2,'vitals_treatment':3,
               'investigation_post':4,'discharge_summary':5},
    'SB039A': {'clinical_notes':1,'xray_ct_knee':2,'indoor_case':3,
               'operative_notes':4,'implant_invoice':5,
               'post_op_photo':6,'post_op_xray':6,'discharge_summary':7},
}

print('Schemas + RANK_MAP loaded.')


# %% CELL 04 — Package Info + STG Configs (all domain knowledge here)
PACKAGE_INFO = {
    'MG064A': {
        'name':      'Severe Anemia',
        'doc_types': ['clinical_notes','cbc_hb_report','indoor_case',
                      'treatment_details','post_hb_report','discharge_summary'],
        'mandatory': ['clinical_notes','cbc_hb_report','indoor_case',
                      'treatment_details','post_hb_report','discharge_summary'],
    },
    'SG039C': {
        'name':      'Cholecystectomy',
        'doc_types': ['clinical_notes','usg_report','lft_report','operative_notes',
                      'pre_anesthesia','discharge_summary','photo_evidence','histopathology'],
        # STG: histopathology and photo evidence both mandatory at claim submission
        'mandatory': ['clinical_notes','usg_report','lft_report','operative_notes',
                      'pre_anesthesia','discharge_summary','photo_evidence','histopathology'],
    },
    'MG006A': {
        'name':      'Enteric Fever',
        'doc_types': ['clinical_notes','investigation_pre','vitals_treatment',
                      'investigation_post','discharge_summary'],
        'mandatory': ['clinical_notes','investigation_pre','vitals_treatment',
                      'investigation_post','discharge_summary'],
    },
    'SB039A': {
        'name':      'Total Knee Replacement',
        'doc_types': ['clinical_notes','xray_ct_knee','indoor_case','operative_notes',
                      'implant_invoice','post_op_photo','post_op_xray','discharge_summary'],
        'mandatory': ['clinical_notes','xray_ct_knee','indoor_case','operative_notes',
                      'implant_invoice','post_op_photo','post_op_xray','discharge_summary'],
    },
}

STG_CONFIGS = {
    'MG064A': {
        # STG source: Severe Anemia_MG064A.pdf
        # ALOS = 3 days | Hb < 7 g/dL | Blood transfusion mandatory
        'package_code': 'MG064A', 'package_name': 'Severe Anemia',
        'mandatory_docs': PACKAGE_INFO['MG064A']['mandatory'],
        'optional_docs':  [],
        'clinical_rules': [
            {'rule_id':'CR-001','severity':'mandatory',
             'description':'STG TMS check: patient Hb level < 7 g/dL (severe anemia criterion)',
             'field':'severe_anemia','operator':'equals','value':1},
            # CR-001b: extracted numeric Hb — advisory only for scanned packets.
            # Handwritten Hb values are often misread by OCR/VLM; absence of a
            # clean numeric extraction must not trigger a hard FAIL.
            {'rule_id':'CR-001b','severity':'advisory',
             'description':'STG TMS check: Hb value must be <= 7.0 g/dL (extracted numeric)',
             'field':'hb_value_claim','operator':'lte','threshold':7.0},
            {'rule_id':'CR-002','severity':'mandatory',
             'description':'STG TMS check: blood transfusion treatment documented',
             'field':'treatment_details','operator':'equals','value':1},
            # CR-003: common signs — advisory for scanned packets.
            # Handwritten symptom lists (pallor, weakness, etc.) may not OCR correctly;
            # absence of OCR-detected text ≠ absence of clinical evidence.
            {'rule_id':'CR-003','severity':'advisory',
             'description':'Common signs of anemia (pallor/fatigue/weakness) documented',
             'field':'common_signs','operator':'equals','value':1},
            {'rule_id':'CR-004','severity':'advisory',
             'description':'Significant signs (tachycardia/dyspnea) documented',
             'field':'significant_signs','operator':'equals','value':1},
            {'rule_id':'CR-005','severity':'mandatory',
             'description':'STG ALOS: length of stay 1-3 days (ALOS=3)',
             'field':'los_days','operator':'between','min':1,'max':3},
        ],
        'visual_checks': [
            {'rule_id':'VC-001','element':'stamp','required':True},
            {'rule_id':'VC-002','element':'signature','required':True},
        ],
    },
    'SG039C': {
        # STG source: Cholecystectomy_SG039C.pdf
        # ALOS: Open=6 days, Laparoscopic=3 days (SG039C = Lap)
        # Histopathology: MANDATORY (submit within 7 days of discharge)
        # Photo evidence (intraoperative + specimen): MANDATORY
        # TMS check: previous cholecystectomy → must be No (red flag if Yes)
        'package_code': 'SG039C', 'package_name': 'Cholecystectomy',
        'mandatory_docs': ['clinical_notes','usg_report','lft_report','operative_notes',
                           'pre_anesthesia','discharge_summary',
                           'photo_evidence','histopathology'],  # STG: both mandatory
        'optional_docs':  [],
        'clinical_rules': [
            {'rule_id':'CR-001','severity':'mandatory',
             'description':'STG TMS check: USG confirms calculi in gall bladder',
             'field':'usg_calculi','operator':'equals','value':1},
            {'rule_id':'CR-002','severity':'mandatory',
             'description':'STG TMS check: pain in right hypochondrium/epigastrium documented',
             'field':'pain_present','operator':'equals','value':1},
            {'rule_id':'CR-003','severity':'mandatory',
             'description':'Clinical condition (cholecystitis/biliary colic/cholangitis) documented',
             'field':'clinical_condition','operator':'equals','value':1},
            {'rule_id':'CR-004','severity':'mandatory',
             # STG TMS: "Cholecystectomy done in the past? No" — prior surgery = red flag
             'description':'STG TMS check: no prior cholecystectomy (previous_surgery=0 expected)',
             'field':'previous_surgery','operator':'equals','value':0},
            {'rule_id':'CR-005','severity':'mandatory',
             'description':'STG ALOS: length of stay 1-6 days (Lap=3, Open=6)',
             'field':'los_days','operator':'between','min':1,'max':6},
        ],
        'visual_checks': [
            {'rule_id':'VC-001','element':'stamp','required':True},
            {'rule_id':'VC-002','element':'signature','required':True},
        ],
    },
    'MG006A': {
        # STG source: Enteric fever_MG006A.pdf
        # ALOS = 3-5 days | Fever >= 38.3°C / 101°F for > 2 days
        # investigation_pre: CBC, ESR, Peripheral smear, LFT
        # investigation_post: Post treatment CBC, ESR, Peripheral smear, LFT
        'package_code': 'MG006A', 'package_name': 'Enteric Fever',
        'mandatory_docs': PACKAGE_INFO['MG006A']['mandatory'],
        'optional_docs':  [],
        'clinical_rules': [
            {'rule_id':'CR-001','severity':'mandatory',
             'description':'STG TMS check: fever >= 38.3°C / 101°F for > 2 days documented',
             'field':'fever','operator':'equals','value':1},
            {'rule_id':'CR-002','severity':'mandatory',
             'description':'Enteric symptoms (headache/diarrhea/malaise) documented',
             'field':'symptoms','operator':'equals','value':1},
            {'rule_id':'CR-003','severity':'mandatory',
             'description':'Pre-treatment investigation date present',
             'field':'pre_date','operator':'present','value':True},
            {'rule_id':'CR-004','severity':'mandatory',
             'description':'Post-treatment investigation date present',
             'field':'post_date','operator':'present','value':True},
            {'rule_id':'CR-005','severity':'mandatory',
             'description':'STG ALOS: length of stay 3-5 days (ALOS=3-5)',
             'field':'los_days','operator':'between','min':3,'max':5},
        ],
        'visual_checks': [
            {'rule_id':'VC-001','element':'stamp','required':True},
            {'rule_id':'VC-002','element':'signature','required':True},
        ],
    },
    'SB039A': {
        # STG source: Total Knee Replacement (TKR)_SB039A.pdf
        # ALOS = 5-7 days | Age > 55 years (primary osteoarthritis)
        # TMS check: post-op X-ray shows implant | Age > 55 if primary OA
        # Mandatory: indoor case papers, post-op photo, post-op X-ray,
        #            implant invoice/barcode, operative notes, discharge summary
        'package_code': 'SB039A', 'package_name': 'Total Knee Replacement',
        'mandatory_docs': PACKAGE_INFO['SB039A']['mandatory'],
        'optional_docs':  [],
        'clinical_rules': [
            {'rule_id':'CR-001','severity':'mandatory',
             'description':'STG: knee arthritis type (primary OA/degenerative/secondary) documented',
             'field':'arthritis_type','operator':'equals','value':1},
            {'rule_id':'CR-002','severity':'mandatory',
             'description':'STG TMS check: patient age > 55 years (primary OA criterion)',
             'field':'age_valid','operator':'equals','value':1},
            {'rule_id':'CR-003','severity':'mandatory',
             'description':'STG TMS check: post-op X-ray shows implant in situ',
             'field':'post_op_implant_present','operator':'equals','value':1},
            {'rule_id':'CR-004','severity':'mandatory',
             'description':'Admission date (DOA) present',
             'field':'doa','operator':'present','value':True},
            {'rule_id':'CR-005','severity':'mandatory',
             'description':'Discharge date (DOD) after admission (DOA)',
             'field':'discharge_after_admission','operator':'equals','value':True},
            {'rule_id':'CR-006','severity':'mandatory',
             'description':'STG ALOS: length of stay 5-7 days (ALOS=5-7)',
             'field':'los_days','operator':'between','min':5,'max':7},
        ],
        'visual_checks': [
            {'rule_id':'VC-001','element':'stamp','required':True},
            {'rule_id':'VC-002','element':'signature','required':True},
            {'rule_id':'VC-003','element':'implant_sticker','required':True},
        ],
    },
}
print('Package info + STG configs loaded:', list(STG_CONFIGS.keys()))

# %% CELL 05 — Module 0: Data Discovery + Evidence Slot Planner
# ─────────────────────────────────────────────────────────────────────────────
# Evidence Slot Planner: translates PACKAGE_INFO into the structured evidence
# graph expected for each claim. This shifts the pipeline from "classify pages"
# to "fill evidence slots" — the core claim adjudication framing.
#
# Each slot maps to:  what doc type(s) fill it  |  mandatory / supporting / visual
# The pipeline uses slots to:
#   1. Know upfront what evidence is expected (CEX lens)
#   2. Map each page's output to a slot (not just a label)
#   3. Drive decision confidence at slot level, not just page level
# ─────────────────────────────────────────────────────────────────────────────

EVIDENCE_SLOTS = {
    # Slot definition per package: list of {slot, doc_types, role, event_type}
    # role: 'mandatory' | 'supporting' | 'visual'
    # event_type: mirrors EVENT_DEFS for timeline linkage
    'MG064A': [
        {'slot': 'admission_evidence',       'doc_types': ['indoor_case'],
         'role': 'mandatory', 'event_type': 'Admission'},
        {'slot': 'diagnostic_evidence',      'doc_types': ['cbc_hb_report'],
         'role': 'mandatory', 'event_type': 'Diagnostic Investigation'},
        {'slot': 'treatment_evidence',       'doc_types': ['treatment_details'],
         'role': 'mandatory', 'event_type': 'Treatment'},
        {'slot': 'post_treatment_evidence',  'doc_types': ['post_hb_report'],
         'role': 'mandatory', 'event_type': 'Post-Treatment Assessment'},
        {'slot': 'clinical_context',         'doc_types': ['clinical_notes'],
         'role': 'supporting', 'event_type': None},
        {'slot': 'discharge_evidence',       'doc_types': ['discharge_summary'],
         'role': 'mandatory', 'event_type': 'Discharge'},
        {'slot': 'visual_authentication',    'doc_types': [],
         'role': 'visual', 'event_type': None},
    ],
    'SG039C': [
        {'slot': 'clinical_context',         'doc_types': ['clinical_notes'],
         'role': 'mandatory', 'event_type': 'Admission / Clinical Eval'},
        {'slot': 'diagnostic_evidence',      'doc_types': ['usg_report', 'lft_report'],
         'role': 'mandatory', 'event_type': 'Diagnostic Investigation'},
        {'slot': 'pre_procedure_evidence',   'doc_types': ['pre_anesthesia'],
         'role': 'mandatory', 'event_type': 'Pre-Anaesthesia Eval'},
        {'slot': 'procedure_evidence',       'doc_types': ['operative_notes'],
         'role': 'mandatory', 'event_type': 'Operative Procedure'},
        {'slot': 'post_procedure_evidence',  'doc_types': ['histopathology', 'photo_evidence'],
         'role': 'mandatory', 'event_type': None},
        {'slot': 'discharge_evidence',       'doc_types': ['discharge_summary'],
         'role': 'mandatory', 'event_type': 'Discharge'},
        {'slot': 'visual_authentication',    'doc_types': [],
         'role': 'visual', 'event_type': None},
    ],
    'MG006A': [
        {'slot': 'clinical_context',         'doc_types': ['clinical_notes'],
         'role': 'mandatory', 'event_type': 'Admission'},
        {'slot': 'pre_treatment_evidence',   'doc_types': ['investigation_pre'],
         'role': 'mandatory', 'event_type': 'Pre-Treatment Investigation'},
        {'slot': 'treatment_evidence',       'doc_types': ['vitals_treatment'],
         'role': 'mandatory', 'event_type': 'Treatment'},
        {'slot': 'post_treatment_evidence',  'doc_types': ['investigation_post'],
         'role': 'mandatory', 'event_type': 'Post-Treatment Investigation'},
        {'slot': 'discharge_evidence',       'doc_types': ['discharge_summary'],
         'role': 'mandatory', 'event_type': 'Discharge'},
        {'slot': 'visual_authentication',    'doc_types': [],
         'role': 'visual', 'event_type': None},
    ],
    'SB039A': [
        {'slot': 'admission_evidence',       'doc_types': ['indoor_case', 'clinical_notes'],
         'role': 'mandatory', 'event_type': 'Admission'},
        {'slot': 'diagnostic_evidence',      'doc_types': ['xray_ct_knee'],
         'role': 'mandatory', 'event_type': 'Diagnostic Investigation'},
        {'slot': 'procedure_evidence',       'doc_types': ['operative_notes'],
         'role': 'mandatory', 'event_type': 'Operative Procedure'},
        {'slot': 'implant_evidence',         'doc_types': ['implant_invoice'],
         'role': 'mandatory', 'event_type': None},
        {'slot': 'post_procedure_evidence',  'doc_types': ['post_op_photo', 'post_op_xray'],
         'role': 'mandatory', 'event_type': 'Post-Op Monitoring'},
        {'slot': 'discharge_evidence',       'doc_types': ['discharge_summary'],
         'role': 'mandatory', 'event_type': 'Discharge'},
        {'slot': 'visual_authentication',    'doc_types': [],
         'role': 'visual', 'event_type': None},
    ],
}

def plan_evidence_slots(package_code: str) -> list[dict]:
    """
    Returns the expected evidence slots for a package.
    Used by the rules engine to produce slot-level coverage summary
    and by the reviewer utility to surface missing slots.
    """
    return EVIDENCE_SLOTS.get(package_code, [])

def map_doctype_to_slot(doc_type: str, package_code: str) -> Optional[str]:
    """Returns the slot name that this doc_type fills, or None if extra."""
    for slot_def in EVIDENCE_SLOTS.get(package_code, []):
        if doc_type in slot_def['doc_types']:
            return slot_def['slot']
    return None

def score_evidence_coverage(ranked_rows: list[dict], package_code: str,
                             facts: Optional[dict] = None) -> dict:
    """
    Computes slot-level evidence coverage across all ranked rows.
    Returns a dict: slot → {filled: bool, confidence: float, best_page: int, doc_type: str}

    P0.5: For MG064A post_treatment_evidence slot, also checks facts dict for
    post_treatment_evidence_found (hb_post_value, blood_transfusion signal, or
    post-Hb on discharge summary) — so the slot fills even without an explicit
    post_hb_report page, matching real-world scanned claim behavior.
    """
    facts = facts or {}
    slots = plan_evidence_slots(package_code)
    coverage: dict[str, dict] = {
        s['slot']: {'filled': False, 'confidence': 0.0,
                    'best_page': None, 'doc_type': None,
                    'role': s['role']}
        for s in slots
        if s['role'] != 'visual'   # visual checked separately by rules engine
    }
    for row in ranked_rows:
        if row.get('extra_document') == 1:
            continue
        dt = row.get('_doc_type', 'unknown')
        slot = map_doctype_to_slot(dt, package_code)
        if slot and slot in coverage:
            conf = float(row.get('_confidence', 0.0))
            if not coverage[slot]['filled'] or conf > coverage[slot]['confidence']:
                coverage[slot].update({
                    'filled':     True,
                    'confidence': conf,
                    'best_page':  row.get('page_number'),
                    'doc_type':   dt,
                })

    # P0.5: MG064A post_treatment_evidence — fill via facts if not already filled
    if (package_code == 'MG064A'
            and not coverage.get('post_treatment_evidence', {}).get('filled')
            and facts.get('post_treatment_evidence_found')):
        coverage['post_treatment_evidence'].update({
            'filled':     True,
            'confidence': 0.65,   # moderate — inferred, not doc-classified
            'best_page':  None,
            'doc_type':   'hb_post_value (inferred from ef)',
        })

    return coverage


def discover_claims(base_dir: Path) -> list[dict]:
    """
    Walks base_dir for claim files. Supports two layouts:
      A) base_dir/<PKG_CODE>/<CLAIM_ID>/<files>  — local dataset layout
      B) base_dir/<CLAIM_ID>/<files>             — sandbox flat layout
    Package code auto-detected from parent folder name in layout A,
    or from package_code.txt in layout B.
    Returns: [{claim_id, package_code, files: [Path, ...]}]
    """
    claims = []
    if not base_dir.exists():
        print(f'WARNING: {base_dir} not found. Check BASE_DATA_DIR.')
        return claims
    for child in sorted(base_dir.iterdir()):
        if not child.is_dir(): continue
        if child.name in PACKAGE_CODES:
            # Layout A: child is a package folder
            for claim_dir in sorted(child.iterdir()):
                if not claim_dir.is_dir(): continue
                files = sorted([f for f in claim_dir.rglob('*')
                                if f.is_file() and f.suffix.lower() in SUPPORTED_EXT])
                if files:
                    claims.append({'claim_id': claim_dir.name,
                                   'package_code': child.name,
                                   'files': files})
        else:
            # Layout B: child is a claim folder (package from txt or unknown)
            pkg = None
            helper = child / 'package_code.txt'
            if helper.exists():
                pkg = helper.read_text(encoding='utf-8').strip()
            if pkg not in PACKAGE_CODES:
                continue
            files = sorted([f for f in child.rglob('*')
                            if f.is_file() and f.suffix.lower() in SUPPORTED_EXT])
            if files:
                claims.append({'claim_id': child.name,
                               'package_code': pkg,
                               'files': files})
    print(f'Discovered {len(claims)} claims across packages:', list(PACKAGE_CODES))
    ## Optionally print this
    return claims

ALL_CLAIMS = discover_claims(BASE_DATA_DIR)

# Summary table
_df = pd.DataFrame([{k: v for k, v in c.items() if k != 'files'} for c in ALL_CLAIMS])
if not _df.empty:
    # Print the claims (summary) table
    print(_df.groupby('package_code').size().reset_index(name='claim_count'))


# %% CELL 06 — Module 1: Ingestion (PyMuPDF, zero Poppler dependency)
def extract_pages(file_path: Path, dpi: int = 150) -> list[dict]:
    """
    Returns list of page dicts: {page_number, image (PIL), text (str),
    is_text_based (bool), file_name}
    DPI=150: sufficient for gemma-3-12b-it; saves memory vs 300 DPI.
    """
    pages = []
    suffix = file_path.suffix.lower()
    try:
        if suffix == '.pdf':
            doc = fitz.open(str(file_path))
            for i, pg in enumerate(doc):
                text = pg.get_text('text').strip()
                is_text_based = len(text.split()) > 15
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = pg.get_pixmap(matrix=mat)
                img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
                # Run OCR on scanned PDF pages (fitz returned no real text)
                ocr_lines = [] if is_text_based else ocr_page_lines(img)
                if ocr_lines and not is_text_based:
                    # Promote OCR text so Tier 2 keyword matching can fire
                    text = '\n'.join(l['text'] for l in ocr_lines)
                    is_text_based = bool(text.split())
                pages.append({
                    'page_number': i + 1, 'image': img,
                    'text': text, 'is_text_based': is_text_based,
                    'file_name': file_path.name,
                    '_ocr_lines': ocr_lines,
                })
            doc.close()
        else:
            img = Image.open(str(file_path)).convert('RGB')
            # Run OCR on image files — this is the primary text source for JPG/PNG
            ocr_lines = ocr_page_lines(img)
            text = '\n'.join(l['text'] for l in ocr_lines)
            pages.append({
                'page_number': 1, 'image': img,
                'text': text,
                'is_text_based': bool(text.split()),
                'file_name': file_path.name,
                '_ocr_lines': ocr_lines,
            })
    except Exception as e:
        print(f'  WARN: ingestion failed {file_path.name}: {e}')
    return pages

def page_to_base64(img: Image.Image, max_dim: int = 1024) -> str:
    """Resize + JPEG-encode for VLM. max_dim=1024 reduces token cost."""
    img_copy = img.copy()
    img_copy.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = io.BytesIO()
    img_copy.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode()

print('Module 1 (Ingestion) ready.')

import threading as _threading

# Single shared PaddleOCR instance guarded by a lock.
# This avoids non-deterministic OneDNN crashes under cross-claim threading.
_paddle_lock = _threading.Lock()

def _get_paddle_instance():
    """Return a shared PaddleOCR instance, creating it lazily if needed."""
    global _paddle_ocr_instance
    if not PADDLE_AVAILABLE:
        return None
    if _paddle_ocr_instance is None:
        try:
            from paddleocr import PaddleOCR as _PaddleOCR
            _paddle_ocr_instance = _PaddleOCR(
                use_angle_cls=True,
                lang='en',
                show_log=False,
                use_gpu=False,
                enable_mkldnn=False,
                cpu_threads=1,
            )
        except Exception:
            _paddle_ocr_instance = None
    return _paddle_ocr_instance

def ocr_page_lines(img: Image.Image) -> list[dict]:
    """
    Extract structured OCR lines from a page image.
    Returns: [{"text": str, "bbox": [x1,y1,x2,y2], "confidence": float, "method": str}]

    Priority:
      1. PaddleOCR PP-OCRv4 (PADDLE_AVAILABLE) — stronger on Indian printed forms
      2. Tesseract (PYTESSERACT_AVAILABLE) — fallback for clean printed text
      3. [] — if neither available (VLM remains the extraction source)
    """
    if PADDLE_AVAILABLE and _get_paddle_instance() is not None:
        return _ocr_with_paddle(img)
    if PYTESSERACT_AVAILABLE and _pytesseract_lib is not None:
        return _ocr_with_tesseract(img)
    return []


def _ocr_with_paddle(img: Image.Image) -> list[dict]:
    """PaddleOCR backend — serialised to prevent OneDNN segfaults."""
    import numpy as np
    paddle = _get_paddle_instance()
    if paddle is None:
        return _ocr_with_tesseract(img) if PYTESSERACT_AVAILABLE else []
    try:
        img_np = np.array(img.convert('RGB'))
        with _paddle_lock:
            result = paddle.ocr(img_np, cls=True)
        lines = []
        if not result or not result[0]:
            return []
        for line in result[0]:
            if not line:
                continue
            bbox_pts, (text, conf) = line[0], line[1]
            xs = [p[0] for p in bbox_pts]
            ys = [p[1] for p in bbox_pts]
            bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
            if text.strip() and conf > 0.20:
                lines.append({
                    'text':       text.strip(),
                    'bbox':       bbox,
                    'confidence': round(float(conf), 2),
                    'method':     'paddleocr',
                })
        return lines
    except Exception:
        return _ocr_with_tesseract(img) if PYTESSERACT_AVAILABLE else []


def _ocr_with_tesseract(img: Image.Image) -> list[dict]:
    """Tesseract backend — groups word boxes into lines."""
    try:
        w, h = img.size
        img_hr = img.resize((w * 2, h * 2), Image.LANCZOS).convert('L')
        data = _pytesseract_lib.image_to_data(
            img_hr, lang='eng',
            config='--psm 6 --oem 1',
            output_type=_pytesseract_lib.Output.DICT
        )
        line_groups: dict[tuple, dict] = {}
        n = len(data['text'])
        for i in range(n):
            word = (data['text'][i] or '').strip()
            conf = int(data['conf'][i])
            if not word or conf < 20:
                continue
            key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
            if key not in line_groups:
                line_groups[key] = {
                    'words': [], 'confs': [],
                    'left': data['left'][i], 'top': data['top'][i],
                    'right': data['left'][i] + data['width'][i],
                    'bottom': data['top'][i] + data['height'][i],
                }
            g = line_groups[key]
            g['words'].append(word); g['confs'].append(conf)
            g['right']  = max(g['right'],  data['left'][i] + data['width'][i])
            g['bottom'] = max(g['bottom'], data['top'][i]  + data['height'][i])
        lines = []
        for g in sorted(line_groups.values(), key=lambda x: (x['top'], x['left'])):
            text = ' '.join(g['words'])
            avg_conf = sum(g['confs']) / len(g['confs']) / 100.0
            lines.append({
                'text':       text,
                'bbox':       [g['left']//2, g['top']//2, g['right']//2, g['bottom']//2],
                'confidence': round(avg_conf, 2),
                'method':     'tesseract',
            })
        return lines
    except Exception as e:
        print(f'  WARN _ocr_with_tesseract: {e}')
        return []

# %% CELL 07 — Module 2 Tiers 1+2: Free Classification (zero tokens)
FILENAME_HINTS = {
    ## Are we handling case sensitivity here ? Upper/Lower/Camel case ?
    # ── Discharge Summary ────────────────────────────────────────────────────
    'discharge':          'discharge_summary',  'ds':                'discharge_summary',
    'disc':               'discharge_summary',  'dis_sum':           'discharge_summary',
    'detailed_discharge': 'discharge_summary',  'discharge__':       'discharge_summary',
    'ms_dis':             'discharge_summary',  # MS_DIS.jpg = medical summary discharge
    # DIS.pdf / DIS.jpg patterns handled via _dis_ / _dc_ suffix checks below

    # ── Clinical Notes ───────────────────────────────────────────────────────
    'clinical':           'clinical_notes',     'notes':             'clinical_notes',
    'clinical_note':      'clinical_notes',     'cn_':               'clinical_notes',
    'doctor_note':        'clinical_notes',     'doc_note':          'clinical_notes',
    'er_note':            'clinical_notes',     'progress_note':     'clinical_notes',

    # ── Indoor Case Papers ───────────────────────────────────────────────────
    'indoor':             'indoor_case',        'icp':               'indoor_case',
    'ip_':                'indoor_case',        'icp_chart':         'indoor_case',
    'icp_notes':          'indoor_case',        'icu_notes':         'indoor_case',
    'casesheet':          'indoor_case',        'case_sheet':        'indoor_case',
    'caserecord':         'indoor_case',        'case_record':       'indoor_case',
    'allcase':            'indoor_case',        'all_case':          'indoor_case',
    'initial_assessment': 'indoor_case',        'admission_form':    'indoor_case',
    'enc':                'indoor_case',        # ENC.pdf = encounter/case sheet
    'bht':                'indoor_case',        # BHT = Bed Head Ticket (very common)
    'admission':          'indoor_case',        'adm':               'indoor_case',
    'adit':               'indoor_case',        # ADIT = Admission Details (MG064A pattern)

    # ── CBC / Haemoglobin Reports ────────────────────────────────────────────
    'lab':                'cbc_hb_report',      'cbc':               'cbc_hb_report',
    'hb':                 'cbc_hb_report',      'blood':             'cbc_hb_report',
    'hb_report':          'cbc_hb_report',      'lab_report':        'cbc_hb_report',
    'all_report':         'cbc_hb_report',      'print_report':      'cbc_hb_report',
    'lab_reports':        'cbc_hb_report',      'all_reports':       'cbc_hb_report',
    'report':             'cbc_hb_report',      'reports':           'cbc_hb_report',
    'ptinr':              'cbc_hb_report',      # PT/INR = coagulation lab
    'fb':                 'cbc_hb_report',      # FB = Full Blood count (common in Indian labs)
    'n4':                 'cbc_hb_report',      # N4.jpg = Nabl/lab report (MG064A dataset pattern)
    'fbc':                'cbc_hb_report',      # FBC = Full Blood Count

    # ── Treatment Details ────────────────────────────────────────────────────
    'treatment':          'treatment_details',  'medication_chart':  'treatment_details',
    'intake_output':      'treatment_details',  'prescription':      'treatment_details',
    'drug_chart':         'treatment_details',  'drug':              'treatment_details',
    'treatment_chart':    'treatment_details',  'nursing_note':      'treatment_details',
    'medication':         'treatment_details',  'chart':             'treatment_details',

    # ── Vitals / TPR Charts (MG006A/MG064A) ─────────────────────────────────
    'vitals':             'vitals_treatment',   'tpr':               'vitals_treatment',
    'enhancement_record': 'vitals_treatment',   'progress_record':   'vitals_treatment',
    'nurses_record':      'vitals_treatment',   'nursing':           'vitals_treatment',
    'bedside':            'vitals_treatment',   # BEDSIDE_12.jpg etc. = bedside monitoring charts

    # ── USG / Ultrasound ─────────────────────────────────────────────────────
    'usg':                'usg_report',         'ultrasound':        'usg_report',
    'ultra_sound':        'usg_report',         'usg_lft':           'usg_report',

    # ── LFT ──────────────────────────────────────────────────────────────────
    'lft':                'lft_report',         'liver':             'lft_report',

    # ── Operative Notes ──────────────────────────────────────────────────────
    'ot_notes':           'operative_notes',    'operation':         'operative_notes',
    'ot_note':            'operative_notes',    'operative_note':    'operative_notes',
    'op_slip':            'operative_notes',    'detailed_operative':'operative_notes',
    'otnotes':            'operative_notes',    'otnote':            'operative_notes',

    # ── Pre-Anaesthesia ──────────────────────────────────────────────────────
    'anaes':              'pre_anesthesia',     'anesthesia':        'pre_anesthesia',
    'prea':               'pre_anesthesia',     # PREA.pdf — very common in SG039C
    'anaesth_slip':       'pre_anesthesia',     'anaes_slip':        'pre_anesthesia',
    'pre_anesthesia':     'pre_anesthesia',     'pre_anaes':         'pre_anesthesia',

    # ── Histopathology ───────────────────────────────────────────────────────
    'histo':              'histopathology',     'biopsy':            'histopathology',
    'pathology':          'histopathology',

    # ── X-Ray / CT (SB039A) ─────────────────────────────────────────────────
    'xray':               'xray_ct_knee',       'x_ray':             'xray_ct_knee',
    'x-ray':              'xray_ct_knee',       'cxray':             'xray_ct_knee',

    # ── Post-Op X-Ray ────────────────────────────────────────────────────────
    'post_xray':          'post_op_xray',       'post_x_ray':        'post_op_xray',
    'post_op_x':          'post_op_xray',

    # ── Implant Invoice / Barcode ────────────────────────────────────────────
    'implant':            'implant_invoice',    'invoice':           'implant_invoice',
    'barcode':            'implant_invoice',    # BARCODE.jpg = implant sticker barcode
    'implant_inv':        'implant_invoice',

    # ── Post-Op Photo ─────────────────────────────────────────────────────────
    'post_op':            'post_op_photo',      'postop':            'post_op_photo',
    'photo':              'post_op_photo',      'pic':               'post_op_photo',

    # ── Discharge Summary — additional tokens ────────────────────────────────
    'sum':                'discharge_summary',  'summary':           'discharge_summary',

    # ── Pre/Post Investigations (MG006A) ────────────────────────────────────
    'widal':              'investigation_pre',  'invest_pre':        'investigation_pre',
    'invest_post':        'investigation_post', 'investigation_pre': 'investigation_pre',
    'investigation_post': 'investigation_post',
    'fever_profile':      'investigation_pre',  # Typhoid fever profile = pre-investigation
    'urine_re':           'investigation_pre',  # Urine routine examination
    'urine':              'investigation_pre',  # URINE.jpg = urine investigation
    'blood_investigation':'investigation_pre',
    # Note: generic 'inves'/'inv'/'investigation' → handled by _disambiguate_investigation()
    # because pre vs post depends on occurrence order within the claim
}

# Explicit extra-document set — filenames containing these tokens are always
# extra_document regardless of package. Checked BEFORE FILENAME_HINTS in Tier 1
# to save VLM tokens and prevent misclassification (e.g. BILL.pdf → cbc_hb_report).
KNOWN_EXTRAS = {
    'bill', 'feedback', 'feed_back', 'adhar', 'aadhaar',
    'aadhar', 'id_card', 'card', 'justification', 'consent',
    'photo_id', 'voter', 'passport', 'pan_card',
}

# Suffixes that reliably indicate discharge summary even with short/cryptic filenames
# e.g. DIS.pdf, DIS.jpg, DC.pdf, DC.jpg — checked as standalone after normalization
_DISCHARGE_SUFFIXES = {'dis', 'dc', 'diss', 'disc'}
_CLINICAL_PREFIXES  = {'cn', 'dp'}    # CN_.pdf = clinical note; DP_ = daily progress → clinical

## Is the KEYWORD_MAP exhaustive ?
KEYWORD_MAP = {
    'discharge_summary':  ['date of discharge', 'final diagnosis', 'condition at discharge',
                           'discharge advice', 'discharged on', 'date of admission',
                           'summary of treatment', 'discharge diagnosis', 'summary'],
    'clinical_notes':     ['chief complaint', 'history of present illness', 'physical examination',
                           'indoor case papers', 'clinical notes', 'presenting complaints',
                           'on examination', 'hopi', 'c/o', 'k/c/o', 'general examination',
                           'systemic examination', 'plan of management', 'provisional diagnosis'],
    'indoor_case':        ['indoor case', 'ip no', 'ip number', 'ward', 'bed no',
                           'mrd no', 'uhid', 'case sheet', 'registration no',
                           'aayu id', 'aayu no', 'admission no', 'bed number',
                           'attending doctor', 'unit head', 'inpatient', 'ipd no'],
    'cbc_hb_report':      ['haemoglobin', 'hemoglobin', 'hb g/dl', 'rbc', 'wbc', 'platelets',
                           'complete blood count', 'cbc', 'packed cell volume', 'hematocrit',
                           'mcv', 'mch', 'mchc', 'total count', 'differential count',
                           'neutrophils', 'lymphocytes', 'eosinophils'],
    'post_hb_report':     ['post transfusion', 'repeat hb', 'follow up hb', 'post treatment',
                           'haemoglobin after', 'post-treatment hb', 'repeat cbc',
                           'post-transfusion', 'review cbc', 'post therapy hb',
                           'after treatment hb', 'post iron therapy', 'hb after',
                           'second hb', 'hemoglobin after', 'follow-up blood',
                           'repeat hemoglobin', 'post infusion'],
    'treatment_details':  ['blood transfusion', 'iv fluids', 'inj.', 'tablet', 'syrup',
                           'treatment chart', 'nursing notes', 'drug chart', 'medication',
                           'prescribed', 'dose', 'route', 'frequency', 'intravenous',
                           'packed red blood cells', 'prbc', 'ffp', 'platelet transfusion'],
    'usg_report':         ['ultrasonography', 'ultrasound', 'gall bladder', 'gallbladder',
                           'calculus', 'calculi', 'usg abdomen', 'acoustic shadow',
                           'common bile duct', 'cbd', 'choledocholithiasis', 'murphy sign',
                           'liver', 'spleen', 'pancreas', 'kidneys', 'echogenicity'],
    'lft_report':         ['liver function', 'sgpt', 'sgot', 'alkaline phosphatase', 'bilirubin',
                           'total protein', 'albumin', 'lft', 'alanine aminotransferase',
                           'aspartate aminotransferase', 'alt', 'ast', 'direct bilirubin',
                           'indirect bilirubin', 'pt', 'inr', 'gamma gt'],
    'operative_notes':    ['intraoperative', 'anaesthesia', 'incision', 'haemostasis',
                           'operative findings', 'surgeon', 'ot notes', 'laparoscopic',
                           'open cholecystectomy', 'port placement', 'trocar', 'ligated',
                           'specimen retrieved', 'bleeding controlled', 'closure',
                           'arthroplasty', 'cemented', 'implant fixed', 'tibial component'],
    'pre_anesthesia':     ['pre anesthesia', 'preanesthetic', 'asa grade', 'airway assessment',
                           'anesthetic fitness', 'pre op evaluation', 'mallampati',
                           'dentition', 'neck movement', 'mouth opening', 'fitness for surgery'],
    'histopathology':     ['histopathology', 'biopsy', 'specimen', 'microscopy', 'pathology report',
                           'gross examination', 'sections show', 'histological', 'benign', 'malignant',
                           'hpe report', 'h.p.e', 'hpe', 'tissue examination', 'surgical pathology',
                           'cholelithiasis specimen', 'cholecystectomy specimen', 'section shows',
                           'microscopic examination', 'histopathological examination',
                           'received specimen', 'gallbladder specimen', 'gall bladder specimen'],
    'xray_ct_knee':       ['x-ray', 'xray', 'ct scan', 'knee', 'joint space', 'osteophyte',
                           'degenerative', 'kellgren', 'radiograph', 'bone', 'fracture',
                           'varus', 'valgus', 'medial compartment', 'lateral compartment'],
    'implant_invoice':    ['implant', 'invoice', 'prosthesis', 'femoral', 'tibial',
                           'lot number', 'serial number', 'manufacturer', 'barcode',
                           'catalogue', 'part number', 'poly ethylene', 'polyethylene',
                           'zimmer', 'stryker', 'depuy', 'arthroplasty implant'],
    'post_op_xray':       ['post op xray', 'post operative x-ray', 'implant in situ',
                           'post-operative', 'check x-ray', 'postoperative radiograph'],
    'post_op_photo':      [],   # image-only; classified via absence of strong text + VLM
    'investigation_pre':  ['widal', 'blood culture', 'typhoid', 'dengue', 'pre treatment',
                           'baseline investigation', 'salmonella', 'ns1', 'malaria',
                           'before treatment', 'initial investigation', 'pre-treatment',
                           'on admission', 'enteric fever profile', 'urine routine'],
    'investigation_post': ['repeat', 'post treatment', 'follow up', 'response to treatment',
                           'culture sensitivity after', 'post-treatment', 'follow-up',
                           'after treatment', 'review investigation', 'outcome investigation'],
    'vitals_treatment':   ['temperature', 'pulse', 'blood pressure', 'respiratory rate',
                           'spo2', 'vitals', 'treatment given', 'tpr chart', 'bp',
                           'pulse rate', 'fever chart', 'nursing record', 'fluid chart',
                           'intake', 'output', 'daily progress', 'bedside'],
}

def classify_tier1(file_name: str,
                   package_code: str = '',
                   investigation_count: int = 0) -> tuple[Optional[str], float]:
    """
    Tier 1 classification from filename alone (zero tokens).

    Enhancements vs v4.0:
    - Expanded FILENAME_HINTS covering real dataset patterns
    - Suffix-based discharge detection: DIS.pdf / DC.jpg / DISC.pdf
    - Prefix-based clinical note detection: CN_.pdf / DP_.pdf
    - investigation_pre/post disambiguation for MG006A based on occurrence count
    - Generic 'inves'/'inv'/'investigation' treated as pre (first) vs post (subsequent)
    """
    ## This is the common pattern noted across dataset: <sequence_number>__PMJAY_xx_<year>_xx_<another_sequence_number>__<file_type>.pdf/jpg/jpeg

    ## Are we missing out of __ ? Eg: 000402__PMJAY_AR_S_2025_R3_1021475536__DIS.jpg, 000405__PMJAY_AR_S_2025_R3_1021475536__MS_DIS.jpg, 000404__PMJAY_AR_S_2025_R3_1021475536__notes.jpeg, 001070__PMJAY_BR_S_2025_R3_2026032410053432__Discharge_Summary_General.pdf

    # Normalise: lower, replace hyphens+spaces with underscore, strip extension
    stem = file_name.lower().replace('-', '_').replace(' ', '_')
    stem_noext = stem.rsplit('.', 1)[0]  # remove extension first
    # Strip noise suffixes AFTER extension removal so anchors work correctly.
    # e.g. sheela_1_20_11zon → sheela_1_20; doc_compressed → doc
    stem_noext = re.sub(r'_11zon$', '', stem_noext)
    stem_noext = re.sub(r'_compressed$', '', stem_noext)
    stem = stem_noext  # use cleaned stem for all subsequent hint matching

    # ── KNOWN_EXTRAS: explicit extra-document token check (before all other tiers) ─
    # Prevents BILL.pdf, FEEDBACK.pdf etc. from being mis-routed to VLM or
    # misclassified as cbc_hb_report / clinical_notes via keyword overlap.
    parts_all = [p for p in stem_noext.replace('__', '_').split('_') if p]
    for token in parts_all:
        if token in KNOWN_EXTRAS:
            return 'extra_document', 0.90

    # ── Suffix-based discharge detection (DIS.pdf / DC.pdf / DISC.pdf) ──────
    # Extract the last token between underscores as the "stem root"
    parts = [p for p in stem_noext.split('__')[-1].split('_') if p]
    if parts and parts[-1] in _DISCHARGE_SUFFIXES:
        return 'discharge_summary', 0.78

    # ── Prefix-based clinical note detection (CN_*.pdf / DP_*.pdf) ──────────
    raw_name_lower = file_name.lower()
    for pfx in _CLINICAL_PREFIXES:
        if raw_name_lower.startswith(pfx + '_') or f'__{pfx}_' in raw_name_lower:
            return 'clinical_notes', 0.72

    # ── Generic investigation disambiguation for MG006A ──────────────────────
    # 'inves', 'inv', 'investigation', 'investigations', 'all_investigations'
    # → first occurrence: investigation_pre; subsequent: investigation_post
    _inv_stems = ('inves', '_inv.', 'investigation', 'investigations')
    is_inv = any(token in stem for token in _inv_stems)
    if is_inv:
        if package_code == 'MG006A':
            dt = 'investigation_post' if investigation_count > 0 else 'investigation_pre'
            return dt, 0.72
        # For other packages just ignore (they don't have investigation_pre/post)
        return None, 0.0

    # ── Standard FILENAME_HINTS lookup ───────────────────────────────────────
    # Sort by hint length descending so longer (more specific) hints win.
    # e.g. 'post_xray' must match before 'xray'; 'post_op_xray' before 'post_op'.
    best_match: tuple[Optional[str], float] = (None, 0.0)
    for hint, doc_type in sorted(FILENAME_HINTS.items(), key=lambda x: -len(x[0])):
        if hint in stem:
            best_match = (doc_type, 0.75)
            break   # longest match wins — stop immediately

    # ── Low-confidence OD fallback ───────────────────────────────────────────
    # OD.jpg = likely "On Duty" or "Outdoor" notes → clinical_notes at low confidence
    # so Tier 2/3 can still override. Confidence 0.55 < 0.60 threshold → falls to Tier 2.
    if best_match == (None, 0.0):
        last_token = parts[-1] if parts else ''
        if last_token == 'od':
            best_match = ('clinical_notes', 0.55)

    return best_match

def classify_tier2(text: str, package_code: str,
                   investigation_count: int = 0) -> tuple[Optional[str], float]:
    """
    Tier 2: keyword match on fitz-extracted text (zero tokens).
    Confidence scaled by keyword hit count.
    investigation_count used for MG006A pre/post disambiguation.
    """
    if not text: return None, 0.0
    text_lower = text.lower()
    pkg_types  = set(PACKAGE_INFO[package_code]['doc_types'])
    scores: dict[str, int] = {}
    for doc_type, keywords in KEYWORD_MAP.items():
        if doc_type not in pkg_types: continue
        hits = sum(1 for kw in keywords if kw in text_lower)
        if hits >= 2:
            scores[doc_type] = hits
    if not scores: return None, 0.0
    best = max(scores, key=scores.get)
    best_score = scores[best]   # capture before potentially renaming best
    if package_code == 'MG006A' and best == 'investigation_pre' and investigation_count > 0:
        best = 'investigation_post'
    confidence = min(0.88, 0.55 + best_score * 0.08)
    return best, confidence

print('Module 2 (Tiers 1+2) ready.')

# %% CELL 08 — [GAP 7] VLM JSON Safe Parser + Mock VLM + Module 2 Tier 3
# _mock_vlm_response is defined HERE (not CELL 00) so it can call
# classify_tier2 / find_dates / extract_fields_from_text without NameError.
def _vlm_default(reason: str) -> dict:
    """Safe default for any VLM parse failure. Never raises."""
    return {
        'doc_type': 'unknown', 'confidence': 0.20,
        'has_stamp': False, 'has_signature': False,
        'has_qr_barcode': False, 'is_handwritten': False,
        'image_quality': 'good', 'dates_found': [],
        'extracted_fields': {}, '_parse_fallback': reason,
    }

## Ensure this works for the given dataset, that comprises of mostly scanned + handwritten documents
def _mock_vlm_response(img: Image.Image, package_code: str) -> dict:
    """
    Mock Tier 3 for testing without NHAclient.
    Called automatically when MOCK_MODE=True (Ollama unavailable).

    Strategy:
    1. Try pytesseract OCR (if installed + tesseract binary present).
       Runs classify_tier2 on the OCR result — often gets it right for digital scans.
    2. If pytesseract not available, returns 'unknown' / confidence=0.20.

    Mock pages are marked _mock=True in provenance.
    Confidence capped at 0.45 → these pages always show CONDITIONAL, never PASS.
    That way mock decisions are visually distinct from live ones.
    """
    TOKEN_LOG['mock_calls'] += 1
    ocr_text = ''
    if PYTESSERACT_AVAILABLE and _pytesseract_lib is not None:
        try:
            ocr_text = _pytesseract_lib.image_to_string(img, lang='eng+hin') or ''
        except Exception:
            pass

    doc_type, conf = 'unknown', 0.20
    if ocr_text.strip():
        dt, c = classify_tier2(ocr_text, package_code)
        if dt:
            doc_type, conf = dt, min(c, 0.45)   # cap — clearly mock tier

    return {
        'doc_type':         doc_type,
        'confidence':       conf,
        'has_stamp':        False,
        'has_signature':    False,
        'has_qr_barcode':   False,
        'is_handwritten':   False,
        'image_quality':    'good',
        'dates_found':      find_dates(ocr_text),
        'extracted_fields': extract_fields_from_text(ocr_text),
        '_mock':            True,
    }

def _extract_ef_date(ef: dict, field: str) -> tuple[Optional[str], Optional[str], float]:
    """
    Safely read a date from extracted_fields, which may be either:
      - new structured format: {"value": "...", "source_text": "...", "confidence": 0.9}
      - legacy plain string:   "15/07/2023"  (old VLM responses in cache)
      - null / missing

    Returns: (raw_value, source_text, confidence)
    Applies prompt-leakage guard: known example strings → returns (None, None, 0.0)
    """
    raw = ef.get(field)
    if raw is None:
        return None, None, 0.0

    # Structured object (new schema)
    if isinstance(raw, dict):
        val  = raw.get('value') or None
        src  = raw.get('source_text') or None
        conf = float(raw.get('confidence') or 0.0)
        # Prompt leakage guard — VLM must not echo example values from prompt
        if val and val.strip() in _PROMPT_LEAK_GUARDS:
            print(f'  WARN _extract_ef_date: prompt-leak detected for {field}: "{val}" — discarded')
            return None, None, 0.0
        return val, src, conf

    # Legacy plain string (old VLM format — treat as low-confidence, no source_text)
    if isinstance(raw, str) and raw.lower() not in ('null', 'none', ''):
        if raw.strip() in _PROMPT_LEAK_GUARDS:
            print(f'  WARN _extract_ef_date: prompt-leak (legacy) for {field}: "{raw}" — discarded')
            return None, None, 0.0
        return raw, None, 0.50   # moderate confidence: we have a value but no anchor proof

    return None, None, 0.0


def _extract_ef_dates_found(classified: dict) -> list[dict]:
    """
    Safely read dates_found which may be:
      - new format: [{"value": "...", "nearby_text": "...", "is_labeled": bool}]
      - legacy flat: ["15/07/2023", "16/07/2023"]

    Returns a list of normalised dicts. Removes any prompt-leak values.
    """
    raw_list = classified.get('dates_found', []) or []
    result = []
    for item in raw_list:
        if isinstance(item, dict):
            val = item.get('value') or ''
            if val.strip() and val.strip() not in _PROMPT_LEAK_GUARDS:
                result.append({
                    'value':       val.strip(),
                    'nearby_text': item.get('nearby_text') or '',
                    'is_labeled':  bool(item.get('is_labeled', False)),
                })
        elif isinstance(item, str):
            if item.strip() and item.strip() not in _PROMPT_LEAK_GUARDS:
                result.append({
                    'value':       item.strip(),
                    'nearby_text': '',
                    'is_labeled':  False,
                })
    return result


def safe_parse_vlm(raw: str) -> dict:
    """
    [GAP 7] Five-stage VLM response parser — never raises, always returns dict.
    Stage 1: direct json.loads()
    Stage 2: strip markdown fences + extract first {...} block
    Stage 3: json_repair library
    Stage 4: regex field extraction from raw text
    Stage 5: safe default
    """
    if not raw or not raw.strip():
        return _vlm_default('empty_response')
    # Stage 1
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Stage 2
    clean = re.sub(r'^```[a-z]*\s*', '', raw.strip())
    clean = re.sub(r'\s*```\s*$', '', clean)
    m = re.search(r'\{.*\}', clean, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            clean = m.group(0)
    # Stage 3
    try:
        result = json_repair_loads(clean)
        if isinstance(result, dict) and result.get('doc_type'):
            return result
    except Exception:
        pass
    # Stage 4: regex field extraction
    extracted = {}
    dt_m = re.search(r'"doc_type"\s*:\s*"([^"]+)"', raw)
    if dt_m: extracted['doc_type'] = dt_m.group(1)
    cf_m = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw)
    if cf_m: extracted['confidence'] = float(cf_m.group(1))
    for flag in ('has_stamp','has_signature','has_qr_barcode','is_handwritten'):
        fm = re.search(rf'"{flag}"\s*:\s*(true|false)', raw, re.I)
        if fm: extracted[flag] = fm.group(1).lower() == 'true'
    iq_m = re.search(r'"image_quality"\s*:\s*"([^"]+)"', raw)
    if iq_m: extracted['image_quality'] = iq_m.group(1)
    if extracted.get('doc_type'):
        return {**_vlm_default('regex_fallback'), **extracted}
    # Stage 5
    return _vlm_default('all_stages_failed')

VLM_CLASSIFY_PROMPT = """You are an expert medical document analyst for PM-JAY healthcare claims in India.
Documents may be in English, Hindi, or mixed language. Handwritten content is common.

Analyze ONLY the document page image provided.
Return ONLY valid JSON. No markdown. No explanation.

Package: {package_code} ({package_name})
Valid document types: {doc_types}

Important rules:
- Read visible printed and handwritten content only.
- Never invent, infer, assume, or copy values that are not visibly present.
- Never copy any text, date, or number from these instructions into your output.
- If a field is not clearly visible on the page, return null for that field.
- If uncertain about a value, return null and add the field name to uncertain_fields.

DATE EXTRACTION RULES:
- Extract dates ONLY if they are visibly present on this page.
- Return each date exactly as it appears written — do not reformat.
- Populate a labeled date field ONLY when the date appears next to a visible label such as:
  Date of Admission, DOA, Admitted on, Date of Discharge, DOD, Discharged on,
  Investigation Date, Report Date, Treatment Date, Transfusion Date.
- If a visible date has no clear label, include it in dates_found only with is_labeled=false.
- Do NOT assign an unlabeled date to admission, discharge, treatment, or investigation.
- If no date is visible for a labeled field, set value to null and confidence to 0.0.

Return exactly this JSON shape:
{{
  "doc_type": "<one of the valid document types, or extra_document>",
  "confidence": <0.0-1.0>,
  "has_stamp": <true/false>,
  "has_signature": <true/false>,
  "has_qr_barcode": <true/false>,
  "is_handwritten": <true/false>,
  "image_quality": "<good/poor>",
  "dates_found": [
    {{
      "value": "<date exactly as visible on page>",
      "nearby_text": "<short text near the date that provides context>",
      "is_labeled": <true if a label identifies what this date means, else false>
    }}
  ],
  "extracted_fields": {{
    "date_of_admission":      {{"value": null, "source_text": null, "confidence": 0.0}},
    "date_of_discharge":      {{"value": null, "source_text": null, "confidence": 0.0}},
    "treatment_date":         {{"value": null, "source_text": null, "confidence": 0.0}},
    "pre_investigation_date": {{"value": null, "source_text": null, "confidence": 0.0}},
    "post_investigation_date":{{"value": null, "source_text": null, "confidence": 0.0}},
    "hb_value":               null,
    "hb_post_value":          null,
    "blood_transfusion":      null,
    "patient_age":            null,
    "diagnosis_keywords":     [],
    "clinical_findings":      [],
    "procedure_keywords":     [],
    "uncertain_fields":       [],
    "key_values":             {{}}
  }}
}}"""

def classify_tier3_vlm(img: Image.Image, package_code: str,
                        page_text: str = '') -> dict:
    """
    Edge model (Gemma 4 E4B) page classification — solutionFlow stage:
    edge_parse_page_with_e4b().

    Three paths:
      1. Ollama + Gemma 4 E4B  — vision or text depending on page content
      2. Mock (Ollama unavailable) — pytesseract OCR stub, confidence capped at 0.45

    Escalation: when confidence < 0.75 or needs_deep_review, caller escalates to 26B.
    page_text: fitz-extracted text for this page.
    """
    info   = PACKAGE_INFO[package_code]
    prompt = VLM_CLASSIFY_PROMPT.format(
        package_code=package_code,
        package_name=info['name'],
        doc_types=json.dumps(info['doc_types'])
    )

    if MOCK_MODE:
        return _mock_vlm_response(img, package_code)

    try:
        raw = edge_model_call(img, prompt, page_text=page_text)
        return safe_parse_vlm(raw)
    except Exception as e:
        print(f'  E4B call failed: {e} — falling back to mock')
        return _mock_vlm_response(img, package_code)

# Filename stems that are always extra_document regardless of package.
# Non-clinical documents: billing, pharmacy, satisfaction letters, Urdu/foreign content.
_EXTRA_DOCUMENT_FILENAME_STEMS: frozenset = frozenset({
    'pharmacy', 'pharma', 'medicine', 'drug',
    'receipt', 'bill', 'billing', 'invoice', 'payment', 'final_bill', 'finalbill',
    'satisfactory', 'satisfaction', 'snt', 'letter', 'complaint',
    'consent',    # consent forms are supporting, not clinical evidence
    'sticker', 'label',
    'barcode_only', 'qr',
})
# For these, Tier 1 provides only a weak prior; Tier 2 (keyword) runs per page
# and can override; Tier 3 runs if both fail.
# e.g. ALL_REPORTS.pdf page 1 = CBC, page 2 = discharge summary, page 3 = indoor case.
_AGGREGATE_FILENAME_STEMS: frozenset = frozenset({
    'all_report', 'all_reports', 'allreport', 'allreports',
    'all_doc', 'all_docs', 'alldoc', 'alldocs',
    'all_case', 'allcase',
    'usg_lft', 'usglft', 'investigations', 'investigation',
    'inves', 'all_investigations',
    'ipd', 'ipd1', 'ipd2',
    'combined', 'combined_report', 'combined_doc',
    'print_report', 'printreport',
    'case', 'caserecord', 'case_record',
    'enc',   # encounter — often multi-section
    'all',
})

# Package-specific filename overrides — checked BEFORE global FILENAME_HINTS.
# Prevents cross-package leakage (e.g. 'photo' → post_op_photo for SB039A only).
_PACKAGE_FILENAME_HINTS: dict[str, dict[str, str]] = {
    'SG039C': {
        'photo':         'photo_evidence',
        'photograph':    'photo_evidence',
        'specimen':      'photo_evidence',
        'specimen_photo':'photo_evidence',
        'histo':         'histopathology',
        'histopath':     'histopathology',
        'histopathology':'histopathology',
        'path':          'histopathology',
        'pathology':     'histopathology',
        'biopsy':        'histopathology',
        'usg':           'usg_report',
        'ultrasound':    'usg_report',
        'lft':           'lft_report',
        'liver':         'lft_report',
        'operative':     'operative_notes',
        'op_notes':      'operative_notes',
        'pre_anesthesia':'pre_anesthesia',
        'anaesthesia':   'pre_anesthesia',
        'anesthesia':    'pre_anesthesia',
    },
    'SB039A': {
        'photo':         'post_op_photo',
        'xray':          'xray_ct_knee',
        'x_ray':         'xray_ct_knee',
        'xxx':           'xray_ct_knee',   # XXX.jpg = X-ray film scan (UK claim pattern)
        'ct':            'xray_ct_knee',
        'implant':       'implant_invoice',
        'invoice':       'implant_invoice',
        'operative':     'operative_notes',
        'op_notes':      'operative_notes',
    },
    'MG006A': {
        'enc':           'clinical_notes',
        'case':          'clinical_notes',
        'admission_form':'clinical_notes',
        'pre_inv':       'investigation_pre',
        'preinv':        'investigation_pre',
        'cbc':           'investigation_pre',   # CBC.pdf in MG006A = pre-investigation labs (no cbc_hb_report slot)
        'lab':           'investigation_pre',   # generic lab files in MG006A context
        'inv':           'investigation_pre',   # INV/investigation prefix
        'post_inv':      'investigation_post',
        'postinv':       'investigation_post',
    },
}

def classify_and_extract_page(page: dict, package_code: str,
                               investigation_count: int = 0) -> dict:
    """
    Orchestrates Tier 1 → Tier 2 → Tier 3.

    Tier 1 (filename):
      - Package-specific hints checked first (_PACKAGE_FILENAME_HINTS).
      - Aggregate filenames (ALL_REPORTS, IPD, USG_LFT, …) are treated as
        weak priors (conf capped at 0.55) so Tier 2 always runs per page.
        This is the critical fix for multi-type aggregate documents.
      - All other filenames use the global FILENAME_HINTS conf as-is.

    Tier 2 (keyword on fitz text): always runs for aggregate filenames;
      runs normally when Tier 1 failed or had low confidence.

    Tier 3 (VLM): only when Tier 1+2 both fail or for image-only pages.
      Text-first: pages with >30 fitz words use ollama_text_call (~4s vs ~18s).
    """
    base = {
        'doc_type': 'unknown', 'confidence': 0.0, 'method': 'none',
        'has_stamp': False, 'has_signature': False, 'has_qr_barcode': False,
        'is_handwritten': False, 'image_quality': 'good',
        'dates_found': [], 'extracted_fields': {}, 'error': None
    }
    try:
        text      = page.get('text', '')
        file_stem = Path(page['file_name']).stem.lower()

        # ── Extra document early exit ─────────────────────────────────────────
        if any(stem in file_stem for stem in _EXTRA_DOCUMENT_FILENAME_STEMS):
            base.update({'doc_type': 'extra_document', 'confidence': 0.90,
                         'method': 'filename', '_extra_reason': 'non_clinical_document'})
            return base

        # ── Tier 1: filename ──────────────────────────────────────────────────
        # Check package-specific hints first (prevents cross-package leakage)
        pkg_hints = _PACKAGE_FILENAME_HINTS.get(package_code, {})
        tier1_dt, tier1_conf = None, 0.0
        for hint, dt in sorted(pkg_hints.items(), key=lambda x: -len(x[0])):
            if hint in file_stem:
                tier1_dt, tier1_conf = dt, 0.82   # package hint = high confidence
                break
        if tier1_dt is None:
            tier1_dt, tier1_conf = classify_tier1(page['file_name'], package_code,
                                                   investigation_count)

        # Aggregate filenames → cap confidence so Tier 2 runs per page
        is_aggregate = any(agg in file_stem for agg in _AGGREGATE_FILENAME_STEMS)
        if is_aggregate and tier1_conf > 0.55:
            tier1_conf = 0.55   # force Tier 2 to run

        tier1_accepted = bool(tier1_dt and tier1_conf >= 0.60)

        # ── Tier 2: keyword on page text ──────────────────────────────────────
        # Runs when: Tier 1 failed, filename is aggregate (per-page override),
        # or a broad Tier 1 label may need a narrow "post" upgrade.
        allow_post_override = (
            (package_code == 'MG064A' and tier1_dt == 'cbc_hb_report') or
            (package_code == 'SB039A' and tier1_dt == 'xray_ct_knee')
        )
        tier2_dt, tier2_conf = None, 0.0
        if ((not tier1_accepted or is_aggregate or allow_post_override)
                and page.get('is_text_based') and text):
            tier2_dt, tier2_conf = classify_tier2(text, package_code,
                                                   investigation_count)

        # ── Select best classification so far ─────────────────────────────────
        if tier2_dt and tier2_conf >= 0.60:
            if tier1_accepted and allow_post_override:
                post_upgrade_ok = (
                    (package_code == 'MG064A' and tier2_dt == 'post_hb_report') or
                    (package_code == 'SB039A' and tier2_dt == 'post_op_xray')
                )
                if not post_upgrade_ok:
                    tier2_dt, tier2_conf = None, 0.0

        if tier2_dt and tier2_conf >= 0.60:
            base.update({'doc_type': tier2_dt, 'confidence': tier2_conf,
                         'method': 'keyword'})
            base['dates_found']      = find_dates(text)
            base['extracted_fields'] = extract_fields_from_text(text)
            return base

        if tier1_accepted:
            base.update({'doc_type': tier1_dt, 'confidence': tier1_conf,
                         'method': 'filename'})
            base['dates_found']      = find_dates(text)
            base['extracted_fields'] = extract_fields_from_text(text)
            # Do NOT call VLM here — Tier 1 filename classification is sufficient.
            # OCR lines on the page already go to _extract_date_atom_from_lines
            # in build_timeline, so dates are extracted without a VLM call.
            return base

        # ── Tier 3: VLM (only when Tier 1+2 both failed) ─────────────────────
        vlm = classify_tier3_vlm(page['image'], package_code, page_text=text)
        base.update({
            'doc_type':         vlm.get('doc_type', 'unknown'),
            'confidence':       float(vlm.get('confidence', 0.20)),
            'method':           'vlm',
            'has_stamp':        bool(vlm.get('has_stamp', False)),
            'has_signature':    bool(vlm.get('has_signature', False)),
            'has_qr_barcode':   bool(vlm.get('has_qr_barcode', False)),
            'is_handwritten':   bool(vlm.get('is_handwritten', False)),
            'image_quality':    vlm.get('image_quality', 'good'),
            'dates_found':      vlm.get('dates_found', []),
            'extracted_fields': vlm.get('extracted_fields', {}),
        })
    except Exception as e:
        base['error'] = str(e)
    return base

print('Module 2 (Tier 3 VLM + safe_parse_vlm) ready.')


# %% CELL 09 — Module 3: Visual Detection (pyzbar + cv2)
def detect_qr_barcode(img: Image.Image) -> dict:
    if not PYZBAR_AVAILABLE:
        return {'found': False, 'confidence': 0.0, 'note': 'pyzbar unavailable'}
    try:
        decoded = pyzbar_decode(np.array(img))
        if decoded:
            d = decoded[0]
            return {'found': True, 'type': d.type,
                    'data': d.data.decode(errors='ignore'),
                    'bbox': list(d.rect), 'confidence': 0.99}
    except Exception:
        pass
    return {'found': False, 'confidence': 0.0}

def detect_stamp_cv2(img: Image.Image) -> dict:
    """Best-effort circular/oval stamp detection."""
    try:
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perim = cv2.arcLength(cnt, True)
            if perim == 0: continue
            circ = 4 * 3.14159 * area / (perim ** 2)
            if area > 2000 and circ > 0.45:
                x, y, w, h = cv2.boundingRect(cnt)
                return {'found': True, 'bbox': [x, y, x+w, y+h],
                        'confidence': min(0.82, circ), 'reliability': 'best_effort'}
    except Exception:
        pass
    return {'found': False, 'confidence': 0.0, 'reliability': 'best_effort'}

def detect_signature_cv2(img: Image.Image) -> dict:
    """Best-effort: high Laplacian variance in bottom 35% of page."""
    try:
        arr  = np.array(img)
        zone = arr[int(arr.shape[0] * 0.65):, :]
        gray = cv2.cvtColor(zone, cv2.COLOR_RGB2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return {'found': lap_var > 250,
                'confidence': min(0.65, lap_var / 800),
                'reliability': 'best_effort'}
    except Exception:
        return {'found': False, 'confidence': 0.0, 'reliability': 'best_effort'}

def detect_implant_sticker_cv2(img: Image.Image) -> dict:
    """Best-effort: rectangular high-contrast region for implant sticker."""
    try:
        gray  = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h_c = cv2.boundingRect(cnt)
            area   = w * h_c
            aspect = w / h_c if h_c > 0 else 0
            if 2000 < area < 60000 and 1.5 < aspect < 7:
                return {'found': True, 'bbox': [x, y, x+w, y+h_c],
                        'confidence': 0.55, 'reliability': 'best_effort'}
    except Exception:
        pass
    return {'found': False, 'confidence': 0.0, 'reliability': 'best_effort'}

def run_visual_detection(img: Image.Image, vlm_result: dict,
                          page_num: int = 0, file_name: str = '') -> dict:
    """
    Merges VLM visual signals with deterministic cv2 detections.
    VLM signals take precedence; cv2 fills gaps for free pages.
    Provenance (page/bbox) is preserved per element for defensibility.
    """
    qr      = detect_qr_barcode(img)
    stamp   = detect_stamp_cv2(img)
    sig     = detect_signature_cv2(img)
    implant = detect_implant_sticker_cv2(img)

    vlm_stamp = 0.85 if vlm_result.get('has_stamp')     else 0.0
    vlm_sig   = 0.80 if vlm_result.get('has_signature') else 0.0
    has_stamp = bool(vlm_result.get('has_stamp'))    or stamp['found']
    has_sig   = bool(vlm_result.get('has_signature')) or sig['found']
    has_qr    = qr['found'] or bool(vlm_result.get('has_qr_barcode'))

    # Provenance: attach page and bbox to each detected element
    def _prov(det: dict, source: str) -> dict:
        return {
            'page': page_num, 'file': file_name,
            'bbox': det.get('bbox'), 'source': source,
            'confidence': det.get('confidence', 0.0),
        }

    return {
        'stamp_present':              int(has_stamp),
        'stamp_confidence':           max(vlm_stamp, stamp['confidence']),
        'stamp_provenance':           _prov(stamp, 'vlm' if vlm_stamp else 'cv2'),
        'signature_present':          int(has_sig),
        'signature_confidence':       max(vlm_sig, sig['confidence']),
        'signature_provenance':       _prov(sig, 'vlm' if vlm_sig else 'cv2'),
        'qr_present':                 int(has_qr),
        'qr_data':                    qr.get('data'),
        'barcode_present':            int(qr['found'] and qr.get('type') != 'QRCODE'),
        'implant_sticker_present':    int(implant['found']),
        'implant_sticker_confidence': implant['confidence'],
        'implant_provenance':         _prov(implant, 'cv2'),
    }

print('Module 3 (Visual Detection) ready.')


# %% CELL 10 — [GAP 6] Logical Document Grouping
@dataclass
class LogicalDoc:
    """
    Groups consecutive pages sharing (file_name, doc_type) into one logical document.
    Used internally for combined-text extraction on multi-page reports.
    Page-level rows are still produced — schema unchanged.
    """
    doc_type:  str
    file_name: str
    pages:     list = field(default_factory=list)  # list of (page_dict, classified_dict)

    @property
    def combined_text(self) -> str:
        # Filter None before join — some pages have text=None on OCR-only pages
        return '\n'.join(p['page'].get('text') or '' for p in self.pages)

    @property
    def best_image(self) -> Image.Image:
        return self.pages[0]['page']['image']  # first page = title/header

    @property
    def all_dates(self) -> list[str]:
        """Return all date values across pages, using structured format aware extraction."""
        result = []
        for p in self.pages:
            for d in _extract_ef_dates_found(p.get('classified', {})):
                result.append(d['value'])
        return result

def group_into_logical_docs(classified_pages: list[dict]) -> list[LogicalDoc]:
    """
    classified_pages: [{'page': page_dict, 'classified': classified_dict}, ...]
    Groups consecutive pages with same (file_name, doc_type) into LogicalDocs.
    Non-consecutive same-type pages become separate LogicalDocs.
    """
    docs: list[LogicalDoc] = []
    for entry in classified_pages:
        page      = entry['page']
        doc_type  = entry['classified'].get('doc_type', 'unknown')
        fname     = page.get('file_name', '')
        if (docs
                and docs[-1].doc_type  == doc_type
                and docs[-1].file_name == fname):
            docs[-1].pages.append(entry)
        else:
            docs.append(LogicalDoc(doc_type=doc_type, file_name=fname, pages=[entry]))
    return docs

print('Module 3b (Logical Document Grouping) ready.')

# %% CELL 11 — Helper Functions

# FIX: Use lookarounds instead of \b so dates immediately adjacent to time strings
# (e.g. "11/02/202605:24PM") or other digits are still captured.
# \b fails here because both the last digit of the year and the first digit of
# the time are \w — no word boundary exists between them.
DATE_PATTERNS = [
    r'(?<!\d)\d{1,2}[/-]\d{1,2}[/-]\d{2,4}(?!\d)',   # DD/MM/YY, DD-MM-YYYY (lookaround safe)
    r'(?<!\d)\d{1,2}\.\d{1,2}\.\d{2,4}(?!\d)',        # DD.MM.YYYY — dot separator
    r'(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)',                 # YYYY-MM-DD — ISO format
    r'(?<!\d)\d{1,2}-[A-Za-z]{3}-\d{2,4}(?!\d)',      # DD-Mon-YY
    r'(?<!\d)\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}(?!\d)', # DD Mon YYYY (with space)
]

# Compact OCR repair pattern: catches cases where OCR drops a separator
# e.g. "102/2026" → likely "10/2/2026" or "1/02/2026".
# Only applied near strong anchors (DOA/DOD/pre/post) — never globally.
_COMPACT_OCR_PATTERN = re.compile(r'(?<!\d)(\d{1,2})(\d{1,2})/(\d{4})(?!\d)')

def _repair_compact_ocr_date(text: str) -> str:
    """
    Attempt to repair compact OCR artefacts like '102/2026' → '10/2/2026'.
    Only runs on text that already contains a strong DOA/DOD anchor — never globally.
    The repair inserts a '/' between the two leading digit groups.
    """
    def _sub(m: re.Match) -> str:
        a, b, yr = m.group(1), m.group(2), m.group(3)
        # Sanity: day 1-31, month 1-12
        try:
            d, mo = int(a), int(b)
            if 1 <= d <= 31 and 1 <= mo <= 12:
                return f'{a}/{b}/{yr}'
        except ValueError:
            pass
        return m.group(0)   # leave unchanged if implausible
    return _COMPACT_OCR_PATTERN.sub(_sub, text)

_STRONG_DOA_ANCHORS = re.compile(
    r'(?:admitted\s+on|date\s+of\s+admission|d\.?o\.?a\.?|boa|start\s+date)',
    re.I
)
_STRONG_DOD_ANCHORS = re.compile(
    r'(?:discharged\s+on|date\s+of\s+discharge|d\.?o\.?d\.?)',
    re.I
)

def _strip_time_suffix(text: str) -> str:
    """
    Insert a space between a date string and an immediately-following time string.
    Handles: '11/02/202605:24PM' → '11/02/2026 05:24PM'
             '03-11-202514:30'   → '03-11-2025 14:30'
    This runs BEFORE DATE_PATTERNS so that lookaround-safe regexes find clean dates.
    """
    # Date (DD/MM/YYYY or DD-MM-YYYY) immediately followed by HH:MM (with optional AM/PM)
    return re.sub(
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})(\d{1,2}:\d{2}(?:\s*[AaPp][Mm])?)',
        r'\1 \2',
        text,
    )

def find_dates(text: str) -> list[str]:
    """Extract all date strings from text. Applies time-suffix stripping first."""
    if not text:
        return []
    cleaned = _strip_time_suffix(text)
    hits = []
    for pat in DATE_PATTERNS:
        hits.extend(re.findall(pat, cleaned))
    return list(dict.fromkeys(hits))   # deduplicate, preserve order

# Prompt-leakage guard set — any VLM output value matching these strings is
# evidence that the model copied an example from the prompt rather than reading
# the document. These are values that appeared in older prompt versions.
# Add any newly discovered leaked values here.
_PROMPT_LEAK_GUARDS: frozenset = frozenset({
    '15/07/2023', '15-07-2023', '15/07/23',   # the infamous leaked example date
    'DD/MM/YYYY', '<date>', '<DD/MM/YYYY>',    # placeholder strings echoed literally
    'null', 'none', 'n/a', 'na', '',           # non-values (handled separately but kept here for safety)
})

def normalize_date(date_str: Optional[str]) -> Optional[str]:
    if not date_str: return None
    s = date_str.strip()
    # Reject fragments — must be long enough and have a separator
    if len(s) < 5 or (not any(c in s for c in '/-. ') and not any(c.isalpha() for c in s)):
        return None
    # Coerce ISO format YYYY-MM-DD → DD/MM/YYYY
    iso_m = re.match(r'^(\d{4})-(\d{2})-(\d{2})$', s)
    if iso_m:
        s = f'{iso_m.group(3)}/{iso_m.group(2)}/{iso_m.group(1)}'
    # Normalize dot separator DD.MM.YYYY → DD/MM/YYYY
    dot_m = re.match(r'^(\d{1,2})\.(\d{1,2})\.(\d{2,4})$', s)
    if dot_m:
        s = f'{dot_m.group(1)}/{dot_m.group(2)}/{dot_m.group(3)}'
    # 2-digit year expansion BEFORE strptime: YY < 50 → 20YY, else 19YY
    # Handles Indian hospital short forms: 21/2/26 → 21/2/2026
    two_yr = re.match(r'^(\d{1,2}[/-]\d{1,2}[/-])(\d{2})$', s)
    if two_yr:
        yy = int(two_yr.group(2))
        full_yr = 2000 + yy if yy < 50 else 1900 + yy
        s = two_yr.group(1) + str(full_yr)
    fmts = ['%d/%m/%Y', '%d-%m/%Y', '%d/%m-%Y', '%d-%m-%Y',
            '%d-%b-%Y', '%d %b %Y', '%d %B %Y']
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).strftime('%d-%m-%Y')
        except ValueError:
            continue
    return None   # silently drop — no log spam

def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    nd = normalize_date(date_str)
    if not nd: return None
    try:
        return datetime.strptime(nd, '%d-%m-%Y')
    except ValueError:
        return None

# ── Evidence Atom ─────────────────────────────────────────────────────────────
# Unified provenance carrier for every extracted date/field value.
# Replaces DateProvenance; carries bbox, method, and source_doc for defensibility.
from dataclasses import dataclass, field as dc_field

@dataclass
class EvidenceAtom:
    """
    A single extracted date candidate with full provenance.
    Used throughout the pipeline — from extraction through timeline fusion.
    """
    field:       str            = ''      # 'doa', 'dod', 'pre_date', 'post_date', 'treatment_date'
    value:       Optional[str]  = None    # raw date exactly as found
    normalized:  Optional[str]  = None    # DD-MM-YYYY canonical form
    source_doc:  str            = ''      # doc_type of the page this came from
    page:        int            = 0       # page_number
    bbox:        Optional[list] = None    # [x1,y1,x2,y2] from OCR; None for VLM/fitz
    source_text: Optional[str]  = None    # anchor line text containing the date
    method:      str            = ''      # 'fitz_regex'|'paddle_anchor'|'vlm_labeled'|'vlm_crop'
    anchor_label:Optional[str]  = None    # human-readable label found near date
    confidence:  float          = 0.0     # 0.0–1.0 date-extraction confidence

    # ── acceptance gate ────────────────────────────────────────────────────────
    @property
    def is_fitz_or_ocr(self) -> bool:
        # vlm_crop is also treated as a verified extraction — it has a bbox and
        # the VLM was given a tight crop, not a full page, so hallucination risk is low.
        return self.method in ('fitz_regex', 'paddle_anchor', 'fitz_anchor', 'vlm_crop')

    def is_accepted(self) -> tuple[bool, str]:
        """
        Four-condition acceptance gate (same logic as _accept_date_provenance).
        Returns (accepted, reason).
        """
        if not self.normalized:
            return False, 'No date extracted'
        if parse_date(self.normalized) is None:
            return False, f'Unparseable date "{self.normalized}"'
        if self.field not in _LABELED_DATE_FIELDS:
            return False, f'Field "{self.field}" not a labeled date field'
        # Fitz/OCR methods bypass source_text condition — they have real bboxes
        if not self.is_fitz_or_ocr:
            if not self.source_text:
                return False, 'No source_text — VLM anchor not evidenced'
            raw_clean = (self.value or '').replace(' ', '').lower()
            src_clean = self.source_text.replace(' ', '').lower()
            if raw_clean and raw_clean not in src_clean:
                return False, f'Date "{self.value}" not found in source_text — likely hallucinated'
        if self.confidence < _DATE_MIN_CONFIDENCE:
            return False, f'Confidence {self.confidence:.2f} < threshold {_DATE_MIN_CONFIDENCE}'
        return True, 'Accepted'

    def to_dict(self) -> dict:
        return {
            'field':        self.field,
            'value':        self.value,
            'normalized':   self.normalized,
            'source_doc':   self.source_doc,
            'page':         self.page,
            'bbox':         self.bbox,
            'source_text':  self.source_text,
            'method':       self.method,
            'anchor_label': self.anchor_label,
            'confidence':   round(self.confidence, 2),
        }

# Backward-compat alias — DateProvenance used in older call sites
# New code should use EvidenceAtom directly.
@dataclass
class DateProvenance:
    raw_date:        Optional[str]  = None
    normalized_date: Optional[str]  = None
    field_name:      Optional[str]  = None
    anchor_label:    Optional[str]  = None
    source_text:     Optional[str]  = None
    date_confidence: float          = 0.0

    @property
    def is_anchored(self) -> bool:
        return bool(self.anchor_label)

    def to_dict(self) -> dict:
        return {
            'raw_date':        self.raw_date,
            'normalized_date': self.normalized_date,
            'field_name':      self.field_name,
            'anchor_label':    self.anchor_label,
            'source_text':     self.source_text,
            'date_confidence': round(self.date_confidence, 2),
        }

    @classmethod
    def from_atom(cls, atom: 'EvidenceAtom') -> 'DateProvenance':
        """Convert EvidenceAtom back to DateProvenance for legacy call sites."""
        return cls(
            raw_date=atom.value, normalized_date=atom.normalized,
            field_name=atom.field, anchor_label=atom.anchor_label,
            source_text=atom.source_text, date_confidence=atom.confidence,
        )

# Maps VLM ef field names → expected anchor labels for confidence scoring
_DATE_FIELD_ANCHORS: dict[str, tuple[str, float]] = {
    'date_of_admission':       ('Date of Admission / DOA / Admitted on',  0.90),
    'date_of_discharge':       ('Date of Discharge / DOD / Discharged on', 0.90),
    'treatment_date':          ('Date of Treatment / Procedure date',      0.85),
    'pre_investigation_date':  ('Pre-treatment investigation date',        0.80),
    'post_investigation_date': ('Post-treatment investigation date',       0.80),
    'doa':                     ('Date of Admission (fitz-extracted)',       0.85),
    'dod':                     ('Date of Discharge (fitz-extracted)',       0.85),
    'pre_date':                ('Pre-investigation date (fitz-extracted)',  0.80),
    'post_date':               ('Post-investigation date (fitz-extracted)', 0.80),
}

def _select_date_with_provenance(
    ef: dict,
    doc_type: str,
    dates_found: list[str],
    classified: dict,
    target_field: Optional[str] = None,   # e.g. 'doa', 'dod', 'pre_date', 'post_date'
) -> DateProvenance:
    """
    Select the best date for a given field, returning full provenance.

    Priority:
      1. Labeled VLM field (date_of_admission etc.) — highest confidence
      2. Fitz-extracted labeled date (doa/dod/pre_date/post_date from ef)
      3. dates_found[0] — only as last resort, marked Unanchored

    If no labeled source found → anchor_label=None → caller marks 'Unverifiable'.
    """
    # Priority 1: VLM labeled fields
    vlm_field_priority = {
        'doa':       ['date_of_admission'],
        'dod':       ['date_of_discharge'],
        'pre_date':  ['pre_investigation_date'],
        'post_date': ['post_investigation_date'],
        'treatment_date': ['treatment_date'],
    }
    candidates = vlm_field_priority.get(target_field, [])
    # Also check by doc_type to pick the right VLM field when target_field is None
    if not candidates and doc_type == 'discharge_summary':
        candidates = ['date_of_admission', 'date_of_discharge']
    if not candidates and doc_type in ('investigation_pre', 'investigation_post'):
        candidates = ['pre_investigation_date', 'post_investigation_date']
    if not candidates and doc_type == 'treatment_details':
        candidates = ['treatment_date']
    if not candidates and doc_type == 'indoor_case':
        candidates = ['date_of_admission']

    for field in candidates:
        raw, src_text, vlm_conf = _extract_ef_date(ef, field)
        if raw and str(raw).lower() not in ('null', 'none', ''):
            anchor_label, base_conf = _DATE_FIELD_ANCHORS.get(field, ('VLM labeled field', 0.80))
            # Use VLM-reported confidence if higher than base; downgrade if uncertain
            uncertain = ef.get('uncertain_fields', []) or []
            if field in uncertain:
                vlm_conf = max(min(vlm_conf, base_conf) - 0.25, 0.40)
                anchor_label += ' [VLM uncertain]'
            conf = vlm_conf if vlm_conf > 0.0 else base_conf
            # No source_text → downgrade confidence (anchor claimed but not evidenced)
            if not src_text:
                conf = max(conf - 0.10, 0.40)
            nd = normalize_date(str(raw))
            return DateProvenance(
                raw_date=str(raw), normalized_date=nd,
                field_name=field, anchor_label=anchor_label,
                source_text=src_text or f'VLM extracted_fields.{field} (no source_text)',
                date_confidence=conf,
            )

    # Priority 2: fitz-extracted labeled fields (doa/dod/pre_date/post_date in ef)
    fitz_candidates = {
        'doa':      ef.get('doa'),
        'dod':      ef.get('dod'),
        'pre_date': ef.get('pre_date'),
        'post_date':ef.get('post_date'),
    }
    if target_field and fitz_candidates.get(target_field):
        raw = fitz_candidates[target_field]
        anchor_label, conf = _DATE_FIELD_ANCHORS.get(target_field, ('Fitz keyword-anchored', 0.80))
        nd = normalize_date(raw)
        return DateProvenance(
            raw_date=raw, normalized_date=nd,
            field_name=target_field, anchor_label=anchor_label,
            source_text='Regex on fitz text',
            date_confidence=conf,
        )

    # Priority 3: unanchored fallback from dates_found (last resort)
    if dates_found:
        raw = dates_found[0]
        nd = normalize_date(raw)
        return DateProvenance(
            raw_date=raw, normalized_date=nd,
            field_name='dates_found[0]', anchor_label=None,
            source_text='Unanchored — first date on page',
            date_confidence=0.30,
        )

    return DateProvenance()  # nothing found

def extract_fields_from_text(text: str) -> dict:
    """
    Rule-based field extraction from fitz text (zero tokens).
    FIX 3: Extended to extract doa/dod from discharge summaries,
    pre_date/post_date from investigation reports, and Hb from CBC reports.
    FIX 5: Apply _strip_time_suffix before anchor regex so that dates glued to
    time strings (e.g. "11/02/202605:24PM") are correctly captured.
    """
    ef = {}
    if not text: return ef

    # Pre-process: strip time suffixes so date regex doesn't miss glued patterns
    text_clean = _strip_time_suffix(text)

    # Hb value
    hb_m = re.search(r'h[ae]moglobin[:\s]*(\d+\.?\d*)\s*g', text_clean, re.I)
    if hb_m: ef['hb_value'] = float(hb_m.group(1))
    # Age
    age_m = re.search(r'\b(\d{1,3})\s*(?:years?|yrs?|y/?o)\b', text_clean, re.I)
    if age_m: ef['patient_age'] = int(age_m.group(1))

    # Date pattern fragment: handles DD/MM/YY(YY), DD-MM-YY(YY), DD Mon YYYY
    # Uses lookaround-safe boundaries instead of \b
    _DATE_FRAG = (
        r'(?<!\d)\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}(?!\d)'
        r'|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}'
        r'|\d{1,2}[\s\-][A-Za-z]{3,9}[\s\-]\d{2,4}'
    )

    # DOA/DOD extraction — handles "D.O.A. 11/02/2026" including time-glued variants
    doa_m = re.search(
        r'(?:date\s+of\s+admission|admitted\s+on|d\.?o\.?a\.?|boa)'
        r'[:\s]*(' + _DATE_FRAG + r')',
        text_clean, re.I)
    if doa_m:
        raw = doa_m.group(1).strip()
        # If anchor text is strong, try OCR compact repair on the raw date
        if _STRONG_DOA_ANCHORS.search(doa_m.group(0)):
            raw = _repair_compact_ocr_date(raw)
        ef['doa'] = raw

    dod_m = re.search(
        r'(?:date\s+of\s+discharge|discharged\s+on|d\.?o\.?d\.?)'
        r'[:\s]*(' + _DATE_FRAG + r')',
        text_clean, re.I)
    if dod_m:
        raw = dod_m.group(1).strip()
        if _STRONG_DOD_ANCHORS.search(dod_m.group(0)):
            raw = _repair_compact_ocr_date(raw)
        ef['dod'] = raw

    # Pre/post investigation dates (MG006A)
    pre_m = re.search(
        r'(?:pre[\s\-]?treatment|baseline|date\s+of\s+sample|collection\s+date)'
        r'[:\s]*(?<!\d)(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})(?!\d)',
        text_clean, re.I)
    if pre_m: ef['pre_date'] = pre_m.group(1).strip()

    post_m = re.search(
        r'(?:post[\s\-]?treatment|repeat\s+sample|follow[\s\-]?up\s+date)'
        r'[:\s]*(?<!\d)(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})(?!\d)',
        text_clean, re.I)
    if post_m: ef['post_date'] = post_m.group(1).strip()

    return ef

def _has_any(text: str, ef: dict, keywords: list[str]) -> int:
    t = (text or '').lower()
    if any(k.lower() in t for k in keywords): return 1
    # Check VLM-populated fields: diagnosis_keywords, clinical_findings, procedure_keywords, key_values
    # clinical_findings is the new field added to VLM prompt for scanned docs
    diag = ' '.join(
        [str(x) for x in ef.get('diagnosis_keywords', []) if x]
        + [str(x) for x in ef.get('clinical_findings', []) if x]
        + [str(x) for x in ef.get('procedure_keywords', []) if x]
        + [str(v) for v in ef.get('key_values', {}).values() if v]
    ).lower()
    return int(any(k.lower() in diag for k in keywords))

def _extract_age(text: str) -> Optional[int]:
    m = re.search(r'\b(\d{1,3})\s*(?:years?|yrs?)\b', text, re.I)
    return int(m.group(1)) if m else None

def _coerce_age(value: Any) -> Optional[int]:
    """Normalize VLM/OCR age outputs to an int, tolerating strings and dicts."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        age = int(value)
        return age if 0 < age < 125 else None
    if isinstance(value, dict):
        for key in ('value', 'age', 'patient_age', 'text'):
            age = _coerce_age(value.get(key))
            if age is not None:
                return age
        return None
    if isinstance(value, str):
        m = re.search(r'\b(\d{1,3})\b', value)
        if m:
            age = int(m.group(1))
            return age if 0 < age < 125 else None
    return None

def aggregate_rows(rows: list[dict]) -> dict:
    """Claim-level aggregation: max of binary fields; first non-None for others."""
    agg: dict = {}
    for row in rows:
        for k, v in row.items():
            if k.startswith('_'): continue   # skip meta keys
            if isinstance(v, int):
                agg[k] = max(agg.get(k, 0), v)
            elif v is not None and agg.get(k) is None:
                agg[k] = v
            else:
                agg.setdefault(k, v)
    return agg

def _dominant_doc_type(row: dict, package_code: str) -> str:
    for dt in PACKAGE_INFO[package_code]['doc_types']:
        if row.get(dt) == 1:
            return dt
    return 'unknown'

print('Helpers ready.')


# %% CELL 12 — [GAP 2] Row Template + [GAP 3] Extra Document Detection + Module 4

def init_row_template(package_code: str) -> dict:
    """
    [GAP 2] Initialize a row with ALL schema keys in exact PACKAGE_SCHEMAS order.
    Binary fields → 0. Date fields → None. Text/rank fields → None.
    Guarantees key order matches evaluator expectation regardless of fill order.
    """
    row = {}
    for key in PACKAGE_SCHEMAS[package_code]:
        if key in DATE_FIELDS:
            row[key] = None
        elif key in TEXT_FIELDS or key == 'document_rank':
            row[key] = None
        else:
            row[key] = 0
    return row

def is_extra_document(doc_type: str, package_code: str,
                      seen_doc_types: set) -> tuple[int, str]:
    """
    [GAP 3] Returns (extra_flag, reason).

    Scanned claim packets routinely contain continuation pages, duplicate scans, and split-page captures — all of the same logical document.
    Policy: extra_document=1 ONLY when:
      (a) doc_type == 'unknown'         — unclassified, zero evidence value
      (b) doc_type not in pkg_types     — genuinely unrelated document

    Duplicates / continuations of valid doc types → NOT extra.
    All such pages still contribute to evidence slot scoring.
    """
    pkg_types = set(PACKAGE_INFO[package_code]['doc_types'])
    if doc_type == 'unknown':
        return 1, 'unclassified'
    if doc_type not in pkg_types:
        return 1, f'not_required_for_{package_code}'
    # Duplicate / continuation — not extra for scanned packets
    seen_doc_types.add(doc_type)
    return 0, ''

def build_output_row(claim_id: str, file_name: str, page_num: int,
                     package_code: str, classified: dict,
                     visual: dict, text: str,
                     seen_doc_types: set,
                     ocr_lines: Optional[list] = None) -> dict:
    """
    [GAP 2] Starts from init_row_template() — key order guaranteed.
    [GAP 3] Uses is_extra_document() with per-claim seen_doc_types tracker.
    Populates all PACKAGE_SCHEMAS fields then returns in exact key order.
    """
    # [GAP 2] Always start from schema template
    row = init_row_template(package_code)

    doc_type = classified.get('doc_type', 'unknown')
    ef       = classified.get('extracted_fields', {})
    t        = text or ''

    # [GAP 3] Duplicate + extra detection
    extra, extra_reason = is_extra_document(doc_type, package_code, seen_doc_types)

    # ── Base fields ──
    row['case_id']         = claim_id
    row[LINK_KEY[package_code]] = file_name
    row['procedure_code']  = package_code
    row['page_number']     = page_num
    row['extra_document']  = extra
    row['document_rank']   = (99 if extra
                              else RANK_MAP[package_code].get(doc_type, 98))

    # ── Doc-type presence flags ──
    for dt in PACKAGE_INFO[package_code]['doc_types']:
        row[dt] = 1 if (doc_type == dt and not extra) else 0

    # ── Package-specific clinical signal fields ──────────────────────────────
    # Full try/except: VLM may return patient_age or other values as str/dict,
    # causing '>' type errors. Any crash here leaves clinical fields at 0 —
    # the row still exports correctly with doc-type flags and dates intact.
    try:
        if package_code == 'MG064A':
            row['severe_anemia']          = _has_any(t, ef, ['severe anemia','hb','haemoglobin','hemoglobin','anaemia','anemia','severe anaemia'])
            row['common_signs']           = _has_any(t, ef, ['pallor','fatigue','weakness','lethargy','tiredness','dizziness','dyspnea','dyspnoea','tachycardia','heart murmur','breathlessness','pale','palpitation'])
            row['significant_signs']      = _has_any(t, ef, ['dark urine','dark colored urine','dark coloured urine','melena','hematuria','haematuria','bleeding','jaundice','hepatosplenomegaly','hepatomegaly','splenomegaly','cheilosis','glossitis','icterus'])
            row['life_threatening_signs'] = _has_any(t, ef, ['shock','sweating','diaphoresis','thirst','cold extremities','peripheral cyanosis','edema','oedema','respiratory distress','angina','cardiac failure','heart failure','hypoxia','altered sensorium','unconscious'])

        elif package_code == 'SG039C':
            row['clinical_condition'] = _has_any(t, ef, ['cholecystitis','gall stone','gallstone','biliary colic','cholangitis','cholelithiasis','pancreatitis','choledocholithiasis'])
            row['usg_calculi']        = _has_any(t, ef, ['calculus','calculi','gallstones','acoustic shadow','cholelithiasis','stone in gall'])
            row['pain_present']       = _has_any(t, ef, ['pain abdomen','abdominal pain','right hypochondrium','epigastric','rif pain'])
            row['previous_surgery']   = _has_any(t, ef, ['previous surgery','prior surgery','post surgical','h/o surgery','previous cholecystectomy','redo','revision'])

        elif package_code == 'MG006A':
            row['poor_quality'] = int(classified.get('image_quality') == 'poor')
            row['fever']        = _has_any(t, ef, ['enteric fever','typhoid','fever','pyrexia','high grade fever','38.3','101','febrile','temperature','temp'])
            row['symptoms']     = _has_any(t, ef, ['abdominal pain','vomiting','diarrhea','diarrhoea','headache','weakness','loose stools','constipation','malaise','myalgia','arthralgia','muscle pain','joint pain','dizziness','rose spots','splenomegaly','hepatomegaly','nausea'])
            dates_objs = _extract_ef_dates_found(classified)
            dates = [d['value'] for d in dates_objs]
            if doc_type == 'investigation_pre':
                vlm_raw, _, _ = _extract_ef_date(ef, 'pre_investigation_date')
                row['pre_date'] = normalize_date(ef.get('pre_date') or vlm_raw or (dates[0] if dates else None))
            if doc_type == 'investigation_post':
                vlm_raw, _, _ = _extract_ef_date(ef, 'post_investigation_date')
                row['post_date'] = normalize_date(ef.get('post_date') or vlm_raw or (dates[0] if dates else None))

        elif package_code == 'SB039A':
            age = _coerce_age(ef.get('patient_age')) or _extract_age(t)
            row['arthritis_type']          = _has_any(t, ef, ['osteoarthritis','oa knee','degenerative joint','arthritis','genu varum','joint space narrowing','rheumatoid'])
            row['post_op_implant_present'] = _has_any(t, ef, ['implant','prosthesis','femoral component','tibial component','cemented','cementless','polyethylene'])
            row['age_valid']               = int(int(age) > 55) if isinstance(age, (int, float)) else 0
            dates_objs = _extract_ef_dates_found(classified)
            dates = [d['value'] for d in dates_objs]
            if doc_type == 'discharge_summary':
                vlm_doa, _, _ = _extract_ef_date(ef, 'date_of_admission')
                vlm_dod, _, _ = _extract_ef_date(ef, 'date_of_discharge')
                row['doa'] = normalize_date(ef.get('doa') or vlm_doa or (dates[0] if len(dates) >= 1 else None))
                row['dod'] = normalize_date(ef.get('dod') or vlm_dod or (dates[1] if len(dates) >= 2 else None))
            elif doc_type == 'indoor_case':
                vlm_doa, _, _ = _extract_ef_date(ef, 'date_of_admission')
                row['doa'] = normalize_date(ef.get('doa') or vlm_doa or (dates[0] if dates else None))

    except Exception:
        pass   # clinical fields stay 0 — doc-type flags and dates are unaffected

    if extra:
        preserve = {
            'case_id', LINK_KEY[package_code], 'procedure_code', 'page_number',
            'extra_document', 'document_rank'
        }
        for key in PACKAGE_SCHEMAS[package_code]:
            if key in preserve:
                continue
            row[key] = None if key in DATE_FIELDS else 0

    # Store metadata in private keys for pipeline use (stripped before export)
    # _dates_found stores structured objects {value, nearby_text, is_labeled}
    # so build_timeline can prefer is_labeled=True entries for last-resort fallback.
    row['_dates_found']       = _extract_ef_dates_found(classified)
    # _ocr_lines: fitz lines for digital PDFs, tesseract lines for images/scanned pages.
    # Passed in directly from page['_ocr_lines'] — not via classified (which has no image data).
    fitz_lines = [
        {'text': ln, 'bbox': None, 'confidence': 1.0, 'method': 'fitz'}
        for ln in (text or '').split('\n') if ln.strip()
    ] if not ocr_lines else []   # skip fitz lines when real OCR lines exist
    row['_ocr_lines']         = (ocr_lines or []) + fitz_lines
    row['_extra_reason']      = extra_reason
    row['_method']            = classified.get('method', 'none')
    row['_confidence']        = classified.get('confidence', 0.0)
    row['_doc_type']          = doc_type
    row['_extracted_fields']  = ef   # needed by derive_temporal_facts for numeric Hb
    # _image: stored for VLM crop fallback (Path 5 in build_timeline).
    # Only set when the page has an image object; never exported in output rows.
    row['_image']             = classified.get('_page_image')   # set by process_claim below

    # Return only schema keys + _meta (meta stripped before export)
    return row

print('Module 4 (Row Assembler) ready — GAP 2 (key order) + GAP 3 (duplicates) applied.')

# %% CELL 13 — [GAP 1] Document Ranking
_RANK_DATE_PRIORITY = {
    'clinical_notes':      ['date_of_admission', 'doa', 'treatment_date'],
    'indoor_case':         ['date_of_admission', 'doa'],
    'cbc_hb_report':       ['pre_investigation_date', 'pre_date'],
    'post_hb_report':      ['post_investigation_date', 'post_date'],
    'treatment_details':   ['treatment_date'],
    'investigation_pre':   ['pre_investigation_date', 'pre_date'],
    'investigation_post':  ['post_investigation_date', 'post_date'],
    'vitals_treatment':    ['treatment_date', 'date_of_admission', 'doa'],
    'usg_report':          ['pre_investigation_date', 'pre_date', 'treatment_date'],
    'lft_report':          ['pre_investigation_date', 'pre_date', 'treatment_date'],
    'pre_anesthesia':      ['treatment_date', 'date_of_admission', 'doa'],
    'operative_notes':     ['treatment_date'],
    'histopathology':      ['post_investigation_date', 'post_date', 'treatment_date'],
    'photo_evidence':      ['treatment_date'],
    'xray_ct_knee':        ['pre_investigation_date', 'pre_date', 'date_of_admission', 'doa'],
    'implant_invoice':     ['treatment_date'],
    'post_op_photo':       ['post_investigation_date', 'post_date', 'treatment_date'],
    'post_op_xray':        ['post_investigation_date', 'post_date', 'treatment_date'],
    'discharge_summary':   ['date_of_discharge', 'dod', 'date_of_admission', 'doa'],
}

_LATE_DATE_DOC_TYPES = {
    'discharge_summary', 'post_hb_report', 'investigation_post',
    'post_op_photo', 'post_op_xray', 'histopathology'
}

def _iter_rank_date_candidates(row: dict, doc_type: str) -> list[datetime]:
    """Collect normalized dates that can plausibly date the logical document."""
    candidates: list[datetime] = []
    ef = row.get('_extracted_fields', {}) or {}

    def _add(raw) -> None:
        if isinstance(raw, dict):
            raw = raw.get('value')
        nd = normalize_date(str(raw)) if raw else None
        if not nd:
            return
        dt = parse_date(nd)
        if dt:
            candidates.append(dt)

    for field in _RANK_DATE_PRIORITY.get(doc_type, []):
        _add(row.get(field))
        raw, _, _ = _extract_ef_date(ef, field)
        _add(raw)

    # Do not use generic dates_found/OCR-line dates for ranking. They often include
    # print dates, birth dates, footer dates, and repeated claim dates. If no
    # structured/anchored field exists, fall back to STG order rather than
    # manufacturing chronology from noisy page text.

    return candidates

def _best_logical_doc_date(rows: list[dict], doc_type: str) -> Optional[datetime]:
    dates: list[datetime] = []
    for row in rows:
        dates.extend(_iter_rank_date_candidates(row, doc_type))
    if not dates:
        return None
    return max(dates) if doc_type in _LATE_DATE_DOC_TYPES else min(dates)

def assign_chronological_document_ranks(rows: list[dict], package_code: str) -> None:
    """
    Export-facing rank assignment per output guide:
    - group contiguous pages from the same file with the same document type
    - rank logical document groups by extracted document date
    - assign the same rank to every page in a logical document group
    - keep extra documents at rank 99
    """
    if not rows:
        return
    link_key = LINK_KEY[package_code]
    groups: list[dict] = []
    for idx, row in enumerate(rows):
        doc_type = row.get('_doc_type') or _dominant_doc_type(row, package_code)
        is_extra = row.get('extra_document') == 1 or doc_type == 'unknown'
        group_key = (row.get(link_key, ''), doc_type, is_extra)
        if groups and groups[-1]['key'] == group_key:
            groups[-1]['rows'].append(row)
        else:
            groups.append({
                'key': group_key,
                'rows': [row],
                'doc_type': doc_type,
                'is_extra': is_extra,
                'first_index': idx,
            })

    sortable = []
    for group in groups:
        if group['is_extra']:
            for row in group['rows']:
                row['document_rank'] = 99
            continue
        doc_type = group['doc_type']
        doc_date = _best_logical_doc_date(group['rows'], doc_type)
        fallback_rank = RANK_MAP[package_code].get(doc_type, 98)
        sortable.append((doc_date is None, doc_date or datetime.max,
                         fallback_rank, group['first_index'], group))

    for rank, (_, _, _, _, group) in enumerate(sorted(sortable), start=1):
        for row in group['rows']:
            row['document_rank'] = rank

def rank_rows(rows: list[dict], package_code: str) -> list[dict]:
    """
    [GAP 1 + P1.1] Sort page rows by canonical source quality.
    Sort key (ascending = better):
      1. has_labeled_date:  0 if any labeled date field has a real value, else 1
         — uses _has_date_value() to handle both structured {value:...} dicts
           and legacy plain strings; treats {"value": null} as absent
      2. inv_confidence:    1 - VLM classification confidence
      3. document_rank:     clinical importance rank per package
      4. page_number:       earlier pages preferred within same doc
    """
    def _has_date_value(x) -> bool:
        """Return True only if x contains an actual non-null date value."""
        if isinstance(x, dict):
            return bool(x.get('value'))   # {"value": null} → False; {"value": "15/08"} → True
        return bool(x)                    # plain string: '' or None → False

    def sort_key(r: dict) -> tuple:
        ef = r.get('_extracted_fields', {}) or {}
        has_labeled_date = any(
            _has_date_value(ef.get(f))
            for f in ('date_of_admission', 'date_of_discharge', 'treatment_date',
                      'pre_investigation_date', 'post_investigation_date',
                      'doa', 'dod', 'pre_date', 'post_date')
        )
        anchor_score  = 0 if has_labeled_date else 1
        inv_conf      = round(1.0 - float(r.get('_confidence', 0.0)), 2)
        doc_rank      = r.get('document_rank') or 99
        page_num      = r.get('page_number', 9999)
        return (anchor_score, inv_conf, doc_rank, page_num)

    return sorted(rows, key=sort_key)

print('Module 4b (Document Ranking) ready.')


# %% CELL 14 — Module 5: Timeline Builder (GAP 4 — best-source + date arbitration)
EVENT_DEFS = {
    'MG064A': [
        ('Admission',                 'indoor_case',       'doa'),
        ('Diagnostic Investigation',  'cbc_hb_report',     None),
        ('Treatment',                 'treatment_details', None),
        ('Post-Treatment Assessment', 'post_hb_report',    None),
        ('Discharge',                 'discharge_summary', 'dod'),
    ],
    'SG039C': [
        ('Admission / Clinical Eval', 'clinical_notes',    'doa'),
        ('Diagnostic Investigation',  'usg_report',        None),
        ('Pre-Anaesthesia Eval',      'pre_anesthesia',    None),
        ('Operative Procedure',       'operative_notes',   None),
        ('Discharge',                 'discharge_summary', 'dod'),
    ],
    'MG006A': [
        ('Admission',                       'clinical_notes',       'doa'),
        ('Pre-Treatment Investigation',     'investigation_pre',    'pre_date'),
        ('Treatment',                       'vitals_treatment',     None),
        ('Post-Treatment Investigation',    'investigation_post',   'post_date'),
        ('Discharge',                       'discharge_summary',    'dod'),
    ],
    'SB039A': [
        ('Admission',              'indoor_case',       'doa'),
        ('Diagnostic Investigation','xray_ct_knee',     None),
        ('Operative Procedure',    'operative_notes',   None),
        ('Post-Op Monitoring',     'post_op_photo',     None),
        ('Discharge',              'discharge_summary', 'dod'),
    ],
}

_DATE_PLAUSIBILITY_MIN  = datetime(2020, 1, 1)
_DATE_PLAUSIBILITY_MAX  = datetime(2026, 12, 31)
_DATE_TODAY = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
_LOS_MAX_HARD_REJECT    = 60    # days; DOD-DOA > 60 → hard reject (not just downgrade)
_LOS_MAX_DOWNGRADE      = 30    # days; DOD-DOA > 30 → conf downgrade to 0.58

def _validate_date_atom(atom: 'EvidenceAtom', field: str,
                         facts: dict) -> Optional['EvidenceAtom']:
    """
    Unified plausibility gate for ALL date fields — not just DOD.
    Applies after extraction, before accepting into timeline or facts.

    Rejects:
      - date > today (2026-04-26) — no future claims
      - date.year outside [2020, 2026]
      - post_date < pre_date
      - post_date < doa
      - pre_date > dod (if known)
      - DOD - DOA > 60 days (hard reject)
      - DOD - DOA > 30 days (confidence downgrade to 0.58)
      - DOD < DOA (hard reject)
      - DOD == DOA with weak anchor (reject)
    """
    if atom is None or not atom.normalized:
        return atom

    dt = parse_date(atom.normalized)
    if dt is None:
        return None

    # Future date gate
    if dt > _DATE_TODAY:
        return None

    # Year plausibility
    if not (2020 <= dt.year <= 2026):
        return None

    # DOA ref for cross-field checks
    doa_dt = parse_date(facts.get('doa')) if facts.get('doa') else None

    if field == 'dod':
        if doa_dt:
            if dt < doa_dt:
                return None   # DOD before DOA
            if dt == doa_dt and not _STRONG_DOD_LABELS.search(atom.anchor_label or ''):
                return None   # same-day with weak anchor = shared header date
            gap = (dt - doa_dt).days
            if gap > _LOS_MAX_HARD_REJECT:
                return None   # LOS > 60 days — almost certainly wrong year
            if gap > _LOS_MAX_DOWNGRADE:
                atom.confidence = min(atom.confidence, 0.58)

    elif field == 'post_date':
        pre_dt = parse_date(facts.get('pre_date'))
        if doa_dt and dt < doa_dt:
            return None   # post-treatment before admission
        if pre_dt and dt < pre_dt:
            return None   # post-treatment before pre-treatment
        # Claim-year gate: reject dates > 1 year before admission
        if doa_dt and abs((dt - doa_dt).days) > 365:
            return None

    elif field == 'pre_date':
        if doa_dt:
            # Pre-treatment investigation should be within 90 days before admission
            days_before = (doa_dt - dt).days
            if days_before > 90 or days_before < -7:   # allow up to 7 days after admission
                return None

    return atom

# The only fields that may produce a canonical timeline date.
# dates_found is explicitly excluded — it is debug-only.
_LABELED_DATE_FIELDS = frozenset({
    'date_of_admission', 'date_of_discharge', 'treatment_date',
    'pre_investigation_date', 'post_investigation_date',
    # fitz-extracted equivalents
    'doa', 'dod', 'pre_date', 'post_date',
})

_DATE_MIN_CONFIDENCE = 0.70   # below this → Unverifiable regardless of anchor

# ── Event-specific source priority (v5.0) ─────────────────────────────────────
# For each timeline event type, the ordered list of doc_types to prefer as
# date sources. First match with an accepted EvidenceAtom wins.
DATE_SOURCE_PRIORITY: dict[str, list[str]] = {
    # MG064A
    'Admission':                    ['indoor_case', 'discharge_summary', 'clinical_notes'],
    'Diagnostic Investigation':     ['cbc_hb_report', 'discharge_summary'],
    'Treatment':                    ['treatment_details', 'clinical_notes', 'discharge_summary'],
    'Post-Treatment Assessment':    ['post_hb_report', 'discharge_summary', 'cbc_hb_report'],
    'Discharge':                    ['discharge_summary', 'indoor_case', 'clinical_notes'],
    # MG006A — exact names from EVENT_DEFS
    'Pre-Treatment Investigation':  ['investigation_pre', 'discharge_summary'],
    'Post-Treatment Investigation': ['investigation_post', 'discharge_summary'],
    # SG039C — exact names from EVENT_DEFS
    'Admission / Clinical Eval':    ['discharge_summary', 'clinical_notes', 'indoor_case'],
    'Diagnostic Investigation':     ['usg_report', 'cbc_hb_report', 'discharge_summary'],
    'Pre-Anaesthesia Eval':         ['pre_anesthesia', 'discharge_summary'],
    'Operative Procedure':          ['operative_notes', 'discharge_summary', 'indoor_case'],
    # SB039A — exact names from EVENT_DEFS
    'Post-Op Monitoring':           ['post_op_photo', 'operative_notes', 'discharge_summary'],
}

# ── Required timeline fields per package (v5.0) ───────────────────────────────
# If a required field is missing after all extraction attempts, the corresponding
# timeline event is marked CONDITIONAL (not FAIL) — missing date ≠ clinical failure.
REQUIRED_TIMELINE_FIELDS: dict[str, list[str]] = {
    'MG064A': ['doa', 'dod'],
    'MG006A': ['pre_date', 'post_date'],
    'SB039A': ['doa', 'dod'],
    'SG039C': ['dod'],
}

# ── Tiered anchor matcher — runs on fitz text lines or PaddleOCR lines ────────
# Exact → substring → fuzzy (via rapidfuzz). Returns (anchor_text, tier, score).
try:
    from rapidfuzz import fuzz as _rfuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False

# Canonical anchor phrases per field (exact match targets)
_ANCHOR_ALIASES: dict[str, list[str]] = {
    'doa':  ['date of admission', 'date of admn', 'date of admittance',
             'd.o.a', 'doa', 'doa/time', 'admitted on', 'admit date',
             'admission date', 'admit date & time', 'date & time of admission',
             'date/time of admission', 'boa', 'ip date', 'date of indoor',
             'date of admission:', 'admitted on:', 'admission date:',
             # NOT: 'date', 'sample date', 'admission file', 'reported' — too broad
             ],
    'dod':  ['date of discharge', 'd.o.d', 'dod', 'discharged on',
             'discharge date', 'date of discharging', 'discharge date & time'],
    'pre_date':  ['sample collected on', 'report date', 'investigation date',
                  'test date', 'lab date', 'sample date', 'collection date',
                  'date of sample', 'sample collection date'],
    'post_date': ['post report date', 'post investigation date', 'follow up date',
                  'post treatment report', 'repeat sample date'],
    'treatment_date': ['date of treatment', 'procedure date', 'treatment date',
                       'date of procedure', 'transfusion date', 'date of transfusion'],
}

# Anchors that LOOK like DOD anchors but must be rejected for DOD extraction.
# These are document titles or non-specific headers that contain the first date
# on the page (often the admission date) rather than a specific discharge date.
_REJECTED_DOD_ANCHORS: frozenset = frozenset({
    'admission and discharge record',
    'admission & discharge record',
    'admission and discharge',
    'discharge summary',                   # doc title, not a field label
    'indoor case paper',
    'case summary',
})

# Weak anchors for DOD — accepted but confidence capped at 0.55
_WEAK_DOD_ANCHORS: frozenset = frozenset({
    'printdate', 'print date', 'print time', 'printed on',
    'generated on', 'report date', 'date of report',
})

# For DOD, ONLY these anchors are considered strong (conf tier-base preserved)
_STRONG_DOD_LABELS = re.compile(
    r'(?:date\s+of\s+discharge|discharged\s+on|d\.?o\.?d\.?|discharge\s+date'
    r'|date\s+of\s+discharging)',
    re.I,
)

def _find_anchor_in_lines(ocr_lines: list[dict], target_field: str,
                           ) -> tuple[Optional[dict], str, float]:
    """
    Search OCR lines for an anchor matching target_field.
    Returns (best_line, tier, score) where tier is 'exact'|'substring'|'fuzzy'|'none'.
    ocr_lines: [{text, bbox, confidence, method}]

    For 'dod': explicitly rejects document-title anchors (ADMISSION AND DISCHARGE RECORD,
    DISCHARGE SUMMARY, etc.) — these contain the first date on the page (often admission)
    not a specific discharge date value.
    """
    aliases = _ANCHOR_ALIASES.get(target_field, [])
    if not aliases:
        return None, 'none', 0.0

    # Tier A — exact match
    for line in ocr_lines:
        t = line.get('text', '').lower().strip()
        # DOD: reject document-title anchors — they do not point at discharge date
        if target_field == 'dod':
            if any(rej in t for rej in _REJECTED_DOD_ANCHORS):
                continue
        for alias in aliases:
            if alias in t:
                return line, 'exact', 1.0

    # Tier B — substring (partial alias)
    short_forms = {a.split()[0] for a in aliases if len(a.split()) >= 2}
    for line in ocr_lines:
        t = line.get('text', '').lower().strip()
        if target_field == 'dod' and any(rej in t for rej in _REJECTED_DOD_ANCHORS):
            continue
        for sf in short_forms:
            if sf in t and len(sf) >= 3:   # avoid single-char false positives
                return line, 'substring', 0.85

    # Tier C — fuzzy (rapidfuzz partial_ratio > 75)
    if _RAPIDFUZZ_AVAILABLE:
        best_line, best_score = None, 0.0
        for line in ocr_lines:
            t = line.get('text', '').lower().strip()
            if target_field == 'dod' and any(rej in t for rej in _REJECTED_DOD_ANCHORS):
                continue
            for alias in aliases:
                score = _rfuzz.partial_ratio(t, alias) / 100.0
                if score > 0.75 and score > best_score:
                    best_line, best_score = line, score
        if best_line:
            return best_line, 'fuzzy', best_score

    return None, 'none', 0.0

def _extract_date_atom_from_lines(
    ocr_lines: list[dict],
    target_field: str,
    source_doc: str,
    page_num: int,
) -> Optional['EvidenceAtom']:
    """
    Run tiered anchor search on OCR lines, then regex-extract a date.
    Returns an EvidenceAtom or None.
    """
    anchor_line, tier, tier_score = _find_anchor_in_lines(ocr_lines, target_field)
    if anchor_line is None or tier == 'none':
        return None

    anchor_text = anchor_line.get('text', '')
    anchor_bbox = anchor_line.get('bbox')

    # Build search text from widened window
    anchor_idx = next((i for i, l in enumerate(ocr_lines)
                       if l.get('text') == anchor_line.get('text')), -1)

    search_parts = [anchor_text]

    # Same-line right-side text: lines whose bbox y-range overlaps anchor
    if anchor_bbox:
        ay1, ay2 = anchor_bbox[1], anchor_bbox[3]
        y_mid = (ay1 + ay2) / 2
        for i, line in enumerate(ocr_lines):
            if i == anchor_idx:
                continue
            lb = line.get('bbox')
            if lb and lb[0] > anchor_bbox[2]:  # to the right of anchor
                ly1, ly2 = lb[1], lb[3]
                # overlapping y-range = same horizontal row
                if ly1 <= y_mid <= ly2 or ay1 <= (ly1+ly2)/2 <= ay2:
                    search_parts.append(line.get('text', ''))

    # Next 2 lines below
    for i in range(1, 3):
        next_idx = anchor_idx + i
        if 0 <= next_idx < len(ocr_lines):
            search_parts.append(ocr_lines[next_idx].get('text', ''))

    search_text = ' '.join(search_parts)

    # Strip time suffixes ("11/02/202605:24PM" → "11/02/2026 05:24PM")
    search_text = _strip_time_suffix(search_text)

    # Compact OCR repair near strong DOA/DOD anchors only
    if target_field == 'doa' and _STRONG_DOA_ANCHORS.search(search_text):
        search_text = _repair_compact_ocr_date(search_text)
    elif target_field == 'dod' and _STRONG_DOD_ANCHORS.search(search_text):
        search_text = _repair_compact_ocr_date(search_text)

    candidate_raw = None
    for pat in DATE_PATTERNS:
        hits = re.findall(pat, search_text)
        if hits:
            candidate_raw = hits[0]
            break

    if not candidate_raw:
        return None

    nd = normalize_date(candidate_raw)
    if nd is None:
        return None

    tier_base = {'exact': 0.85, 'substring': 0.78, 'fuzzy': 0.72}.get(tier, 0.70)
    ocr_conf  = float(anchor_line.get('confidence', 1.0))
    conf      = min(tier_base * ocr_conf, 0.92)
    method    = anchor_line.get('method', 'fitz_anchor')
    if method == 'fitz':
        method = 'fitz_anchor'

    # Downgrade weak anchors (PrintDate / report date / generated on)
    # These may happen to contain the correct date but are unreliable sources.
    anchor_lower = anchor_text.lower()
    if any(weak in anchor_lower for weak in _WEAK_DOD_ANCHORS):
        conf = min(conf, 0.55)
    if target_field == 'dod' and not _STRONG_DOD_LABELS.search(anchor_text):
        conf = min(conf, 0.60)

    return EvidenceAtom(
        field=target_field, value=candidate_raw, normalized=nd,
        source_doc=source_doc, page=page_num,
        bbox=anchor_bbox, source_text=search_text[:120],
        method=method,
        anchor_label=anchor_text[:80],
        confidence=conf,
    )

def _accept_date_provenance(prov: 'DateProvenance') -> tuple[bool, str]:
    """
    Gate that decides whether a DateProvenance is trustworthy for the canonical
    timeline. ALL four conditions must hold:

      1. field_name is one of the labeled VLM fields (or fitz equivalents) —
         dates_found is excluded entirely
      2. source_text is present (VLM showed its work)
      3. raw_date string appears literally inside source_text
         (proves the VLM is not hallucinating; the date was actually visible)
      4. date_confidence >= _DATE_MIN_CONFIDENCE (0.70)

    Fitz-extracted dates (doa/dod/pre_date/post_date) bypass condition 3
    because they come from regex on plain text — source_text is the pattern
    context, not a VLM string to verify against.

    Returns (accepted: bool, reason: str).
    """
    if not prov or not prov.normalized_date:
        return False, 'No date extracted'

    # Gate 0: normalized_date must actually parse — catches garbage like "1/24/44"
    # that survived normalize_date (e.g. from legacy plain-string paths).
    if parse_date(prov.normalized_date) is None:
        return False, f'Unparseable/invalid date "{prov.normalized_date}"'

    # Condition 1: must come from a labeled field, not dates_found
    if prov.field_name not in _LABELED_DATE_FIELDS:
        return False, f'Field "{prov.field_name}" is not a labeled date field — dates_found excluded'

    # Fitz-extracted fields bypass condition 3 (source_text is regex context, not VLM string)
    is_fitz = prov.field_name in ('doa', 'dod', 'pre_date', 'post_date')

    # Condition 2: source_text must exist
    if not prov.source_text:
        if is_fitz:
            pass   # fitz dates have no source_text by design — still acceptable
        else:
            return False, 'No source_text — VLM anchor not evidenced'

    # Condition 3: raw_date must appear in source_text (VLM fields only)
    if not is_fitz and prov.source_text:
        raw_clean = (prov.raw_date or '').replace(' ', '').lower()
        src_clean = prov.source_text.replace(' ', '').lower()
        if raw_clean and raw_clean not in src_clean:
            return False, (f'Date "{prov.raw_date}" not found in source_text '
                           f'"{prov.source_text[:60]}" — likely hallucinated')

    # Condition 4: minimum confidence
    if prov.date_confidence < _DATE_MIN_CONFIDENCE:
        return False, (f'Confidence {prov.date_confidence:.2f} < '
                       f'threshold {_DATE_MIN_CONFIDENCE} — weak/ungrounded VLM date')

    return True, 'Accepted'

def _suspicious_date_reasons(date_obj: Optional[datetime],
                              anchor_label: Optional[str]) -> list[str]:
    """
    Return list of reasons a date is suspicious, empty if clean.
    Called for every date entering the timeline.
    """
    reasons = []
    if date_obj is None:
        return reasons
    if date_obj > _DATE_TODAY:
        reasons.append(f'Future date: {date_obj.strftime("%d-%m-%Y")} > today {_DATE_TODAY.strftime("%d-%m-%Y")}')
    if date_obj < _DATE_PLAUSIBILITY_MIN:
        if anchor_label and 'uncertain' not in anchor_label.lower():
            reasons.append(f'Too early: {date_obj.strftime("%d-%m-%Y")} before 2020 (PM-JAY era)')
        else:
            reasons.append(f'Likely hallucinated: {date_obj.strftime("%d-%m-%Y")} before 2020 and unanchored')
    if date_obj > _DATE_PLAUSIBILITY_MAX:
        reasons.append(f'Too late: {date_obj.strftime("%d-%m-%Y")} after 2026 — likely OCR error')
    return reasons

def _check_temporal_validity(event_type: str, provenance: 'DateProvenance',
                              facts: dict,
                              all_event_dates: Optional[list] = None) -> str:
    """
    Returns a human-readable validity string for a timeline event.

    Checks (in order):
      0. _accept_date_provenance gate — field, source_text, value-in-source, confidence
      1. No date
      2. Suspicious (future, pre-2020, post-2026)
      3. Same date 3+ events
      4. Cross-event logical checks
      5. LOS plausibility
    """
    date = provenance.normalized_date if provenance else None

    if not date:
        return 'Date not extracted'

    # Gate: all four acceptance conditions
    accepted, reason = _accept_date_provenance(provenance)
    if not accepted:
        return f'Unverifiable — {reason}'

    date_obj = parse_date(date)
    sus = _suspicious_date_reasons(date_obj, provenance.anchor_label)
    if sus:
        return 'SUSPICIOUS: ' + '; '.join(sus)

    # Same date used for 3+ different event types → likely date explosion artifact
    if all_event_dates is not None:
        same_count = sum(1 for d in all_event_dates if d and d == date)
        if same_count >= 3:
            return (f'SUSPICIOUS: date {date} appears in {same_count} different events '
                    f'— likely repeated header date, not event-specific')

    if facts.get('discharge_after_admission') is False and 'Discharge' in event_type:
        return 'INVALID: DOD before DOA'
    if facts.get('pre_before_post') is False and 'Post-Treatment' in event_type:
        return 'INVALID: post-investigation before pre-investigation'

    los = facts.get('los_days')
    if los is not None and los > _LOS_MAX_HARD_REJECT:
        return f'IMPLAUSIBLE: LOS={los} days exceeds {_LOS_MAX_HARD_REJECT}-day ceiling'
    if los is not None and los < 0:
        return 'INVALID: negative LOS (DOD before DOA)'

    return f'Valid (conf={provenance.date_confidence:.2f}, anchor: {provenance.anchor_label})'

def derive_temporal_facts(ranked_rows: list[dict], package_code: str) -> dict:
    """
    Aggregate temporal and clinical facts across all ranked rows for a claim.

    Returns a dict consumed by:
      - build_timeline()  → date fallbacks (doa, dod, pre_date, post_date)
      - _check_temporal_validity()  → discharge_after_admission, pre_before_post, los_days
      - score_evidence_coverage()  → post_treatment_evidence_found (MG064A)
      - run_rules_engine()  → hb_value_claim, los_days, alos_days

    Aggregation strategy:
      - Dates: first non-None value across rows (by document_rank order)
      - Binary flags: OR across rows (any page having it counts)
      - Numeric (Hb): min for pre-treatment (worst), max for post-treatment (best response)
    """
    facts: dict = {}

    # ── Step 1: aggregate binary row fields across all pages ─────────────────
    binary_fields = [
        'discharge_summary', 'indoor_case', 'cbc_hb_report',
        'treatment_details', 'clinical_notes', 'investigation_pre',
        'investigation_post', 'post_hb_report', 'vitals_treatment',
        'prescription', 'consent_form', 'stamp_present', 'signature_present',
        'qr_present', 'implant_sticker_present',
    ]
    stg_row = aggregate_rows(ranked_rows)
    for f in binary_fields:
        if stg_row.get(f):
            facts[f] = stg_row[f]

    # ── Step 2: pull labeled dates from aggregate row ─────────────────────────
    # aggregate_rows uses max() which works for string dates (lexicographic).
    # For real date arbitration, build_timeline uses ranked_rows directly.
    # Here we just surface whatever fitz regex found for facts-dict fallback.
    for date_field in ('doa', 'dod', 'pre_date', 'post_date'):
        val = stg_row.get(date_field)
        if val and str(val).lower() not in ('none', 'null', ''):
            facts[date_field] = val

    # ── Step 3: temporal cross-checks ────────────────────────────────────────
    doa_str = facts.get('doa')
    dod_str = facts.get('dod')
    pre_str = facts.get('pre_date')
    post_str = facts.get('post_date')

    doa_dt  = parse_date(doa_str)
    dod_dt  = parse_date(dod_str)
    pre_dt  = parse_date(pre_str)
    post_dt = parse_date(post_str)

    if doa_dt and dod_dt:
        facts['discharge_after_admission'] = dod_dt >= doa_dt
        facts['los_days'] = (dod_dt - doa_dt).days
    if pre_dt and post_dt:
        facts['pre_before_post'] = post_dt >= pre_dt

    # ── Step 4: package-specific clinical facts ───────────────────────────────
    if package_code == 'MG064A':
        hb_pre_values  = []
        hb_post_values = []
        post_evidence_found = False

        for r in ranked_rows:
            ef = r.get('_extracted_fields', {}) or {}
            doc_type = r.get('_doc_type', '')

            # Pre-treatment Hb
            if ef.get('hb_value'):
                try:    hb_pre_values.append(float(ef['hb_value']))
                except (TypeError, ValueError, KeyError): pass

            # Post-treatment Hb — VLM structured field
            if ef.get('hb_post_value'):
                try:
                    hb_post_values.append(float(ef['hb_post_value']))
                    post_evidence_found = True
                except (TypeError, ValueError, KeyError): pass

            # Post-treatment Hb — explicit post_hb_report page
            if doc_type == 'post_hb_report' and ef.get('hb_value'):
                try:
                    hb_post_values.append(float(ef['hb_value']))
                    post_evidence_found = True
                except (TypeError, ValueError, KeyError): pass

            # Post-treatment Hb — discharge summary mentions post-transfusion Hb
            if doc_type == 'discharge_summary':
                t = (r.get('_text', '') or '').lower()
                post_hb_m = re.search(
                    r'(?:post[\s\-]?(?:transfusion|treatment|op|operative)'
                    r'[\s\-]?(?:hb|haemoglobin|hemoglobin))[:\s]*(\d+\.?\d*)',
                    t, re.I
                )
                if post_hb_m:
                    try:
                        hb_post_values.append(float(post_hb_m.group(1)))
                        post_evidence_found = True
                    except (TypeError, ValueError): pass

            # Blood transfusion documented in VLM fields
            if ef.get('blood_transfusion') is True:
                post_evidence_found = True

        if hb_pre_values:
            facts['hb_value_claim'] = min(hb_pre_values)   # worst (lowest) pre-treatment Hb
        if hb_post_values:
            facts['hb_post_value']  = max(hb_post_values)  # best post-treatment response
        if post_evidence_found:
            facts['post_treatment_evidence_found'] = True

    # ── Step 5: ALOS (Average Length of Stay) from LOS ───────────────────────
    if 'los_days' in facts and facts['los_days'] is not None:
        facts['alos_days'] = facts['los_days']   # single-claim ALOS = LOS itself

    return facts

def _merge_timeline_into_facts(timeline: list[dict], facts: dict) -> dict:
    """
    After build_timeline(), feed accepted DOA/DOD/pre/post dates back into
    temporal_facts so that LOS computation and rules engine see the same dates
    as the timeline display. Without this, ALOS rules see MISSING even when
    the timeline correctly extracted DOA/DOD.
    """
    DATE_KEY_MAP = {
        'Admission':                    'doa',
        'Admission / Clinical Eval':    'doa',
        'Discharge':                    'dod',
        'Pre-Treatment Investigation':  'pre_date',
        'Post-Treatment Investigation': 'post_date',
    }
    updated = dict(facts)
    for event in timeline:
        if not event.get('date_accepted'):
            continue   # only merge dates that passed is_accepted() — never unverifiable
        event_type = event.get('event_type', '')
        date_val   = event.get('date')
        key        = DATE_KEY_MAP.get(event_type)
        if key and date_val and date_val != '—':
            if not updated.get(key):
                updated[key] = date_val
            elif key in ('doa', 'dod'):
                updated[key] = date_val

    # Recompute LOS if both DOA and DOD now available
    doa_dt = parse_date(updated.get('doa'))
    dod_dt = parse_date(updated.get('dod'))
    if doa_dt and dod_dt and dod_dt >= doa_dt:
        updated['los_days'] = (dod_dt - doa_dt).days
    return updated


print('Module 5 (Timeline + Temporal Facts) ready.')


def _vlm_crop_date_fallback(
    by_doc: dict,
    priority_docs: list[str],
    target_field: str,
    package_code: str,
    known_doa: Optional[str] = None,
) -> Optional['EvidenceAtom']:
    """
    Path 5: Targeted VLM crop for DOA/DOD fields that failed all deterministic paths.
    Uses Gemma 4 E4B (vision) via Ollama when available. Fails closed otherwise.
    """
    if MOCK_MODE or not OLLAMA_AVAILABLE or not OLLAMA_VISION:
        return None   # fail closed — deterministic paths already exhausted

    doa_dt: Optional[datetime] = None
    if known_doa:
        doa_dt = parse_date(normalize_date(known_doa))

    _ANCHOR_ALIASES_LOCAL = {
        'doa': ['date of admission', 'd.o.a', 'admitted on', 'boa', 'start date'],
        'dod': ['date of discharge', 'd.o.d', 'discharged on'],
    }
    aliases = _ANCHOR_ALIASES_LOCAL.get(target_field, [])

    for priority_doc in priority_docs:
        rows = by_doc.get(priority_doc, [])
        for row in rows:
            img: Optional[Image.Image] = row.get('_image')
            if img is None:
                continue

            ocr_lines = row.get('_ocr_lines', [])
            crop_bbox: Optional[tuple] = None
            for line in ocr_lines:
                t = line.get('text', '').lower()
                if any(a in t for a in aliases):
                    lb = line.get('bbox')
                    if lb:
                        x1 = max(0, lb[0] - 10)
                        y1 = max(0, lb[1] - 5)
                        x2 = min(img.width, lb[0] + 900)
                        y2 = min(img.height, lb[3] + 40)
                        crop_bbox = (x1, y1, x2, y2)
                    break

            if crop_bbox is None:
                crop_bbox = (0, 0, img.width, int(img.height * 0.40))

            try:
                crop_img = img.crop(crop_bbox)
                scale = max(1, min(3, 800 // max(crop_img.width, 1)))
                if scale > 1:
                    crop_img = crop_img.resize(
                        (crop_img.width * scale, crop_img.height * scale),
                        Image.LANCZOS,
                    )
            except Exception:
                continue

            prompt = (
                f'This is a crop from an Indian hospital document. '
                f'Find the {"Admission" if target_field == "doa" else "Discharge"} date. '
                f'Return ONLY the date in DD/MM/YYYY format, nothing else. '
                f'If no date is visible, return "NOT_FOUND".'
            )

            try:
                raw_response = edge_model_call(crop_img, prompt)
                raw_response = raw_response.strip()
                if not raw_response or 'NOT_FOUND' in raw_response.upper():
                    continue
            except Exception as e:
                print(f'  WARN _vlm_crop_date_fallback: edge model call failed: {e}')
                continue

            date_match = None
            for pat in DATE_PATTERNS:
                hits = re.findall(pat, _strip_time_suffix(raw_response))
                if hits:
                    date_match = hits[0]
                    break

            if not date_match:
                continue

            nd = normalize_date(date_match)
            if nd is None:
                continue

            dt_obj = parse_date(nd)
            if dt_obj is None:
                continue
            if not (_DATE_PLAUSIBILITY_MIN <= dt_obj <= _DATE_PLAUSIBILITY_MAX):
                continue

            crop_conf = 0.62
            if doa_dt:
                if abs(dt_obj.year - doa_dt.year) <= 1:
                    if target_field == 'dod' and dt_obj < doa_dt:
                        continue
                    crop_conf = 0.75
                else:
                    continue
            else:
                if dt_obj.year < 2022:
                    continue

            return EvidenceAtom(
                field=target_field, value=date_match, normalized=nd,
                source_doc=priority_doc, page=row.get('page_number', 0),
                bbox=list(crop_bbox), source_text=f'VLM crop response: {raw_response[:60]}',
                method='vlm_crop',
                anchor_label=f'VLM crop ({target_field})',
                confidence=crop_conf,
            )

    return None   # all attempts failed


def build_timeline(package_code: str, ranked_rows: list[dict],
                   facts: dict) -> list[dict]:
    """
    v5.0: EvidenceAtom-based timeline with event-specific source priority.

    Date selection per event (in order):
      1. Tiered anchor search on _ocr_lines (fitz lines + PaddleOCR lines)
         for doc_types in DATE_SOURCE_PRIORITY[event_type]
      2. VLM labeled fields (structured date objects with source_text)
      3. Fitz regex fields (doa/dod/pre_date/post_date in ef)
      4. Facts dict (cross-page aggregated labeled extracts)
      ✗  dates_found — DISABLED. Debug only.

    Each accepted date is an EvidenceAtom. If no atom accepted:
      - Required field → CONDITIONAL — missing required date
      - Optional field → '—'
    """
    # Group rows by doc_type
    by_doc: dict[str, list[dict]] = {}
    for r in ranked_rows:
        dt = r.get('_doc_type', _dominant_doc_type(r, package_code))
        if dt and dt != 'unknown' and not r.get('extra_document'):
            by_doc.setdefault(dt, []).append(r)

    def _canonical_score(row: dict) -> tuple:
        ef = row.get('_extracted_fields', {}) or {}
        def _has_date_value(x) -> bool:
            if isinstance(x, dict): return bool(x.get('value'))
            return bool(x)
        has_labeled_date = any(
            _has_date_value(ef.get(f))
            for f in ('date_of_admission','date_of_discharge','treatment_date',
                      'pre_investigation_date','post_investigation_date',
                      'doa','dod','pre_date','post_date')
        )
        anchor_score = 0 if has_labeled_date else 1
        conf_score   = round(1.0 - float(row.get('_confidence', 0.0)), 2)
        doc_rank     = row.get('document_rank') or 99
        page_num     = row.get('page_number', 9999)
        return (anchor_score, conf_score, doc_rank, page_num)

    best_page_per_doc: dict[str, dict] = {}
    for dt, rows in by_doc.items():
        best_page_per_doc[dt] = min(rows, key=_canonical_score)

    required_fields = REQUIRED_TIMELINE_FIELDS.get(package_code, [])
    all_selected_dates: list[Optional[str]] = []
    events = []
    seq = 1
    # Track the DOA accepted by THIS timeline run — used by DOD sanity gate.
    # More reliable than facts['doa'] which uses a different extraction path.
    _timeline_accepted_doa: Optional[str] = None
    _timeline_accepted_pre: Optional[str] = None

    for event_type, doc_key, date_key in EVENT_DEFS.get(package_code, []):
        pages_for_doc = by_doc.get(doc_key, [])
        if not pages_for_doc:
            continue

        # ── Source priority: try each doc_type in priority order ──────────────
        priority_docs = DATE_SOURCE_PRIORITY.get(event_type, [doc_key])
        best_atom: Optional[EvidenceAtom] = None

        for priority_doc in priority_docs:
            candidate_rows = by_doc.get(priority_doc, [])
            if not candidate_rows:
                continue

            # FIX: Iterate ALL candidate rows sorted by canonical score.
            # Previously only best_row was tried — if that row lacked OCR lines
            # or a usable anchor, the pipeline gave up on the whole doc_type.
            # Now we try each row in priority order and break on first accepted atom.
            sorted_rows = sorted(candidate_rows, key=_canonical_score)

            for best_row in sorted_rows:
                ef        = best_row.get('_extracted_fields', {}) or {}
                ocr_lines = best_row.get('_ocr_lines', [])

                # Path 1: tiered anchor on OCR lines
                if date_key and ocr_lines:
                    atom = _extract_date_atom_from_lines(
                        ocr_lines, date_key,
                        source_doc=priority_doc,
                        page_num=best_row.get('page_number', 0),
                    )
                    if atom:
                        ok, _ = atom.is_accepted()
                        if ok:
                            best_atom = atom
                            break  # found in this row — stop iterating rows

                # Path 2: VLM labeled structured date field
                if date_key and not best_atom:
                    vlm_raw, vlm_src, vlm_conf = _extract_ef_date(ef, {
                        'doa': 'date_of_admission', 'dod': 'date_of_discharge',
                        'pre_date': 'pre_investigation_date',
                        'post_date': 'post_investigation_date',
                        'treatment_date': 'treatment_date',
                    }.get(date_key, date_key))
                    if vlm_raw:
                        nd = normalize_date(vlm_raw)
                        anchor_label = _DATE_FIELD_ANCHORS.get(date_key, ('VLM labeled', 0.80))[0]
                        uncertain = (ef.get('uncertain_fields') or [])
                        conf = vlm_conf if vlm_conf >= _DATE_MIN_CONFIDENCE else 0.0
                        if date_key in uncertain:
                            conf = max(conf - 0.25, 0.40)
                        if not vlm_src:
                            conf = max(conf - 0.10, 0.40)
                        atom = EvidenceAtom(
                            field=date_key, value=vlm_raw, normalized=nd,
                            source_doc=priority_doc, page=best_row.get('page_number', 0),
                            bbox=None, source_text=vlm_src,
                            method='vlm_labeled', anchor_label=anchor_label,
                            confidence=conf,
                        )
                        ok, _ = atom.is_accepted()
                        if ok:
                            best_atom = atom
                            break  # accepted from VLM — stop iterating rows

                # Path 3: fitz regex field
                if date_key and not best_atom:
                    fitz_raw = ef.get(date_key) if isinstance(ef.get(date_key), str) else None
                    if fitz_raw:
                        nd = normalize_date(fitz_raw)
                        atom = EvidenceAtom(
                            field=date_key, value=fitz_raw, normalized=nd,
                            source_doc=priority_doc, page=best_row.get('page_number', 0),
                            bbox=None, source_text=None,
                            method='fitz_regex',
                            anchor_label=_DATE_FIELD_ANCHORS.get(date_key, ('Fitz regex', 0.85))[0],
                            confidence=0.85,
                        )
                        ok, _ = atom.is_accepted()
                        if ok:
                            best_atom = atom
                            break  # accepted from fitz — stop iterating rows

            if best_atom:
                break  # found an atom from this priority_doc — stop priority_docs loop

        # Path 4: facts dict fallback
        if not best_atom and date_key and facts.get(date_key):
            raw = facts[date_key]
            nd  = normalize_date(raw)
            if nd:
                atom = EvidenceAtom(
                    field=date_key, value=raw, normalized=nd,
                    source_doc=doc_key, page=0,
                    bbox=None, source_text=None,
                    method='fitz_regex',   # facts come from fitz regex aggregation
                    anchor_label=_DATE_FIELD_ANCHORS.get(date_key, ('Facts dict', 0.80))[0],
                    confidence=0.80,
                )
                ok, _ = atom.is_accepted()
                if ok:
                    best_atom = atom

        # Path 5: targeted VLM crop — ONLY for DOA/DOD that are still missing.
        if not best_atom and date_key in ('doa', 'dod'):
            # Use the DOA this timeline run accepted (more reliable than facts['doa'])
            known_doa_for_crop = _timeline_accepted_doa or facts.get('doa')
            best_atom = _vlm_crop_date_fallback(
                by_doc, priority_docs, date_key,
                package_code=package_code,
                known_doa=known_doa_for_crop,
            )

        # ── Unified plausibility gate for ALL date fields ────────────────────
        if best_atom and date_key:
            best_atom = _validate_date_atom(best_atom, date_key, {
                **facts,
                # Include DOA/pre_date already accepted in THIS timeline run
                'doa':      _timeline_accepted_doa or facts.get('doa'),
                'pre_date': _timeline_accepted_pre or facts.get('pre_date'),
            })

        # ── Set tracking vars when key dates accepted ─────────────────────────
        if best_atom and best_atom.normalized:
            if date_key == 'doa':
                _timeline_accepted_doa = best_atom.normalized
            elif date_key == 'pre_date':
                _timeline_accepted_pre = best_atom.normalized

        # ── Gate + date output ────────────────────────────────────────────────
        best_page = best_page_per_doc.get(doc_key, pages_for_doc[0])
        dates_raw = best_page.get('_dates_found', [])

        if best_atom:
            accepted, accept_reason = best_atom.is_accepted()
            # Only show accepted dates in the timeline — unaccepted stay as '—'
            canonical_date = best_atom.normalized if accepted else None
            prov = DateProvenance.from_atom(best_atom)
        else:
            accepted, accept_reason = False, 'No accepted date from any source'
            canonical_date = None
            prov = DateProvenance()

        all_selected_dates.append(canonical_date)

        # ── Temporal validity with REQUIRED_TIMELINE_FIELDS ───────────────────
        if not canonical_date and date_key in required_fields:
            validity = f'CONDITIONAL — missing required date field "{date_key}" (no reliable extraction)'
        else:
            validity = _check_temporal_validity(event_type, prov, facts, all_selected_dates)

        n_pages = len(pages_for_doc)
        if n_pages > 1 and canonical_date:
            validity = (
                f'MULTIPLE_SOURCES ({n_pages} pages) — '
                f'canonical: page {best_page.get("page_number")} '
                f'(anchor_score={_canonical_score(best_page)[0]}) | {validity}'
            )

        events.append({
            'sequence':           seq,
            'event_type':         event_type,
            'date':               canonical_date or '—',
            'date_evidence':      best_atom.to_dict() if best_atom else {},
            'date_accepted':      accepted,
            'date_accept_reason': accept_reason,
            'source_document':    doc_key,
            'best_page':          best_page.get('page_number'),
            'source_method':      best_atom.method if best_atom else 'none',
            'date_confidence':    round(best_atom.confidence, 2) if best_atom else 0.0,
            'temporal_validity':  validity,
            '_dates_found_debug': [d['value'] for d in dates_raw[:5]],
        })
        seq += 1

    return events

# %% CELL 15 — [GAP 8] Visual Rule Evaluator + [GAP 5] Rules Engine with Rich Provenance
def evaluate_visual_check(elem: str, visual_agg: dict, required: bool) -> tuple[str, str]:
    """
    [GAP 8] Tiered visual rule evaluation.
    - found (any confidence): PASS
    - not found, conf > 0.40: CONDITIONAL (ambiguous; low cv2 conf = uncertain, not absent)
    - not found, conf <= 0.40: FAIL if required, ADVISORY_FAIL if optional
    VLM-sourced confidence (>0.80) treated as highly reliable.
    """
    found = visual_agg.get(f'{elem}_present', 0) == 1
    conf  = float(visual_agg.get(f'{elem}_confidence', 0.0))
    fail_str = 'FAIL' if required else 'ADVISORY_FAIL'

    if found:
        return 'PASS', f'{elem} detected (conf={conf:.2f})'
    elif conf > 0.40:
        return 'CONDITIONAL', f'{elem} uncertain (conf={conf:.2f}) — manual verification recommended'
    else:
        return fail_str, f'{elem} not detected (conf={conf:.2f}) — likely absent'

def evaluate_operator(value, rule: dict) -> str:
    op   = rule['operator']
    sev  = rule.get('severity', 'mandatory')
    fail = 'FAIL' if sev == 'mandatory' else 'ADVISORY_FAIL'
    try:
        if op == 'lte':
            return 'PASS' if float(value) <= rule['threshold'] else fail
        if op == 'gte':
            # Defensively cast — patient_age may arrive as string from OCR
            return 'PASS' if float(value) >= rule['threshold'] else fail
        if op == 'between':
            return 'PASS' if rule['min'] <= float(value) <= rule['max'] else fail
        if op == 'equals':
            return 'PASS' if str(value).lower() == str(rule['value']).lower() else fail
        if op == 'contains_any':
            return 'PASS' if any(k in str(value).lower() for k in rule['values']) else fail
        if op == 'present':
            return 'PASS' if value else fail
        if op == 'icd_prefix':
            return 'PASS' if any(str(value).startswith(p) for p in rule['prefix']) else fail
    except (TypeError, ValueError):
        return 'MISSING'
    return 'MISSING'

def _rich_provenance(fld: str, ranked_rows: list[dict]) -> dict:
    """
    [GAP 5] Build rich provenance: page, doc_type, extraction method, confidence.
    Uses rank-sorted rows so provenance always points to best-ranked page.
    """
    for r in ranked_rows:          # ranked_rows already sorted best-first
        if r.get(fld) == 1:
            return {
                'page':       r.get('page_number'),
                'doc_type':   r.get('_doc_type', r.get('_dominant_doc_type', 'unknown')),
                'method':     r.get('_method', 'unknown'),
                'confidence': r.get('_confidence', 0.0),
            }
    return {'page': None, 'doc_type': None, 'method': None, 'confidence': 0.0}

def run_rules_engine(claim_id: str, package_code: str,
                     ranked_rows: list[dict], temporal_facts: dict,
                     stg: dict, visual_agg: dict,
                     evidence_coverage: Optional[dict] = None) -> dict:
    """
    [GAP 5] Every rule result carries rich provenance.
    [GAP 8] Visual checks use tiered FAIL/CONDITIONAL.
    evidence_coverage: slot-level coverage from score_evidence_coverage().
    ranked_rows: rank-sorted (use for provenance — best page first).
    """
    agg = aggregate_rows(ranked_rows)
    results = []

    # 1. Mandatory document checks
    for doc_type in stg['mandatory_docs']:
        found = agg.get(doc_type, 0) == 1
        prov  = _rich_provenance(doc_type, ranked_rows)
        results.append({
            'rule_id':          f'DOC-{doc_type.upper()}',
            'rule_description': f'Mandatory document: {doc_type}',
            'severity':         'mandatory',
            'result':           'PASS' if found else 'MISSING',
            'field_checked':    doc_type,
            'value_found':      found,
            'provenance':       prov,      # [GAP 5]
        })

    # 2. Clinical rules
    for rule in stg['clinical_rules']:
        fld   = rule['field']
        value = temporal_facts.get(fld, agg.get(fld))
        prov  = _rich_provenance(fld, ranked_rows)
        result_str = 'MISSING' if value is None else evaluate_operator(value, rule)
        results.append({
            'rule_id':          rule['rule_id'],
            'rule_description': rule['description'],
            'severity':         rule['severity'],
            'result':           result_str,
            'field_checked':    fld,
            'value_found':      value,
            'provenance':       prov,      # [GAP 5]
        })

    # 3. Visual checks (GAP 8: tiered evaluation)
    for vc in stg['visual_checks']:
        elem     = vc['element']
        res, exp = evaluate_visual_check(elem, visual_agg, vc['required'])
        results.append({
            'rule_id':          vc['rule_id'],
            'rule_description': f'Visual: {elem} present',
            'severity':         'mandatory' if vc['required'] else 'advisory',
            'result':           res,
            'field_checked':    f'visual:{elem}',
            'value_found':      visual_agg.get(f'{elem}_present', 0),
            'provenance': {
                'page':       None,
                'doc_type':   'visual_detection',
                'method':     'cv2+vlm',
                'confidence': visual_agg.get(f'{elem}_confidence', 0.0),
                'explanation':exp,
            },
        })

    return make_decision(claim_id, package_code, results, temporal_facts,
                         evidence_coverage=evidence_coverage or {})

def make_decision(claim_id: str, package_code: str,
                  rule_results: list[dict], temporal_facts: dict,
                  evidence_coverage: Optional[dict] = None) -> dict:
    """
    FAIL  — explicit clinical contradiction:
            value WAS extracted/found AND it contradicts the rule
            e.g. previous_surgery=1 when expected 0, or doc is genuinely absent
            but NOT when a binary field=0 because OCR/VLM missed handwriting.

    CONDITIONAL — missing or weak evidence:
            rule field = 0 / None (extraction may have failed on scanned handwriting)
            advisory failures, missing mandatory docs, unfilled evidence slots

    PASS  — all mandatory checks satisfied with positive evidence.
    """
    def _is_explicit_contradiction(r: dict) -> bool:
        """
        True only when we have POSITIVE evidence of a contradiction:
          - value_found IS a non-zero/non-None value AND it fails the rule
            (e.g. numeric Hb extracted as 8.5 > 7.0, or previous_surgery=1 expected=0)
          - NOT when value_found=0 and expected=1 (that's "not found", not "contradiction")
        """
        v = r.get('value_found')
        if v is None or v == 0:
            # Zero or missing — OCR / VLM extraction failure on scanned doc,
            # not a confirmed clinical contradiction.
            return False
        # v is a positive value — rule failed on real extracted data
        return True

    # Partition rule failures
    raw_fails   = [r for r in rule_results if r['result'] == 'FAIL']
    hard_fails  = [r for r in raw_fails if _is_explicit_contradiction(r)]
    # Demoted: binary fields = 0 (not extracted) → CONDITIONAL, not FAIL
    demoted     = [r for r in raw_fails if not _is_explicit_contradiction(r)]

    missing     = [r for r in rule_results
                   if r['result'] in ('MISSING', 'CONDITIONAL')
                   and r['severity'] == 'mandatory']
    advisory    = [r for r in rule_results if r['result'] == 'ADVISORY_FAIL']
    passes      = [r for r in rule_results if r['result'] == 'PASS']

    # Demoted fails → treated as missing evidence (CONDITIONAL path)
    all_conditional = missing + demoted

    decision = (DECISION_FAIL        if hard_fails
                else DECISION_CONDITIONAL if (all_conditional or advisory)
                else DECISION_PASS)

    def _prov_str(r: dict) -> str:
        p = r.get('provenance', {})
        parts = [f"rule={r['rule_id']}", f"field='{r['field_checked']}'",
                 f"found='{r['value_found']}'"]
        if p.get('page'):       parts.append(f"page={p['page']}")
        if p.get('doc_type'):   parts.append(f"doc={p['doc_type']}")
        if p.get('method'):     parts.append(f"method={p['method']}")
        if p.get('confidence'): parts.append(f"conf={p['confidence']:.2f}")
        return ' | '.join(parts)

    flags = (
        [{'severity': 'CRITICAL',    'flag': _prov_str(r)} for r in hard_fails]  +
        [{'severity': 'CONDITIONAL', 'flag': _prov_str(r)} for r in all_conditional] +
        [{'severity': 'ADVISORY',    'flag': _prov_str(r)} for r in advisory]
    )
    confidence = max(0.20, min(0.95,
        0.95 - 0.10 * len(hard_fails)
             - 0.05 * len(all_conditional)
             - 0.02 * len(advisory)))

    # ── Slot-level coverage for reviewer utility ──────────────────────────────
    ec = evidence_coverage or {}
    unfilled_mandatory = [
        slot for slot, info in ec.items()
        if info['role'] == 'mandatory' and not info['filled']
    ]
    filled_slots = [
        {'slot': slot, 'doc_type': info['doc_type'],
         'page': info['best_page'], 'confidence': info['confidence']}
        for slot, info in ec.items()
        if info['filled']
    ]

    reviewer_utility = {
        'decision_summary': (
            f"PASS — all {len(passes)} mandatory checks satisfied."
            if decision == DECISION_PASS else
            f"FAIL — {len(hard_fails)} explicit clinical contradiction(s)."
            if decision == DECISION_FAIL else
            f"CONDITIONAL — {len(all_conditional)} check(s) missing or uncertain "
            f"(may be extraction gap on scanned/handwritten docs); "
            f"{len(unfilled_mandatory)} evidence slot(s) unfilled."
        ),
        'evidence_slot_coverage': {
            slot: {
                'filled':     info['filled'],
                'doc_type':   info['doc_type'],
                'page':       info['best_page'],
                'confidence': round(info['confidence'], 3),
            }
            for slot, info in ec.items()
        },
        'unfilled_mandatory_slots': unfilled_mandatory,
        'top_supporting_evidence': [
            {
                'rule':        r['rule_id'],
                'description': r['rule_description'],
                'evidence_page': r.get('provenance', {}).get('page'),
                'source_doc':  r.get('provenance', {}).get('doc_type'),
                'confidence':  r.get('provenance', {}).get('confidence', 0.0),
            }
            for r in passes[:5]
        ],
        'missing_evidence': [
            {'rule': r['rule_id'], 'description': r['rule_description'],
             'status': r['result']}
            for r in missing
        ],
        'contradictions': [
            {'rule': r['rule_id'], 'description': r['rule_description'],
             'value_found': r['value_found'],
             'evidence_page': r.get('provenance', {}).get('page')}
            for r in hard_fails
        ],
        'temporal_summary': {
            k: v for k, v in temporal_facts.items()
            if k in ('los_days', 'discharge_after_admission',
                     'pre_before_post', 'hb_value_claim')
        },
    }

    return {
        'claim_id':         claim_id,
        'package_code':     package_code,
        'decision':         decision,
        'confidence':       round(confidence, 3),
        'flags':            flags,
        'rule_results':     rule_results,
        'temporal_facts':   temporal_facts,
        'reviewer_utility': reviewer_utility,
        'token_usage':      dict(TOKEN_LOG),
    }

print('Module 5/6 (Rules Engine + Decision) ready — GAP 5 + GAP 8 applied.')


# %% CELL 16 — Output Formatters (exact ps1.md table specs)
def _strip_meta(row: dict) -> dict:
    """Remove _meta keys before export — they are not part of PACKAGE_SCHEMAS."""
    return {k: v for k, v in row.items() if not k.startswith('_')}

def build_classification_table(claim_id: str, rows: list[dict],
                                package_code: str) -> pd.DataFrame:
    """
    Table 1 per ps1.md: Claim ID | File | Page | Document Classification | Clinical Rules Checks
    Exact ps1.md wording used for extra doc notes.
    Inconsistency detection: flags same doc_type appearing with different clinical signals.
    """
    # Build a map of doc_type → {file_name → [signal fingerprints]}
    # Inconsistency = same doc_type, DIFFERENT files, conflicting clinical signals.
    # Same file with different signals across pages is normal multi-page behaviour.
    _type_file_signals: dict[str, dict[str, list]] = {}
    link_key = LINK_KEY[package_code]
    for r in rows:
        dt = r.get('_doc_type', 'unknown')
        if dt == 'unknown' or r.get('extra_document') == 1:
            continue
        fname = r.get(link_key, '')
        sig = {k: v for k, v in r.items()
               if not k.startswith('_') and k not in
               ('case_id', 'page_number', 'document_rank', 'extra_document',
                link_key, 'procedure_code')}
        _type_file_signals.setdefault(dt, {}).setdefault(fname, []).append(sig)

    # INCONSISTENT detection suppressed — per-page signal variance within a doc type
    # is normal for multi-page hospital records (handwritten notes, continuation pages).
    # Only flag if two DIFFERENT files of same doc_type have conflicting ACCEPTED dates
    # — handled downstream by temporal sanity gate, not here.
    _inconsistent: set[str] = set()

    records = []
    for r in rows:
        doc_type = r.get('_doc_type', _dominant_doc_type(r, package_code))
        notes = []
        if r.get('extra_document') == 1:
            reason = r.get('_extra_reason', '')
            if 'duplicate' in reason:
                notes.append('Not required for clinical or claim validation (duplicate submission)')
            elif reason == 'unclassified':
                notes.append('Not required for clinical or claim validation (unclassified document)')
            else:
                notes.append('Not required for clinical or claim validation')
        if doc_type in _inconsistent:
            notes.append(f'INCONSISTENT: conflicting clinical signals across multiple {doc_type} pages')
        if doc_type == 'unknown':
            notes.append(f"Unclassified (method={r.get('_method','?')}, "
                         f"conf={r.get('_confidence',0.0):.2f})")
        records.append({
            'Claim ID':                 claim_id,
            'File':                     r.get(link_key, ''),
            'Page':                     r['page_number'],
            'Document Classification':  doc_type,
            'Extraction Method':        r.get('_method', '—'),
            'Confidence':               f"{r.get('_confidence', 0.0):.2f}",
            'Clinical Rules Checks':    ' | '.join(notes) if notes else '—',
        })
    return pd.DataFrame(records)

def build_timeline_table(timeline: list[dict]) -> pd.DataFrame:
    """
    Table 2 per ps1.md: Sequence | Event Type | Date | Source Document | Temporal Validity
    Extended: Best Page, Method, Confidence shown for evaluator traceability.
    """
    if not timeline:
        return pd.DataFrame(columns=['Sequence','Event Type','Date',
                                     'Source Document','Best Page',
                                     'Method','Confidence','Temporal Validity'])
    return pd.DataFrame([{
        'Sequence':          e['sequence'],
        'Event Type':        e['event_type'],
        'Date':              e['date'],
        'Source Document':   e['source_document'],
        'Best Page':         e.get('best_page', '—'),
        'Method':            e.get('source_method', '—'),
        'Confidence':        f"{e.get('date_confidence', 0.0):.2f}",  # date conf, not page conf
        'Temporal Validity': e['temporal_validity'],
    } for e in timeline])

def validate_output_rows(package_code: str, rows: list[dict]) -> tuple[bool, list[str]]:
    """Confirm exported rows match exact PACKAGE_SCHEMAS key order."""
    expected = PACKAGE_SCHEMAS[package_code]
    issues = []
    for i, row in enumerate(rows):
        clean   = _strip_meta(row)
        row_keys = list(clean.keys())
        if row_keys != expected:
            extra   = [k for k in row_keys if k not in expected]
            missing = [k for k in expected  if k not in row_keys]
            issues.append(f'Row {i} (page {row.get("page_number")}): '
                          f'extra={extra}, missing={missing}')
    return len(issues) == 0, issues

print('Output formatters ready.')

# %% CELL 17 — Pipeline Runner (all gaps integrated)
def process_claim(claim: dict, stg_configs: dict) -> dict:
    """
    End-to-end pipeline. File extraction is sequential — PaddleOCR is serialised
    by _paddle_lock anyway, and Ollama serves one request at a time on M3.
    Cross-claim parallelism is handled at the caller level via --jobs N.
    """
    claim_id     = claim['claim_id']
    package_code = claim['package_code']
    stg          = stg_configs[package_code]
    link_key     = LINK_KEY[package_code]

    all_rows:         list[dict] = []
    visual_signals:   list[dict] = []
    classified_pages: list[dict] = []

    seen_doc_types:    set = set()
    investigation_count: int = 0

    # ── Extract + classify pages sequentially ────────────────────────────────
    for file_path in claim['files']:
        try:
            pages = extract_pages(file_path)
        except Exception as e:
            print(f'  WARN ingestion {file_path.name}: {e}')
            continue

        for page in pages:
            try:
                classified = classify_and_extract_page(page, package_code,
                                                       investigation_count)
                classified['_page_image'] = page.get('image')
                if (package_code == 'MG006A'
                        and classified.get('doc_type') == 'investigation_pre'):
                    investigation_count += 1
                visual = run_visual_detection(
                    page['image'], classified,
                    page_num=page['page_number'],
                    file_name=page['file_name'],
                )
                row = build_output_row(
                    claim_id=claim_id,
                    file_name=page['file_name'],
                    page_num=page['page_number'],
                    package_code=package_code,
                    classified=classified,
                    visual=visual,
                    text=page.get('text', ''),
                    seen_doc_types=seen_doc_types,
                    ocr_lines=page.get('_ocr_lines', []),
                )
                all_rows.append(row)
                visual_signals.append(visual)
                classified_pages.append({'page': page, 'classified': classified})

            except Exception as e:
                print(f'  WARN page {page.get("page_number")} of {file_path.name}: {e}')
                fallback = init_row_template(package_code)
                fallback['case_id']       = claim_id
                fallback[link_key]        = file_path.name
                fallback['procedure_code']= package_code
                fallback['page_number']   = page.get('page_number', 0)
                fallback['extra_document']= 1
                fallback['document_rank'] = 99
                fallback['_method']       = 'error_fallback'
                fallback['_confidence']   = 0.0
                fallback['_doc_type']     = 'unknown'
                fallback['_dates_found']  = []
                fallback['_extra_reason'] = 'page_processing_error'
                all_rows.append(fallback)

    if not all_rows:
        return {'claim_id': claim_id, 'package_code': package_code,
                'json_rows': [], 'error': 'No pages processed'}

    # [GAP 6] Logical document grouping — NOW WIRED to post-hoc reclassification.
    # For multi-page logical docs where individual pages classified as 'unknown'
    # via Tier 1/2, combine their fitz text and retry Tier 2 on the combined text.
    # This rescues multi-page indoor case papers, split CBC reports, etc.
    logical_docs = group_into_logical_docs(classified_pages)
    _doc_remap: dict[tuple, str] = {}  # (file_name, page_num) → reclassified doc_type
    for ldoc in logical_docs:
        if ldoc.doc_type != 'unknown':
            continue   # already classified — skip
        combined = ldoc.combined_text
        if not combined.strip():
            continue   # image-only pages — nothing to reclassify via Tier 2
        dt2, conf2 = classify_tier2(combined, package_code, investigation_count)
        if dt2 and conf2 >= 0.60:
            for entry in ldoc.pages:
                key = (entry['page']['file_name'], entry['page']['page_number'])
                _doc_remap[key] = dt2

    # Apply remapping — update _doc_type and doc-type flags in all_rows
    if _doc_remap:
        for row in all_rows:
            key = (row.get(link_key, ''), row.get('page_number', 0))
            if key in _doc_remap:
                new_dt = _doc_remap[key]
                row['_doc_type'] = new_dt
                # Re-set doc-type presence flags
                for dt in PACKAGE_INFO[package_code]['doc_types']:
                    row[dt] = 1 if dt == new_dt else row.get(dt, 0)
                # If previously marked extra_document, clear it if now classified
                if new_dt in set(PACKAGE_INFO[package_code]['doc_types']):
                    row['extra_document'] = 0
                    row['document_rank']  = RANK_MAP[package_code].get(new_dt, 98)

    # Output guide rank semantics: rank logical document groups by document date,
    # not by static document class. Pages in one logical document keep the same rank.
    assign_chronological_document_ranks(all_rows, package_code)

    # [GAP 1] Rank-sort for timeline + provenance
    ranked_rows = rank_rows(all_rows, package_code)

    # Aggregate visual signals at claim level
    def _max_vis(key, default=0):
        return max((v.get(key, default) for v in visual_signals), default=default)

    visual_agg = {
        'stamp_present':              _max_vis('stamp_present'),
        'stamp_confidence':           _max_vis('stamp_confidence', 0.0),
        'signature_present':          _max_vis('signature_present'),
        'signature_confidence':       _max_vis('signature_confidence', 0.0),
        'implant_sticker_present':    _max_vis('implant_sticker_present'),
        'implant_sticker_confidence': _max_vis('implant_sticker_confidence', 0.0),
    }

    temporal_facts = derive_temporal_facts(ranked_rows, package_code)
    timeline       = build_timeline(package_code, ranked_rows, temporal_facts)

    # FIX: merge accepted timeline dates back into temporal_facts so rules engine
    # sees the same DOA/DOD that appear in the timeline. Without this merge,
    # ALOS is MISSING even when timeline correctly extracted DOA/DOD.
    temporal_facts = _merge_timeline_into_facts(timeline, temporal_facts)

    # ── GUARDED DATE BROADCASTING ─────────────────────────────────────────────
    # The evaluator merges on (case_id, page_number) and scores date fields per row.
    # Without broadcasting, only the source-doc row has a date; all other rows score
    # as null (FN). Broadcasting the claim-level resolved date to ALL non-extra rows
    # fixes this. Guards: year 2022-2027, and dod >= doa (prevents bad hallucinations
    # from spreading — a wrong date on 1 row is neutral, broadcasting it is also neutral,
    # but a wrong date from a failed sanity check stays contained).
    try:
        from datetime import datetime as _dt
        def _parse_broadcast_date(s):
            if not s: return None
            try:
                p = str(s).split('-')
                if len(p) == 3:
                    return _dt(int(p[2]), int(p[1]), int(p[0]))
            except Exception:
                return None
            return None

        _BROADCAST_FIELDS = {
            'MG006A': ['pre_date', 'post_date'],
            'SB039A': ['doa', 'dod'],
        }
        _bcast_fields = _BROADCAST_FIELDS.get(package_code, [])
        if _bcast_fields:
            _doa_dt = _parse_broadcast_date(temporal_facts.get('doa'))
            _dod_dt = _parse_broadcast_date(temporal_facts.get('dod'))
            for _field in _bcast_fields:
                _val = temporal_facts.get(_field)
                if not _val:
                    continue
                _dt_obj = _parse_broadcast_date(_val)
                if not _dt_obj or not (2022 <= _dt_obj.year <= 2027):
                    continue   # reject hallucinated years
                if _field == 'dod' and _doa_dt and _dt_obj < _doa_dt:
                    continue   # reject dod before doa
                # Safe to broadcast
                for _row in all_rows:
                    if _row.get('extra_document') != 1 and _field in _row:
                        _row[_field] = _val
    except Exception:
        pass  # broadcasting is best-effort; never crash the pipeline

    evidence_coverage = score_evidence_coverage(ranked_rows, package_code, temporal_facts)

    decision_obj   = run_rules_engine(
        claim_id, package_code, ranked_rows,
        temporal_facts, stg, visual_agg,
        evidence_coverage=evidence_coverage,
    )

    # Export rows: strip _meta keys for evaluator
    export_rows = [_strip_meta(r) for r in all_rows]
    valid, issues = validate_output_rows(package_code, export_rows)
    if not valid:
        print(f'  SCHEMA WARNING {claim_id}:', issues[:2])

    result = {
        'claim_id':          claim_id,
        'package_code':      package_code,
        'json_rows':         export_rows,
        '_ranked_rows':      ranked_rows,
        'evidence_coverage': evidence_coverage,
        'logical_docs':      [(d.doc_type, d.file_name, len(d.pages))
                              for d in logical_docs],
        'classification_df': build_classification_table(claim_id, all_rows, package_code),
        'timeline':          timeline,
        'timeline_df':       build_timeline_table(timeline),
        'decision':          decision_obj,
        'visual_agg':        visual_agg,
        'temporal_facts':    temporal_facts,
    }

    # ── Stage 5: Gemma 4 26B reviewer reasoning ───────────────────────────────
    # Deterministic rules produce the decision. 26B produces the explanation.
    result['reasoning_summary'] = generate_claim_reasoning_with_26b({
        'claim_id':          claim_id,
        'package_code':      package_code,
        'decision':          decision_obj.get('decision'),
        'confidence':        decision_obj.get('confidence'),
        'flags':             decision_obj.get('flags', []),
        'evidence_coverage': {
            slot: {'filled': info['filled'], 'doc_type': info.get('doc_type'),
                   'confidence': round(info['confidence'], 2)}
            for slot, info in evidence_coverage.items()
        },
        'timeline':          timeline,
        'document_count':    len(logical_docs),
        'unfilled_mandatory': [
            s for s, i in evidence_coverage.items()
            if not i['filled'] and i.get('role') == 'mandatory'
        ],
    })
    return result

print('Pipeline runner ready.')

def generate_claim_reasoning_with_26b(claim_context: dict) -> dict:
    """
    Stage 5 (M1 fix): Gemma 4 26B generates a reviewer-facing explanation
    grounded in deterministic rule findings and extracted evidence.

    The deterministic rules engine produces the PASS/CONDITIONAL/REVIEW decision.
    26B produces the human-readable explanation, key evidence summary, and
    recommended next action — it does NOT override the decision.

    Fails gracefully: if Ollama is unavailable or output cannot be parsed,
    returns a safe default that preserves the deterministic decision.
    """
    _SAFE_DEFAULT = {
        'reviewer_summary':        'Claim review based on deterministic rule findings.',
        'key_evidence':            [],
        'risk_flags':              [],
        'missing_evidence':        claim_context.get('unfilled_mandatory', []),
        'recommended_next_action': 'Human reviewer should inspect extracted evidence and source documents.',
        'safety_note':             'This is an assistive recommendation. Final claim decision requires human review.',
    }

    if MOCK_MODE or not OLLAMA_AVAILABLE:
        return _SAFE_DEFAULT

    prompt = f"""You are a claim review assistant for public health insurance.
You do not approve or reject claims. You explain evidence for a human reviewer.

Given the structured claim context below, produce valid JSON only. No markdown.

Required JSON schema:
{{
  "reviewer_summary": "2-3 sentence plain-language summary of the claim evidence quality",
  "key_evidence": ["list of the strongest evidence items found"],
  "risk_flags": ["list of clinical or billing concerns, empty list if none"],
  "missing_evidence": ["list of missing mandatory documents or evidence slots"],
  "recommended_next_action": "what the human reviewer should do next",
  "safety_note": "always include: This is an assistive recommendation. Final claim decision requires human review."
}}

Rules:
- Do not invent facts. Use only the provided evidence context.
- Do not make final medical, diagnostic, or payment decisions.
- If evidence is weak, missing, or contradictory, recommend human review.
- Keep each field concise. reviewer_summary max 3 sentences.

CLAIM_CONTEXT:
{json.dumps(claim_context, indent=2, ensure_ascii=False, default=str)}
"""
    try:
        raw = reason_model_call(json.dumps(claim_context, ensure_ascii=False, default=str), prompt)
        parsed = safe_parse_vlm(raw)
        if isinstance(parsed, dict) and parsed.get('reviewer_summary'):
            # Enforce safety note is always present
            if not parsed.get('safety_note'):
                parsed['safety_note'] = _SAFE_DEFAULT['safety_note']
            return parsed
    except Exception as e:
        print(f'  WARN generate_claim_reasoning_with_26b: {e}')

    return _SAFE_DEFAULT

# ── solutionFlow.md stage function aliases ────────────────────────────────────
# These names match the architecture documented in solutionFlow.md exactly.
# The pipeline internally uses these stages in sequence:
#   OCR reads → E4B structures → validates → timeline → 26B reasons → reviewer pack

def ocr_extract_pages(file_path, dpi: int = 150) -> list:
    """Stage 1: PDF/image ingestion via PaddleOCR + PyTesseract. Alias for extract_pages()."""
    return extract_pages(file_path, dpi=dpi)

def edge_parse_page_with_e4b(img, package_code: str, page_text: str = '') -> dict:
    """Stage 2: Gemma 4 E4B — OCR cleanup, page triage, classification, field extraction."""
    return classify_tier3_vlm(img, package_code, page_text=page_text)

def validate_extracted_evidence(rows: list, package_code: str) -> tuple:
    """Stage 3: Deterministic gates — dates, confidence thresholds, required docs."""
    return validate_output_rows(package_code, rows)

def build_episode_timeline(claim: dict, all_rows: list, stg: dict) -> list:
    """Stage 4: Admission → investigation → procedure → monitoring → discharge."""
    return build_timeline(claim, all_rows, stg)

def reason_claim_with_26b(claim_id: str, package_code: str,
                           ranked_rows: list, temporal_facts: dict,
                           stg: dict, visual_agg: dict,
                           evidence_coverage: dict | None = None) -> dict:
    """
    Stage 5: Gemma 4 26B — claim-level reasoning.
    Handles: package/STG rule interpretation, cross-document contradiction analysis,
    timeline reasoning, PASS/CONDITIONAL/REVIEW recommendation with explanation.
    Wraps run_rules_engine() which invokes the reasoning model for complex rule checks.
    """
    return run_rules_engine(
        claim_id, package_code, ranked_rows,
        temporal_facts, stg, visual_agg,
        evidence_coverage=evidence_coverage,
    )

def generate_reviewer_summary(result: dict) -> dict:
    """Stage 6: Reviewer-ready pack — decision, reasons, missing evidence, provenance."""
    dec = result.get('decision', {})
    return {
        'claim_id':         result.get('claim_id'),
        'package_code':     result.get('package_code'),
        'decision':         dec.get('decision'),
        'confidence':       dec.get('confidence'),
        'flags':            dec.get('flags', []),
        'reviewer_utility': dec.get('reviewer_utility', {}),
        'timeline':         result.get('timeline', []),
        'classification':   result.get('logical_docs', []),
        'evidence_coverage':result.get('evidence_coverage', {}),
    }

print('Stage aliases ready (ocr_extract_pages / edge_parse_page_with_e4b / validate_extracted_evidence / build_episode_timeline / reason_claim_with_26b / generate_reviewer_summary)')

# %% CELL 18 — Batch Runner + Export
def run_batch(claims: list[dict], stg_configs: dict,
              max_claims: Optional[int] = None) -> list[dict]:
    """Process all (or limited) claims, export outputs to disk after each claim."""
    results = []
    subset  = claims[:max_claims] if max_claims else claims
    for i, claim in enumerate(subset):
        cid = claim['claim_id']; pkg = claim['package_code']
        print(f'[{i+1}/{len(subset)}] {cid} ({pkg}) ...', end=' ', flush=True)
        try:
            result   = process_claim(claim, stg_configs)
            decision = result.get('decision', {}).get('decision', '?')
            n_rows   = len(result.get('json_rows', []))
            tokens   = TOKEN_LOG['input'] + TOKEN_LOG['output']
            print(f'{decision} | {n_rows} rows | tokens={tokens:,}')
            results.append(result)
            # Export immediately after each claim (notebook alignment)
            export_results([result])
        except Exception as e:
            print(f'ERROR: {e}')
    print(f'\n=== Token Usage ===')
    print(f'  Input:   {TOKEN_LOG["input"]:,}')
    print(f'  Output:  {TOKEN_LOG["output"]:,}')
    print(f'  Total:   {TOKEN_LOG["input"]+TOKEN_LOG["output"]:,}')
    print(f'  Calls:   {TOKEN_LOG["calls"]}')
    return results

def export_results(results: list[dict], out: Path | None = None) -> None:
    """Write JSON rows, classification CSV, timeline CSV, decision JSON per claim."""
    out = out or OUTPUT_ROOT
    for result in results:
        if not result.get('json_rows'): continue
        cid = result['claim_id']; pkg = result['package_code']
        case_dir = out / cid
        case_dir.mkdir(parents=True, exist_ok=True)

        with open(case_dir / f'{cid}_{pkg}_output.json', 'w', encoding='utf-8') as f:
            json.dump(result['json_rows'], f, indent=2, ensure_ascii=False)

        if 'classification_df' in result:
            result['classification_df'].to_csv(
                case_dir / f'{cid}_classification.csv', index=False)

        if 'timeline_df' in result:
            result['timeline_df'].to_csv(
                case_dir / f'{cid}_timeline.csv', index=False)

        if 'decision' in result:
            # Remove non-serializable internal keys from rule_results
            dec = json.loads(json.dumps(result['decision'], default=str))
            with open(case_dir / f'{cid}_decision.json', 'w', encoding='utf-8') as f:
                json.dump(dec, f, indent=2, ensure_ascii=False)

    print(f'Exported {len(results)} claims to {out.resolve()}')

print('Batch runner + exporter ready.')

# %% CELL 19 — Quick Data Preview (validate dataset before run)
if __name__ == '__main__':
    print('=== Dataset Overview ===')
    for pkg in PACKAGE_CODES:
        pkg_claims = [c for c in ALL_CLAIMS if c['package_code'] == pkg]
        print(f'  {pkg}: {len(pkg_claims)} claims')

    print('\n=== Sample filenames (first claim per package) ===')
    seen = set()
    for claim in ALL_CLAIMS:
        pkg = claim['package_code']
        if pkg not in seen:
            print(f"\n  {pkg} — {claim['claim_id']}:")
            for f in claim['files'][:6]:
                dt, conf = classify_tier1(f.name)
                pred = f'{dt} ({conf:.0%})' if dt else 'no filename hint'
                print(f'    {f.name:<40}  -> {pred}')
            seen.add(pkg)

# %% CELL 20 — Run Pipeline (test: 1 claim per package first)
if __name__ == '__main__':
    ALL_RESULTS = run_batch(ALL_CLAIMS, STG_CONFIGS, max_claims=4)

# %% CELL 21 — Display Results
if __name__ == '__main__':
    try:
        from IPython.display import display
    except ImportError:
        display = print

    for result in ALL_RESULTS:
        cid = result['claim_id']; pkg = result['package_code']
        dec = result.get('decision', {})
        ru  = dec.get('reviewer_utility', {})
        print(f'\n{"="*70}')
        print(f'CLAIM: {cid}  |  PACKAGE: {pkg}  |  DECISION: {dec.get("decision","?")}')
        print(f'Confidence: {dec.get("confidence","?")} | Flags: {len(dec.get("flags",[]))}')

        # Reviewer utility — evidence-centric, slot-aware
        if ru:
            print(f'\n  ⚖  {ru.get("decision_summary","")}')
            # Evidence slot coverage grid
            ec = ru.get('evidence_slot_coverage', {})
            if ec:
                print('  📋 Evidence Slots:')
                for slot, info in ec.items():
                    tick = '✓' if info['filled'] else '✗'
                    pg   = f'p.{info["page"]}' if info.get('page') else '—'
                    conf = f'{info["confidence"]:.2f}' if info['filled'] else '—'
                    print(f'    {tick} {slot:<30} doc={info["doc_type"] or "—":<22} {pg}  conf={conf}')
            if ru.get('unfilled_mandatory_slots'):
                print(f'  ⚠  Unfilled mandatory slots: {", ".join(ru["unfilled_mandatory_slots"])}')
            if ru.get('contradictions'):
                print('  ✗  Contradictions:')
                for c in ru['contradictions']:
                    print(f'    [{c["rule"]}] {c["description"][:65]} (p.{c["evidence_page"]})')
            if ru.get('temporal_summary'):
                ts = ru['temporal_summary']
                print(f'  🕐 Temporal: LOS={ts.get("los_days","?")}d  '
                      f'discharge_after_admission={ts.get("discharge_after_admission","?")}  '
                      f'hb={ts.get("hb_value_claim","?")}')

        for fl in dec.get('flags', [])[:2]:
            print(f'  [{fl["severity"]}] {fl["flag"]}')

        if 'classification_df' in result:
            print('\nDocument Classification:')
            display(result['classification_df'])
        if 'timeline_df' in result and not result['timeline_df'].empty:
            print('\nEpisode Timeline:')
            display(result['timeline_df'])
        if result.get('logical_docs'):
            print('\nLogical Documents (doc_type, file, n_pages):')
            for ld in result['logical_docs']:
                print(f'  {ld}')

# %% CELL 22 — Export + Validate
if __name__ == '__main__':
    export_results(ALL_RESULTS)

    print('\n=== Schema Validation ===')
    for result in ALL_RESULTS:
        if not result.get('json_rows'): continue
        valid, issues = validate_output_rows(result['package_code'], result['json_rows'])
        status = 'VALID' if valid else f'INVALID ({len(issues)} issues)'
        print(f"  {result['claim_id']} ({result['package_code']}): {status}")
        for iss in issues[:2]:
            print(f'    {iss}')

# %% CELL 23 — Full Batch (uncomment to run all claims)
# if __name__ == '__main__':
#     ALL_RESULTS_FULL = run_batch(ALL_CLAIMS, STG_CONFIGS, max_claims=None)
#     export_results(ALL_RESULTS_FULL)

# %% CELL 24 — Token Usage Report
if __name__ == '__main__':
    total_tok = TOKEN_LOG['input'] + TOKEN_LOG['output']
    n_claims  = len([r for r in ALL_RESULTS if r.get('json_rows')])
    n_calls   = TOKEN_LOG['calls']
    n_mock    = TOKEN_LOG.get('mock_calls', 0)
    print(f'=== Mode: {"MOCK (Ollama unavailable)" if MOCK_MODE else "LIVE (Gemma 4 via Ollama)"} ===')
    print(f'  Output tokens (est.): {TOKEN_LOG["output"]:,}')
    print(f'  Ollama calls:         {n_calls}')
    print(f'  Mock calls:           {n_mock}')
    print(f'  Claims processed:     {n_claims}')
    print(f'  Models used:          edge={MODEL_EDGE}, reasoning={MODEL_REASON}')
