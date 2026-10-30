# Implementation Plan — Cultural Inclusive Story Generation from Images using Reasoning

**Version:** 0.1 (Phase 0 — Inspect and Plan)  
**Date:** 2026-10-30  
**Author:** Mohan (Final Year Project)  
**Status:** DRAFT — Pending Phase 1 execution

---

## 1. Workspace and Environment Assessment

### 1.1 Current Project State

The workspace (`c:\Users\mohan\Documents\FinalYear`) is **empty** — no prior code, dependencies, or notebooks exist. This is a greenfield project. All structure will be created from scratch.

### 1.2 Confirmed Runtime Environment

| Property | Value |
|---|---|
| OS | Windows 11 |
| Python | 3.11.9 |
| pip | 26.1.2 |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| VRAM | 6.44 GB |
| CUDA Version (PyTorch build) | 12.1 |
| System RAM | ~24 GB |
| CUDA Available in PyTorch | ✅ Yes |

### 1.3 Pre-installed Libraries (Confirmed Working)

| Library | Version | Role |
|---|---|---|
| torch | 2.5.1+cu121 | Model inference |
| torchvision | 0.20.1+cu121 | Image transforms |
| transformers | 5.12.1 | VLM / embedding models |
| sentence-transformers | 5.6.0 | Semantic embeddings |
| chromadb | 1.5.9 | Vector store |
| pydantic | 2.11.10 | Data validation |
| pydantic-settings | 2.14.2 | Config from env |
| streamlit | 1.55.0 | Web UI |
| pillow | 11.3.0 | Image I/O |
| numpy | 2.2.2 | Array ops |
| pandas | 2.3.3 | Tabular data |
| pytest | 8.3.4 | Testing |
| pytest-asyncio | 1.4.0 | Async tests |
| langchain | 1.3.4 | LLM orchestration |
| langchain-google-genai | 4.2.6 | Gemini LLM integration |
| google-genai | 2.8.0 | Google Gemini API |
| openai | 2.44.0 | OpenAI API |
| huggingface_hub | 1.21.0 | Model downloads |

### 1.4 Missing Libraries Requiring Installation

| Library | Purpose |
|---|---|
| faiss-cpu | Alternative vector index (optional, ChromaDB is primary) |
| python-dotenv | Load .env files |
| loguru | Structured logging |
| tqdm | Progress bars |

**Decision:** Use **ChromaDB** as the primary vector store (already installed). Install `faiss-cpu` as a secondary, testable backend satisfying the "replaceable adapter" requirement.

### 1.5 Hardware Constraints and Decisions

- **VRAM: 6.44 GB** — rules out large local VLMs (LLaVA-34B, InstructBLIP-13B).
- **Decision:** Use **Google Gemini 2.0 Flash** (via `google-genai`) as the default VLM and LLM. Avoids large model weight downloads, has a generous free tier (~1500 req/day), and is natively multimodal.
- **Local fallback:** Deterministic mock adapters run all tests without API keys or GPU.
- **Embedding model:** `all-MiniLM-L6-v2` — 22 MB, CPU-compatible, no API key needed.

---

## 2. Research Gap Analysis

### 2.1 Literature Review Summary

**What exists (2023–2024):**

1. **Standard image captioning / story generation** — LLaVA, InstructBLIP, GPT-4V generate text from images but with documented Western-centric bias (CulturalVQA benchmark, Google Research 2024; MOSAIC-1.5k, 2024).
2. **RAG for factual QA** — text-only retrieval-augmented LLMs (Lewis et al. 2020 and follow-up work) not connected to visual input.
3. **Culturally-aware image captioning (CIC)** — VQA + LLM pipelines (IJCAI 2023) improve cultural naming but do not generate full coherent stories and lack structured evidence alignment.
4. **Cultural bias benchmarks** — CulturalVQA, MOSAIC-1.5k, Cultural Awareness Score (CAS 2024) expose the problem but provide no generation pipeline solution.

**What does not yet exist in an integrated, open, reproducible form:**

- A full pipeline connecting: image visual observation → explicit cultural evidence retrieval → evidence-to-observation alignment and uncertainty analysis → story planning → evidence-aware generation → automated cultural consistency checking → systematic baseline comparison.
- Explicit **inspectable intermediate reasoning** (structured Pydantic representations) at every stage, making cultural grounding transparent and auditable.
- A **reproducible evaluation protocol** comparing direct generation, fixed-example prompting, and evidence-grounded generation on the same image set with documented metrics.

### 2.2 Defensible Research Gap Statement

> *Current vision-language story generation pipelines lack an explicit, inspectable evidence-grounding stage that retrieves and aligns cultural knowledge with visual observations before narrative generation. Generated stories may assert cultural claims unsupported by retrievable evidence, repeat pre-training stereotypes, and provide no mechanism for users to verify cultural claims. This project addresses that gap by implementing and empirically comparing a structured evidence-grounded pipeline against two common baselines.*

### 2.3 Testable Research Hypotheses

**Primary hypothesis (H1):**  
Incorporating dynamic retrieval of cultural evidence and structured evidence-to-observation alignment reduces the rate of unsupported cultural assertions compared with direct image-to-story generation, as measured by a documented annotation protocol on a fixed evaluation dataset.

**Secondary hypothesis (H2):**  
The evidence-grounded pipeline produces stories rated as more culturally consistent than fixed-example prompting, without a statistically significant decrease in story coherence.

**Null hypothesis (H0):**  
The evidence-grounded pipeline does not significantly differ from baselines on the unsupported-assertion rate or cultural consistency score.

> ⚠️ **Integrity note:** H1 and H2 are hypotheses to test. Neither is assumed true. Negative and null results will be reported faithfully.

---

## 3. System Architecture

### 3.1 Data Flow

```
[Image Upload] → [Input Validation] → [VLM Adapter] → [VisualObservation]
                                                              ↓
                              [KB Records JSONL] → [ChromaDB Index]
                                                              ↓
                                               [RetrievalEngine] → [RetrievedEvidence[]]
                                                              ↓
                                            [ReasoningPipeline (7 steps)]
                                              → ReasoningTrace (Pydantic)
                                                              ↓
                                              [LLM Adapter] → [GeneratedStory]
                                                              ↓
                                             [ConsistencyChecker] → [ConsistencyReport]
                                                              ↓
                                                    [Streamlit UI Display]

Baselines (parallel):
  Baseline A: Image → LLM → Story (no retrieval)
  Baseline B: Image + Fixed examples → LLM → Story

Evaluation:
  All outputs → MetricsEngine → ExperimentLogger → experiments/results/
```

### 3.2 Module Interface Summary

```
src/
├── config/        ← AppConfig (pydantic-settings), model registry
├── vision/        ← VLMAdapter protocol, GeminiVLM, MockVLM, VisualObservation schema
├── knowledge_base/← KBRecord schema, ingestion, validation
├── embeddings/    ← EmbeddingAdapter protocol, MiniLMEmbedder, MockEmbedder
├── retrieval/     ← RetrievalEngine (ChromaDB/FAISS), query builder, RetrievedEvidence schema
├── reasoning/     ← 7-step ReasoningPipeline, EvidenceAligner, ReasoningTrace schema
├── generation/    ← LLMAdapter protocol, GeminiLLM, MockLLM, story planner, GeneratedStory schema
├── verification/  ← ConsistencyChecker, rule engine, ConsistencyReport schema
├── evaluation/    ← MetricsEngine, BaselineRunner, ExperimentLogger
└── utils/         ← logger, errors, image_utils, text_utils
```

### 3.3 Pydantic Schema Boundaries

Every stage boundary carries a validated Pydantic model:

| Stage boundary | Schema |
|---|---|
| VLM output | `VisualObservation` |
| KB record | `KBRecord` |
| Retrieval output | `RetrievedEvidence` |
| Reasoning output | `ReasoningTrace` |
| Generation output | `GeneratedStory` |
| Verification output | `ConsistencyReport` |
| Evaluation output | `ExperimentResult` |

---

## 4. Proposed Folder Structure

```
FinalYear/
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
│
├── app/
│   ├── main.py
│   ├── components/
│   │   ├── image_upload.py
│   │   ├── story_display.py
│   │   ├── evidence_panel.py
│   │   ├── reasoning_trace.py
│   │   ├── consistency_warnings.py
│   │   └── baseline_comparison.py
│   └── pages/
│       ├── 01_generate.py
│       ├── 02_evaluate.py
│       └── 03_about.py
│
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   └── model_registry.py
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── adapter.py
│   │   ├── gemini_vlm.py
│   │   ├── mock_vlm.py
│   │   └── schemas.py
│   ├── knowledge_base/
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   ├── ingestion.py
│   │   ├── validator.py
│   │   └── sources.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── adapter.py
│   │   ├── minilm_embedder.py
│   │   └── mock_embedder.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── query_builder.py
│   │   ├── ranker.py
│   │   └── schemas.py
│   ├── reasoning/
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   ├── aligner.py
│   │   └── schemas.py
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── adapter.py
│   │   ├── gemini_llm.py
│   │   ├── mock_llm.py
│   │   ├── story_planner.py
│   │   └── schemas.py
│   ├── verification/
│   │   ├── __init__.py
│   │   ├── checker.py
│   │   ├── rules.py
│   │   └── schemas.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── baselines.py
│   │   ├── experiment_logger.py
│   │   └── schemas.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── errors.py
│       ├── image_utils.py
│       └── text_utils.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── sample/
│   └── annotations/
│
├── experiments/
│   ├── configs/
│   ├── results/
│   └── reports/
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── evaluation/
│
├── scripts/
│   ├── build_knowledge_base.py
│   └── run_evaluation.py
│
├── docs/
│   ├── implementation_plan.md
│   ├── architecture.md
│   ├── methodology.md
│   ├── experiment_protocol.md
│   └── limitations.md
│
└── notebooks/
    └── exploratory_analysis.ipynb
```

---

## 5. Dependency and Model Selection Plan

### 5.1 VLM / LLM

| Option | VRAM | API Key | Selected |
|---|---|---|---|
| Google Gemini 2.0 Flash | 0 (API) | Required | ✅ Default |
| moondream2 | ~2 GB | No | Local fallback |
| Mock VLM / Mock LLM | 0 | No | ✅ Tests |

### 5.2 Embedding Model

`sentence-transformers/all-MiniLM-L6-v2` — 22 MB, 384-dim, CPU, no API key.

### 5.3 Vector Store

Primary: ChromaDB 1.5.9 | Secondary (optional): faiss-cpu

### 5.4 Additional Packages to Install

```
faiss-cpu
python-dotenv
loguru
tqdm
```

---

## 6. Cultural Knowledge Base Plan

### 6.1 KB Record Schema Fields

`record_id`, `topic`, `cultural_region`, `description`, `evidence_text`,  
`source_title`, `source_url`, `source_type`, `confidence`, `known_limitations`, `tags`

### 6.2 Confidence Enum

- `verified` — traceable to named, stable source
- `plausible` — from secondary/summarised source
- `synthetic_demo` — demonstration record, not a verified fact

### 6.3 Initial Coverage (~50–100 records)

South Asian, East Asian, African, Indigenous/First Nations, European, Middle Eastern cultural practices, festivals, clothing, architecture, food.

**Source priority:** UNESCO ICH descriptions → Encyclopaedia Britannica → Government cultural ministry publications → CC-licensed Wikipedia summaries → Synthetic demo records (explicitly labelled).

---

## 7. Evaluation Protocol Summary

### 7.1 Dataset

- 20–30 CC0/public domain images from Wikimedia Commons / Unsplash
- Ground-truth cultural elements annotated per image
- Sentence-level supported/unsupported claim labels per story

### 7.2 Metrics

| Metric | Method |
|---|---|
| Unsupported-claim rate (UCR) | Annotation protocol |
| Cultural consistency score | Human / LLM rating (1–5) |
| Story coherence score | Human / LLM rating (1–5) |
| Retrieval Precision@k | KB ground truth |
| Latency (sec/image) | Wall-clock timing |
| Warning detection rate | Automated vs. annotation |

### 7.3 Three Methods Compared

| Method | Description |
|---|---|
| Baseline A | Image → Gemini Flash → story (no retrieval) |
| Baseline B | Image + 3 fixed cultural examples → Gemini Flash → story |
| Proposed | Image → visual obs → retrieval → reasoning → story |

---

## 8. Phased Implementation Checklist

### Phase 0 — Inspect and Plan ← CURRENT
- [x] Workspace inspected (empty)
- [x] Environment assessed (Python 3.11.9, RTX 4050 6.44 GB, all key libs)
- [x] Research gap identified and documented
- [x] Architecture and data flow designed
- [x] Folder structure defined
- [x] Model and dependency selection justified
- [x] Risks and assumptions documented
- [x] Plan saved to docs/implementation_plan.md

**✅ Phase 0 COMPLETE**

---

### Phase 1 — Project Foundation
**Acceptance criteria:**
- Full folder structure exists
- `python -c "from src.config.settings import AppConfig"` succeeds
- `.env.example` has placeholder values only — no real keys
- `pytest tests/` exits with 0 failures

### Phase 2 — Data and Knowledge Base
**Acceptance criteria:**
- All sample KB records pass validation
- Invalid records raise specific, catchable errors
- Duplicate detection works
- Empty file handled gracefully

### Phase 3 — Visual Understanding
**Acceptance criteria:**
- Mock VLM returns valid `VisualObservation` for any PIL Image
- Invalid image types raise `ImageValidationError`
- All unit tests pass without API key

### Phase 4 — Retrieval Pipeline
**Acceptance criteria:**
- Retrieval returns `RetrievedEvidence[]` with source references
- Empty KB returns `no_evidence_found`, not an error
- Top-k configurable via settings
- All tests pass using mock embedder

### Phase 5 — Reasoning and Story Generation
**Acceptance criteria:**
- `ReasoningTrace` produced at each step, serializable to JSON
- Unsupported claims trigger specific `ConsistencyReport` warnings
- Story generation works with mock LLM

### Phase 6 — Baselines and Evaluation
**Acceptance criteria:**
- All three methods produce output for same image
- Results saved to JSON with full config
- Pending results explicitly marked "PENDING"

### Phase 7 — User Interface
**Acceptance criteria:**
- App loads: `streamlit run app/main.py`
- Upload → generate end-to-end works
- All error states handled visibly

### Phase 8 — Integration and QA
**Acceptance criteria:**
- All unit + integration tests pass
- 0 unhandled exceptions for any documented input type

### Phase 9 — Documentation and Delivery
**Acceptance criteria:**
- Another student can set up from README only
- All cited results are real, with configs
- Limitations section is honest and complete

---

## 9. Risks, Assumptions, and External Requirements

### 9.1 Risks

| Risk | Mitigation |
|---|---|
| Gemini API rate limit during eval | Cache all results; batch evaluation |
| Gemini API key unavailable | Full mock pipeline works without key |
| ChromaDB version incompatibility | Pinned version; FAISS fallback |
| 6.44 GB VRAM insufficient for local VLMs | Gemini API default; moondream2 fallback |
| KB records contain inaccuracies | Every record has source URL; synthetic labelled |
| Human annotation is time-consuming | Limit to 20 images; LLM-assisted secondary metric |

### 9.2 Assumptions

1. A Gemini API key is available for integration testing and evaluation.
2. Sample KB (~50–100 records) is sufficient to demonstrate retrieval.
3. 20–30 CC-licensed test images are obtainable from Wikimedia Commons.
4. A single researcher can annotate the initial evaluation set.

### 9.3 External Requirements

| Requirement | Status |
|---|---|
| `GEMINI_API_KEY` | Required for real runs; mock for tests |
| Internet connection | Required for API mode |
| ~500 MB disk space | Available |

### 9.4 Unresolved Decisions

1. **LLM-based consistency checking:** Use Gemini (quality, costs quota) or deterministic rules only by default?
2. **Reading-level configurability:** Phase 5 or deferred?
3. **Report format:** LaTeX or Markdown?

> Assumed defaults if no response: (1) LLM check optional/configurable, (2) Deferred, (3) Markdown with LaTeX-compatible structure.

---

## 10. Phase 0 Quality Gate

| Criterion | Status |
|---|---|
| Workspace inspected | ✅ PASS |
| Environment fully assessed | ✅ PASS |
| Research gap defensible | ✅ PASS |
| System architecture designed | ✅ PASS |
| Folder structure defined | ✅ PASS |
| Model selection justified | ✅ PASS |
| Phased checklist with acceptance criteria | ✅ PASS |
| Risks and assumptions documented | ✅ PASS |
| No code generated prematurely | ✅ PASS |
| No fabricated citations or results | ✅ PASS |

**Phase 0 Status: ✅ COMPLETE — Ready to begin Phase 1**
