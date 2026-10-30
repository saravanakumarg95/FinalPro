# Cultural Inclusive Story Generation from Images using Reasoning

> **Final Year AI/ML Engineering Project**  
> A research-oriented system that generates culturally grounded, inclusive stories from input images using Evidence-Grounded Cultural Reasoning.

---

## Project Overview

This system investigates whether retrieving relevant cultural evidence and incorporating it into a structured story-generation pipeline improves cultural grounding and contextual consistency compared with direct image-to-story generation.

**Research Hypothesis:** Incorporating dynamic retrieval of cultural evidence and structured evidence-to-observation alignment reduces the rate of unsupported cultural assertions compared with direct image-to-story generation.

### Pipeline Stages

1. **Image Input** — Validation and preprocessing
2. **Visual Understanding** — VLM extracts observations, cultural clues, and uncertainty
3. **Cultural Knowledge Retrieval** — Semantic search over a curated cultural KB
4. **Structured Reasoning** — 7-step evidence-alignment and story planning
5. **Story Generation** — Evidence-aware, culturally grounded narrative
6. **Consistency Checking** — Automated detection of unsupported claims and stereotypes
7. **Evaluation** — Comparison against two baselines with documented metrics

---

## Quick Start

### 1. Prerequisites

- Python 3.11.x
- NVIDIA GPU with CUDA (optional but recommended)
- A Google Gemini API key ([get free key](https://aistudio.google.com/apikey))

### 2. Clone and install

```bash
git clone <your-repo-url>
cd FinalYear
pip install -r requirements.txt
```

### 3. Configure environment

```bash
# Copy the example file
copy .env.example .env

# Edit .env and add your Gemini API key:
# GEMINI_API_KEY=your_actual_key_here
```

### 4. Run in mock mode (no API key required)

```bash
# Verify installation
python -c "from src.config.settings import AppConfig; print(AppConfig())"

# Run tests (all mock — no API calls)
pytest tests/ -v
```

### 5. Build the knowledge base

```bash
python scripts/build_knowledge_base.py
```

### 6. Launch the UI

```bash
streamlit run app/main.py
```

### 7. Launch the React UI

The React client is the preferred frontend. Run the Python API and Vite client
in separate terminals:

```bash
# Terminal 1 — API (use APP_MODE=live for Gemini-backed generation)
set APP_MODE=mock
uvicorn api.main:app --reload --port 8000

# Terminal 2 — React development server
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The client sends image uploads to
`http://localhost:8000/api/generate` and displays visual observations,
retrieved evidence, the generated story, consistency checks, and optional
reasoning/baseline panels.

---

## Project Structure

```
FinalYear/
├── app/                    # Streamlit UI
├── src/
│   ├── config/             # AppConfig (pydantic-settings)
│   ├── vision/             # VLM adapters (Gemini, mock)
│   ├── knowledge_base/     # KB schema, ingestion, validation
│   ├── embeddings/         # Embedding adapters (MiniLM, mock)
│   ├── retrieval/          # ChromaDB/FAISS retrieval engine
│   ├── reasoning/          # 7-step reasoning pipeline
│   ├── generation/         # LLM adapters, story planner
│   ├── verification/       # Consistency & inclusivity checker
│   ├── evaluation/         # Metrics, baselines, experiment logger
│   └── utils/              # Logger, errors, image/text utilities
├── data/
│   ├── raw/                # Original KB source JSONL
│   ├── processed/          # ChromaDB index
│   ├── sample/             # Test images (CC-licensed)
│   └── annotations/        # Human annotation files
├── experiments/            # Configs, results, reports
├── tests/                  # Pytest tests (unit, integration, eval)
├── scripts/                # build_knowledge_base.py, run_evaluation.py
└── docs/                   # Architecture, methodology, limitations
```

---

## Environment Variables

See [`.env.example`](.env.example) for all options. Key settings:

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(empty)* | Google Gemini API key |
| `APP_MODE` | `mock` | `mock` = no API calls; `live` = real models |
| `RETRIEVAL_TOP_K` | `5` | Number of KB records to retrieve |
| `STORY_TARGET_WORDS` | `400` | Target story length |
| `VECTOR_STORE` | `chromadb` | Vector store backend |

---

## Running Tests

```bash
# All tests (mock mode — no API key required)
pytest tests/ -v

# Specific module
pytest tests/unit/test_image_validation.py -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## Research Methods

### Three compared methods

| Method | Description |
|---|---|
| **Baseline A** | Image → Gemini Flash → story (no retrieval) |
| **Baseline B** | Image + fixed cultural examples → Gemini Flash → story |
| **Proposed** | Image → visual obs → cultural KB retrieval → evidence reasoning → story |

### Evaluation metrics

- Unsupported-claim rate (UCR) — annotated per sentence
- Cultural consistency score — human/LLM rating (1–5)
- Story coherence score — human/LLM rating (1–5)
- Retrieval Precision@k
- Latency (sec/image)

---

## Academic Integrity

- All benchmark scores are from actual experiments, not fabricated.
- Synthetic KB records are explicitly labelled `confidence: synthetic_demo`.
- All citations include verifiable sources.
- Negative results are reported faithfully.

See [`docs/limitations.md`](docs/limitations.md) for known limitations.

---

## Hardware Requirements

| Mode | GPU | API Key | Notes |
|---|---|---|---|
| Mock (tests) | Not required | Not required | All deterministic |
| Development | Recommended | Required | Gemini Flash (free tier) |
| Full evaluation | RTX 4050+ | Required | ~1–2 hrs for 20-image eval |

---

## License

This project is for academic research purposes.  
Images in `data/sample/` are CC0 or used with documented attribution.  
The cultural knowledge base records include source URLs for every non-synthetic entry.

---

## Phased Implementation Status

| Phase | Description | Status |
|---|---|---|
| 0 | Inspect and Plan | ✅ Complete |
| 1 | Project Foundation | ✅ Complete |
| 2 | Data and Knowledge Base | 🔄 In Progress |
| 3 | Visual Understanding | ⏳ Pending |
| 4 | Retrieval Pipeline | ⏳ Pending |
| 5 | Reasoning and Story Generation | ⏳ Pending |
| 6 | Baselines and Evaluation | ⏳ Pending |
| 7 | User Interface | ⏳ Pending |
| 8 | Integration and QA | ⏳ Pending |
| 9 | Documentation and Delivery | ⏳ Pending |
