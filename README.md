# Medical Research Agent

LLM-agnostic, graph-based orchestration system for clinical question answering. Retrieves evidence from PubMed, FDA, ClinicalTrials.gov, and local FAISS indexes, then synthesises citation-rich answers with quantified confidence scores.

> **Version history:** The pre-LangGraph implementation is frozen on branch [`legacy/v1`](https://github.com/tanayanand12/Medical-Research-Agent/tree/legacy/v1) and tag [`v1.0.0-legacy`](https://github.com/tanayanand12/Medical-Research-Agent/releases/tag/v1.0.0-legacy). The [`main`](https://github.com/tanayanand12/Medical-Research-Agent/tree/main) branch contains the current precision evidence orchestration system evaluated in the accompanying paper.

## Architecture

The system uses a **LangGraph StateGraph** with 8 nodes and 2 conditional routing edges:

```
classify_intent --> discover_skills --> parallel_retrieve --> synthesise
    |                                                            |
    | (non-medical)                                    score_confidence
    |                                                            |
    v                                                  evaluate_coherence
format_response <----------------------------------------------+
    ^                                                  |
    |                                       (low coherence)
    +------ fallback_regen <-----------------------+
```

| Component | Technology |
|-----------|------------|
| LLM Layer | LiteLLM router (OpenAI, Anthropic, Ollama) |
| Orchestration | LangGraph StateGraph with typed `AgentState` |
| Tool Interface | MCP tools with JSON Schema validation |
| Tool Selection | YAML skill manifests + semantic/keyword scoring |
| Observability | LangSmith tracing, Prometheus metrics, structured JSON logging |
| Evaluation | RAGAS-style metrics (faithfulness, relevancy, citation fidelity, hallucination rate) |

## Quick Start

### Prerequisites

- Python 3.10+
- API key for at least one LLM provider (OpenAI, Anthropic, or local Ollama)

### Install

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### Configure

Create `.env` in this directory:

```env
# Required: at least one LLM provider
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# OLLAMA_BASE_URL=http://localhost:11434

# Model selection (defaults shown)
DEFAULT_LLM_MODEL=gpt-4o
DEFAULT_EMBEDDING_MODEL=text-embedding-3-large

# Optional: observability
# LANGSMITH_API_KEY=ls_...
# LANGSMITH_PROJECT=medical-research-agent
```

### Run

```bash
python -m uvicorn research_agent_api_v2:app --host 0.0.0.0 --port 8000
```

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the latest findings on SGLT2 inhibitors in heart failure?"}'
```

Response includes: `answer`, `citations`, `confidence`, `trace_id`, `execution_time_sec`, `fallback_triggered`.

### Health Check

```bash
curl http://localhost:8000/health
```

### Container

Versioned images are published to GitHub Container Registry after the release workflow succeeds. The `v2.0.0` tag becomes pullable only once that workflow completes for tag `v2.0.0`—not before.

```bash
docker pull ghcr.io/tanayanand12/medical-research-agent:v2.0.0
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  ghcr.io/tanayanand12/medical-research-agent:v2.0.0
```

Provider credentials are injected at runtime and are never baked into the image. Default model registry and configuration files such as `models.yaml` are included in the image, while model selection and provider settings can be overridden through runtime environment variables.

## Continuous integration and delivery

Pull requests and pushes to `main` run the deterministic offline test suite and build the container image without publishing it.

Tags matching `v*` trigger the release workflow, which publishes versioned tags and `latest` to `ghcr.io/tanayanand12/medical-research-agent`. The same workflow accepts manual dispatch with image tags `manual` or `manual-*` for ad-hoc test builds.

Live-provider evaluations are intentionally excluded from CI because they require credentials and query mutable external evidence services.

## Project Structure

```
Medical-Research-Agent/         # repository root
+-- research_agent_api_v2.py    # FastAPI server (/query, /health, /graph/*)
+-- graph.py                    # LangGraph StateGraph (8 nodes, 2 conditional edges)
+-- agent_state.py              # AgentState TypedDict (30+ fields)
+-- edges.py                    # Conditional routing functions
+-- llm_client.py               # LiteLLM singleton (provider-agnostic LLM calls)
+-- models.yaml                 # Model registry (cost, context window, tier)
+-- skill_router.py             # YAML-based skill discovery engine
+-- query_classifier.py         # Medical domain filtering
+-- aggregator.py               # Multi-source response synthesis
+-- fallback.py                 # Coherence evaluation + fallback regeneration
+-- citation_formatter.py       # AMA-style citation formatting
+-- nodes/                      # LangGraph node implementations
|   +-- classify_intent.py
|   +-- discover_skills.py
|   +-- parallel_retrieve.py
|   +-- synthesise.py
|   +-- score_confidence.py
|   +-- evaluate_coherence.py
|   +-- fallback_regen.py
|   +-- format_response.py
+-- tools/                      # MCP tool implementations
|   +-- mcp_base.py             # MCPToolBase + auto-discovery registry
|   +-- pubmed_tool.py
|   +-- fda_tool.py
|   +-- clinical_trials_tool.py
|   +-- local_index_tool.py
|   +-- pubmed_deep_tool.py
|   +-- rag_retrieve_tool.py
+-- skills/                     # YAML skill manifests (triggers, domains, cost)
+-- rag_engine/                 # RAG pipeline (chunker, embedder, dense/sparse index)
+-- agents/                     # LangGraph sub-agent wrappers per data source
+-- observability/              # LangSmith, Prometheus, structured logging, middleware
+-- eval/                       # Evaluation harness (MedQA, BioASQ, custom datasets)
+-- scripts/                    # Reproducibility and utility scripts
```

## Switching LLM Backends

Change `DEFAULT_LLM_MODEL` in `.env` -- no code changes required:

```env
# OpenAI
DEFAULT_LLM_MODEL=gpt-4o

# Anthropic
DEFAULT_LLM_MODEL=claude-3-5-sonnet

# Local Ollama
DEFAULT_LLM_MODEL=ollama/mistral
OLLAMA_BASE_URL=http://localhost:11434
```

## Running Evaluations

```bash
# Quick sanity check
python eval/run_eval.py --dataset medqa --n_samples 10 --agents pubmed --output results/quick.json

# Full benchmark across models
python eval/run_eval.py --dataset medqa --n_samples 200 \
  --model gpt-4o --model claude-3-5-sonnet \
  --output results/comparison.json
```

## Observability

- **LangSmith**: Set `LANGSMITH_API_KEY` to enable automatic tracing of every graph run. Each response includes a `trace_id` for replay.
- **Prometheus**: Metrics exposed on port 9000. Use `docker-compose up` to start Prometheus scraping.
- **Structured logging**: JSON logs written to `logs/observability.log`.

## Reproducibility

```bash
bash scripts/reproduce_baseline.sh
```

This script runs a deterministic 10-sample MedQA evaluation with seed=42 and writes results to `results/baseline_repro.json`.

## License

Research use. See repository root for license details.
