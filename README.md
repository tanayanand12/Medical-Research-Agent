# 🧠 Agentic Pipeline Clinical

![Agentic Pipeline Clinical Banner](https://images.unsplash.com/photo-1515378791036-0648a3ef77b2?auto=format&fit=crop&w=1200&q=80)

---

## Overview

**Agentic Pipeline Clinical** is a modular, agent-based research assistant platform for advanced medical and clinical literature retrieval, synthesis, and question answering.  
It leverages state-of-the-art Retrieval-Augmented Generation (RAG), vector search, and LLM orchestration to provide evidence-based, citation-rich answers to clinical and biomedical queries.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [GCP Integration](#gcp-integration)
- [Agents](#agents)
- [Installation](#installation)
- [GCP Integration](#GCP-credentials)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [Indexing Pipeline](#indexing-pipeline)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features

- **Multi-Agent Orchestration:** Modular agents for PubMed, FDA, ClinicalTrials, and more.
- **Retrieval-Augmented Generation (RAG):** Combines vector search with LLMs for grounded, evidence-based answers.
- **FAISS Vector Indexing:** Fast, scalable similarity search over millions of medical documents.
- **OpenAI GPT-4/5 Integration:** Supports latest LLMs with advanced prompt engineering.
- **Fallback Mechanisms:** Ensures robust answers even when primary agents fail.
- **Citation & Evidence Tracking:** All answers are traceable to original literature.
- **Extensible:** Easily add new agents, data sources, or LLM backends.
- **GCP Integration:** Uses Google Cloud Platform for scalable storage and management of indexes and datasets.

---

## Architecture

![Architecture Diagram](https://raw.githubusercontent.com/your-org/agentic-pipeline-clinical/main/docs/architecture.png)

1. **User/API** sends a clinical question.
2. **Orchestrator** routes the query to specialized agents (PubMed, FDA, etc.).
3. **Agents** perform vector search, retrieve relevant documents, and use LLMs for synthesis.
4. **Fallback** and **Aggregator** modules ensure answer quality and completeness.
5. **Response** is returned with citations, confidence scores, and evidence summaries.

---

## ☁️ GCP Integration

This project makes extensive use of **Google Cloud Platform (GCP)** for scalable, secure, and efficient management of medical research data and indexes.

### GCP Usage Highlights

- **Cloud Storage:**  
  All vector indexes, document stores, and large datasets are stored in Google Cloud Storage buckets (e.g., `gcp-indexes/`). This enables fast, distributed access and easy scaling.
- **Index Management:**  
  FAISS indexes and metadata are uploaded, versioned, and retrieved from GCP, supporting collaborative workflows and robust backup.
- **Compute:**  
  (If applicable) Indexing and embedding pipelines can be run on GCP Compute Engine or Vertex AI for high-throughput processing.
- **Security:**  
  GCP IAM and bucket policies are used to control access to sensitive medical data and indexes.

### Example: Using GCP Storage for Indexes

```
gcp-indexes/
├── randy-data-testing.documents
├── randy-data-testing.index
```

Indexes are automatically loaded from GCP buckets at runtime, ensuring the latest data is always available to the agents.

### Configuration

- Set your GCP credentials in the environment or via `.env`:
  ```
  GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account.json
  GCP_BUCKET_NAME=your-bucket-name
  ```
- The `.gitignore` is configured to exclude all GCP index files and credentials.

---

## 🧑‍⚕️ Agents

The system is built around specialized “agents,” each designed to handle a specific domain or data source. This modular approach allows for robust, explainable, and extensible clinical research assistance.

### 1. **PubMedAgent**
- **Purpose:** Answers medical and scientific questions using academic literature indexed from PubMed.
- **How it works:**  
  Retrieves relevant research papers using vector similarity search, then synthesizes answers using an LLM.  
- **Strength:** Provides evidence-based, citation-rich responses from peer-reviewed biomedical literature.

### 2. **FDAAgent**
- **Purpose:** Handles queries related to drug/device approvals, regulatory status, and FDA datasets.
- **How it works:**  
  Searches FDA databases and synthesizes regulatory insights using LLMs.
- **Strength:** Delivers up-to-date regulatory and approval information.

### 3. **ClinicalTrialsAgent**
- **Purpose:** Answers questions about ongoing or completed clinical trials.
- **How it works:**  
  Searches clinical trial registries and summarizes findings using LLMs.
- **Strength:** Provides trial outcomes, eligibility criteria, and study summaries.

### 4. **LocalAgent**
- **Purpose:** Handles proprietary or institution-specific datasets not covered by public agents.
- **How it works:**  
  Uses local vector indexes and LLMs to answer queries from custom datasets.
- **Strength:** Enables private, organization-specific knowledge retrieval.

Each agent implements a common interface, making it easy to add new agents for other data sources (e.g., insurance claims, EHRs, or specialty registries) as your needs grow.

---

## Installation

### 1. Clone the Repository

```sh
git clone https://github.com/your-org/agentic-pipeline-clinical.git
cd agentic-pipeline-clinical
```

### 2. Set Up Python Environment

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html):

```sh
conda create -n pranav-medical-agent python=3.10
conda activate pranav-medical-agent
```

### 3. Install Dependencies

```sh
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your API keys and configuration:

```sh
cp .env.example .env
# Edit .env with your OpenAI API key, GCP credentials, model IDs, etc.
```
---
### GCP Credentials

If you use Google Cloud Platform features, place your GCP service account key as `local_agent/service_account_credentials.json`.  
**This file is already excluded from version control via `.gitignore` for security.**  
You may need to set the following environment variable before running the API:

```sh
export GOOGLE_APPLICATION_CREDENTIALS="local_agent/service_account_credentials.json"
```
Or on Windows:
```sh
set GOOGLE_APPLICATION_CREDENTIALS=local_agent/service_account_credentials.json
```

---

## Quickstart

### 1. Build or Download Vector Indexes

- **To build your own PubMed index:**
  ```sh
  python -m pubmed_local_agent.process --db_name index
  ```
- **Or download prebuilt indexes** (see [docs/indexing.md](docs/indexing.md) for links).

### 2. Start the API

```sh
python research_agent_api.py
```

### 3. Query the API

Send a POST request to `/query`:

```sh
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the typical pain score for patients using a TR Band?"}'
```

---

## Usage

### Example Query

```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"question": "What are the latest findings on SGLT2 inhibitors in heart failure?"}
)
print(response.json())
```

### Example Response

```json
{
  "answer": "...evidence-based summary...",
  "citations": [
    {"title": "...", "pdf": "...", "page": 12}
  ],
  "confidence": 0.92
}
```

---

## Indexing Pipeline

1. **Document Ingestion:** PDFs and metadata are loaded and chunked.
2. **Embedding Generation:** Text chunks are embedded using OpenAI or local models.
3. **FAISS Indexing:** Embeddings are stored in a FAISS vector database.
4. **Metadata Storage:** Citation and provenance data are stored alongside vectors.
5. **GCP Upload:** Indexes and metadata are uploaded to GCP Cloud Storage for distributed access.

![Indexing Pipeline](https://raw.githubusercontent.com/your-org/agentic-pipeline-clinical/main/docs/indexing_pipeline.png)

---

## API Reference

### `/query` (POST)

- **Input:**  
  ```json
  { "question": "..." }
  ```
- **Output:**  
  ```json
  {
    "answer": "...",
    "citations": [...],
    "confidence": 0.85
  }
  ```

### `/status` (GET)

- Returns health and status info.

---

## Configuration

- **.env:** Store your API keys, GCP credentials, and model IDs here.
- **.gitignore:** Ensures sensitive and large files (indexes, logs, credentials) are not tracked.

### Example `.gitignore` additions:
```
# GCP credentials and indexes
gcp-indexes/
*.documents
*.index
logs/
*.json
```

---

## File Structure

```
agentic-pipeline-clinical/
│
├── pubmed_local_agent/         # PubMed vectorization and QA modules
├── local_agent/                # Local dataset agent and RAG module
├── orchestrator.py             # Orchestrator for agent routing
├── research_agent_api.py       # API entrypoint
├── requirements.txt
├── .env.example
├── .gitignore
├── logs/
├── gcp-indexes/
└── README.md
```

---

## Contributing

We welcome contributions!  
Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

- Fork the repo
- Create a feature branch
- Submit a pull request

---

## License

This project is licensed under the MIT License.  
See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [OpenAI](https://openai.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [LangChain](https://github.com/langchain-ai/langchain)
- [PubMed](https://pubmed.ncbi.nlm.nih.gov/)
- [FDA](https://www.fda.gov/)
- [Google Cloud Platform](https://cloud.google.com/)
- All contributors and the open-source community

---

> _Empowering clinicians and researchers with transparent, evidence-based AI._

![Clinical AI](https://images.unsplash.com/photo-1465101046530-73398c7f28ca?auto=format&fit=crop&w=1200&q=80)