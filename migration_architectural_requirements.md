# Migration Architectural Requirements

**Document Version:** 1.0  
**Target System:** `./medical-research-agent/`  
**Date:** 2025-04-12  
**Status:** Non-negotiable technical standards for Phase 1–7 implementation

---

## Executive Summary

This document defines the **non-negotiable technical standards** for the target Medical Research Agent system. Every implementation decision in Phases 1–7 must be traceable to a requirement defined here. The system migrates from OpenAI-locked, ThreadPool-based orchestration to a LLM-agnostic, LangGraph state machine architecture with full observability and evaluation infrastructure.

**Key drivers for the architecture:**
1. **LLM Provider Independence** — Support multiple LLM backends (OpenAI, Anthropic, Ollama, Azure) without code changes
2. **Reproducibility** — Every run is fully traceable, repeatable, and evaluated against benchmarks
3. **Extensibility** — Adding a new data source (MAUDE, PubChem) requires ≤ 1 day of work
4. **Production Readiness** — Full observability (traces, metrics, logs), error handling, and fallback mechanisms

---

## REQ-LLM-*: LLM Agnostic Core

### REQ-LLM-001: Unified LLMClient Interface

**Requirement:** All LLM calls must route through a single `LLMClient` abstraction layer.

**Specification:**

```python
class LLMClient:
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Call LLM with provider-agnostic interface. Returns text response."""
    
    def embed(
        self,
        text: str,
        model: Optional[str] = None
    ) -> List[float]:
        """Embed text. Returns vector of dimension matching model."""
    
    def get_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """Calculate cost of a call."""
```

**Usage Pattern:**
- No direct imports of `openai`, `anthropic`, or model-specific libraries in application code
- All model selection goes through LLMClient constructor (via `DEFAULT_LLM_MODEL` env var or request parameter)
- LLMClient is a singleton per process

**Evidence:** [INV: orchestrator.py:99, aggregator.py:27, fallback.py:99] — Current system hardcodes OpenAI in 8+ places. New system must have zero hardcoded model names outside `models.yaml`.

### REQ-LLM-002: LiteLLM Router Integration

**Requirement:** Use LiteLLM as the underlying LLM routing library.

**Specification:**

LiteLLM configuration via `models.yaml`:

```yaml
models:
  - model_name: gpt-4o
    litellm_params:
      api_key: $OPENAI_API_KEY
      organization_id: $OPENAI_ORG_ID  # optional
    model_info:
      cost_per_1k_input_tokens: 0.005
      cost_per_1k_output_tokens: 0.015
      context_window_tokens: 128000
      supports_vision: true
      tier: enterprise
  
  - model_name: claude-3-5-sonnet
    litellm_params:
      api_key: $ANTHROPIC_API_KEY
    model_info:
      cost_per_1k_input_tokens: 0.003
      cost_per_1k_output_tokens: 0.015
      context_window_tokens: 200000
      supports_vision: true
      tier: enterprise
  
  - model_name: ollama/mistral
    litellm_params:
      base_url: http://localhost:11434
    model_info:
      cost_per_1k_input_tokens: 0.0
      cost_per_1k_output_tokens: 0.0
      context_window_tokens: 32768
      supports_vision: false
      tier: local

embeddings:
  - model_name: text-embedding-3-large
    litellm_params:
      api_key: $OPENAI_API_KEY
    model_info:
      dimension: 3072
      cost_per_1k_tokens: 0.00002
  
  - model_name: ollama/nomic-embed-text
    litellm_params:
      base_url: http://localhost:11434
    model_info:
      dimension: 768
      cost_per_1k_tokens: 0.0
```

**Fallback Order:**
1. Explicitly requested model (from request parameter)
2. Default model (from `DEFAULT_LLM_MODEL` env var)
3. Primary enterprise model (gpt-4o)

**Implementation:**
- `LLMClient` wraps `litellm.completion()` and `litellm.embedding()`
- Load `models.yaml` at startup
- Validate model exists in registry before calling
- Track tokens used per call for cost calculation
- Emit cost metrics to Prometheus per model

**Evidence:** [INV: research_agent_api.py:95-98] — Current system sets `openai.api_key` globally, making it impossible to switch providers. New system must support this via single env var change.

### REQ-LLM-003: Support for Multiple Backends

**Requirement:** Minimum viable set of supported LLM providers.

**Specification:**

| Provider | Models Supported | Status | Notes |
|----------|-----------------|--------|-------|
| OpenAI | gpt-4o, gpt-4-turbo | Primary | Recommended for production |
| Anthropic | claude-3-5-sonnet, claude-3-opus | Primary | Recommended for production |
| Ollama (Local) | mistral, llama2, neural-chat | Secondary | For local development/testing |
| Azure OpenAI | gpt-4, gpt-4-turbo | Optional | Via LiteLLM Azure backend |

**Minimum requirement:** System must be testable with at least one OSS model (Mistral via Ollama) without cost.

**Evidence:** [INV: query_classifier.py:97] — Uses gpt-3.5-turbo for classification; must support cheaper local alternative for testing.

### REQ-LLM-004: Parallel Model Evaluation

**Requirement:** Support running the same query through multiple LLM backends simultaneously.

**Specification:**

```python
# In evaluation harness:
models_to_compare = ["gpt-4o", "claude-3-5-sonnet", "ollama/mistral"]
for query in test_queries:
    results = await asyncio.gather(*[
        graph.ainvoke({
            "query": query,
            "model_id": model
        })
        for model in models_to_compare
    ])
    # results[i] = response from models_to_compare[i]
```

**LangGraph Integration:**
- `AgentState.model_id` parameter selects LLM for this run
- Evaluation harness loops over models, passes to graph
- Results aggregated per model for comparison

**Output:** Side-by-side performance table per model (accuracy, latency, cost)

### REQ-LLM-005: Cost Tracking and Reporting

**Requirement:** Every LLM call logs its token usage and cost.

**Specification:**

```python
# LLMClient must track:
class CallMetrics:
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float
    timestamp: datetime

# Logged to:
# 1. Prometheus metric: mra_llm_cost_usd{model}
# 2. Structured log: {"tokens_in": X, "tokens_out": Y, "cost": Z}
# 3. LangSmith run metadata
```

**Cost Calculation:**
- `cost = (tokens_in * cost_per_1k_in + tokens_out * cost_per_1k_out) / 1000`
- Look up rates from `models.yaml`
- Report cumulative cost per run in response

---

## REQ-MCP-*: MCP Tool Interface

### REQ-MCP-001: Tool Definition Standard

**Requirement:** All retrieval sources implement the MCP tool interface.

**Specification:**

| Data Source | MCP Tool Name | Input Schema | Output Schema |
|-------------|---------------|--------------|---------------|
| PubMed | `search_pubmed` | `{query: str, top_k: int, filter?: str}` | `[{pmid: str, title: str, abstract: str, score: float, url: str}]` |
| FDA | `search_fda` | `{query: str, top_k: int, dataset?: str}` | `[{id: str, label: str, date: str, text: str, score: float}]` |
| Clinical Trials | `search_clinical_trials` | `{condition: str, top_k: int, status?: str}` | `[{nct_id: str, title: str, phase: str, status: str, score: float}]` |
| Local Index (FAISS) | `search_local_index` | `{query: str, top_k: int, model_id: str}` | `[{doc_id: str, text: str, metadata: dict, score: float}]` |
| MAUDE (new) | `search_maude` | `{device_name: str, top_k: int, event_type?: str}` | `[{report_id: str, event: str, text: str, date: str, score: float}]` |

**JSON Schema Validation:**

Each tool must define:

```python
class MCPToolBase(ABC):
    name: str  # e.g., "search_pubmed"
    description: str  # Human-readable purpose
    input_schema: dict  # JSON Schema Draft 7
    output_schema: dict  # JSON Schema Draft 7
    
    def call(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute tool with validated input.
        
        Returns:
            {
                "status": "success" | "error",
                "output": [...],  # matches output_schema if success
                "error": str,  # present if status == "error"
                "execution_time_ms": int
            }
        """
        pass
```

**Validation:**
- MCPToolRegistry validates `input_dict` against `input_schema` before calling tool
- Tool raises `ToolInputValidationError` if invalid
- Tool catches internal errors and returns `{"status": "error", "error": "...message..."}`
- Registry catches and logs all errors; graph continues with other tools

**Evidence:** [INV: local_agent_wrapper.py:57-80, pubmed_local_agent_wrapper.py:28-45] — Current wrappers have inconsistent schemas and error handling. New system must enforce JSON Schema contracts.

### REQ-MCP-002: In-Process Tool Execution

**Requirement:** All MCP tools run as in-process Python modules (single FastAPI process), not as separate HTTP services.

**Specification:**

**Architecture (NOT this):**
```
FastAPI → gRPC → [MCP Tool Server A]
       → gRPC → [MCP Tool Server B]
```

**Architecture (THIS):**
```
FastAPI → LangGraph Graph
         └── MCP Tool Registry
             ├── search_pubmed.py (in-process Python class)
             ├── search_fda.py (in-process Python class)
             ├── search_clinical_trials.py (in-process Python class)
             └── search_local_index.py (in-process Python class)
```

**Rationale:**
- Eliminates inter-process RPC latency (reduces P95 from 8s to <5s)
- Simplifies deployment (single container, single process model)
- Enables LangGraph's parallel edges (fan-out to all tools atomically)
- Easier testing (instantiate tool directly, no HTTP mocking)

**Deployment Model:**
- Medical Research Agent is a **monolithic FastAPI app**
- Single Gunicorn/Uvicorn worker pool
- All tools run in the same Python process as the graph
- Tools can spawn background threads for I/O (requests library, etc.)

**Exception:** If a tool needs to call an external deployed service (e.g., Local Agent as a Cloud Run service), that's a tool *implementation detail*. The tool itself is still in-process; it just makes HTTP calls internally.

### REQ-MCP-003: Tool Registration and Discovery

**Requirement:** Tools are discovered and registered automatically at startup.

**Specification:**

**Module Registry Pattern:**

```python
# mcp_tools/__init__.py
from mcp_tools.search_pubmed import SearchPubMedTool
from mcp_tools.search_fda import SearchFDATool
# ... etc

# MCPToolRegistry auto-discovers:
# 1. Scan mcp_tools/ directory for .py files
# 2. Import each module
# 3. Find classes inheriting from MCPToolBase
# 4. Instantiate and register

registry = MCPToolRegistry()
registry.auto_load("mcp_tools/")
# registry now contains: search_pubmed, search_fda, etc.
```

**Zero Wiring Required:**
- Adding a new tool file (`search_maude.py`) automatically registers it
- No changes to graph logic, API, or orchestrator needed
- Registration happens at FastAPI startup

**Tool Listing Endpoint:**

```python
GET /tools
→ {
    "tools": [
        {"name": "search_pubmed", "description": "..."},
        {"name": "search_fda", "description": "..."},
        ...
    ]
}
```

### REQ-MCP-004: Error Handling and Fallback

**Requirement:** Tool failures do not crash the graph; errors are logged and other tools continue.

**Specification:**

**Error Mapping:**

| Exception | Handling | Graph Behavior |
|-----------|----------|---|
| `ToolInputValidationError` | Log warning, skip tool | Continue to next tool |
| `ToolTimeoutError` | Log error, skip tool | Continue with timeout in output |
| `ToolRetrievalError` (e.g., API down) | Log error, skip tool | Continue; may trigger fallback if all tools fail |
| `ToolConfigError` (e.g., API key missing) | Log fatal error, fail fast | Orchestrator rejects request with 500 error |

**In LangGraph:**
```python
# Node: parallel_retrieve
# Calls: registry.call_parallel(tools, input_dict)
# Returns: Dict[tool_name] → {status, output or error}
# Never raises exception; always completes

tool_outputs = {
    "search_pubmed": {"status": "success", "output": [...]},
    "search_fda": {"status": "error", "error": "API rate limit"},
    "search_clinical_trials": {"status": "success", "output": [...]}
}
# Graph continues with available outputs
```

**Evidence:** [INV: orchestrator.py:114-133] — Current system tries/excepts individual agents but doesn't handle partial failures. New system must continue with available data.

---

## REQ-GRAPH-*: LangGraph Orchestration

### REQ-GRAPH-001: State Schema Definition

**Requirement:** Single typed `AgentState` TypedDict governs all graph state.

**Specification:**

```python
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime

class AgentState(TypedDict):
    # Input
    query: str
    model_id: str  # LLM to use
    embedding_model: str  # Embedding model to use
    
    # Intermediate: Classification
    classified_intent: str  # "medical" | "non_medical"
    intent_confidence: float  # 0.0-1.0
    rejection_reason: Optional[str]  # if non_medical
    
    # Intermediate: Skill Discovery
    selected_skills: List[str]  # e.g., ["search_pubmed", "search_fda"]
    skill_scores: Dict[str, float]  # {tool_name: similarity_score}
    
    # Intermediate: Retrieval
    tool_outputs: Dict[str, Dict[str, Any]]  # {tool_name: {status, output, error}}
    retrieval_count: int  # total docs retrieved
    
    # Intermediate: Synthesis
    synthesis: str  # LLM-generated answer text
    raw_citations: List[Dict[str, Any]]  # citation objects from retriever
    
    # Output
    answer: str  # final formatted answer
    citations: List[Dict[str, Any]]  # AMA-formatted citations
    confidence_score: float  # 0.0-1.0
    fallback_triggered: bool  # was fallback used?
    fallback_reason: Optional[str]  # why fallback was triggered
    
    # Observability
    trace_id: str  # unique run identifier
    execution_time_ms: int
    total_tokens_in: int  # LLM input tokens
    total_tokens_out: int  # LLM output tokens
    total_cost_usd: float
    
    # Error handling
    error: Optional[str]  # if any fatal error
    error_step: Optional[str]  # which node failed
```

**Immutability Guarantee:**
- State is immutable across nodes (LangGraph enforces this)
- Nodes return modified copies of state
- No shared mutable state between parallel tools

### REQ-GRAPH-002: Graph Node Definitions

**Requirement:** LangGraph StateGraph with 8 required nodes.

**Specification:**

#### Node 1: `classify_intent`

**Input:** `query`  
**Output:** `classified_intent`, `intent_confidence`, `rejection_reason`  
**Logic:**
```python
async def classify_intent(state: AgentState) -> dict:
    """Determine if query is medical research question."""
    prompt = """Classify if this query is related to medical research.
    Query: {query}
    
    Respond with JSON:
    {
        "is_medical": true/false,
        "confidence": 0.0-1.0,
        "reason": "brief explanation"
    }"""
    
    response = llm_client.chat([
        {"role": "user", "content": prompt.format(query=state["query"])}
    ], model=state["model_id"])
    
    # Parse response, update state
    return {
        "classified_intent": "medical" if is_medical else "non_medical",
        "intent_confidence": confidence,
        "rejection_reason": reason if not is_medical else None
    }
```

**Conditional Gate:** If `classified_intent == "non_medical"`, route to `format_response` with rejection message. Else continue.

---

#### Node 2: `discover_skills`

**Input:** `query`, `skill_scores` (will be computed here)  
**Output:** `selected_skills`, `skill_scores`  
**Logic:**
```python
async def discover_skills(state: AgentState) -> dict:
    """Select relevant MCP tools based on query."""
    # Load skill manifests from skills/*.yaml
    manifests = skill_router.load_manifests()
    
    # Embed query
    query_embedding = llm_client.embed(
        state["query"],
        model=state["embedding_model"]
    )
    
    # Score each skill
    skill_scores = {}
    for tool_name, manifest in manifests.items():
        # 1. Semantic similarity
        skill_embedding = llm_client.embed(
            manifest["description"],
            model=state["embedding_model"]
        )
        semantic_score = cosine_similarity(query_embedding, skill_embedding)
        
        # 2. Keyword matching
        keyword_score = skill_router.score_keywords(
            query=state["query"],
            triggers=manifest["triggers"]
        )
        
        # 3. Combine scores (weighted average)
        skill_scores[tool_name] = 0.7 * semantic_score + 0.3 * keyword_score
    
    # Select top-K
    top_k = 3  # configurable
    selected = sorted(
        skill_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    
    return {
        "selected_skills": [name for name, _ in selected],
        "skill_scores": dict(selected)
    }
```

---

#### Node 3: `parallel_retrieve`

**Input:** `selected_skills`, `query`, `tool_outputs` (dict to populate)  
**Output:** `tool_outputs`, `retrieval_count`  
**Logic:**
```python
async def parallel_retrieve(state: AgentState) -> dict:
    """Call selected MCP tools in parallel."""
    # Prepare inputs for each tool
    tool_inputs = {}
    for tool_name in state["selected_skills"]:
        tool_inputs[tool_name] = {
            "query": state["query"],
            "top_k": 5,  # configurable
            # Additional tool-specific params
        }
    
    # LangGraph parallel edges execute concurrently
    # (handled by graph architecture, not this node)
    # Results collected in tool_outputs
    
    tool_outputs = {}
    retrieval_count = 0
    for tool_name in state["selected_skills"]:
        result = mcp_registry.call(tool_name, tool_inputs[tool_name])
        tool_outputs[tool_name] = result
        if result["status"] == "success":
            retrieval_count += len(result["output"])
    
    return {
        "tool_outputs": tool_outputs,
        "retrieval_count": retrieval_count
    }
```

---

#### Node 4: `synthesise`

**Input:** `query`, `tool_outputs`, `model_id`  
**Output:** `synthesis`, `raw_citations`, `total_tokens_out`  
**Logic:**
```python
async def synthesise(state: AgentState) -> dict:
    """Synthesize answer from retrieved documents."""
    # Collect all documents from tool outputs
    documents = []
    for tool_name, result in state["tool_outputs"].items():
        if result["status"] == "success":
            documents.extend(result["output"])
    
    if not documents:
        return {
            "synthesis": "No relevant documents found.",
            "raw_citations": [],
            "total_tokens_out": 0
        }
    
    # Build context with citations
    context = "\n\n".join([
        f"[{i+1}] {doc.get('title', 'Unknown')}: {doc.get('text', doc.get('abstract', ''))}"
        for i, doc in enumerate(documents[:10])  # top 10
    ])
    
    prompt = """Synthesize an evidence-based medical research answer.
    
Query: {query}

Relevant Research:
{context}

Requirements:
1. Ground every claim in the provided research
2. Use [1], [2] etc. for citations
3. Maintain scientific objectivity
4. Structure as: Executive Summary, Key Findings, Supporting Evidence, Limitations"""
    
    response = llm_client.chat(
        [{"role": "user", "content": prompt.format(query=state["query"], context=context)}],
        model=state["model_id"]
    )
    
    return {
        "synthesis": response,
        "raw_citations": documents,
        "total_tokens_out": count_tokens(response)
    }
```

---

#### Node 5: `score_confidence`

**Input:** `tool_outputs`, `synthesis`  
**Output:** `confidence_score`  
**Logic:**
```python
async def score_confidence(state: AgentState) -> dict:
    """Compute confidence score from retrieval coverage."""
    # Factors:
    # 1. Number of tools that succeeded
    # 2. Number of documents retrieved
    # 3. Average relevance scores
    
    successful_tools = sum(
        1 for r in state["tool_outputs"].values() if r["status"] == "success"
    )
    total_tools = len(state["selected_skills"])
    tool_coverage = successful_tools / max(total_tools, 1)
    
    avg_relevance = 0.0
    total_docs = 0
    for result in state["tool_outputs"].values():
        if result["status"] == "success":
            scores = [doc.get("score", 1.0) for doc in result["output"]]
            if scores:
                avg_relevance += sum(scores) / len(scores)
                total_docs += len(scores)
    
    if total_docs > 0:
        avg_relevance /= len([r for r in state["tool_outputs"].values() if r["status"] == "success"])
    
    # Combine factors
    confidence = 0.5 * tool_coverage + 0.5 * avg_relevance
    
    return {"confidence_score": float(confidence)}
```

---

#### Node 6: `evaluate_coherence`

**Input:** `synthesis`, `query`, `model_id`  
**Output:** Conditional gate: triggers fallback or continues  
**Logic:**
```python
async def evaluate_coherence(state: AgentState) -> dict:
    """Check if synthesis is coherent and faithful."""
    prompt = """Evaluate the coherence and medical accuracy of this response.
    
Query: {query}

Response: {synthesis}

Score (0-1):
- Does it directly answer the query?
- Are claims grounded in evidence?
- Is the tone appropriate for clinicians?
- Are there contradictions or hallucinations?

Return JSON: {{"coherence_score": 0.0-1.0, "issues": ["..."]}}"""
    
    response = llm_client.chat(
        [{"role": "user", "content": prompt.format(query=state["query"], synthesis=state["synthesis"])}],
        model=state["model_id"]
    )
    
    coherence_score = parse_json(response)["coherence_score"]
    
    return {"coherence_score": coherence_score}
```

**Conditional Gate:**
```python
if coherence_score < fallback_threshold:  # default 0.5
    route to "fallback_regen"
else:
    route to "format_response"
```

---

#### Node 7: `fallback_regen`

**Input:** `query`, `coherence_score`, `tool_outputs`, `model_id`  
**Output:** `synthesis`, `fallback_triggered`, `fallback_reason`  
**Logic:**
```python
async def fallback_regen(state: AgentState) -> dict:
    """Regenerate answer with relaxed constraints if primary fails."""
    # Strategy: Re-synthesize with different prompt + direct LLM if needed
    
    # Option 1: Broader prompt (remove grounding requirement)
    prompt = """Provide a research-informed answer to this medical question.
    
Query: {query}

You may use general medical knowledge in addition to specific research."""
    
    response = llm_client.chat(
        [{"role": "user", "content": prompt.format(query=state["query"])}],
        model=state["model_id"]
    )
    
    return {
        "synthesis": response,
        "fallback_triggered": True,
        "fallback_reason": f"Primary synthesis coherence {state['coherence_score']:.2f} < threshold"
    }
```

---

#### Node 8: `format_response`

**Input:** `synthesis`, `raw_citations`, `confidence_score`, `fallback_triggered`, `error`, all state  
**Output:** `answer`, `citations`, (prepare HTTP response)  
**Logic:**
```python
async def format_response(state: AgentState) -> dict:
    """Format final response with AMA citations and disclaimers."""
    # Handle error case
    if state.get("error"):
        answer = f"Error: {state['error']}"
        citations = []
    else:
        # Format citations
        citations = citation_formatter.format_citations_to_ama(state["raw_citations"])
        
        # Build answer with disclaimer
        answer = state["synthesis"]
        if state["fallback_triggered"]:
            answer = f"[FALLBACK] {answer}"
        
        answer += f"\n\n**Confidence:** {state['confidence_score']:.1%}"
        answer += f"\n\n**Clinical Disclaimer:** This response is for research purposes only..."
    
    return {
        "answer": answer,
        "citations": citations,
        "execution_time_ms": int((now - start_time).total_seconds() * 1000)
    }
```

---

### REQ-GRAPH-003: Conditional Edge Routing

**Requirement:** Explicit conditional edges for fallback and rejection.

**Specification:**

**Edge 1: After `classify_intent`**
```python
def route_classification(state: AgentState) -> str:
    if state["classified_intent"] == "non_medical":
        return "format_response"  # Reject non-medical query
    return "discover_skills"  # Continue to skill discovery
```

**Edge 2: After `evaluate_coherence`**
```python
def route_coherence(state: AgentState) -> str:
    if state.get("coherence_score", 1.0) < 0.5:  # configurable threshold
        return "fallback_regen"
    return "format_response"
```

**Edge 3: After `fallback_regen`**
```python
def route_fallback_outcome(state: AgentState) -> str:
    return "format_response"  # Always continue after fallback
```

### REQ-GRAPH-004: Checkpointing and Replay

**Requirement:** Every run is checkpointed and replayable.

**Specification:**

```python
graph = graph_builder.compile(
    checkpointer=SqliteSaver("graph_checkpoints.db"),
    # Saves state after every node execution
)

# Usage:
config = {"configurable": {"thread_id": trace_id}}
result = await graph.ainvoke(initial_state, config=config)

# Replay:
for event in graph.aiter_history(thread_id=trace_id):
    print(event)  # Every node execution, state change
```

**LangSmith Integration:**
- Checkpointer also publishes to LangSmith for remote access
- Trace ID = thread_id (can replay run from any machine)

---

## REQ-SKILL-*: Skill Discovery System

### REQ-SKILL-001: Manifest Schema

**Requirement:** Tools are discovered via YAML skill manifests.

**Specification:**

File: `skills/search_pubmed.yaml`

```yaml
# Metadata
name: search_pubmed
version: "1.0"
description: "Searches PubMed for peer-reviewed biomedical literature."

# Tool binding
mcp_tool: search_pubmed  # Must match registered tool class name

# Skill triggering
triggers:
  - drug efficacy
  - clinical evidence
  - treatment outcomes
  - meta-analysis
  - peer-reviewed research
  - pharmacology
  - efficacy studies

# Categorization
domains:
  - pharmacology
  - oncology
  - cardiology
  - general_medicine
  - infectious_disease

# Cost/latency estimation
cost_estimate: medium  # low | medium | high
latency_p95_ms: 1200
success_rate_percent: 95

# Access control
requires_api_key: false
requires_context: false  # e.g., patient data, user auth

# Availability
available_since: "2025-01-01"
deprecated: false

# Example
example_query: "What are the latest findings on SGLT2 inhibitors in heart failure?"
example_output:
  - pmid: "12345678"
    title: "SGLT2 inhibitors reduce hospitalizations..."
    score: 0.95
```

**Router Usage:**

```python
router = SkillRouter(manifests_dir="skills/")
selected_tools = router.select_skills(
    query="What are the effects of statins on LDL?",
    top_k=3
)
# Returns: ["search_pubmed", "search_clinical_trials"]
```

**Scoring Algorithm:**
1. Embed query: `q_emb = embed(query)`
2. For each tool, embed description: `s_emb = embed(manifest.description)`
3. Semantic score: `semantic = cosine(q_emb, s_emb)`
4. Keyword score: `keyword = max(word_overlap(query.lower(), triggers))`
5. Combined: `score = 0.7 * semantic + 0.3 * keyword`
6. Return top-K by score

---

## REQ-EXT-*: Extensibility Contract

### REQ-EXT-001: Adding a New Data Source Checklist

**Requirement:** Adding a new retrieval source (e.g., MAUDE) must require exactly 2 files + configuration.

**Specification:**

**Checklist to add MAUDE database:**

1. ✅ **Create skill manifest** (`skills/search_maude.yaml`)
   - Copy `skills/search_pubmed.yaml` as template
   - Update: `name`, `description`, `triggers`, `domains`, `mcp_tool`, `latency_p95_ms`
   - No code changes required

2. ✅ **Implement MCP tool** (`mcp_tools/search_maude.py`)
   ```python
   class SearchMAUDETool(MCPToolBase):
       name = "search_maude"
       description = "Searches MAUDE database for medical device adverse events."
       input_schema = {...}  # JSON Schema
       output_schema = {...}  # JSON Schema
       
       def call(self, input_dict):
           # Fetch from MAUDE API
           # Validate results against output_schema
           # Return {"status": "success", "output": [...]}
   ```

3. ❌ **No changes to:**
   - `graph.py` (orchestrator)
   - `research_agent_api.py` (API)
   - `orchestrator.py` (legacy, frozen)
   - `models.yaml` (no config needed)

**Validation:**

After adding tool:
```bash
python test_mcp_tools.py --tool search_maude
# Should pass: input validation, output schema, error handling

# Manual test:
curl -X POST http://localhost:8000/query \
  -d '{"question": "What are adverse events for pacemakers?", "agents_to_use": ["search_maude"]}'
# Should return results from MAUDE if matching query
```

**Estimated Time:** ≤ 1 day (implement API call, validate schema, write unit test)

---

## REQ-NFR-*: Non-Functional Requirements

### REQ-NFR-001: Latency SLA

**Requirement:** P95 end-to-end response time for typical query.

**Specification:**

| Scenario | Target P95 | Notes |
|----------|-----------|-------|
| 1 tool (local only) | < 2s | FAISS search only, no LLM |
| 2 tools (PubMed + FDA) | < 5s | Parallel retrieval + synthesis |
| 3 tools (PubMed + FDA + Trials) | < 8s | 3-way parallel + synthesis |
| With fallback | < 12s | Re-synthesis if coherence fails |

**Breakdown (typical 3-tool case):**
- Classify intent: 200ms (LLM call)
- Discover skills: 100ms (embedding + routing)
- Parallel retrieve: 2s (3 tools × 700ms avg)
- Synthesize: 2.5s (LLM synthesis)
- Score + format: 300ms
- **Total: ~5.1s** (within SLA)

**Monitoring:**
- Prometheus histogram: `mra_latency_seconds{node="synthesise", model_id="gpt-4o"}`
- Alerts if P95 > 8s for 2-tool queries

### REQ-NFR-002: Availability

**Requirement:** Graceful degradation if some tools fail.

**Specification:**

| Scenario | Behavior |
|----------|----------|
| 1 of 3 tools down | Continue with 2 tools, confidence may be lower |
| 2 of 3 tools down | Continue with 1 tool or fallback |
| All tools down | Trigger fallback (direct LLM) |
| LLM API down (all models) | Return 503 error with "LLM service unavailable" |

**No failure threshold:** There is no minimum number of tools required. System always attempts to answer with available data.

### REQ-NFR-003: Reproducibility

**Requirement:** Every experiment run is fully reproducible from configuration.

**Specification:**

**Run Configuration (saved with results):**
```json
{
  "trace_id": "uuid",
  "timestamp": "2025-04-12T10:30:00Z",
  "query": "What are the effects of metformin...",
  "config": {
    "model_id": "gpt-4o",
    "embedding_model": "text-embedding-3-large",
    "selected_skills": ["search_pubmed", "search_fda"],
    "fallback_threshold": 0.5,
    "top_k": 5
  },
  "results": {...},
  "metrics": {...},
  "langsmith_url": "https://smith.langchain.com/..."
}
```

**Replay:**
```bash
# From trace ID:
python eval/replay_run.py --trace_id <uuid>
# Fetches config from LangSmith, re-runs graph
# Compares results to original (should be identical)
```

### REQ-NFR-004: Safety and Disclaimers

**Requirement:** Every response includes clinical disclaimer.

**Specification:**

```python
def format_response(state):
    disclaimer = """
⚠️ **CLINICAL DISCLAIMER:** 
This response is for research and educational purposes only. 
It is NOT a substitute for professional medical advice. 
Always consult with a licensed healthcare provider before making medical decisions. 
The information may be outdated or incomplete. 
No liability is assumed for accuracy or appropriateness of content.
    """
    answer = state["synthesis"] + "\n\n" + disclaimer
    return {"answer": answer}
```

**Additionally:**
- Log every query and response (for audit trail)
- Include confidence score (0-1) so user can assess reliability
- Tag fallback responses with `[FALLBACK]` prefix
- Never provide personal medical advice (reject queries like "Do I have cancer?")

---

## Summary Table: Requirements Cross-Reference

| Req ID | Category | Title | Related Files | Implementation Phase |
|--------|----------|-------|----------------|----------------------|
| REQ-LLM-001 | LLM | Unified LLMClient | `llm_client.py` | Phase 1 |
| REQ-LLM-002 | LLM | LiteLLM Router | `models.yaml`, `llm_client.py` | Phase 1 |
| REQ-LLM-003 | LLM | Multi-backend support | `llm_client.py` | Phase 1 |
| REQ-LLM-004 | LLM | Parallel evaluation | `eval/run_eval.py` | Phase 6 |
| REQ-LLM-005 | LLM | Cost tracking | `observability/metrics.py` | Phase 5 |
| REQ-MCP-001 | Tool | Tool definition | `mcp_tools/*.py` | Phase 2 |
| REQ-MCP-002 | Tool | In-process execution | `mcp_tool_registry.py` | Phase 2 |
| REQ-MCP-003 | Tool | Auto-discovery | `mcp_tool_registry.py` | Phase 2 |
| REQ-MCP-004 | Tool | Error handling | All MCP tools | Phase 2 |
| REQ-GRAPH-001 | Graph | State schema | `graph/state.py` | Phase 3 |
| REQ-GRAPH-002 | Graph | Node definitions | `graph/nodes/` | Phase 3 |
| REQ-GRAPH-003 | Graph | Conditional routing | `graph/graph.py` | Phase 3 |
| REQ-GRAPH-004 | Graph | Checkpointing | `graph/graph.py` | Phase 3 |
| REQ-SKILL-001 | Skill | Manifest schema | `skills/*.yaml` | Phase 3 |
| REQ-EXT-001 | Extend | New tool checklist | All phases | Ongoing |
| REQ-NFR-001 | NFR | Latency SLA | `observability/metrics.py` | Phase 5 |
| REQ-NFR-002 | NFR | Availability | All phases | Ongoing |
| REQ-NFR-003 | NFR | Reproducibility | `eval/replay_run.py` | Phase 6 |
| REQ-NFR-004 | NFR | Safety disclaimers | `graph/nodes/format_response.py` | Phase 3 |

---

**Document Status:** Complete. Ready for Phase 3 implementation planning.
