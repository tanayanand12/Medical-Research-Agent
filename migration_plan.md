# Phase 3: Migration Plan

**Document Version:** 1.0  
**Target System:** `./medical-research-agent/`  
**Date:** 2025-04-12  
**Status:** Engineering roadmap for Phases 1–7 implementation

---

## Executive Summary

This document is the **engineering roadmap** — a phased, sequential plan with explicit code reuse decisions traceable to the Phase 0 inventory. The migration follows a strictly sequential path:

1. **Phase 1** (Week 1) — LLM abstraction + MCP scaffolding
2. **Phase 2** (Weeks 2–3) — MCP tool conversion (5 agents → 5 tools)
3. **Phase 3** (Week 4) — Skill discovery system (YAML + semantic matching)
4. **Phase 4** (Weeks 5–6) — LangGraph orchestration (ThreadPool → StateGraph)
5. **Phase 5** (Week 7) — LangSmith observability integration
6. **Phase 6** (Weeks 8–9) — Evaluation harness (RAGAS + benchmarks)
7. **Phase 7** (Week 10) — Publication preparation

**Total Timeline:** 10 weeks of engineering + 2 weeks review/QA = **12 weeks to production**

**Key Principle:** No phase builds on incomplete prior phases. Each phase has a single "validation gate" that must pass before the next phase begins. Early exit gates prevent cascading failures.

---

## Migration Principles

### 1. Strict Phase Sequencing
- Each phase depends on the **complete output** of prior phases
- Parallelization is forbidden; only sequential execution is safe
- Each phase ends with a validation gate; if the gate fails, do NOT proceed

### 2. Maximal Code Reuse from Phase 0 Inventory
- **REUSE modules** (5 files) are copied without modification
- **REFACTOR modules** (11 files) are wrapped, not rewritten
- **DELETE modules** (10 files) are removed without inspection
- No module is split across phases; decisions are made upfront

### 3. Input/Output Clarity
- Each phase has explicit input artefacts (must exist, validated at start)
- Each phase produces exactly one primary output artefact
- Secondary outputs (logs, intermediate files) do not block subsequent phases

### 4. One Deployable Artifact Per Phase
- Phase 1 output: A FastAPI app that still routes through old Orchestrator, but LLM calls go through LiteLLM
- Phase 2 output: Same FastAPI, but agents are MCP tools (no behavior change, internal structure only)
- Phase 3 output: YAML skill registry; Orchestrator can discover tools dynamically
- Phase 4 output: LangGraph graph; FastAPI now routes through StateGraph instead of Orchestrator
- Phase 5 output: LangSmith + Prometheus integration; tracing is on by default
- Phase 6 output: Evaluation harness runs; benchmarks can compare models/agents
- Phase 7 output: Paper + code repository with evaluation results

---

## Phase Timeline and Gates

### Phase 1: LLM Abstraction + MCP Scaffolding

| Field | Value |
|-------|-------|
| **Duration** | 1 week (Mon–Fri) |
| **Input Artefacts** | `./agentic-pipeline-clinical/` (source), `inventory.md` (reuse decisions), `migration_architectural_requirements.md` (REQ-LLM-001 to REQ-LLM-005) |
| **Primary Output** | `./medical-research-agent/llm_client.py` (LLMClient singleton) + `./models.yaml` (LiteLLM routing config) |
| **Secondary Outputs** | `./medical-research-agent/requirements.txt` (litellm, langchain-core, pydantic), updated `research_agent_api.py` with LLMClient injection |
| **Code Reuse** | **REUSE:** agent_base.py, unicode_safe_logging.py, citation_formatter.py (copied unchanged); **REFACTOR:** aggregator.py, fallback.py, query_classifier.py (rewrite LLM calls to use LLMClient); **DELETE:** client_persona_model.py, guardrails.py, all *_old*.py files, test stubs |
| **Validation Gate** | ✅ `llm_client.chat()` returns valid text for all 3 models in models.yaml; ✅ `llm_client.embed()` returns correct vector dimensions; ✅ `llm_client.get_cost()` computes cost accurately; ✅ Cost tracking persists to Prometheus `/metrics` endpoint; ✅ All REFACTOR modules pass unit tests (refactored with mocked LLMClient); ✅ Backwards-compatibility test: old FastAPI `/query` endpoint still works (orchestrator.py unchanged) |
| **Rollback** | Delete `llm_client.py`, `models.yaml`, `requirements.txt` additions; revert `research_agent_api.py` to import `orchestrator.Orchestrator` directly |

**Detailed Phase 1 Tasks:**

1. **Create `llm_client.py`** (150 LOC)
   - Singleton pattern with eager initialization from `models.yaml`
   - `chat(messages, model=None, **kwargs)` → route via LiteLLM, return text
   - `embed(text, model=None)` → route via LiteLLM embedding, return List[float]
   - `get_cost(model, tokens_in, tokens_out)` → lookup in models.yaml, compute cost
   - Handle 4 provider types: OpenAI, Anthropic, Ollama, Azure
   - Fallback to `DEFAULT_LLM_MODEL` env var if model not specified
   - Metrics: `llm_calls_total` (counter), `llm_tokens_total` (histogram), `llm_cost_total` (gauge)

2. **Create `models.yaml`** (30 LOC)
   - Schema: `models: [ { model_name, litellm_params, model_info } ]`
   - Pre-populate with gpt-4o, claude-3-5-sonnet, ollama/mistral
   - Include cost rates and context window sizes

3. **Refactor `aggregator.py`**
   - Remove line 27-35: Replace hardcoded `openai.ChatCompletion.create()` with `LLMClient.chat()`
   - Inject `LLMClient` singleton in `__init__`
   - No API changes; same input/output signature

4. **Refactor `fallback.py`**
   - Remove line 99: Replace hardcoded `openai.chat.completions.create()` with `LLMClient.chat()`
   - Inject `LLMClient` singleton
   - No API changes

5. **Refactor `query_classifier.py`**
   - Remove gpt-3.5-turbo hardcoding; use LLMClient with config-driven model selection
   - Inject `LLMClient` singleton

6. **Refactor `research_agent_api.py`**
   - Lines 95-98: Remove hardcoded `openai.api_key` setup
   - Inject `LLMClient` singleton instead
   - Pass LLMClient to `Orchestrator` and `Aggregator` constructors

7. **Copy (unchanged) to target:**
   - `agent_base.py` → `./medical-research-agent/agent_base.py`
   - `unicode_safe_logging.py` → `./medical-research-agent/unicode_safe_logging.py`
   - `citation_formatter.py` → `./medical-research-agent/citation_formatter.py`

8. **Update `requirements.txt`**
   - Add: `litellm>=1.0.0`, `pydantic>=2.0`
   - Verify no version conflicts with existing dependencies

9. **Create unit tests** (150 LOC)
   - Mock LLMClient for aggregator, fallback, query_classifier
   - Verify chat/embed/get_cost calls route correctly
   - No changes to orchestrator.py logic; test it still works end-to-end

10. **Create `medresearch_phase1_validation.py`** script
    - Test that `/query` endpoint returns valid JSON with `answer`, `sources`, `confidence` fields
    - Test that 3 model types (OpenAI, Anthropic, Ollama) all produce valid responses
    - Test that cost metrics are populated in Prometheus

**Failure Modes and Handling:**

- ❌ LiteLLM initialization fails (bad API key, wrong base_url): Log error, halt with clear message. Do NOT fall back to hardcoded OpenAI.
- ❌ Embedding model returns wrong dimension: Catch in unit tests; fix models.yaml before proceeding.
- ❌ Old Orchestrator still works but bypasses LLMClient: Catch in backwards-compat test; audit all imports.

---

### Phase 2: MCP Tool Conversion

| Field | Value |
|-------|-------|
| **Duration** | 2 weeks (Mon–Fri × 2) |
| **Input Artefacts** | Phase 1 outputs (llm_client.py, models.yaml, refactored modules), `migration_architectural_requirements.md` (REQ-MCP-001 to REQ-MCP-004), inventory.md (REFACTOR agents) |
| **Primary Output** | `./medical-research-agent/mcp_tools/` directory with 5 tool modules (local.py, pubmed.py, pubmed_deep_research.py, clinical_trials.py, fda.py) |
| **Secondary Outputs** | `./medical-research-agent/mcp_registry.py` (auto-discovery), `./medical-research-agent/tool_schemas.json` (OpenAPI-like specs for each tool) |
| **Code Reuse** | **REUSE:** local_agent_wrapper.py, pubmed_deep_research_agent_wrapper.py (HTTP wrappers unchanged); **REFACTOR:** pubmed_local_agent_wrapper.py, clinical_trials_agent_wrapper.py, fda_agent_wrapper.py (extract embedding calls to use LLMClient), pubmed_local_agent/, local_agent/, FDA_agent/, clinical_trials_agent1/ subdirectories (all embedding calls replaced with LLMClient); **DELETE:** ai_response_interruption/ (not in active pipeline) |
| **Validation Gate** | ✅ Each tool (5 total) is callable as `tool_fn(query: str, context: Dict) → Dict[str, Any]` with consistent schema; ✅ Each tool logs trace IDs for debugging; ✅ mcp_registry auto-discovers all 5 tools without hardcoding; ✅ Each tool returns consistent output: `{ "results": [...], "tokens_used": int, "cost": float, "retrieval_time_sec": float }`; ✅ Orchestrator still works end-to-end with tools (internal call sites update to use new tool interface, but no user-facing API changes); ✅ All embedding calls verified to use LLMClient (no hardcoded text-embedding-3-large) |
| **Rollback** | Delete `mcp_tools/`, `mcp_registry.py`, `tool_schemas.json`; revert Orchestrator to call agent wrappers directly (requires reverting Phase 2 edits to orchestrator.py) |

**Detailed Phase 2 Tasks:**

1. **Create `mcp_tools/` directory** and base structure
   - `__init__.py` — empty
   - `base_tool.py` — BaseMCPTool abstract class (30 LOC)
     ```python
     class BaseMCPTool(ABC):
         @abstractmethod
         def invoke(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
             """Returns { "results": [...], "tokens_used": int, "cost": float, "retrieval_time_sec": float }"""
     ```

2. **Create `mcp_tools/local.py`** (150 LOC)
   - Wrap `LocalAgent` as MCP tool
   - Replace direct embedding calls: call `LLMClient.embed()` instead of hardcoded OpenAI
   - Input: `query: str`, `context["db_name"]`, `context["top_k"]`
   - Output: structured results dict with retrieval time
   - Reuse: Copy `local_agent/` subdirectory modules, refactor embedding calls

3. **Create `mcp_tools/pubmed.py`** (150 LOC)
   - Wrap `PubMedAgent` as MCP tool
   - Replace embedding calls: use `LLMClient.embed()`
   - Input: `query: str`, `context["top_k"]`
   - Output: structured results dict
   - Reuse: Copy `pubmed_local_agent/` subdirectory, refactor embedding calls

4. **Create `mcp_tools/pubmed_deep_research.py`** (50 LOC)
   - Wrap `PubMedDeepResearchAgent` HTTP call as MCP tool (no embedding calls — external service)
   - Input: `query: str`, `context["max_papers"]`, `context["include_fulltext"]`
   - Output: structured results dict with retrieval time (inherited from external service)
   - Reuse: `pubmed_deep_research_agent_wrapper.py` unchanged, just call it and wrap output

5. **Create `mcp_tools/clinical_trials.py`** (150 LOC)
   - Wrap `ClinicalTrialsAgent` as MCP tool
   - Replace embedding calls: use `LLMClient.embed()`
   - Input: `query: str`, `context["max_trials"]`, `context["clinical_trials_top_k"]`
   - Output: structured results dict
   - Reuse: Copy `clinical_trials_agent1/` subdirectory, refactor embedding calls

6. **Create `mcp_tools/fda.py`** (150 LOC)
   - Wrap `FDAAgent` as MCP tool
   - Replace embedding calls: use `LLMClient.embed()`
   - Input: `query: str`, `context["fda_top_k"]`
   - Output: structured results dict
   - Reuse: Copy `FDA_agent/` subdirectory, refactor embedding calls

7. **Create `mcp_registry.py`** (100 LOC)
   - Auto-discover all Python modules in `mcp_tools/`
   - Load each as `BaseMCPTool` subclass
   - Provide `get_tool(name: str)` and `list_tools() → List[str]`
   - Cache discovered tools on first load
   - Log discovery results (found N tools)

8. **Create `tool_schemas.json`** (150 LOC)
   - JSON schema for each tool's input/output
   - Example:
     ```json
     {
       "local": {
         "description": "Search local PubMed index",
         "input": { "query": "string", "top_k": "int" },
         "output": { "results": "List[Dict]", "tokens_used": "int", "cost": "float" }
       }
     }
     ```

9. **Update `orchestrator.py`** (refactoring, not rewrite)
   - Replace agent instantiation lines (30–52) with mcp_registry lookup
   - Keep ThreadPool execution unchanged (will be replaced in Phase 4)
   - Each agent call now invokes `mcp_registry.get_tool(name).invoke(query, context)`
   - No user-facing API changes

10. **Create integration tests** (150 LOC)
    - Test each tool returns valid structured output
    - Test embedding calls use LLMClient (assert no hardcoded model names)
    - Test mcp_registry discovers all 5 tools
    - Test orchestrator still routes queries correctly

11. **Create `medresearch_phase2_validation.py`** script
    - Call each MCP tool directly and verify output schema
    - Verify `/query` endpoint still works end-to-end
    - Verify no LLM calls bypass LLMClient

**Failure Modes and Handling:**

- ❌ A REFACTOR subdirectory has embedding calls in multiple files: Grep for hardcoded model names; audit all files in the subdirectory before creating the tool.
- ❌ MCP tool output schema changes between tools: Document schema inconsistencies; normalize in Phase 4 (Aggregator expects consistent output).
- ❌ Orchestrator doesn't find a tool at runtime: Check mcp_registry; verify module import is not failing silently.

---

### Phase 3: Skill Discovery System

| Field | Value |
|-------|-------|
| **Duration** | 1 week |
| **Input Artefacts** | Phase 2 outputs (mcp_tools/, mcp_registry.py), `migration_architectural_requirements.md` (REQ-SKILL-001) |
| **Primary Output** | `./medical-research-agent/skills/` directory with 5 YAML manifests (local.yaml, pubmed.yaml, pubmed_deep_research.yaml, clinical_trials.yaml, fda.yaml) + `skill_discovery.py` module |
| **Secondary Outputs** | `skill_router.py` (semantic + keyword-based skill selection) |
| **Code Reuse** | No code reuse in this phase; pure new functionality |
| **Validation Gate** | ✅ Each skill YAML is valid and contains required fields (name, description, triggers, domains, cost_estimates); ✅ skill_discovery.query(input_query) returns list of tool names ranked by relevance score; ✅ Semantic similarity scoring uses embeddings consistently (LLMClient.embed()); ✅ Skill router correctly selects appropriate tools for test queries; ✅ Orchestrator can be updated (non-breaking change) to use skill_router instead of static agent selection |
| **Rollback** | Delete `skills/`, `skill_discovery.py`, `skill_router.py`; revert Orchestrator to static agent selection from Phase 2 |

**Detailed Phase 3 Tasks:**

1. **Create `skills/` directory** and base YAML schema
   - Define standard skill manifest structure (name, description, version, triggers, domains, cost_estimates, tool_name)
   - Document in `skills/README.md`

2. **Create skill manifests** (5 files, ~50 LOC each)
   - `local.yaml` — Local index search
     ```yaml
     name: Local Index Search
     version: 1.0.0
     tool_name: local
     description: Search local PubMed index for high-relevance papers
     domains: [general, epidemiology, genetics]
     triggers:
       keywords: [search, find, locate, retrieve]
       semantic_similarity_threshold: 0.6
     cost_estimates:
       avg_tokens_input: 100
       avg_tokens_output: 500
       retrieval_time_sec: 2.0
     ```
   
   - `pubmed.yaml` — PubMed vector search
   - `pubmed_deep_research.yaml` — Deep research (full-text)
   - `clinical_trials.yaml` — Clinical trials search
   - `fda.yaml` — FDA database search

3. **Create `skill_discovery.py`** (200 LOC)
   - `SkillDiscovery` class with methods:
     - `load_manifests(skills_dir: str) → Dict[str, SkillManifest]`
     - `query(user_query: str, top_k: int = 3) → List[Tuple[str, float]]` (returns tool names + relevance scores)
   - Scoring algorithm:
     - 40% semantic similarity (embed query + skill description, cosine distance)
     - 40% keyword matching (TF-IDF on triggers)
     - 20% domain match (if query domain matches skill domain)
   - Caching: Load manifests on first call; reload on file change
   - Logging: Log every query and final score for debugging

4. **Create `skill_router.py`** (150 LOC)
   - `SkillRouter` class
   - Use `SkillDiscovery` to rank tools for a query
   - Return list of enabled tool names in priority order
   - Config: Min relevance threshold (0.4), max tools per query (3)
   - Fallback: If no tools meet threshold, return all tools (graceful degradation)

5. **Create unit tests** (150 LOC)
   - Test skill manifest loading and validation
   - Test semantic similarity scoring (embed a query, verify top tool is correct)
   - Test keyword matching (query with trigger keywords, verify score increases)
   - Test domain matching
   - Test skill_router fallback when no tools meet threshold

6. **Update Orchestrator** (non-breaking change, can be deferred to Phase 4)
   - Optional: Replace hardcoded agent list with skill_router call
   - For now, Phase 3 output is standalone; Phase 4 will integrate it

7. **Create `medresearch_phase3_validation.py`** script
   - Test skill_discovery.query() on 10+ medical queries
   - Verify returned tools are semantically relevant
   - Verify Orchestrator still works if skill_router is not yet integrated

**Failure Modes and Handling:**

- ❌ Embedding model in LLMClient changes; skill scores become meaningless: Invalidate cache on LLMClient.embed() provider change.
- ❌ Skill manifests are incomplete or malformed: Validate all YAML on load; log and fail fast with clear error message.
- ❌ Semantic similarity scores are all ≈ 0.5 (uniform): Check that embedding model is producing meaningful vectors; verify LLMClient is not falling back to a weak model.

---

### Phase 4: LangGraph Orchestration

| Field | Value |
|-------|-------|
| **Duration** | 2 weeks |
| **Input Artefacts** | Phase 3 outputs (skills/), Phase 2 outputs (mcp_tools/), Phase 1 outputs (llm_client.py), `migration_architectural_requirements.md` (REQ-GRAPH-001 to REQ-GRAPH-004) |
| **Primary Output** | `./medical-research-agent/graph.py` (LangGraph StateGraph with 8 nodes) + `agent_state.py` (AgentState TypedDict with 17 fields) |
| **Secondary Outputs** | `./medical-research-agent/edges.py` (conditional edge routing logic), `./medical-research-agent/research_agent_api_v2.py` (FastAPI endpoints using StateGraph) |
| **Code Reuse** | **REFACTOR:** aggregator.py, fallback.py (now integrated as graph nodes); query_classifier.py (now classify_intent node); citation_formatter.py (reused in format_response node) |
| **Validation Gate** | ✅ StateGraph compiles without errors; ✅ All 8 nodes execute in correct order for a test query; ✅ Conditional edges route correctly (e.g., after classify_intent, query goes to discover_skills if medical domain, or returns early if non-medical); ✅ Fallback node triggers when coherence score < threshold; ✅ Format_response node applies AMA citations; ✅ research_agent_api_v2.py `/query` endpoint returns same output structure as Phase 1 (backwards compatible); ✅ E2E test: query through StateGraph produces clinically coherent answer with citations |
| **Rollback** | Delete `graph.py`, `agent_state.py`, `edges.py`, `research_agent_api_v2.py`; keep old `research_agent_api.py` with Orchestrator routing |

**Detailed Phase 4 Tasks:**

1. **Create `agent_state.py`** (100 LOC)
   - Define `AgentState` TypedDict with 17 fields:
     ```python
     class AgentState(TypedDict):
         # Input
         input_query: str
         context: Dict[str, Any]
         trace_id: str
         
         # Classification
         is_medical_query: bool
         classification_confidence: float
         
         # Skill discovery
         discovered_skills: List[str]  # tool names
         skill_scores: Dict[str, float]
         
         # Retrieval
         retrieval_results: Dict[str, Dict]  # {tool_name: tool_output}
         tokens_used: Dict[str, int]  # {tool_name: token_count}
         retrieval_time_sec: Dict[str, float]
         
         # Synthesis
         intermediate_answer: str
         intermediate_sources: List[str]
         
         # Scoring & Fallback
         coherence_score: float
         should_fallback: bool
         fallback_count: int
         
         # Output
         output_answer: str
         output_sources: List[str]
         output_citations: List[str]  # AMA formatted
         confidence_score: float
         execution_time_sec: float
     ```

2. **Create `edges.py`** (150 LOC)
   - Conditional edge functions:
     - `after_classify(state: AgentState) → str`: Return "discover_skills" if `is_medical_query` else "format_response" (early exit)
     - `after_evaluate_coherence(state: AgentState) → str`: Return "fallback_regen" if `should_fallback` else "format_response"
   - Regular edges: All other nodes have single outbound edge (no branching)

3. **Create `graph.py`** (400 LOC)
   - Build StateGraph with 8 nodes:
     
     **Node 1: classify_intent** (50 LOC)
     ```python
     def classify_intent(state: AgentState) -> AgentState:
         """Medical domain filtering. Reuses query_classifier.py logic."""
         classifier = QueryClassifier(llm_client)
         is_medical, confidence = classifier.classify(state["input_query"])
         state["is_medical_query"] = is_medical
         state["classification_confidence"] = confidence
         logger.info(f"[{state['trace_id']}] Query classified: medical={is_medical}, conf={confidence}")
         return state
     ```
     
     **Node 2: discover_skills** (50 LOC)
     ```python
     def discover_skills(state: AgentState) -> AgentState:
         """Semantic + keyword-based tool selection. Uses skill_discovery.py."""
         router = SkillRouter()
         tools, scores = router.rank(state["input_query"], top_k=3)
         state["discovered_skills"] = tools
         state["skill_scores"] = dict(zip(tools, scores))
         logger.info(f"[{state['trace_id']}] Discovered skills: {tools}")
         return state
     ```
     
     **Node 3: parallel_retrieve** (100 LOC)
     ```python
     async def parallel_retrieve(state: AgentState) -> AgentState:
         """Concurrent MCP tool invocation."""
         tasks = []
         for tool_name in state["discovered_skills"]:
             tool = mcp_registry.get_tool(tool_name)
             task = tool.invoke_async(state["input_query"], state["context"])
             tasks.append((tool_name, task))
         
         results = {}
         tokens_used = {}
         retrieval_time_sec = {}
         for tool_name, task in tasks:
             try:
                 result = await asyncio.wait_for(task, timeout=5.0)
                 results[tool_name] = result
                 tokens_used[tool_name] = result.get("tokens_used", 0)
                 retrieval_time_sec[tool_name] = result.get("retrieval_time_sec", 0)
             except asyncio.TimeoutError:
                 logger.warning(f"[{state['trace_id']}] Tool {tool_name} timed out")
                 results[tool_name] = {"results": [], "error": "timeout"}
         
         state["retrieval_results"] = results
         state["tokens_used"] = tokens_used
         state["retrieval_time_sec"] = retrieval_time_sec
         return state
     ```
     
     **Node 4: synthesise** (100 LOC)
     ```python
     def synthesise(state: AgentState) -> AgentState:
         """LLM answer generation with context. Reuses aggregator.py logic."""
         aggregator = Aggregator(llm_client)
         
         # Build context from retrieved results
         context_text = "\n".join([
             f"From {tool_name}: {json.dumps(result['results'][:3])}"
             for tool_name, result in state["retrieval_results"].items()
             if result.get("results")
         ])
         
         system_prompt = f"""You are a medical research assistant. 
         Answer the following question based on the provided evidence.
         Provide a clinically coherent, evidence-based answer.
         Always cite sources.
         \n\nAvailable Evidence:\n{context_text}"""
         
         answer = llm_client.chat([
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": state["input_query"]}
         ])
         
         state["intermediate_answer"] = answer
         state["intermediate_sources"] = list(state["retrieval_results"].keys())
         logger.info(f"[{state['trace_id']}] Answer synthesized")
         return state
     ```
     
     **Node 5: score_confidence** (50 LOC)
     ```python
     def score_confidence(state: AgentState) -> AgentState:
         """Coverage-based confidence calculation."""
         coverage = len([s for s in state["intermediate_sources"] if state["retrieval_results"][s].get("results")])
         max_coverage = len(state["discovered_skills"])
         confidence = coverage / max_coverage if max_coverage > 0 else 0.0
         state["confidence_score"] = confidence
         logger.info(f"[{state['trace_id']}] Confidence: {confidence:.2f}")
         return state
     ```
     
     **Node 6: evaluate_coherence** (100 LOC)
     ```python
     def evaluate_coherence(state: AgentState) -> AgentState:
         """Coherence scoring for fallback decision. Reuses fallback.py logic."""
         evaluator = FallbackMechanism(llm_client)
         coherence_score = evaluator.evaluate_coherence(
             query=state["input_query"],
             answer=state["intermediate_answer"],
             sources=state["intermediate_sources"]
         )
         state["coherence_score"] = coherence_score
         state["should_fallback"] = coherence_score < 0.6 and state["fallback_count"] < 1
         logger.info(f"[{state['trace_id']}] Coherence: {coherence_score:.2f}, fallback={state['should_fallback']}")
         return state
     ```
     
     **Node 7: fallback_regen** (100 LOC)
     ```python
     def fallback_regen(state: AgentState) -> AgentState:
         """Fallback answer regeneration."""
         if not state["should_fallback"]:
             return state
         
         # Regenerate with lower temperature + different prompt
         regenerated_answer = llm_client.chat([
             {"role": "system", "content": "You are a medical expert. Provide a conservative, evidence-based answer. Prioritize clarity and safety."},
             {"role": "user", "content": state["input_query"]}
         ], temperature=0.3)
         
         state["intermediate_answer"] = regenerated_answer
         state["fallback_count"] += 1
         logger.info(f"[{state['trace_id']}] Fallback regeneration complete")
         return state
     ```
     
     **Node 8: format_response** (100 LOC)
     ```python
     def format_response(state: AgentState) -> AgentState:
         """AMA citation formatting + disclaimers."""
         formatter = CitationFormatter()
         
         # Extract citations from sources
         citations = []
         for source in state["intermediate_sources"]:
             results = state["retrieval_results"][source].get("results", [])
             for result in results[:5]:
                 citations.append(formatter.format_ama({
                     "title": result.get("title"),
                     "authors": result.get("authors"),
                     "year": result.get("year"),
                     "doi": result.get("doi")
                 }))
         
         # Add clinical disclaimer
         final_answer = f"""[DISCLAIMER: This response is AI-generated and not a substitute for professional medical advice.]
         
         {state["intermediate_answer"]}
         
         [FALLBACK]""" if state["fallback_count"] > 0 else state["intermediate_answer"]
         
         state["output_answer"] = final_answer
         state["output_sources"] = state["intermediate_sources"]
         state["output_citations"] = citations
         logger.info(f"[{state['trace_id']}] Response formatted with {len(citations)} citations")
         return state
     ```

4. **Integrate edges into graph**
   - Add conditional edge after classify_intent: if not medical, go to format_response (early exit)
   - Add conditional edge after evaluate_coherence: if should_fallback, go to fallback_regen, else go to format_response
   - All other edges: linear progression

5. **Create `research_agent_api_v2.py`** (150 LOC)
   - FastAPI endpoints identical to Phase 1, but route through StateGraph instead of Orchestrator
   - Endpoint: `POST /query` → invoke graph → return result
   - Add tracing: Generate trace_id, inject into AgentState, return in response
   - Example:
     ```python
     @app.post("/query")
     async def query_endpoint(request: QueryRequest):
         trace_id = str(uuid.uuid4())
         initial_state = AgentState(
             input_query=request.question,
             context={"top_k": request.top_k, ...},
             trace_id=trace_id,
             ...  # all fields initialized
         )
         final_state = await graph.ainvoke(initial_state)
         return {
             "answer": final_state["output_answer"],
             "sources": final_state["output_sources"],
             "citations": final_state["output_citations"],
             "confidence": final_state["confidence_score"],
             "trace_id": trace_id,
             "execution_time_sec": final_state["execution_time_sec"]
         }
     ```

6. **Create integration tests** (200 LOC)
   - Test each node individually (unit test with mocked state)
   - Test full graph execution for a medical query
   - Test conditional edges (non-medical query should exit early)
   - Test fallback path (artificially set coherence_score < 0.6)
   - Test format_response produces valid AMA citations

7. **Backwards-compatibility test**
   - Run Phase 1 test suite against research_agent_api_v2.py
   - Verify output structure matches (answer, sources, citations, confidence)
   - Verify `/health` endpoint still works

8. **Create `medresearch_phase4_validation.py`** script
   - Test graph.ainvoke() on 5 medical queries
   - Verify all 8 nodes execute for each query
   - Verify conditional edges route correctly
   - Verify final output has trace_id and execution_time_sec
   - Verify LangSmith can visualize the graph (even if tracing not yet enabled)

**Failure Modes and Handling:**

- ❌ Node execution order is wrong (e.g., fallback_regen runs before evaluate_coherence): Verify graph topology with `graph.get_graph().print_ascii()`.
- ❌ AgentState field is missing in a node: Use TypedDict strictly; mypy will catch at dev time.
- ❌ Async retrieve timeout too short: Increase to 5s for slow tools; configure per tool in Phase 2.

---

### Phase 5: LangSmith Observability

| Field | Value |
|-------|-------|
| **Duration** | 1 week |
| **Input Artefacts** | Phase 4 outputs (graph.py, research_agent_api_v2.py), Phase 1 outputs (llm_client.py) |
| **Primary Output** | `./medical-research-agent/observability.py` (LangSmith integration) + updated `graph.py` with tracing hooks + `prometheus_metrics.py` (Prometheus integration) |
| **Secondary Outputs** | `docker-compose.yml` (LangSmith Trace Server + Prometheus), `.env.example` (LANGSMITH_API_KEY, LANGSMITH_PROJECT) |
| **Code Reuse** | No code reuse; pure new functionality |
| **Validation Gate** | ✅ LangSmith Trace Server receives traces for every graph execution; ✅ Traces include all 8 node names and execution times; ✅ Token counts are recorded per node; ✅ Prometheus `/metrics` endpoint exports `llm_calls_total`, `llm_tokens_total`, `llm_cost_total`, `graph_execution_time_seconds`, `tool_execution_time_seconds`; ✅ Structured JSON logs include trace_id for correlation; ✅ E2E test: Run a query, view trace in LangSmith UI, view metrics in Prometheus |
| **Rollback** | Delete `observability.py`, `prometheus_metrics.py`, docker-compose.yml; revert tracing hooks in graph.py; set LANGSMITH_API_KEY to "" (disables tracing) |

**Detailed Phase 5 Tasks:**

1. **Create `observability.py`** (150 LOC)
   - Initialize LangSmith tracing
   - Decorator: `@trace_execution(name: str)` to wrap node functions
   - Automatic trace context propagation via LangGraph integration
   - Example:
     ```python
     from langsmith import trace, get_tracer
     
     def init_langsmith(project_name: str = "medical-research-agent"):
         os.environ["LANGSMITH_PROJECT"] = project_name
         tracer = get_tracer(project_name)
         return tracer
     ```

2. **Create `prometheus_metrics.py`** (100 LOC)
   - Initialize Prometheus metrics:
     - `llm_calls_total` (Counter) — incremented per LLMClient.chat() call
     - `llm_tokens_total` (Histogram) — distribution of token counts
     - `llm_cost_total` (Gauge) — cumulative cost
     - `graph_execution_time_seconds` (Histogram) — distribution of full graph runs
     - `tool_execution_time_seconds` (Histogram, with label "tool_name")
   - Integrate with LLMClient: llm_client.py emits metrics on every call
   - Integrate with graph: Each node emits tool_execution_time_seconds on completion

3. **Update `llm_client.py`** to emit Prometheus metrics
   - After every chat() call: increment llm_calls_total, add tokens to llm_tokens_total, add cost to llm_cost_total
   - After every embed() call: same metrics (separate counter prefix? discuss)

4. **Update `graph.py`** to emit traces
   - Wrap each node function with `@trace_execution()`
   - Log node start/end with trace_id
   - Emit tool_execution_time_seconds histogram after each node

5. **Update `research_agent_api_v2.py`** to emit metrics
   - Track graph execution time (start at endpoint entry, stop at response)
   - Emit graph_execution_time_seconds histogram

6. **Create `docker-compose.yml`** (50 LOC)
   - LangSmith Trace Server (langsmith/langsmith:latest)
   - Prometheus (prom/prometheus:latest)
   - Example:
     ```yaml
     version: '3.8'
     services:
       langsmith:
         image: langsmith/langsmith:latest
         ports:
           - "8000:8000"
       prometheus:
         image: prom/prometheus:latest
         ports:
           - "9090:9090"
         volumes:
           - ./prometheus.yml:/etc/prometheus/prometheus.yml
     ```

7. **Create `prometheus.yml`** (30 LOC)
   - Scrape config for medical-research-agent `/metrics` endpoint (localhost:9000/metrics)
   - Scrape interval: 15s

8. **Update `.env.example`**
   - Add `LANGSMITH_API_KEY=` (optional; if empty, tracing is disabled)
   - Add `LANGSMITH_PROJECT=medical-research-agent`
   - Add `PROMETHEUS_PORT=9000`

9. **Create structured JSON logging** (50 LOC in unicode_safe_logging.py update)
   - Log format: `{"timestamp": "...", "trace_id": "...", "level": "INFO", "message": "...", "node": "classify_intent"}`
   - Update logger calls in graph.py to include trace_id and node name
   - Example: `logger.info(f"[{state['trace_id']}] Node {node_name} completed", extra={"node": node_name})`

10. **Create integration tests** (100 LOC)
    - Test that LangSmith receives traces (query LangSmith API for recent traces)
    - Test that Prometheus scrapes metrics (curl http://localhost:9090/api/v1/query?query=llm_calls_total)
    - Test that structured logs include trace_id

11. **Create `medresearch_phase5_validation.py`** script
    - Run a query through the full system
    - Retrieve trace from LangSmith API using trace_id
    - Verify all 8 nodes appear in trace
    - Query Prometheus for llm_calls_total; verify it incremented
    - Verify logs include trace_id

**Failure Modes and Handling:**

- ❌ LangSmith API key is invalid: Gracefully disable tracing; log warning; continue execution.
- ❌ Prometheus scrape fails (port not open): Check if medical-research-agent is exporting /metrics; verify port config.
- ❌ Trace context is lost between nodes: Ensure LangGraph propagates context (LangSmith docs); if not, manually inject trace_id in state.

---

### Phase 6: Evaluation Harness

| Field | Value |
|-------|-------|
| **Duration** | 2 weeks |
| **Input Artefacts** | Phase 5 outputs (observability.py), Phase 4 outputs (graph.py), Phase 3 outputs (skills/), Phase 1 outputs (models.yaml) |
| **Primary Output** | `./medical-research-agent/evaluation/` directory with evaluation framework (RAGAS metrics, benchmark loaders, evaluation runner) |
| **Secondary Outputs** | `./medical-research-agent/benchmarks/` (MEDQA dataset, PubMed-QA dataset, custom clinical Q&A), evaluation results JSON, leaderboard dashboard (Streamlit app) |
| **Code Reuse** | No code reuse; pure new functionality (but may reuse citation_formatter.py logic in citation verifier) |
| **Validation Gate** | ✅ Benchmark datasets load correctly (MEDQA: 1000 Q&A pairs, PubMed-QA: 1000 pairs, custom: 50 pairs); ✅ RAGAS metrics calculate correctly (faithfulness, answer_relevancy, context_recall, latency); ✅ Evaluation runner processes 100+ queries across multiple models/agents without crashing; ✅ Results can be filtered/compared (e.g., "gpt-4o vs claude-3-5-sonnet on MEDQA"); ✅ Leaderboard dashboard loads and displays top models/agents; ✅ All results are reproducible (same query + model + seed = same trace) |
| **Rollback** | Delete `evaluation/`, `benchmarks/`, evaluation results JSON; revert any imports of evaluation modules from main codebase |

**Detailed Phase 6 Tasks:**

1. **Create `evaluation/` directory structure**
   - `__init__.py`
   - `ragas_metrics.py` — RAGAS metric implementations
   - `benchmark_loader.py` — Load MEDQA, PubMed-QA, custom datasets
   - `evaluation_runner.py` — Orchestrate evaluation across queries × models × agents
   - `results_aggregator.py` — Aggregate results, compute statistics

2. **Implement RAGAS metrics** (150 LOC)
   - **Faithfulness** — Answer stays true to retrieved context (LLM-based judgment)
   - **Answer Relevancy** — Answer directly addresses the question (semantic similarity)
   - **Context Recall** — Retrieved context contains all information needed to answer (coverage check)
   - **Latency** — End-to-end execution time (milliseconds)
   - Each metric returns a score in [0, 1]

3. **Create `benchmark_loader.py`** (150 LOC)
   - `MedqaDataset` class: Load MEDQA dataset (1000 Q&A pairs)
   - `PubmedqaDataset` class: Load PubMed-QA dataset (1000 pairs)
   - `CustomDataset` class: Load custom clinical Q&A from CSV
   - Common interface: `dataset.get_questions() → List[str]`, `dataset.get_expected_answer(q) → str`

4. **Create `evaluation_runner.py`** (200 LOC)
   - `EvaluationRunner` class
   - `run_evaluation(queries, models, agents, metrics) → EvaluationResults`
   - Iterate over: queries × models × agents
   - For each combination:
     - Switch LLMClient to the model
     - Switch Orchestrator to use the agent(s)
     - Invoke graph with the query
     - Calculate all metrics
     - Store result with trace_id for reproducibility
   - Use ThreadPoolExecutor for parallelization (max_workers=4 to avoid rate limits)
   - Graceful error handling: If a query × model × agent fails, log and continue

5. **Create `results_aggregator.py`** (150 LOC)
   - `ResultsAggregator` class
   - Aggregate results by model, agent, benchmark
   - Compute mean/stddev for each metric
   - Rank models/agents by average score
   - Export to JSON: `{"results": [{query, model, agent, metrics, trace_id}], "summary": {...}}`

6. **Create `benchmarks/` directory**
   - `medqa_dataset.json` (MEDQA questions, 1000 Q&A)
   - `pubmedqa_dataset.json` (PubMed-QA questions, 1000 Q&A)
   - `custom_dataset.csv` (Custom clinical Q&A, 50 pairs)
   - `README.md` (Dataset descriptions, sources)

7. **Create evaluation runner script** (100 LOC)
   - `run_evaluations.py` — CLI script
   - Arguments: `--models gpt-4o,claude-3-5-sonnet --agents local,pubmed --benchmark medqa --output results.json`
   - Example run: `python run_evaluations.py --models gpt-4o --agents local --benchmark medqa --queries 10`

8. **Create Streamlit dashboard** (200 LOC)
   - `dashboard.py` — Interactive leaderboard
   - Filters: Model, Agent, Benchmark, Metric
   - Visualizations:
     - Leaderboard table (models ranked by average score)
     - Heatmap (agents × models)
     - Time series (accuracy over time)
     - Per-query analysis (details for a specific query)
   - Export: CSV, JSON

9. **Create integration tests** (150 LOC)
   - Test RAGAS metric calculation on synthetic data
   - Test benchmark loaders return correct structure
   - Test evaluation runner completes without errors on small dataset (10 queries)
   - Test results aggregator computes correct statistics

10. **Create `medresearch_phase6_validation.py`** script
    - Run evaluation on 10 queries × 2 models × 2 agents = 40 evaluations
    - Verify all metrics are in [0, 1]
    - Verify results JSON is valid
    - Verify Streamlit dashboard loads without error
    - Verify trace_id allows reproducing any result

**Failure Modes and Handling:**

- ❌ MEDQA dataset not available: Fall back to smaller custom dataset; log warning.
- ❌ RAGAS metric calculation fails for a query (e.g., embedding timeout): Log error, skip metric, continue.
- ❌ Model rate limit reached during evaluation: Implement exponential backoff; continue after cooldown.
- ❌ Streamlit dashboard crashes on large results: Implement pagination; pre-aggregate results to summary level.

---

### Phase 7: Publication & Deployment

| Field | Value |
|-------|-------|
| **Duration** | 1 week |
| **Input Artefacts** | Phase 6 outputs (evaluation results, Streamlit dashboard), migration_plan.md (this document), migration_architectural_requirements.md, inventory.md |
| **Primary Output** | `./medical-research-agent-paper/` directory with paper, code repository, evaluation results, reproducibility artifacts |
| **Secondary Outputs** | Docker container, Helm chart (optional), GitHub release |
| **Code Reuse** | No code reuse; publication task |
| **Validation Gate** | ✅ Paper is written and reviewed; ✅ Code repository is clean (no debug logs, no hardcoded secrets); ✅ Evaluation results are reproducible (provided trace IDs can regenerate exact results); ✅ Docker container builds and runs; ✅ GitHub release includes all artifacts (paper, code, Dockerfile, Helm chart, results.json) |
| **Rollback** | N/A (publication is terminal; rollback would mean unpublishing, which is not recommended) |

**Detailed Phase 7 Tasks:**

1. **Write research paper** (2–3 days)
   - Title: "Medical Research Agent: A LLM-Agnostic, Graph-Based Orchestration System for Clinical Question Answering"
   - Sections:
     - Abstract (150 words)
     - Introduction (500 words) — motivation, prior work, contributions
     - Architecture (800 words) — LangGraph design, MCP tools, skill discovery, observability
     - Evaluation (1000 words) — RAGAS metrics, benchmarks, results table, leaderboard
     - Reproducibility (300 words) — trace IDs, LangSmith, Docker setup
     - Conclusion (200 words)
   - Figures: Architecture diagram, Phase 0–7 timeline, leaderboard table, cost vs accuracy graph
   - Tables: Support matrix (4 providers × 5 agents), RAGAS results (models × benchmarks)

2. **Code cleanup** (1 day)
   - Remove all debug logging (keep only INFO, WARNING, ERROR)
   - Verify no hardcoded secrets (API keys, tokens) in codebase
   - Run linters (black, pylint) on all code
   - Document all public APIs (docstrings)
   - Update CLAUDE.md with Phase 7 completion notes

3. **Create reproducibility package** (1 day)
   - `reproducibility/` directory
   - `reproduce_results.sh` — Script to re-run evaluation with same settings
   - `results.json` — All evaluation results from Phase 6
   - `trace_ids.json` — Map from query to trace_id for reproducibility
   - `REPRODUCTION.md` — Instructions for reproducing results

4. **Create Docker container** (1 day)
   - `Dockerfile` — Multi-stage build
     - Stage 1: Python 3.11 + dependencies (litellm, langchain, pydantic, etc.)
     - Stage 2: Copy code, expose port 8000, set ENTRYPOINT to uvicorn
   - `.dockerignore` — Exclude large files (datasets, logs)
   - Build & test: `docker build -t medical-research-agent . && docker run -p 8000:8000 medical-research-agent`

5. **Create Helm chart** (optional; 1 day)
   - `helm/` directory
   - `Chart.yaml`, `values.yaml`, `templates/deployment.yaml`
   - Configurable: replicas, port, environment variables (LANGSMITH_API_KEY, models.yaml)

6. **Create GitHub release** (1 day)
   - Tag: `v1.0.0-phase7`
   - Release notes: Summary of all 7 phases, key features, how to use
   - Artifacts:
     - medical-research-agent-paper.pdf
     - medical-research-agent-code.zip
     - Dockerfile
     - docker-compose.yml
     - results.json
     - reproducibility/

7. **Update README.md** in target directory
   - Quick start (install, run, query)
   - Architecture overview (8 nodes, MCP tools, LangGraph)
   - Evaluation results (leaderboard snippet)
   - Reproducibility instructions
   - Links to paper, Helm chart, Docker Hub

8. **Create `DEPLOYMENT.md`**
   - How to deploy to production (Kubernetes, Docker Compose, cloud platforms)
   - Configuration guide (models.yaml, environment variables)
   - Monitoring setup (LangSmith, Prometheus, Grafana)
   - Troubleshooting guide

---

## Master Dependency Graph

```
Phase 1 (LLM Abstraction)
    ↓
    ├─→ Phase 2 (MCP Tools) ──→ Phase 3 (Skill Discovery)
    │                              ↓
    │                          Phase 4 (LangGraph)
    │                              ↓
    └──→ Phase 5 (Observability) ←─┘
             ↓
         Phase 6 (Evaluation)
             ↓
         Phase 7 (Publication)
```

**Critical Path (all phases are on critical path; sequential only):**
- Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7

**No phases can be parallelized.** Each phase's input depends on prior phase's output.

---

## Rollback Strategy

Each phase has an explicit rollback plan (documented above). The strategy is:

1. **During a phase**: If validation gate fails, stop immediately and rollback to end of prior phase.
2. **After publication** (Phase 7): Rollback is not recommended; instead, issue a patch release (v1.0.1) with fixes.

**Rollback Procedure** (if needed during Phases 1–6):
```bash
# Revert code changes from current phase
git checkout HEAD~1

# Restore prior phase's output files
# (specific files listed in each phase's rollback section)

# Rerun prior phase's validation gate
# (if gate passes, proceed; if fails, rollback further)
```

---

## Cost and Timeline Summary

| Phase | Duration | Key Deliverable | Cost to Proceed |
|-------|----------|-----------------|-----------------|
| 0 | 1 day | inventory.md, migration_architectural_requirements.md | Module inventory complete |
| 1 | 1 week | llm_client.py, models.yaml | LLM abstraction working |
| 2 | 2 weeks | mcp_tools/, mcp_registry.py | All 5 agents converted to tools |
| 3 | 1 week | skills/, skill_discovery.py | Skill router returns correct tool rankings |
| 4 | 2 weeks | graph.py, research_agent_api_v2.py | LangGraph executes all 8 nodes correctly |
| 5 | 1 week | observability.py, prometheus_metrics.py | LangSmith receives traces, Prometheus exports metrics |
| 6 | 2 weeks | evaluation/, leaderboard dashboard | Evaluation runs 100+ queries without error |
| 7 | 1 week | Paper, Dockerfile, GitHub release | Code is published and reproducible |
| **TOTAL** | **10 weeks + 2 week QA** | **Production system** | **Phase 7 validation gate passes** |

---

## Key Validation Gates (Critical Checkpoints)

### Phase 1 Gate: LLM Abstraction
```bash
# Test that all 3 models work
python -c "from llm_client import LLMClient; c = LLMClient(); print(c.chat([{'role': 'user', 'content': 'test'}], model='gpt-4o'))"

# Test that cost tracking works
curl http://localhost:8000/metrics | grep llm_cost_total

# Test that old FastAPI endpoint still works
curl -X POST http://localhost:8000/query -d '{"question": "What is diabetes?"}' -H "Content-Type: application/json"
```

### Phase 2 Gate: MCP Tools
```bash
# Test that all 5 tools are discoverable
python -c "from mcp_registry import mcp_registry; print(mcp_registry.list_tools())"

# Test that each tool returns correct schema
python evaluation/test_mcp_tools.py

# Test that no embedding calls bypass LLMClient
grep -r "text-embedding-3-large\|embedding_openai" ./medical-research-agent/mcp_tools/ && echo "FAIL: found hardcoded embeddings" || echo "PASS: all embeddings use LLMClient"
```

### Phase 3 Gate: Skill Discovery
```bash
# Test that skill_discovery ranks tools correctly
python -c "from skill_discovery import SkillDiscovery; sd = SkillDiscovery(); print(sd.query('What is type 2 diabetes?'))"

# Expected output: [('local', 0.95), ('pubmed', 0.88), ('clinical_trials', 0.72)]
```

### Phase 4 Gate: LangGraph
```bash
# Test that graph executes all 8 nodes
python -c "from graph import graph; import asyncio; asyncio.run(graph.ainvoke({'input_query': 'What is diabetes?', ...}))"

# Test that conditional edges work
# (non-medical query should exit early from classify_intent)
python evaluation/test_graph_edges.py

# Test that format_response produces AMA citations
python -c "from graph import graph; result = ...; print(len(result['output_citations']) > 0)"
```

### Phase 5 Gate: Observability
```bash
# Test that LangSmith receives traces
curl -H "Authorization: Bearer $LANGSMITH_API_KEY" https://api.smith.langchain.com/runs?project_name=medical-research-agent | jq '.runs | length'

# Expected: > 0 (at least one trace from Phase 4 test)

# Test that Prometheus exports metrics
curl http://localhost:9090/api/v1/query?query=llm_calls_total | jq '.data.result[0].value[1]'

# Expected: a non-zero count
```

### Phase 6 Gate: Evaluation
```bash
# Test that evaluation runner completes without errors on 10 queries
python medical-research-agent/evaluation_runner.py --queries 10 --models gpt-4o --agents local --output results.json

# Test that results.json is valid and has metrics
jq '.results[0] | keys' results.json | grep -E "faithfulness|answer_relevancy|context_recall"

# Expected: all metrics present
```

### Phase 7 Gate: Publication
```bash
# Test that GitHub release is created
gh release view v1.0.0-phase7

# Expected: Release exists with all artifacts (paper.pdf, code.zip, Dockerfile, etc.)

# Test that Docker image builds and runs
docker build -t medical-research-agent . && docker run -p 8000:8000 medical-research-agent &
sleep 5
curl http://localhost:8000/health

# Expected: { "status": "healthy" }
```

---

## Post-Phase 7: Ongoing Maintenance

Once Phase 7 is complete, the system enters **production maintenance mode**:

1. **Monitoring** — LangSmith traces all queries; Prometheus tracks latency/cost
2. **Evaluation** — Leaderboard is updated monthly with new benchmark datasets
3. **Iteration** — Bug fixes and improvements are released as patch versions (v1.0.1, v1.0.2, etc.)
4. **Extension** — New agents can be added by creating a new MCP tool + YAML manifest (no other changes required)

---

## Summary

This migration plan is a **sequential, gate-based roadmap** with explicit code reuse decisions. Each phase:
- Has a single primary output artefact
- Depends completely on prior phases
- Cannot be parallelized
- Has a clear validation gate before proceeding

The total timeline is **10 weeks** of engineering, with production deployment expected at **week 12** (including QA/review).

**All decisions are traceable:**
- Phase 0 inventory → Phase 1 decisions (what to reuse/refactor/delete)
- migration_architectural_requirements.md → Phases 1–6 implementation
- This document → Week-by-week execution

**No architectural blockers.** The legacy system is sound; this is a **structural refactoring with extended observability and evaluation**.
