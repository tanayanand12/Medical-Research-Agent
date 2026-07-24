# Phase 4 Quick Reference

**Implementation:** LangGraph StateGraph with 8 nodes  
**Files Created:** 15 new files, ~2500 LOC  
**Status:** ✅ Complete and validated

---

## 8-Node Pipeline

| # | Node | Input | Output | Condition |
|---|------|-------|--------|-----------|
| 1 | **classify_intent** | query | is_medical_query | - |
| 2 | **discover_skills** | query, context | discovered_skills | if is_medical_query |
| 3 | **parallel_retrieve** | query, skills | retrieval_results | - |
| 4 | **synthesise** | query, results | intermediate_answer | - |
| 5 | **score_confidence** | skills, results | confidence_score | - |
| 6 | **evaluate_coherence** | query, answer, results | coherence_score | - |
| 7 | **fallback_regen** | query, context | fallback_answer | if coherence < 0.6 |
| 8 | **format_response** | answer, results | output_answer + citations | - |

---

## File Organization

```
medical-research-agent/
├── agent_state.py                 # ← TypedDict (30+ fields)
├── edges.py                       # ← Routing logic
├── graph.py                       # ← StateGraph definition
├── research_agent_api_v2.py       # ← FastAPI server
├── nodes/                         # ← 8 node modules
│   ├── classify_intent.py
│   ├── discover_skills.py
│   ├── parallel_retrieve.py
│   ├── synthesise.py
│   ├── score_confidence.py
│   ├── evaluate_coherence.py
│   ├── fallback_regen.py
│   └── format_response.py
├── test_phase4_integration.py     # ← Tests
├── medresearch_phase4_validation.py # ← Validation
└── PHASE_4_COMPLETION.md          # ← Full report
```

---

## Key Data Structures

### AgentState (TypedDict)
```python
# INPUT
input_query: str
context: Dict[str, Any]
trace_id: str

# OUTPUT
output_answer: str
output_citations: List[str]
confidence_score: float
execution_time_sec: float
```

### Query Request/Response
```python
# Request
question: str
model_id: str = "gpt-4o"
agents_to_use: Optional[List[str]] = None

# Response
answer: str
citations: List[str]
trace_id: str
execution_time_sec: float
```

---

## Conditional Routes

### Route 1: After classify_intent
```
if is_medical_query:
    → discover_skills
else:
    → format_response (early exit)
```

### Route 2: After evaluate_coherence
```
if coherence_score < 0.6 and fallback_count < 1:
    → fallback_regen
else:
    → format_response
```

---

## Running Phase 4

### Start API Server
```bash
python research_agent_api_v2.py
# Server on http://localhost:8000
```

### Send Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is type 2 diabetes?",
    "model_id": "gpt-4o"
  }'
```

### Run Tests
```bash
pytest test_phase4_integration.py -v
```

### Run Validation
```bash
python medresearch_phase4_validation.py
# Exit code 0 = all checks pass
# Exit code 1 = fix errors
```

### View Graph
```bash
# ASCII art
curl http://localhost:8000/graph/ascii

# Mermaid diagram
curl http://localhost:8000/graph/diagram
```

---

## Phase 1 Dependencies

### LLMClient (Injected)
```python
# In any node
llm_client = LLMClient()
answer = llm_client.chat(
    messages=[...],
    model="gpt-4o",  # or None to use default
    temperature=0.7
)
```

### MCP Registry (Phase 2)
```python
# In parallel_retrieve
tool = mcp_registry.get_tool("pubmed")
result = tool.invoke(query=question, context=ctx)
```

### Skill Router (Phase 3)
```python
# In discover_skills
router = SkillRouter()
tools, scores = router.rank_tools(query, top_k=3)
```

---

## Error Handling Pattern

Every node:
1. Tries main operation
2. Catches exceptions
3. Sets `error_occurred = True`
4. Appends to `error_messages[]`
5. Returns graceful fallback (doesn't crash)

Example:
```python
try:
    result = classifier.classify(query)
except Exception as e:
    logger.error(f"Classification failed: {e}")
    state["error_occurred"] = True
    state["error_messages"].append(f"Classification error: {e}")
    # Fallback: assume medical
    state["is_medical_query"] = True
    return state
```

---

## Validation Checklist

Before Phase 5, verify:

- [ ] `python medresearch_phase4_validation.py` returns exit code 0
- [ ] All 10 validation checks pass
- [ ] `pytest test_phase4_integration.py` passes (15+ tests)
- [ ] API server starts: `python research_agent_api_v2.py`
- [ ] `/query` endpoint responds with valid JSON
- [ ] `/health` returns `{"status": "healthy"}`
- [ ] Graph ASCII visualization works: `curl http://localhost:8000/graph/ascii`
- [ ] Graph Mermaid diagram works: `curl http://localhost:8000/graph/diagram`

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **P95 Latency** | < 8s | For 3-tool case |
| **Confidence** | 0.6–0.95 | Coverage-based |
| **Tokens/Query** | 500–1500 | Across all LLM calls |
| **Cost/Query** | $0.005–0.020 | Depends on model |
| **Memory** | ~10 MB | Per query (state + results) |

---

## Troubleshooting

### Issue: Graph fails to compile
**Fix:** Check all imports in `graph.py` and nodes/

### Issue: Node execution fails
**Fix:** Check logs in `logs/api.log` for specific error; verify mocking in tests

### Issue: Classification always returns non-medical
**Fix:** Check `QueryClassifier` is using `LLMClient` (not hardcoded OpenAI)

### Issue: Parallel retrieve times out
**Fix:** Increase timeout in `parallel_retrieve.py` from 5.0s to 10.0s (or more)

### Issue: Fallback never triggers
**Fix:** Verify `coherence_score < 0.6` and `fallback_count < 1` in edge logic

---

## Phase 4 → Phase 5 Handoff

**Phase 5 will add:**
1. LangSmith trace integration
2. Prometheus metrics export
3. Structured JSON logging with trace_id
4. Docker Compose (LangSmith + Prometheus servers)

**Phase 4 provides:**
- ✅ trace_id in every request
- ✅ execution_time_sec in response
- ✅ error_messages for debugging
- ✅ All nodes callable and mockable

**No changes to Phase 4 code needed for Phase 5** — Phase 5 adds observability on top without refactoring.

---

## Summary

✅ **Phase 4 Deliverables**
- 8 production-ready nodes
- Full state machine with typed contract
- Backwards-compatible API
- Comprehensive error handling
- 15+ integration tests
- 10 validation checks

✅ **Ready For**
- Testing (unit + integration)
- Backwards-compatibility comparison with Phase 1
- Production deployment (after Phase 5 observability)

⏳ **Next: Phase 5 (LangSmith + Prometheus)**
