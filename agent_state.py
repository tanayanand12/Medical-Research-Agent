"""
AgentState TypedDict for LangGraph StateGraph.

Defines the complete state contract for the medical research agent orchestration.
All 8 nodes read from and write to this state dictionary.
"""

from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime


class AgentState(TypedDict):
    """
    Complete state for medical research agent graph execution.

    Fields organized by lifecycle stage:
    - INPUT: Query and context from user
    - CLASSIFICATION: Medical domain filtering result
    - SKILL_DISCOVERY: Tool selection via semantic matching
    - RETRIEVAL: Results from parallel MCP tool execution
    - SYNTHESIS: LLM-generated answer
    - SCORING: Confidence and coherence metrics
    - FALLBACK: Secondary answer if coherence is low
    - OUTPUT: Final formatted response ready for user
    """

    # ============================================================================
    # INPUT STAGE
    # ============================================================================

    input_query: str
    """The user's medical research question."""

    context: Dict[str, Any]
    """
    Request context with parameters:
    - top_k: int (default 5) — documents to retrieve per tool
    - model_id: str (default "gpt-4o") — LLM model for this query
    - agents_to_use: List[str] (optional) — explicit agent names, overrides discovery
    - max_agent_retries: int — bounded retrieval retries per sub-agent
    - max_agent_synthesis_repairs: int — frozen-evidence repairs per sub-agent
    - max_synthesis_repairs: int — top-level frozen-evidence repairs
    - Other backend-specific parameters (db_name, max_trials, etc.)
    """

    trace_id: str
    """
    Unique trace ID for this execution. Used for LangSmith correlation,
    result reproducibility, and debugging.
    """

    timestamp_start: datetime
    """Query start timestamp."""

    # ============================================================================
    # CLASSIFICATION STAGE
    # ============================================================================

    is_medical_query: bool
    """
    True if query is within medical/clinical domain; False otherwise.
    Used for early exit gate.
    """

    classification_confidence: float
    """
    Confidence score for is_medical_query classification [0, 1].
    If < 0.3, query is borderline medical (may be rejected).
    """

    classification_reason: str
    """Human-readable explanation of classification decision."""

    # ============================================================================
    # SKILL DISCOVERY STAGE
    # ============================================================================

    discovered_skills: List[str]
    """
    List of MCP tool names selected by skill router, ranked by relevance.
    Example: ["local", "pubmed", "clinical_trials"]
    Empty list means no tools matched query (fallback: use all tools).
    """

    skill_scores: Dict[str, float]
    """
    Relevance score for each discovered skill [0, 1].
    Example: {"local": 0.95, "pubmed": 0.88, "clinical_trials": 0.72}
    """

    # ============================================================================
    # RETRIEVAL STAGE
    # ============================================================================

    retrieval_results: Dict[str, Dict[str, Any]]
    """
    Results from parallel MCP tool execution.

    Structure:
    {
      "local": {
        "results": [{"title": "...", "authors": [...], "year": 2024, "doi": "..."}],
        "error": None,
        "tokens_used": 150
      },
      "pubmed": {
        "results": [...],
        "error": None,
        "tokens_used": 200
      },
      "clinical_trials": {
        "results": [...],
        "error": "timeout",
        "tokens_used": 0
      }
    }

    Tools with errors still appear in the dict; error field is non-None.
    Downstream nodes can choose to use or ignore errored results.
    """

    tokens_used: Dict[str, int]
    """Token count for each tool. Used for cost tracking."""

    retrieval_time_sec: Dict[str, float]
    """Execution time (seconds) for each tool."""

    total_retrieval_time_sec: float
    """Sum of all retrieval times."""

    runtime_executor_metrics: Dict[str, Any]
    """Bounded worker capacity, saturation, completion, and timeout counters."""

    # ============================================================================
    # SYNTHESIS STAGE
    # ============================================================================

    intermediate_answer: str
    """
    Raw LLM-generated answer before any post-processing.
    Contains the synthesized response based on retrieved documents.
    """

    intermediate_sources: List[str]
    """
    Tool names that contributed documents to the synthesis.
    Example: ["local", "pubmed"]
    """

    intermediate_model_used: str
    """
    The LLM model that generated intermediate_answer.
    Recorded for reproducibility.
    """

    synthesis_tokens_in: int
    """Input tokens used by LLM for synthesis."""

    synthesis_tokens_out: int
    """Output tokens generated by LLM for synthesis."""

    synthesis_time_sec: float
    """LLM call latency."""

    last_synthesis_cost_usd: float
    """Cost of the latest synthesis call; used for per-attempt trace accounting."""

    synthesis_context: List[Dict[str, Any]]
    """Ordered evidence spans actually supplied to synthesis/repair."""

    # ============================================================================
    # SCORING STAGE
    # ============================================================================

    confidence_score: float
    """
    Coverage-based confidence [0, 1].
    Computed as: (# tools with results) / (# tools selected)
    Example: 2/3 = 0.67 (2 out of 3 tools returned results)
    """

    coverage_explanation: str
    """Human-readable explanation of confidence score."""

    confidence_components: Dict[str, float]
    """
    Separate qrel-free runtime quality components.
    """

    runtime_quality_score: float
    """
    Documented weighted combination of confidence_components. This is not a
    clinically calibrated probability and does not replace confidence_score.
    """

    runtime_quality_explanation: str
    """Formula description for runtime_quality_score."""

    # ============================================================================
    # COHERENCE EVALUATION STAGE
    # ============================================================================

    coherence_score: float
    """
    Backward-compatible routing score populated from runtime quality [0, 1].
    It is not a clinically calibrated coherence probability.
    """

    coherence_explanation: str
    """Runtime verifier status, failed checks, and score formula."""

    should_fallback: bool
    """
    True when a valid runtime decision requests synthesis repair. Missing or
    malformed decisions conservatively use the legacy coherence threshold.
    """

    coherence_eval_model_used: str
    """The LLM model that evaluated coherence."""

    # ============================================================================
    # FALLBACK STAGE
    # ============================================================================

    fallback_count: int
    """Number of fallback regenerations attempted (0 or 1 in Phase 1)."""

    fallback_answer: Optional[str]
    """
    Regenerated answer if fallback was triggered.
    If no fallback, this is None.
    """

    fallback_triggered: bool
    """True if fallback regeneration occurred."""

    fallback_reason: str
    """
    Explanation of why fallback was triggered.
    Example: "coherence_score=0.45 < threshold=0.6"
    """

    # ============================================================================
    # SHARED EVALUATION TRACE / RUNTIME VERIFICATION
    # ============================================================================

    evaluation_traces: List[Dict[str, Any]]
    """Versioned per-attempt sidecar traces; never contains benchmark qrels."""

    verification_history: List[Dict[str, Any]]
    """All agent and synthesis VerificationDecision records."""

    verification_decision: Optional[Dict[str, Any]]
    """Latest VerificationDecision used for bounded routing."""

    repair_history: List[Dict[str, Any]]
    """Structured feedback and applied changes for bounded retries/repairs."""

    evidence_limited: bool
    """True when available evidence or verification is insufficient."""

    attempt_telemetry: List[Dict[str, Any]]
    """Per-attempt model, tokens, cost, and compute latency."""

    token_usage: Dict[str, int]
    """Aggregate input/output/total tokens across recorded attempts."""

    # ============================================================================
    # FORMATTING STAGE
    # ============================================================================

    output_answer: str
    """
    Final answer returned to user.
    Includes [DISCLAIMER] prefix and [FALLBACK] tag if applicable.
    """

    output_sources: List[str]
    """Tool names that contributed to the final answer."""

    output_citations: List[str]
    """
    AMA-formatted citations for all referenced papers.
    Example: ["1. Smith J, et al. Title. Journal. 2024;10(1):1-10."]
    """

    output_disclaimer: str
    """
    Clinical disclaimer prepended to the answer.
    Standard text: "[DISCLAIMER: This response is AI-generated...]"
    """

    # ============================================================================
    # PERFORMANCE METRICS
    # ============================================================================

    timestamp_end: datetime
    """Query completion timestamp."""

    execution_time_sec: float
    """Total end-to-end execution time."""

    cost_estimate: float
    """
    Estimated cost of this execution (USD).
    Sum of: (LLM calls via LLMClient) + (token usage across tools)
    """

    # ============================================================================
    # ERROR HANDLING
    # ============================================================================

    error_occurred: bool
    """True if any recoverable error occurred (but execution continued)."""

    error_messages: List[str]
    """
    List of error messages from individual stages.
    Example: ["Tool 'clinical_trials' timed out after 5s"]
    """

    is_partial_response: bool
    """
    True if response is based on partial data (some tools failed).
    Indicates reduced confidence in final answer.
    """
