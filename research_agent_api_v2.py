"""
Medical Research Agent API (Phase 4: LangGraph Version)

FastAPI endpoints for query processing using LangGraph StateGraph orchestration.
Replaces legacy research_agent_api.py which used ThreadPoolExecutor.

Key differences:
- Routes through LangGraph StateGraph instead of ThreadPool + Orchestrator
- Full tracing via trace_id (for LangSmith, Phase 5)
- Structured responses with execution_time_sec, cost_estimate
- No user persona tracking (explicitly removed per requirements)
"""

from typing import Optional, Dict, Any, List
import os
import uuid
from datetime import datetime

from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from graph import get_graph
from agent_state import AgentState
from unicode_safe_logging import configure_all_loggers

import logging

# Configure logging
configure_all_loggers()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Medical Research Agent API",
    description="LangGraph-based medical research question answering with evidence synthesis",
    version="2.0.0-phase4"
)


# ============================================================================
# Request/Response Models
# ============================================================================

class DeepResearchParams(BaseModel):
    """Parameters for deep PubMed research."""
    max_papers: int = Field(100, description="Maximum papers to fetch")
    include_fulltext: bool = Field(True, description="Include PMC full text")
    search_recent: bool = Field(True, description="Prioritize recent publications")
    search_foundational: bool = Field(True, description="Include foundational papers")
    top_k: int = Field(8, description="Top-K chunks for context")
    rerank: bool = Field(True, description="Apply re-ranking on results")

    class Config:
        schema_extra = {
            "example": {
                "max_papers": 100,
                "include_fulltext": True,
                "search_recent": True,
                "search_foundational": True,
                "top_k": 8,
                "rerank": True
            }
        }


class QueryRequest(BaseModel):
    """Query request model."""
    question: str = Field(..., description="Medical research question")
    model_id: str = Field("gpt-4o", description="LLM model to use")
    db_name: str = Field("index", description="Vector index name")
    top_k: int = Field(5, description="Documents per tool")
    clinical_trials_top_k: int = Field(10, description="Clinical trials results")
    fda_top_k: int = Field(10, description="FDA results")
    max_trials: int = Field(25, description="Max trials to retrieve")

    # Agent selection (overrides discovery)
    agents_to_use: Optional[List[str]] = Field(
        None,
        description="Explicit agents to use (overrides discovery). "
        "Options: local, pubmed, pubmed_deep_research, clinical_trials, fda"
    )

    # Deep research params
    pubmed_deep_research_params: DeepResearchParams = Field(
        default_factory=DeepResearchParams,
        description="Parameters for deep PubMed research"
    )

    class Config:
        schema_extra = {
            "example": {
                "question": "What are the latest findings on SGLT2 inhibitors in heart failure?",
                "model_id": "gpt-4o",
                "db_name": "index",
                "top_k": 5,
                "agents_to_use": None  # Use discovery
            }
        }


class QueryResponse(BaseModel):
    """Query response model."""
    answer: str = Field(..., description="Final synthesized answer")
    sources: List[str] = Field(..., description="Tool names that contributed")
    citations: List[str] = Field(..., description="AMA-formatted citations")
    confidence: float = Field(..., description="Confidence score [0, 1]")
    trace_id: str = Field(..., description="Unique trace ID for reproducibility")
    execution_time_sec: float = Field(..., description="Total execution time")
    cost_estimate: float = Field(..., description="Estimated cost (USD)")
    fallback_triggered: bool = Field(..., description="Whether fallback was used")
    is_partial_response: bool = Field(..., description="Whether response is partial")
    error_occurred: bool = Field(..., description="Whether any errors occurred")

    class Config:
        schema_extra = {
            "example": {
                "answer": "[DISCLAIMER: ...] SGLT2 inhibitors...",
                "sources": ["pubmed", "clinical_trials"],
                "citations": ["1. Smith J, et al. ..."],
                "confidence": 0.85,
                "trace_id": "550e8400-e29b-41d4-a716-446655440000",
                "execution_time_sec": 4.23,
                "cost_estimate": 0.012,
                "fallback_triggered": False,
                "is_partial_response": False,
                "error_occurred": False
            }
        }


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Process a medical research query using LangGraph orchestration.

    Args:
        request: QueryRequest with question and optional parameters

    Returns:
        QueryResponse with synthesized answer, citations, metrics

    Raises:
        HTTPException: If query processing fails completely
    """
    trace_id = str(uuid.uuid4())

    logger.info(
        f"[{trace_id}] Processing query: {request.question[:100]}... "
        f"Model: {request.model_id}"
    )

    try:
        # Build context dict from request
        context = {
            "model_id": request.model_id,
            "db_name": request.db_name,
            "top_k": request.top_k,
            "clinical_trials_top_k": request.clinical_trials_top_k,
            "fda_top_k": request.fda_top_k,
            "max_trials": request.max_trials,
            "pubmed_deep_research_params": request.pubmed_deep_research_params.model_dump(),
        }

        # Add explicit agents if provided
        if request.agents_to_use:
            context["agents_to_use"] = request.agents_to_use

        # Initialize AgentState
        initial_state = AgentState(
            # Input
            input_query=request.question,
            context=context,
            trace_id=trace_id,
            timestamp_start=datetime.utcnow(),

            # Classification (defaults)
            is_medical_query=True,
            classification_confidence=0.0,
            classification_reason="",

            # Skill discovery (defaults)
            discovered_skills=[],
            skill_scores={},

            # Retrieval (defaults)
            retrieval_results={},
            tokens_used={},
            retrieval_time_sec={},
            total_retrieval_time_sec=0.0,

            # Synthesis (defaults)
            intermediate_answer="",
            intermediate_sources=[],
            intermediate_model_used="",
            synthesis_tokens_in=0,
            synthesis_tokens_out=0,
            synthesis_time_sec=0.0,

            # Scoring (defaults)
            confidence_score=0.0,
            coverage_explanation="",

            # Coherence (defaults)
            coherence_score=0.0,
            coherence_explanation="",
            should_fallback=False,
            coherence_eval_model_used="",

            # Fallback (defaults)
            fallback_count=0,
            fallback_answer=None,
            fallback_triggered=False,
            fallback_reason="",

            # Output (defaults)
            output_answer="",
            output_sources=[],
            output_citations=[],
            output_disclaimer="",

            # Performance (defaults)
            timestamp_end=datetime.utcnow(),
            execution_time_sec=0.0,
            cost_estimate=0.0,

            # Error handling (defaults)
            error_occurred=False,
            error_messages=[],
            is_partial_response=False,
        )

        # Execute graph
        graph = get_graph()
        final_state = graph.invoke(initial_state)

        # Build response
        response = QueryResponse(
            answer=final_state["output_answer"],
            sources=final_state["output_sources"],
            citations=final_state["output_citations"],
            confidence=final_state["confidence_score"],
            trace_id=trace_id,
            execution_time_sec=final_state["execution_time_sec"],
            cost_estimate=final_state["cost_estimate"],
            fallback_triggered=final_state.get("fallback_triggered", False),
            is_partial_response=final_state.get("is_partial_response", False),
            error_occurred=final_state.get("error_occurred", False),
        )

        logger.info(
            f"[{trace_id}] Query complete. "
            f"Time: {response.execution_time_sec:.2f}s, "
            f"Confidence: {response.confidence:.2f}, "
            f"Fallback: {response.fallback_triggered}"
        )

        return response

    except Exception as e:
        logger.error(
            f"[{trace_id}] Query processing failed: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.

    Returns:
        Status dict
    """
    return {"status": "healthy", "version": "2.0.0-phase4"}


@app.get("/graph/diagram")
async def get_graph_diagram() -> Dict[str, str]:
    """
    Get Mermaid diagram of the graph structure.

    Returns:
        Diagram definition
    """
    from graph import get_graph_diagram
    return {"diagram": get_graph_diagram()}


@app.get("/graph/ascii")
async def get_graph_ascii() -> Dict[str, str]:
    """
    Get ASCII representation of the graph.

    Returns:
        ASCII art
    """
    from graph import print_graph_ascii
    return {"ascii": print_graph_ascii()}


# ============================================================================
# Server Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("Medical Research Agent API v2.0 (Phase 4: LangGraph) starting up")
    # Pre-load graph
    graph = get_graph()
    logger.info("LangGraph compiled and ready")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Medical Research Agent API shutting down")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Run the FastAPI server using uvicorn."""
    logger.info("Starting Medical Research Agent API server")
    uvicorn.run(
        "research_agent_api_v2:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
