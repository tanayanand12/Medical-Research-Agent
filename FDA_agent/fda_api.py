import os
import logging
from fastapi import FastAPI, HTTPException # type: ignore
from pydantic import BaseModel, Field # type: ignore
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv  # type: ignore
from openai import OpenAI  # type: ignore
from fda_rag_pipeline import FdaRAGPipeline

# ----------------------------------------------------------------------------
# Load environment & configure logging
# ----------------------------------------------------------------------------
load_dotenv()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="[%(levelname)s] %(asctime)s - %(name)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger("fda_api")

# ----------------------------------------------------------------------------
# FastAPI app and models
# ----------------------------------------------------------------------------
app = FastAPI(
    title="FDA RAG Agent API",
    description="Retrieve FDA data via an AI-powered RAG pipeline",
    version="1.0.0",
)

class RagRequest(BaseModel):
    query: str = Field(..., description="Natural-language query for FDA data")
    top_k: Optional[int] = Field(10, ge=1, le=50, description="Number of top chunks to retrieve and cite")

class RagResponse(BaseModel):
    success: bool = Field(...)
    answer: Optional[str] = Field(None)
    citations: Optional[List[str]] = Field(None)
    metadata: Optional[Dict[str, Any]] = Field(None)
    error: Optional[str] = Field(None)

# ----------------------------------------------------------------------------
# OpenAI client & pipeline initialization
# ----------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not set. Exiting.")
    raise RuntimeError("Missing OPENAI_API_KEY environment variable")
client = OpenAI(api_key=OPENAI_API_KEY)
pipeline = FdaRAGPipeline(openai_client=client)
logger.info("FDA RAG Pipeline initialized")

# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}

@app.post("/fda/query", response_model=RagResponse, tags=["FDA"])
async def fda_query(request: RagRequest) -> RagResponse:
    """
    Run the FDA RAG pipeline on a natural-language query.
    """
    try:
        result = pipeline.process_query(request.query, top_k=request.top_k)
    except Exception as exc:
        logger.error("Pipeline error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal pipeline error")

    if not result.get("success"):
        logger.warning("Query processing failed: %s", result.get("error"))
        return RagResponse(success=False, error=result.get("error"))

    return RagResponse(
        success=True,
        answer=result.get("answer"),
        citations=result.get("citations"),
        metadata=result.get("metadata"),
        error=None,
    )

# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fda_api:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
