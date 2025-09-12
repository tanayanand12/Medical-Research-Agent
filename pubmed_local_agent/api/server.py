"""
api.server
~~~~~~~~~~

FastAPI wrapper around:
    • /process   → triggers ETL (keywords → FAISS DB)
    • /ask       → answers questions using existing DB

Run locally:
    uvicorn api.server:app --reload

Docs live at http://127.0.0.1:8000/docs
"""

from __future__ import annotations
import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import List, Dict, Any
import os
from pathlib import Path

# Create logs directory if it doesn't exist
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

# Force UTF-8 encoding for console output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Configure logging with proper encoding
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        # Rotating file handler with UTF-8 encoding
        RotatingFileHandler(
            log_dir / "server.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=5,
            encoding='utf-8'
        ),
        # Stream handler with UTF-8 encoding
        logging.StreamHandler(
            stream=sys.stdout
        )
    ]
)
logger = logging.getLogger("server")

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from process import run_pipeline
from query import PubMedQAEngine

# Import keyword agents if available
try:
    from keyword_agents import OpenAIClusteringAgent
    KEYWORD_AGENTS_AVAILABLE = True
except ImportError:
    KEYWORD_AGENTS_AVAILABLE = False

# ───────────────────────────────────────────────────────────────────── #
# FastAPI instance + CORS
# ───────────────────────────────────────────────────────────────────── #
app = FastAPI(
    title="PubMed RAG Service",
    description="Process PubMed keywords into FAISS DBs and query them via LLM.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────────────────────────────────────────────────────────────────── #
# Pydantic request models
# ───────────────────────────────────────────────────────────────────── #
class AgentParams(BaseModel):
    model: str = "gpt-4o"
    max_batch_size: int = 50
    max_clusters: int = 10
    optimize_boolean: bool = True


class ProcessRequest(BaseModel):
    keywords: List[str]
    db_name: str = "index"
    max_papers: int = Field(200_000, le=200_000)
    fulltext: bool = False
    use_keyword_agent: bool = False
    agent_params: AgentParams = None


class AskRequest(BaseModel):
    question: str
    db_name: str = Field("index")
    top_k: int = Field(8, ge=1, le=20)


# ───────────────────────────────────────────────────────────────────── #
# Endpoints
# ───────────────────────────────────────────────────────────────────── #
@app.post("/process", summary="Build / refresh a FAISS DB")
async def process_endpoint(req: ProcessRequest, bg: BackgroundTasks):
    logger.info("Received /process request: %s", req.dict())
    
    # Check if keyword agent is requested but not available
    if req.use_keyword_agent and not KEYWORD_AGENTS_AVAILABLE:
        logger.warning("Keyword agent requested but not available - falling back to standard processing")
        req.use_keyword_agent = False
    
    # Convert agent params
    agent_params = None
    if req.agent_params:
        agent_params = {
            "init": {
                "model": req.agent_params.model,
                "max_batch_size": req.agent_params.max_batch_size
            },
            "process": {
                "max_clusters": req.agent_params.max_clusters,
                "optimize_boolean": req.agent_params.optimize_boolean
            }
        }
    
    bg.add_task(
        run_pipeline,
        keywords=req.keywords,
        db_name=req.db_name,
        max_papers=req.max_papers,
        include_fulltext=req.fulltext,
        use_keyword_agent=req.use_keyword_agent,
        agent_params=agent_params,
    )
    return {"status": "processing", "db_name": req.db_name}


@app.post("/ask", summary="Answer a question using an existing DB")
async def ask_endpoint(req: AskRequest):
    logger.info("Received /ask request: %s", req.dict())
    try:
        engine = PubMedQAEngine(db_name=req.db_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    result = engine.answer(req.question, top_k=req.top_k)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


# Add endpoint to list available keyword agents
@app.get("/keyword_agents", summary="List available keyword processing agents")
async def list_keyword_agents():
    if KEYWORD_AGENTS_AVAILABLE:
        return {
            "status": "available",
            "agents": [
                {
                    "name": "OpenAIClusteringAgent",
                    "description": "Uses OpenAI to cluster keywords semantically",
                    "parameters": {
                        "model": "Model to use (default: gpt-4o)",
                        "max_batch_size": "Maximum keywords per URL (default: 50)",
                        "max_clusters": "Maximum number of clusters (default: 10)",
                        "optimize_boolean": "Use optimized boolean logic (default: true)"
                    }
                }
            ]
        }
    else:
        return {
            "status": "unavailable",
            "message": "Keyword processing agents are not installed"
        }
