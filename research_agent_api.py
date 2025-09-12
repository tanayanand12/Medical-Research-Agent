from typing import Optional, Dict, Any, List
import os
import requests
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import openai
from orchestrator import Orchestrator
from aggregator import Aggregator
import logging
from unicode_safe_logging import configure_all_loggers
from client_persona_model import *

# Configure all loggers to handle unicode safely
configure_all_loggers()

# Initialize logging
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
    description="API for medical research question answering with domain filtering",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    question: str
    model_id: str = "randy-data-testing"
    db_name: str = "index"
    top_k: int = 5
    clinical_trials_top_k: int = 10
    fda_top_k: int = 10
    max_trials: int = 25
    context: Dict[str, Any] = {
        "model_id": "randy-data-testing",
        "db_name": "index",
        "top_k": 5,
        "clinical_trials_top_k": 10,
        "fda_top_k": 5,
        "max_trials": 25
    }
    local: bool = True
    pubmed: bool = False
    clinical_trials: bool = False
    fda: bool = False
    release: bool = True

class ResearchAgent:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        openai.api_key = self.api_key
        
        self.orchestrator = Orchestrator()
        self.aggregator = Aggregator(None)
        logger.info("Research agent initialized with domain filtering support")

    async def process_query(self, request: QueryRequest) -> Dict[str, Any]:
        try:
            # Prepare context with all necessary parameters
            self.aggregator.model_id = request.model_id
            context = {
                "model_id": request.model_id,
                "db_name": request.db_name,
                "top_k": request.top_k,
                "clinical_trials_top_k": request.clinical_trials_top_k,
                "fda_top_k": request.fda_top_k,
                "max_trials": request.max_trials,
                "release": request.release,
                **request.context,
                "enabled_agents": {
                    "local": request.local,
                    "pubmed": request.pubmed,
                    "clinical_trials": request.clinical_trials,
                    "fda": request.fda
                }
            }
            
            logger.info(f"Processing query: {request.question}")

            logger.info(f"Tracking question to Client Persona Tracker: {request.question}")
            update_persona_data = UpdatePersonaRequest(
                uid=request.model_id,
                question=request.question,
                answer=None,  # No answer yet, will be updated later
                tone_preference=None,  # No tone preference specified
                topics_of_interest=None,  # No topics of interest specified
                metadata={
                    "source": "research_agent",
                    "context": context
                }
            )
            request_url = f"{ClientPersonaTrackerData().base_url}{ClientPersonaTrackerData().update_endpoint}"
            logger.info(f"Sending update request to {request_url} with data: {update_persona_data.model_dump()}")
            response = requests.post(
                request_url,
                json=update_persona_data.model_dump(),
                headers={"Content-Type": "application/json"}
            )
            if response.status_code != 200:
                logger.error(f"Failed to update persona data: {response.status_code} - {response.text}")
                # we don't raise an exception here as this is an internal bug and we don't want user
                # to get affected by this.
            else:
                logger.info(f"Successfully updated persona data for UID: {update_persona_data.uid}")
            
            
            # Get responses from agents asynchronously
            agent_responses = await self.orchestrator.process_query_async(request.question, context)
            
            # Aggregate responses with fallback support
            final_result = self.aggregator.aggregate(agent_responses)
            
            # Log fallback usage if applicable
            if final_result.get("fallback_used", False):
                logger.info(f"Fallback mechanism used. Reason: {final_result.get('fallback_reason')}")

            # serialize final result to str
            final_result_str = str(final_result)
            logger.info(f"Final result: {final_result_str}")

            # Update persona data with the final answer
            update_persona_data.answer = final_result.get("answer", "")
            update_persona_data.tone_preference = final_result.get("tone_preference", "")
            update_persona_data.topics_of_interest = final_result.get("topics_of_interest", [])
            logger.info(f"Updating persona data with final answer: {update_persona_data.answer}")

            response = requests.post(
                request_url,
                json=update_persona_data.model_dump(),
                headers={"Content-Type": "application/json"}
            )
            if response.status_code != 200:
                logger.error(f"Failed to update persona data with final answer: {response.status_code} - {response.text}")
            else:
                logger.info(f"Successfully updated persona data with final answer for UID: {update_persona_data.uid}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

# Initialize research agent
agent = ResearchAgent()

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """
    Process a medical research query with domain filtering support
    """
    return await agent.process_query(request)

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

def main():
    """Run the FastAPI server using uvicorn"""
    logger.info("Starting Medical Research Agent API server")
    uvicorn.run(
        "research_agent_api:app",
        host="0.0.0.0",
        port=9923,
        reload=False
    )

if __name__ == "__main__":
    main()
