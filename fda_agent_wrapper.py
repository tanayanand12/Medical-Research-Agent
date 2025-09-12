from typing import Dict, Any, Optional
import os
import logging
from datetime import datetime
from agent_base import AgentBase

# Import your FDA RAG pipeline
from FDA_agent.fda_rag_pipeline import FdaRAGPipeline

# Configure logging with UTF-8 encoding handling
logger = logging.getLogger(__name__)

# Set environment variable for better Windows console encoding
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

class FDAAgent(AgentBase):
    """
    Agent wrapper for FDA RAG system that answers queries 
    based on FDA drug labels, adverse events, and regulatory data.
    """
    
    def __init__(self):
        """Initialize the FDA agent with its pipeline."""
        try:
            # Initialize OpenAI client
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in environment variables")
                
            from openai import OpenAI #type: ignore
            self.openai_client = OpenAI(api_key=api_key)
            
            # Initialize the FDA RAG pipeline
            self.pipeline = FdaRAGPipeline(
                openai_client=self.openai_client,
                model_name="gpt-4-turbo",
                embedding_model="text-embedding-ada-002",
                max_records=300,
                chunk_size=10000,
                chunk_overlap=400,
                max_context_length=8000
            )
            
            logger.info("FDA agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize FDA agent: {str(e)}")
            raise
        
    def get_summary(self) -> str:
        """Return a summary of the agent's capabilities."""
        return "FDA regulatory data RAG system that answers queries based on drug labels, adverse events, recalls, and regulatory approvals"
        
    def query(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a query against the FDA knowledge base.
        
        Parameters
        ----------
        question : str
            The user's question
        context : Dict[str, Any], optional
            Additional context parameters:
            - top_k : int, number of results to retrieve (default: 10)
            - max_records : int, maximum records to process (default: 300)
            
        Returns
        -------
        Dict[str, Any]
            Response dictionary containing answer, citations, and confidence score
        """
        try:
            # Extract context parameters
            context = context or {}
            top_k = context.get('top_k', 10)
            max_records = context.get('max_records', 300)
            
            logger.info(f"Processing FDA query with top_k={top_k}, max_records={max_records}")
            
            # Update pipeline parameters if different from defaults
            if max_records != 300:
                self.pipeline.max_records = max_records
            
            start_time = datetime.now()
            
            # Process the query using the FDA pipeline
            result = self.pipeline.process_query(
                query=question,
                top_k=top_k
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"FDA query processed in {processing_time:.2f} seconds")
            
            # Check if there was an error in the result
            if not result.get("success", False):
                error_msg = result.get("error", "Unknown error in FDA pipeline")
                raise ValueError(error_msg)
            
            # Extract and format the response
            answer = result.get("answer", "")
            citations = result.get("citations", [])
            
            # If citations is not a list, try to convert or create empty list
            if not isinstance(citations, list):
                if isinstance(citations, str):
                    # If citations is a string, split it or wrap it in a list
                    citations = [citations] if citations else []
                else:
                    citations = []
            
            # Calculate confidence based on result quality
            confidence = self._calculate_confidence(result, processing_time)
            
            return {
                "answer": answer,
                "citations": citations,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error in FDA agent: {str(e)}", exc_info=True)
            return {
                "answer": f"Error processing FDA query: {str(e)}",
                "citations": [],
                "confidence": 0.0
            }
    
    def _calculate_confidence(self, result: Dict[str, Any], processing_time: float) -> float:
        """
        Calculate confidence score based on result quality and processing metrics.
        
        Parameters
        ----------
        result : Dict[str, Any]
            The result from the FDA pipeline
        processing_time : float
            Time taken to process the query
            
        Returns
        -------
        float
            Confidence score between 0.0 and 1.0
        """
        try:
            base_confidence = 0.82  # Base confidence for FDA agent (regulatory data is typically high quality)
            
            # Adjust based on answer length (longer answers might be more comprehensive)
            answer_length = len(result.get("answer", ""))
            if answer_length > 800:
                base_confidence += 0.05
            elif answer_length < 150:
                base_confidence -= 0.1
            
            # Adjust based on number of citations
            citations = result.get("citations", [])
            if isinstance(citations, list) and len(citations) > 5:
                base_confidence += 0.08
            elif isinstance(citations, list) and len(citations) == 0:
                base_confidence -= 0.12
            
            # Adjust based on processing time (very fast might indicate cached/simple response)
            if processing_time < 3.0:
                base_confidence -= 0.03
            elif processing_time > 15.0:
                base_confidence -= 0.08
            
            # Adjust based on records fetched
            records_count = result.get("metadata", {}).get("records_count", 0)
            if records_count > 100:
                base_confidence += 0.05
            elif records_count < 10:
                base_confidence -= 0.1
            
            # Ensure confidence is within bounds
            return max(0.0, min(1.0, base_confidence))
            
        except Exception as e:
            logger.warning(f"Error calculating confidence: {str(e)}")
            return 0.78  # Default confidence on error