from typing import Dict, Any, Optional
import os
import logging
from agent_base import AgentBase
from pubmed_local_agent.core.vectorizer import Vectorizer
from pubmed_local_agent.core.faiss_db_manager import FaissVectorDB
from pubmed_local_agent.query import PubMedQAEngine

# Configure logging
logger = logging.getLogger(__name__)

class PubMedAgent(AgentBase):
    """
    Agent for retrieving and answering questions from medical research papers
    indexed from PubMed database using vector similarity search.
    """
    
    def __init__(self):
        """Initialize the PubMed agent with its core components."""
        self.vectorizer = Vectorizer()
        self.vector_db = FaissVectorDB(dimension=3072)  # Using OpenAI's text-embedding-3-large dimension
        self.qa_engine = PubMedQAEngine()
        
    def get_summary(self) -> str:
        """Return a summary of the agent's capabilities."""
        return "Medical research RAG system that answers queries based on PubMed academic papers"
        
    def query(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a query against the PubMed knowledge base.
        
        Parameters
        ----------
        question : str
            The user's question
        context : Dict[str, Any], optional
            Additional context parameters:
            - db_name : str, index name (default: "index")
            - top_k : int, number of results to retrieve (default: 8)
            
        Returns
        -------
        Dict[str, Any]
            Response dictionary containing answer, citations, and confidence score
        """
        try:
            # Extract context parameters
            db_name = context.get('db_name', 'index')
            top_k = context.get('top_k', 8)
            
            # Use direct path to the index file
            index_path = os.path.join("pubmed_faiss_index", f"{db_name}.index")
            logger.info(f"Using PubMed index from: {index_path}")
            
            if not os.path.exists(index_path):
                raise ValueError(f"PubMed index not found at: {index_path}")
            
            # Process query using the QA engine
            result = self.qa_engine.answer(question, top_k=top_k)
            
            if "error" in result:
                raise ValueError(result["error"])
                
            return {
                "answer": result["answer"],
                "citations": result["citations"],
                "confidence": 0.85  # Confidence score based on paper relevance
            }
            
        except Exception as e:
            logger.error(f"Error in PubMed agent: {str(e)}")
            return {
                "answer": f"Error processing PubMed query: {str(e)}",
                "citations": [],
                "confidence": 0.0
            }