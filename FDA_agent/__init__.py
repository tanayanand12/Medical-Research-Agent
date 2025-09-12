"""
FDA RAG Library
A comprehensive library for querying and analyzing FDA and clinical trial data using RAG (Retrieval-Augmented Generation).

Example usage:
    from fda_agent import FdaRAG
    
    # Initialize with OpenAI API key
    rag = FdaRAG(api_key="your-openai-api-key")
    
    # Process a query
    result = rag.query("What are the recent adverse events for aspirin?")
    print(result.answer)
"""

__version__ = "1.0.0"
__author__ = "FDA RAG Team"
__email__ = "contact@example.com"

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import main components
from .fda_fetcher import *
from .fda_chunker import *
from .fda_context_extractor import *
from .fda_rag_module import *
from .fda_rag_pipeline import *
from .clinical_trials_context_extractor import *
from .clinical_trials_vectorizer import *

# Export main classes
__all__ = [
    'FdaRAG',
    'FdaRAGPipeline',
    'FdaFetcherAgent',
    'FdaChunker',
    'ClinicalTrialsVectorizer',
    'FdaContextExtractor',
    'FdaRAGModule',
    'create_rag_pipeline',
    'RAGResult'
]

class RAGResult:
    """Wrapper class for RAG query results with easy attribute access."""
    
    def __init__(self, result_dict: Dict[str, Any]):
        self._data = result_dict
        self.success = result_dict.get('success', False)
        self.answer = result_dict.get('answer', '')
        self.citations = result_dict.get('citations', [])
        self.metadata = result_dict.get('metadata', {})
        self.records = result_dict.get('records', [])
        self.error = result_dict.get('error', None)
        
    def __repr__(self):
        return f"RAGResult(success={self.success}, citations_count={len(self.citations)})"
    
    def to_dict(self):
        """Convert back to dictionary format."""
        return self._data


class FdaRAG:
    """
    High-level interface for the FDA RAG library.
    
    This class provides a simple API for querying FDA data
    without needing to understand the internal pipeline structure.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = "gpt-4-turbo",
                 embedding_model: str = "text-embedding-ada-002",
                 max_records: int = 300,
                 chunk_size: int = 10000,
                 **kwargs):
        """
        Initialize the FDA RAG system.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model_name: OpenAI model for generation
            embedding_model: Model for embeddings
            max_records: Maximum records to fetch per query
            chunk_size: Size of text chunks
            **kwargs: Additional pipeline configuration
        """
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Pass api_key parameter or set OPENAI_API_KEY environment variable."
            )
        
        # Set environment variable if provided as parameter
        if api_key and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Initialize OpenAI client
        try:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
        
        # Initialize pipeline
        self.pipeline = FdaRAGPipeline(
            openai_client=self.openai_client,
            model_name=model_name,
            embedding_model=embedding_model,
            max_records=max_records,
            chunk_size=chunk_size,
            chunk_overlap=kwargs.get('chunk_overlap', 400),
            max_context_length=kwargs.get('max_context_length', 8000)
        )
        
        logger.info("FDA RAG initialized successfully")
    
    def query(self, 
              question: str, 
              top_k: int = 10,
              return_raw: bool = False) -> RAGResult:
        """
        Query the FDA database with a research question.
        
        Args:
            question: Question about FDA data (drug labels, adverse events, etc.)
            top_k: Number of relevant chunks to use
            return_raw: If True, return raw dictionary instead of RAGResult
            
        Returns:
            RAGResult object with answer and metadata
            
        Example:
            result = rag.query("What are the common side effects of aspirin?")
            print(result.answer)
            print(f"Found {len(result.citations)} relevant citations")
        """
        if not question or len(question.strip()) < 5:
            raise ValueError("Question must be at least 5 characters long")
        
        logger.info(f"Processing query: {question[:100]}...")
        
        result = self.pipeline.process_query(query=question, top_k=top_k)
        
        if return_raw:
            return result
        
        return RAGResult(result)

def create_rag_pipeline(api_key: Optional[str] = None, **kwargs) -> FdaRAG:
    """
    Factory function to create an FDA RAG pipeline.
    
    Args:
        api_key: OpenAI API key
        **kwargs: Additional configuration options
        
    Returns:
        Configured FdaRAG instance
    """
    return FdaRAG(api_key=api_key, **kwargs)


# Convenience function for quick queries
def quick_query(question: str, api_key: Optional[str] = None) -> str:
    """
    Perform a quick query without creating a persistent pipeline.
    
    Args:
        question: Question about FDA data
        api_key: OpenAI API key (optional if env var set)
        
    Returns:
        Answer string
        
    Example:
        answer = quick_query("What are the recent recalls for acetaminophen?")
    """
    rag = FdaRAG(api_key=api_key)
    result = rag.query(question)
    return result.answer if result.success else f"Error: {result.error}"
