# vectorization.py
import os
import time
import logging
import numpy as np
from typing import List, Dict, Any
import backoff
from dotenv import load_dotenv
import openai  # Updated import

logger = logging.getLogger(__name__)

class VectorizationModule:
    """Module for embedding document content and queries using OpenAI embeddings."""
    
    def __init__(self, openai_api_key: str = None, model_name: str = "text-embedding-ada-002"):
        """
        Initialize the vectorization module.
        
        Args:
            openai_api_key: OpenAI API key (will use environment variable if not provided)
            model_name: Embedding model name
        """
        load_dotenv()
        self.model_name = model_name
        self.embedding_dim = 1536  # text-embedding-ada-002 returns 1536-dimensional embeddings
        
        # Get API key from arguments or environment variables
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found in arguments or environment variables")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        logger.info(f"Using OpenAI embedding model: {self.model_name}")
    
    @backoff.on_exception(
        backoff.expo,
        Exception,  # Simplified error handling
        max_tries=5,
        factor=2
    )
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get an embedding for a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding as numpy array
        """
        response = self.client.embeddings.create(
            input=[text],
            model=self.model_name
        )
        embedding = response.data[0].embedding
        return np.array(embedding)
    
    @backoff.on_exception(
        backoff.expo,
        Exception,  # Simplified error handling
        max_tries=5,
        factor=2
    )
    def get_batch_embeddings(self, texts: List[str], batch_size: int = 10) -> List[np.ndarray]:
        """
        Get embeddings for a batch of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process per API call
            
        Returns:
            List of numpy arrays, each representing a text embedding
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = self.client.embeddings.create(
                input=batch,
                model=self.model_name
            )
            
            batch_embeddings = [np.array(item.embedding) for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            if i + batch_size < len(texts):
                time.sleep(0.5)  # Brief pause between batches to respect rate limits
        
        return all_embeddings
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Embed a list of document chunks.
        
        Args:
            chunks: List of document chunk dictionaries
            
        Returns:
            List of document chunks with embeddings added
        """
        texts = [chunk['text'] for chunk in chunks]
        if not texts:
            return []
        
        try:
            embeddings = self.get_batch_embeddings(texts)
            
            for i, embedding in enumerate(embeddings):
                chunks[i]['embedding'] = embedding
            
            logger.info(f"Successfully embedded {len(chunks)} document chunks")
            return chunks
        
        except Exception as e:
            logger.error(f"Error embedding chunks: {e}")
            return []
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string.
        
        Args:
            query: Query string to embed
            
        Returns:
            Query embedding as numpy array
        """
        try:
            return self.get_embedding(query)
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return np.zeros(self.embedding_dim)