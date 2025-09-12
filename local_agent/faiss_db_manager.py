# faiss_db_manager.py
import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Any, Optional, Tuple
import logging
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class FaissVectorDB:
    """FAISS vector database manager for document retrieval."""
    
    def __init__(self, dimension: int = 1536):
        """
        Initialize the FAISS vector database.
        
        Args:
            dimension: Dimension of the vectors to be stored
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
        self.documents = []  # Store document metadata
        self.is_populated = False
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of document dictionaries with embeddings
            
        Returns:
            Success status
        """
        if not documents:
            logger.warning("No documents provided to add to the index")
            return False
        
        try:
            # Extract embeddings and convert to numpy array
            embeddings = [doc['embedding'] for doc in documents if 'embedding' in doc]
            if not embeddings:
                logger.warning("No embeddings found in documents")
                return False
            
            embeddings_array = np.array(embeddings).astype('float32')
            
            # Add to FAISS index
            self.index.add(embeddings_array)
            
            # Store document information without the embedding to save memory
            for doc in documents:
                doc_copy = doc.copy()
                if 'embedding' in doc_copy:
                    del doc_copy['embedding']  # Don't store embedding twice
                self.documents.append(doc_copy)
            
            self.is_populated = True
            logger.info(f"Added {len(documents)} documents to FAISS index")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to FAISS index: {e}")
            return False
    
    def similarity_search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5, 
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Perform similarity search in the vector database.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            metadata_filter: Optional filter for document metadata
            
        Returns:
            Tuple of (matching documents, similarity scores)
        """
        if not self.is_populated:
            logger.warning("Index is empty. No documents to search.")
            return [], []
        
        try:
            query_vector = np.array([query_embedding]).astype('float32')
            
            # Search the index
            distances, indices = self.index.search(query_vector, k * 4)  # Get more results for filtering
            distances = distances[0]
            indices = indices[0]
            
            # Filter results if needed
            results = []
            scores = []
            
            for i, idx in enumerate(indices):
                if idx != -1 and idx < len(self.documents):  # Valid index
                    doc = self.documents[idx]
                    
                    # Apply metadata filtering if specified
                    if metadata_filter:
                        include = True
                        for key, value in metadata_filter.items():
                            if key in doc and doc[key] != value:
                                include = False
                                break
                        
                        if not include:
                            continue
                    
                    results.append(doc)
                    scores.append(float(distances[i]))
                    
                    if len(results) >= k:
                        break
            
            return results, scores
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            return [], []
    
    def save(self, path: str) -> bool:
        """
        Save the vector database to disk.
        
        Args:
            path: Path to save the database
            
        Returns:
            Success status
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save the FAISS index
            faiss.write_index(self.index, f"{path}.index")
            
            # Save the documents metadata
            with open(f"{path}.documents", 'wb') as f:
                pickle.dump(self.documents, f)
            
            logger.info(f"Saved vector database to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving vector database: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """
        Load the vector database from disk.
        
        Args:
            path: Path to load the database from
            
        Returns:
            Success status
        """
        try:
            # Load the FAISS index
            self.index = faiss.read_index(f"{path}.index")
            
            # Load the documents metadata
            with open(f"{path}.documents", 'rb') as f:
                self.documents = pickle.load(f)
            
            self.is_populated = len(self.documents) > 0
            logger.info(f"Loaded vector database from {path} with {len(self.documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector database: {e}")
            return False
    
    def get_langchain_documents(self, results: List[Dict[str, Any]]) -> List[Document]:
        """
        Convert results to LangChain document format.
        
        Args:
            results: List of document dictionaries
            
        Returns:
            List of LangChain documents
        """
        documents = []
        
        for doc in results:
            content = doc.get('text', '')
            metadata = {k: v for k, v in doc.items() if k != 'text'}
            
            documents.append(Document(
                page_content=content,
                metadata=metadata
            ))
        
        return documents