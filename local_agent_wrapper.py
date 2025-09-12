from typing import Dict, Any, Optional
import os
from agent_base import AgentBase
from local_agent.rag_module import RAGModule
from local_agent.vectorization import VectorizationModule
from local_agent.faiss_db_manager import FaissVectorDB
from local_agent.gcp_storage_adapter import GCPStorageAdapter

class LocalAgent(AgentBase):
    def __init__(self):
        self.vectorizer = VectorizationModule()
        self.vector_db = FaissVectorDB()
        self.rag = RAGModule()
        self.gcp_storage = GCPStorageAdapter(
            bucket_name="intraintel-cloudrun-clinical-volume",
            credentials_path="local_agent/service_account_credentials.json"
        )
        
    def get_summary(self) -> str:
        return "Medical research RAG system that answers queries based on academic papers"
        
    def query(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            model_id = context.get('model_id', 'medical_papers')
            top_k = context.get('top_k', 5)
            
            # Download and load index
            index_path = os.path.join("gcp-indexes", model_id)
            if not os.path.exists(index_path):
                if not self.gcp_storage.download_index_using_model_id_for_local(model_id, index_path):
                    raise ValueError(f"Failed to download index: {model_id}")
            
            if not self.vector_db.load(f"gcp-indexes/{model_id}"):
                raise ValueError(f"Failed to load index: {model_id}")
            
            # Process query
            query_embedding = self.vectorizer.embed_query(question)
            results, _ = self.vector_db.similarity_search(query_embedding, k=top_k)
            documents = self.vector_db.get_langchain_documents(results)
            
            # Generate answer
            response = self.rag.generate_answer(question, documents)
            
            return {
                "answer": response["answer"],
                "citations": response["citations"],
                "confidence": 0.8
            }
            
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "citations": [],
                "confidence": 0.0
            }
