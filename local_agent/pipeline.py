# pipeline.py
import argparse
import os
import logging
from typing import List, Dict, Any
from .pdf_processor import PDFProcessor
from .vectorization import VectorizationModule
from .faiss_db_manager import FaissVectorDB
from .rag_module import RAGModule

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Pipeline for Research RAG processing and querying."""
    
    def __init__(self):
        """Initialize the RAG pipeline components."""
        self.pdf_processor = PDFProcessor()
        self.vectorizer = VectorizationModule()
        self.vector_db = FaissVectorDB()
        self.rag = RAGModule()
    
    def process_pdfs(
        self, 
        model_id: str, 
        index_name: str,
        max_chars: int = 5000,
        overlap: float = 0.2
    ) -> bool:
        """
        Process PDF files and create an index.
        
        Args:
            pdf_folder: Path to folder containing PDF files
            index_name: Name for the index
            max_chars: Maximum characters per chunk
            overlap: Overlap percentage between chunks
            
        Returns:
            Success status
        """
        try:
            from gcp_storage_adapter import GCPStorageAdapter
            logger.info("Initializing GCP Storage Adapter")
            gcp_storage = GCPStorageAdapter(
                bucket_name="intraintel-cloudrun-clinical-volume",
                credentials_path=".\service_account_credentials.json"
            )
            logger.info("GCP Storage Adapter initialized")

            logger.info(f"Downloading PDFs for model ID: {model_id}")
            pdf_folder = gcp_storage.download_pdfs_to_temp_using_model_id(model_id=model_id)

            # 1. Process PDFs
            logger.info(f"Processing PDFs in {pdf_folder}")
            # pipeline.py (continued)
            self.pdf_processor = PDFProcessor(max_char_limit=max_chars, overlap_percentage=overlap)
            chunks = self.pdf_processor.process_folder(pdf_folder)
            logger.info(f"Created {len(chunks)} chunks from PDFs")

            # 2. Create embeddings
            logger.info("Creating embeddings for chunks")
            embedded_chunks = self.vectorizer.embed_chunks(chunks)
            
            # 3. Create vector database
            logger.info("Adding documents to vector database")
            self.vector_db.add_documents(embedded_chunks)
            
            # 4. Save the index
            os.makedirs("indexes", exist_ok=True)
            index_path = os.path.join("indexes", index_name)
            self.vector_db.save(index_path)
            logger.info(f"Index saved to {index_path}")

            # 5. Upload index to GCP
            logger.info(f"Uploading index to GCS: {model_id}")
            gcs_path = f"indexes/{model_id}/{model_id}"
            if not gcp_storage.upload_index_to_model_id(model_id=model_id, index_path=index_path):
                raise ValueError(f"Failed to upload index to GCS: {gcs_path}")
            logger.info(f"Index uploaded to GCS: {gcs_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing PDFs: {e}")
            return False
    
    def query(
        self, 
        query: str, 
        index_name: str, 
        top_k: int = 5, 
        metadata_filter: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Query the index and generate an answer.
        
        Args:
            query: User query
            index_name: Name of the index to query
            top_k: Number of documents to retrieve
            metadata_filter: Optional filter for document metadata
            
        Returns:
            Dictionary with answer and citations
        """
        try:
            from .gcp_storage_adapter import GCPStorageAdapter
            # 1. Load the index from GCP
            logger.info("Initializing GCP Storage Adapter for querying")
            gcp_storage = GCPStorageAdapter(
                bucket_name="intraintel-cloudrun-clinical-volume",    
                credentials_path="service_account_credentials.json"
            )
            logger.info("GCP Storage Adapter initialized for querying")

            logger.info(f"Downloading index for model ID: {index_name}")
            index_path = os.path.join("gcp-indexes", index_name)
            logger.info(f"Index path: {index_path}")
            status = gcp_storage.download_index_using_model_id(
                model_id=index_name,
                local_path=index_path
            )
            if not status:
                raise ValueError(f"Failed to download index: {index_name}")
            logger.info(f"Index downloaded to {index_path}")

            # index_path = os.path.join("indexes", index_name)
            # if not self.vector_db.load(index_path):
            #     raise ValueError(f"Failed to load index: {index_name}")
            
            # 2. Embed the query
            query_embedding = self.vectorizer.embed_query(query)
            
            # 3. Retrieve relevant documents
            results, scores = self.vector_db.similarity_search(
                query_embedding, 
                k=top_k, 
                metadata_filter=metadata_filter
            )
            
            # 4. Convert to LangChain documents
            context_docs = self.vector_db.get_langchain_documents(results)
            
            # 5. Generate answer
            result = self.rag.generate_answer(query, context_docs)
            
            return result
            
        except Exception as e:
            logger.error(f"Error querying index: {e}")
            return {
                "answer": f"An error occurred while generating the answer: {str(e)}",
                "citations": [],
                "documents": []
            }

def main():
    """Run the pipeline from command line."""
    parser = argparse.ArgumentParser(description="Research RAG Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process PDFs and create index")
    process_parser.add_argument("--model_id", "-p", required=True, help="Model ID")
    process_parser.add_argument("--index_name", "-i", required=True, help="Name for the index")
    process_parser.add_argument("--max_chars", "-m", type=int, default=1000, help="Max characters per chunk")
    process_parser.add_argument("--overlap", "-o", type=float, default=0.2, help="Chunk overlap percentage")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the index")
    query_parser.add_argument("--query", "-q", required=True, help="Query string")
    query_parser.add_argument("--index_name", "-i", required=True, help="Name of the index to query")
    query_parser.add_argument("--top_k", "-k", type=int, default=5, help="Number of documents to retrieve")
    
    args = parser.parse_args()
    
    pipeline = RAGPipeline()
    
    if args.command == "process":
        success = pipeline.process_pdfs(
            args.model_id,
            args.model_id,
            args.max_chars,
            args.overlap
        )
        if success:
            print(f"Successfully processed PDFs and created index: {args.index_name}")
        else:
            print("Failed to process PDFs")
    
    elif args.command == "query":
        result = pipeline.query(args.query, args.index_name, args.top_k)
        print("\n=== ANSWER ===\n")
        print(result["answer"])
        print("\n=== CITATIONS ===\n")
        for citation in result["citations"]:
            print(citation)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
