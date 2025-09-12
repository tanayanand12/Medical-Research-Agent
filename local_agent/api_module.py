from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import shutil
import uvicorn
import logging
from .pdf_processor import PDFProcessor
from .vectorization import VectorizationModule
from .faiss_db_manager import FaissVectorDB
from .rag_module import RAGModule

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Medical Research RAG API",
    description="API for processing medical research papers and answering queries using RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
pdf_processor = PDFProcessor()
vectorizer = VectorizationModule()
vector_db = FaissVectorDB()
rag = RAGModule()

# Define request/response models
class QueryRequest(BaseModel):
    query: str
    model_id: str
    top_k: int = 5

class GenerateEmbeddingsRequest(BaseModel):
    model_id: str
    max_chars: int = 3000
    overlap: float = 0.2


class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]

# API endpoints
@app.post("/generate_embeddings")
async def generate_embeddings(request: GenerateEmbeddingsRequest):
    """Generate embeddings for a given model ID."""
    try:
        from .pipeline import RAGPipeline
        pipeline = RAGPipeline()

        success = pipeline.process_pdfs(
            request.model_id,
            request.model_id,
            request.max_chars,
            request.overlap
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to generate embeddings")

        return {"message": f"Embeddings generated successfully for model ID: {request.model_id}"}

    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_pdfs(
    files: List[UploadFile] = File(...),
    index_name: str = "default"
):
    """Upload PDFs and create searchable index."""
    try:
        # Create temporary directory for PDFs
        os.makedirs("temp_pdfs", exist_ok=True)
        
        # Save uploaded files
        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Only PDF files are accepted")
            
            file_path = f"temp_pdfs/{file.filename}"
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        
        # Process PDFs
        chunks = pdf_processor.process_folder("temp_pdfs")
        if not chunks:
            raise HTTPException(status_code=400, detail="No valid content extracted from PDFs")
        
        # Create embeddings
        embedded_chunks = vectorizer.embed_chunks(chunks)
        if not embedded_chunks:
            raise HTTPException(status_code=500, detail="Failed to create embeddings")
        
        # Add to vector database
        if not vector_db.add_documents(embedded_chunks):
            raise HTTPException(status_code=500, detail="Failed to add documents to vector database")
        
        # Save vector database
        os.makedirs("indexes", exist_ok=True)
        if not vector_db.save(f"indexes/{index_name}"):
            raise HTTPException(status_code=500, detail="Failed to save vector database")
        
        # Cleanup
        shutil.rmtree("temp_pdfs")
        
        return {"message": f"Successfully processed {len(files)} files and created index '{index_name}'"}
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_index(request: QueryRequest):
    """Query the RAG system for answers."""
    try:
        # Load vector database
        # if not vector_db.load(f"indexes/{request.index_name}"):
        #     raise HTTPException(status_code=404, detail=f"Index '{request.index_name}' not found")
        from .gcp_storage_adapter import GCPStorageAdapter

        logger.info("Initializing GCP Storage Adapter for querying")
        gcp_storage = GCPStorageAdapter(
            bucket_name="intraintel-cloudrun-clinical-volume",    
            credentials_path="service_account_credentials.json"
        )
        logger.info("GCP Storage Adapter initialized for querying")

        logger.info(f"Downloading index for model ID: {request.model_id}")
        index_path = os.path.join("gcp-indexes", request.model_id)
        status = gcp_storage.download_index_using_model_id(
            model_id=request.model_id,
            local_path=index_path
        )
        if not status:
            raise ValueError(f"Failed to download index: {request.model_id}")
        logger.info(f"Index downloaded to {index_path}")

        # Load vector database
        if not vector_db.load(f"gcp-indexes/{request.model_id}"):
            raise HTTPException(status_code=404, detail=f"Index '{request.model_id}' not found")

        # Create query embedding
        query_embedding = vectorizer.embed_query(request.query)
        
        # Search for relevant documents
        results, scores = vector_db.similarity_search(
            query_embedding,
            k=request.top_k
        )
        
        if not results:
            raise HTTPException(status_code=404, detail="No relevant documents found")
        
        # Convert to LangChain documents
        documents = vector_db.get_langchain_documents(results)
        
        # Generate answer
        response = rag.generate_answer(request.query, documents)
        
        return {
            "answer": response["answer"],
            "citations": response["citations"]
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/indexes")
async def list_indexes():
    """List available indexes."""
    try:
        indexes = []
        if os.path.exists("indexes"):
            for file in os.listdir("indexes"):
                if file.endswith(".index"):
                    indexes.append(file[:-6])  # Remove .index extension
        return {"indexes": indexes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/indexes/{index_name}")
async def delete_index(index_name: str):
    """Delete an index."""
    try:
        index_path = f"indexes/{index_name}"
        if os.path.exists(f"{index_path}.index"):
            os.remove(f"{index_path}.index")
            os.remove(f"{index_path}.documents")
            return {"message": f"Index '{index_name}' deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
