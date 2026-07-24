"""
rag_engine — Phase 6: Central RAG Engine.

Provides semantic chunking, LLM-agnostic embeddings, hybrid dense+sparse
retrieval with RRF fusion, and cross-encoder reranking.

Public API::

    from rag_engine.chunker import SemanticChunker
    from rag_engine.embedder import Embedder
    from rag_engine.sparse_index import BM25Index
    from rag_engine.dense_index import DenseIndex
    from rag_engine.hybrid_retriever import HybridRetriever
    from rag_engine.reranker import Reranker
    from rag_engine.mcp_rag_tool import RAGTool
"""

from rag_engine.chunker import SemanticChunker
from rag_engine.embedder import Embedder
from rag_engine.sparse_index import BM25Index
from rag_engine.dense_index import DenseIndex
from rag_engine.hybrid_retriever import HybridRetriever
from rag_engine.reranker import Reranker
from rag_engine.mcp_rag_tool import RAGTool

__all__ = [
    "SemanticChunker",
    "Embedder",
    "BM25Index",
    "DenseIndex",
    "HybridRetriever",
    "Reranker",
    "RAGTool",
]
