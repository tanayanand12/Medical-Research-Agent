import logging
import time
from typing import Dict, Any, List

from .fda_fetcher import FdaFetcherAgent
from .fda_chunker import FdaChunker
from .fda_context_extractor import FdaContextExtractor
from .clinical_trials_vectorizer import ClinicalTrialsVectorizer
from .fda_rag_module import FdaRAGModule

# ----------------------------------------------------------------------------
# Logger Configuration
# ----------------------------------------------------------------------------
logger = logging.getLogger("fda_rag_pipeline")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "[%(levelname)s] %(asctime)s - %(filename)s:%(lineno)d - %(message)s"
    )
    ch.setFormatter(fmt)
    logger.addHandler(ch)

class FdaRAGPipeline:
    """
    Orchestrator: fetch → chunk → embed → retrieve → generate.
    """

    def __init__(
        self,
        openai_client=None,
        model_name: str = "gpt-4o",
        embedding_model: str = "text-embedding-ada-002",
        max_records: int = 300,
        chunk_size: int = 10_000,
        chunk_overlap: int = 400,
        max_context_length: int = 8_000,
    ):
        self.fetcher = FdaFetcherAgent(openai_client=openai_client, model=model_name)
        self.chunker = FdaChunker(max_chunk_size=chunk_size, overlap=chunk_overlap)
        self.vectorizer = ClinicalTrialsVectorizer(openai_model=embedding_model)
        self.ctx = FdaContextExtractor(max_context_length=max_context_length)
        self.rag = FdaRAGModule(model_name=model_name)
        self.max_records = max_records
        logger.info("Initialized FdaRAGPipeline")

    def process_query(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Execute end-to-end RAG for a given query.

        Returns a dict with keys:
          - success: bool
          - answer: str, if success
          - citations: List[str]
          - metadata: timing & token info
        """
        start_time = time.time()
        logger.info("Starting RAG pipeline for query: '%s'", query)

        # 1. Fetch records
        fetch_res = self.fetcher.analyze_user_query(query)
        if not fetch_res.get("success"):
            logger.error("Fetch step failed: %s", fetch_res.get("error"))
            return {"success": False, "error": fetch_res.get("error")}

        records = fetch_res["data"]["records"][: self.max_records]
        logger.info("Fetched %d records", len(records))

        # 2. Chunk records
        chunks = self.chunker.chunk_records(records)
        logger.info("Generated %d chunks", len(chunks))

        # 3. Embed chunks
        embeds = self.vectorizer.embed_chunks(chunks)
        logger.info("Embedded chunks into %d vectors", len(embeds))

        # 4. Embed query
        q_vec = self.vectorizer.embed_query(query)
        logger.info("Generated embedding for query")

        # 5. Compute similarity & select top_k
        sims = self.vectorizer.compute_similarity(q_vec, embeds)
        for cid, score in sims.items():
            embeds[cid]["metadata"]["similarity_score"] = score

        sorted_chunks = sorted(
            embeds.values(),
            key=lambda x: x["metadata"]["similarity_score"],
            reverse=True,
        )[:top_k]
        top_chunks = [c["metadata"] for c in sorted_chunks]
        logger.info("Selected top %d relevant chunks", len(top_chunks))

        # 6. Build context
        context = self.ctx.format_context(top_chunks)
        logger.info("Formatted context for RAG generation")

        # 7. Generate answer
        rag_res = self.rag.generate(query, context, top_chunks)
        rag_res["processing_time"] = time.time() - start_time
        rag_res["records"] = top_chunks

        logger.info("RAG pipeline complete (%.2fs)", rag_res["processing_time"])
        return rag_res
