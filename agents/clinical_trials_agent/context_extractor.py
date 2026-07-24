"""
context_extractor.py — Phase 7: Clinical trials context extraction.

Direct REUSE of the legacy ``ClinicalTrialsContextExtractor`` from
``agentic-pipeline-clinical/clinical_trials_agent1/clinical_trials_context_extractor.py``.

No LLM calls — pure filtering, deduplication, prioritisation, and
formatting logic.  Unchanged from legacy except for log formatting
(f-strings replaced with %-style for consistency with the target
codebase logging conventions).
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np  # type: ignore

logger = logging.getLogger(__name__)


class ClinicalTrialsContextExtractor:
    """Extract and format relevant context from clinical trial chunks for RAG.

    Handles context length limits and provides structured context for
    LLM consumption.

    Parameters
    ----------
    max_context_length : int
        Maximum length of context in characters.
    min_similarity_threshold : float
        Minimum similarity score to include a chunk.
    """

    def __init__(
        self,
        max_context_length: int = 8000,
        min_similarity_threshold: float = 0.3,
    ) -> None:
        self.max_context_length = max_context_length
        self.min_similarity_threshold = min_similarity_threshold

    def filter_chunks_by_similarity(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter chunks based on similarity threshold.

        Parameters
        ----------
        chunks : list[dict]
            List of chunks with similarity scores.

        Returns
        -------
        list[dict]
            Filtered list of chunks.
        """
        filtered_chunks = [
            chunk
            for chunk in chunks
            if chunk.get("similarity_score", 0.0) >= self.min_similarity_threshold
        ]

        logger.info(
            "Filtered %d chunks to %d chunks above similarity threshold %s",
            len(chunks),
            len(filtered_chunks),
            self.min_similarity_threshold,
        )
        return filtered_chunks

    def deduplicate_studies(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deduplicate chunks by study ID, keeping the highest-scoring chunk per study.

        Parameters
        ----------
        chunks : list[dict]
            List of chunks.

        Returns
        -------
        list[dict]
            Deduplicated list of chunks.
        """
        study_chunks: Dict[str, Dict[str, Any]] = {}

        for chunk in chunks:
            study_id = chunk.get("study_id", "unknown")
            current_score = chunk.get("similarity_score", 0.0)

            if (
                study_id not in study_chunks
                or current_score
                > study_chunks[study_id].get("similarity_score", 0.0)
            ):
                study_chunks[study_id] = chunk

        deduplicated = list(study_chunks.values())
        logger.info(
            "Deduplicated %d chunks to %d unique studies",
            len(chunks),
            len(deduplicated),
        )
        return deduplicated

    def prioritize_chunk_types(
        self, chunks: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """Prioritise chunks based on query type and chunk content type.

        Parameters
        ----------
        chunks : list[dict]
            List of chunks.
        query : str
            User query.

        Returns
        -------
        list[dict]
            Prioritised list of chunks.
        """
        query_lower = query.lower()

        # Define priority weights for different chunk types based on query content
        type_priorities: Dict[str, float] = {
            "overview": 1.0,
            "intervention": 1.0,
            "outcomes": 1.0,
            "eligibility": 0.8,
            "detailed_description": 0.6,
            "location": 0.4,
        }

        # Boost priorities based on query keywords
        if any(
            keyword in query_lower
            for keyword in ["treatment", "intervention", "therapy", "drug"]
        ):
            type_priorities["intervention"] = 1.5

        if any(
            keyword in query_lower
            for keyword in ["outcome", "result", "endpoint", "efficacy"]
        ):
            type_priorities["outcomes"] = 1.5

        if any(
            keyword in query_lower
            for keyword in ["eligibility", "criteria", "inclusion", "exclusion"]
        ):
            type_priorities["eligibility"] = 1.5

        if any(
            keyword in query_lower
            for keyword in ["location", "where", "country", "hospital"]
        ):
            type_priorities["location"] = 1.5

        # Apply priority weighting to similarity scores
        for chunk in chunks:
            chunk_type = chunk.get("chunk_type", "").split("_")[0]
            priority_weight = type_priorities.get(chunk_type, 1.0)
            original_score = chunk.get("similarity_score", 0.0)
            chunk["weighted_score"] = original_score * priority_weight

        # Sort by weighted score
        prioritized_chunks = sorted(
            chunks, key=lambda x: x.get("weighted_score", 0.0), reverse=True
        )

        logger.info("Applied chunk type prioritization based on query content")
        return prioritized_chunks

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chunks into a structured context string.

        Parameters
        ----------
        chunks : list[dict]
            List of relevant chunks.

        Returns
        -------
        str
            Formatted context string.
        """
        context_parts: List[str] = []
        current_length = 0

        for i, chunk in enumerate(chunks):
            study_id = chunk.get("study_id", "Unknown")
            chunk_type = chunk.get("chunk_type", "unknown")
            similarity_score = chunk.get("similarity_score", 0.0)

            chunk_header = (
                f"\n--- Clinical Trial {i + 1}: {study_id} "
                f"({chunk_type}, relevance: {similarity_score:.3f}) ---\n"
            )
            chunk_content = chunk.get("content", "")

            chunk_text = chunk_header + chunk_content

            # Check if adding this chunk would exceed length limit
            if current_length + len(chunk_text) > self.max_context_length:
                if current_length == 0:
                    # First chunk is too long — truncate it
                    available_space = (
                        self.max_context_length - len(chunk_header) - 100
                    )
                    truncated_content = (
                        chunk_content[:available_space] + "... [truncated]"
                    )
                    chunk_text = chunk_header + truncated_content
                    context_parts.append(chunk_text)
                    current_length += len(chunk_text)
                break

            context_parts.append(chunk_text)
            current_length += len(chunk_text)

        context = "\n".join(context_parts)

        logger.info(
            "Formatted context from %d chunks, total length: %d characters",
            len(context_parts),
            len(context),
        )
        return context

    def extract_study_metadata(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract study metadata for citation purposes.

        Parameters
        ----------
        chunks : list[dict]
            List of relevant chunks.

        Returns
        -------
        list[dict]
            List of study metadata dictionaries.
        """
        studies: Dict[str, Dict[str, Any]] = {}

        for chunk in chunks:
            study_id = chunk.get("study_id", "Unknown")
            if study_id not in studies:
                content = chunk.get("content", "")

                # Try to extract title from content
                title = "Unknown Title"
                if "Study:" in content:
                    lines = content.split("\n")
                    for line in lines:
                        if line.strip().startswith("Study:"):
                            title = line.replace("Study:", "").strip()
                            break

                studies[study_id] = {
                    "study_id": study_id,
                    "title": title,
                    "similarity_score": chunk.get("similarity_score", 0.0),
                    "chunk_type": chunk.get("chunk_type", "unknown"),
                    "section": chunk.get("section", "unknown"),
                }

        study_list = sorted(
            studies.values(),
            key=lambda x: x["similarity_score"],
            reverse=True,
        )

        logger.info("Extracted metadata for %d unique studies", len(study_list))
        return study_list

    def extract_context(
        self,
        query: str,
        query_embedding: np.ndarray,
        chunk_embeddings: Dict[str, np.ndarray],
        vectorizer: Any,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Main method to extract relevant context from chunks.

        Parameters
        ----------
        query : str
            User query.
        query_embedding : np.ndarray
            Query embedding vector.
        chunk_embeddings : dict
            Dictionary of chunk embeddings.
        vectorizer
            Vectorizer instance for similarity computation.
        top_k : int
            Number of top chunks to consider.

        Returns
        -------
        dict
            Dictionary containing context and metadata.
        """
        try:
            # Find most similar chunks
            similar_chunks = vectorizer.find_most_similar_chunks(
                query_embedding,
                chunk_embeddings,
                top_k=top_k,
            )

            if not similar_chunks:
                logger.warning("No similar chunks found")
                return {
                    "context": "No relevant clinical trial data found for this query.",
                    "studies": [],
                    "chunk_count": 0,
                }

            # Filter by similarity threshold
            filtered_chunks = self.filter_chunks_by_similarity(similar_chunks)

            if not filtered_chunks:
                logger.warning("No chunks passed similarity threshold")
                return {
                    "context": "No sufficiently relevant clinical trial data found for this query.",
                    "studies": [],
                    "chunk_count": 0,
                }

            # Prioritize chunks based on query
            prioritized_chunks = self.prioritize_chunk_types(filtered_chunks, query)

            # Format context
            context = self.format_context(prioritized_chunks)

            # Extract study metadata
            study_metadata = self.extract_study_metadata(prioritized_chunks)

            return {
                "context": context,
                "studies": study_metadata,
                "chunk_count": len(prioritized_chunks),
                "total_similarity_scores": [
                    chunk.get("similarity_score", 0.0)
                    for chunk in prioritized_chunks
                ],
            }

        except Exception as e:
            logger.error("Error extracting context: %s", e)
            return {
                "context": f"Error processing clinical trial data: {e}",
                "studies": [],
                "chunk_count": 0,
            }
