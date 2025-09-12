"""
query.py
~~~~~~~~

End‑to‑end Q&A engine:

1. Load (or create) a FaissVectorDB.
2. Embed the user question via OpenAI `text-embedding-3-large`.
3. Retrieve top‑K chunks (L2) and build IEEE‑style citations.
4. Construct a RAG prompt that *requires* citations.
5. Call OpenAI ChatCompletion and return answer + citations list.

Assumes:
    • pubmed_faiss_index/index.index + .meta.pkl exist
    • core.vectorizer.Vectorizer and core.faiss_db_manager.FaissVectorDB
      are importable.

Run demo:
    python query.py --question "What new evidence links APOE4 to Alzheimer’s?"
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from .core.faiss_db_manager import FaissVectorDB
from .core.vectorizer import Vectorizer, EMBED_DIM  # reuse embedding + tokenizer

# ────────────────────────────────────────────────────────────────────────────────
# Logging config
# ────────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# ────────────────────────────────────────────────────────────────────────────────
# Helper: build IEEE‑style citation from metadata
# ────────────────────────────────────────────────────────────────────────────────
def _format_citation(meta: Dict, idx: int) -> str:
    authors = meta.get("paper_authors", "")
    if authors.count(",") >= 1:  # use et al. if >2 authors
        first = authors.split(",")[0]
        authors = f"{first} et al."
    return (
        f"[{idx}] {authors}. {meta.get('paper_title', '')}. "
        f"{meta.get('paper_journal', '')}, {meta.get('paper_year', '')}. "
        f"PMID: {meta.get('paper_id', '')}"
    )


# ────────────────────────────────────────────────────────────────────────────────
# Core class
# ────────────────────────────────────────────────────────────────────────────────
class PubMedQAEngine:
    def __init__(self, db_name: str = "index") -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set")

        self.embedder = Vectorizer()  # reuse same tokenizer + model
        self.llm = OpenAI(api_key=api_key)
        self.db = FaissVectorDB(dimension=EMBED_DIM)
        self.db.load(db_name)

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #
    def answer(self, question: str, top_k: int = 8) -> Dict:
        logger.info("Embedding user question")
        q_vec = self.embedder._openai_embed([question])[0]  # returns np.ndarray

        logger.info("Searching FAISS (k=%d)", top_k)
        hits, dists = self.db.similarity_search(q_vec, k=top_k)
        if not hits:
            return {"error": "No relevant papers found."}

        # Build context string + citations list
        citations = []
        context_blocks: List[str] = []
        for i, (meta, dist) in enumerate(zip(hits, dists), start=1):
            citations.append(_format_citation(meta, i))
            snippet = meta["text"][:1000]  # keep prompt small
            context_blocks.append(
                f"[{i}] {snippet}\n(Similarity score: {1/(1+dist):.3f})"
            )

        context = "\n\n".join(context_blocks)
        ref_list = "\n".join(citations)

        prompt = self._build_prompt(question, context, ref_list)

        logger.info("Calling OpenAI ChatCompletion")
        try:
            resp = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert biomedical research assistant. "
                            "Ground every claim in the provided context and "
                            "cite using bracketed numbers."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=900,
            )
            answer_text = resp.choices[0].message.content
            return {
                "question": question,
                "answer": answer_text,
                "citations": citations,
            }

        except (
            APIError,
            APITimeoutError,
            RateLimitError,
            APIConnectionError,
            BadRequestError,
        ) as exc:
            logger.error("OpenAI API error: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------ #
    #  Prompt template                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_prompt(question: str, context: str, refs: str) -> str:
        return (
            f"USER QUESTION:\n{question}\n\n"
            "CONTEXT (excerpts from retrieved PubMed papers):\n"
            f"{context}\n\n"
            "You must answer the question **using only the information in the "
            "context above**. When you utilise a piece of information, cite it "
            "with its bracketed number. After the answer, include a "
            "'References' section listing each citation on its own line.\n\n"
            """
            GUIDELINES FOR YOUR RESPONSE:
                1. Prioritize information directly present in the provided papers
                2. Extract and synthesize findings across multiple sources when available
                3. Present a nuanced analysis that acknowledges:
                - Strength and consistency of evidence
                - Methodology quality (study design, sample size, controls)
                - Statistical significance of findings (p-values, confidence intervals)
                - Clinical relevance versus statistical significance
                4. When evaluating evidence quality:
                - Clearly identify study designs (RCT, meta-analysis, cohort, case-control, etc.)
                - Note sample characteristics (size, demographics, inclusion/exclusion criteria)
                - Address potential limitations or biases in methodology
                - Indicate levels of evidence using recognized frameworks (e.g., GRADE)
                5. For numerical data:
                - Include specific effect sizes, risk ratios, hazard ratios, or odds ratios
                - Provide confidence intervals and p-values when available
                - Contextualize percentages with absolute numbers
                - Compare findings across studies when possible
                6. For comparative analyses:
                - Present data from each intervention/group side-by-side
                - Highlight statistical and clinical significance of differences
                - Note heterogeneity in methods or populations that might affect comparability

                FORMAT YOUR RESPONSE AS:
                1. Executive Summary 
                - Concise answer to the primary question
                - Level of confidence in conclusion based on evidence quality

                2. Key Findings 
                - Comprehensive synthesis of main evidence-based insights
                - Integration of findings across multiple papers
                - Clear presentation of consensus and contradictions in the literature

                3. Supporting Evidence 
                - Detailed breakdown of specific data points with precise citations
                - Methodological context for each cited study
                - Quantitative results with statistical measures
                - Comparison of findings across different studies

                4. Clinical Implications 
                - Evidence-based practical applications
                - Considerations for implementation in clinical practice
                - Patient populations most likely to benefit
                - Potential risks or limitations in clinical application

                5. Evidence Quality Assessment (100-150 words)
                - Evaluation of the overall body of evidence
                - Identification of research gaps
                - Methodological strengths and limitations
                - Suggestions for further research needed

                6. References
                - Numbered citation list with full bibliographic information

                Citation format: Use bracketed numbers [1], [2], etc. within the text that correspond to the reference list. Include page or paragraph numbers for direct quotations.

                When multiple papers address the same point, cite all relevant sources [1,3,5].

                If certain aspects cannot be fully addressed with available evidence, focus on related findings that ARE supported by the papers and clearly indicate the limitations of current knowledge without speculative conclusions.

                Remember to maintain scientific objectivity while providing actionable insights based on the best available evidence.
            
            """
            
            f"References:\n{refs}"
        )


# ────────────────────────────────────────────────────────────────────────────────
# CLI interface
# ────────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="PubMed QA over FAISS")
    parser.add_argument("--question", "-q", required=True, help="User question")
    parser.add_argument("--top_k", "-k", type=int, default=8, help="Top‑K chunks")
    parser.add_argument(
        "--db_name", "-d", default="index", help="DB base name under pubmed_faiss_index/"
    )
    args = parser.parse_args()

    engine = PubMedQAEngine(args.db_name)
    result = engine.answer(args.question, top_k=args.top_k)
    if "error" in result:
        print("Error:", result["error"])
    else:
        print("\n" + "=" * 80)
        print("ANSWER:\n")
        print(result["answer"])
        print("\n" + "=" * 80)
        print("CITATIONS:")
        for c in result["citations"]:
            print(c)


if __name__ == "__main__":
    main()
