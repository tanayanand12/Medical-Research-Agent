"""
core.medical_search_agent
~~~~~~~~~~~~~~~~~~~~~~~~~

Builds PubMed ESearch URLs from a *flat* list of medical keywords.

Example
-------
>>> agent = MedicalSearchAgent()
>>> url = agent.format_pubmed_search_url(
...     ["glioblastoma", "temozolomide", "IDH1 mutation"]
... )
>>> print(url)
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=(glioblastoma%20OR%20temozolomide%20OR%20IDH1%20mutation)&retmode=json&datetype=pdat&mindate=2024/01/01&maxdate=2025/04/20&retmax=1000
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import List
from urllib.parse import quote_plus

from dotenv import load_dotenv
from openai import OpenAI  # imported only to verify API key presence

# ────────────────────────────────────────────────────────────────────────────────
# Logging setup (module‑level logger)  |  Best practice per Python docs [1]
# ────────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
if not logger.handlers:  # prevent double‑handlers in interactive sessions
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# ────────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────────
PUBMED_ESEARCH_ENDPOINT = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
)
DATE_FLOOR = "2024/01/01"  # Inclusive lower bound per project spec


class MedicalSearchAgent:
    """Utility class to build PubMed search URLs for *post‑Jan‑2024* literature."""

    def __init__(self) -> None:
        """
        Verify that an OpenAI API key exists (fail‑fast).

        We don't call the client here, but downstream modules will.
        """
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError(
                "OPENAI_API_KEY not found. "
                "Set it in your environment or .env file before continuing."
            )

        # Lazily import to avoid unused‑client warnings in linters
        self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.debug("MedicalSearchAgent initialised successfully.")

    # --------------------------------------------------------------------- #
    #  Public API                                                           #
    # --------------------------------------------------------------------- #
    # def format_pubmed_search_url(self, keywords: List[str], retmax: int = 100000) -> str:
    #     """
    #     Build a PubMed ESearch URL that OR‑combines *all* supplied keywords and
    #     limits the publication date range from 1 Jan 2024 to **today**.

    #     Parameters
    #     ----------
    #     keywords : List[str]
    #         Single‑dimension list of medical terms (no pre‑grouping required).
    #     retmax : int, default 1000
    #         Maximum number of PMIDs to return (PubMed allows up to 100 000).

    #     Returns
    #     -------
    #     str
    #         Fully encoded ESearch URL.

    #     Raises
    #     ------
    #     ValueError
    #         If `keywords` is empty.
    #     """
    #     if not keywords:
    #         raise ValueError("Keyword list is empty; cannot build PubMed query.")

    #     # 1. --- Build the OR‑joined term string --------------------------------
    #     #     Quote each keyword so spaces -> %20, then join with +OR+.
    #     encoded_terms = [quote_plus(term) for term in keywords]
    #     term_clause = "%20OR%20".join(encoded_terms)
    #     term_clause = f"({term_clause})"

    #     # 2. --- Append mandatory ESearch parameters ---------------------------
    #     today = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    #     params = (
    #         f"?db=pubmed"
    #         f"&term={term_clause}"
    #         f"&retmode=json"
    #         f"&datetype=pdat"
    #         f"&mindate={DATE_FLOOR}"
    #         f"&maxdate={today}"
    #         f"&retmax={retmax}"
    #     )

    #     url = f"{PUBMED_ESEARCH_ENDPOINT}{params}"
    #     logger.info("Generated PubMed URL: %s", url)
    #     return url

    def format_pubmed_search_url(self, keywords: List[str]) -> str:
        """
        Format a PubMed ESearch URL for a list of keywords.

        Args:
            keywords (List[str]): Plain list of keywords to OR-combine

        Returns:
            str: PubMed ESearch URL
        """
        from urllib.parse import quote_plus
        from datetime import datetime

        if not keywords:
            raise ValueError("Keyword list is empty")

        # 1. --- Build the OR‑joined term string --------------------------------
        query_string = "+OR+".join([quote_plus(k) for k in keywords])
        today = datetime.utcnow().strftime("%Y/%m/%d")
        url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
            f"db=pubmed&term=({query_string})"
            f"&retmode=json&datetype=pdat"
            f"&mindate=2024/01/01&maxdate={today}"
        )
        return url


# ────────────────────────────────────────────────────────────────────────────────
# Basic smoke‑test (run “python -m core.medical_search_agent” to verify)
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    sample_keywords = [
        "glioblastoma",
        "temozolomide",
        "IDH1 mutation",
    ]
    agent = MedicalSearchAgent()
    print(agent.format_pubmed_search_url(sample_keywords))
