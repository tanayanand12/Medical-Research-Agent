"""
data_fetcher.py — Phase 7: ClinicalTrials.gov data fetcher.

Hybrid architecture:
- LLM extracts search *concepts* only (broad_terms, condition_terms,
  intervention_terms, synonym_terms) — a simple task any small model handles.
- Deterministic URL builder constructs valid ClinicalTrials.gov v2 URLs
  from those concepts — no URL schema knowledge required from the LLM.
- Pydantic validates term lists before URL construction.
- Retry widens terms automatically on empty results.

Changes from legacy ClinicalTrialsFetcherAgent
-----------------------------------------------
* LLM no longer generates URLs — only extracts medical concepts.
* URL construction is deterministic and always valid.
* Pydantic SearchTerms model validates LLM output.
* Fallback broadens search automatically on empty result sets.
* All LLM calls route through LLMClient (LiteLLM — provider-agnostic).
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import requests  # type: ignore
from pydantic import BaseModel, field_validator # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic model — validates LLM term extraction output
# ---------------------------------------------------------------------------

class SearchTerms(BaseModel):
    """Validated search concepts extracted from a user query.

    The LLM is only asked to fill these four lists — no URL knowledge needed.
    Each list is capped at 4 items and sanitised before URL construction.
    """

    broad_terms: List[str] = []
    condition_terms: List[str] = []
    intervention_terms: List[str] = []
    synonym_terms: List[str] = []

    @field_validator("broad_terms", "condition_terms", "intervention_terms", "synonym_terms", mode="before")
    @classmethod
    def clean_and_cap(cls, v: Any) -> List[str]:
        if not isinstance(v, list):
            return []
        cleaned = []
        for item in v:
            if isinstance(item, str):
                # strip punctuation, lowercase, collapse whitespace
                item = re.sub(r"[^\w\s\-]", "", item).strip().lower()
                if item:
                    # Strip ICD codes (e11, k21, j45 pattern - letter followed by digits)
                    item = re.sub(r'\b[a-z]\d+\b', '', item).strip()
                    # Strip tokens under 3 characters
                    if len(item) < 3:
                        continue
                    cleaned.append(item)
        return cleaned[:4]

    @property
    def all_terms(self) -> List[str]:
        """Flat list of all unique terms across all categories."""
        seen: set = set()
        out: List[str] = []
        for t in (
            self.broad_terms
            + self.condition_terms
            + self.intervention_terms
            + self.synonym_terms
        ):
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    @classmethod
    def from_query(cls, query: str) -> "SearchTerms":
        """Naive keyword fallback when LLM extraction fails."""
        stopwords = {
            "the", "for", "and", "or", "in", "of", "a", "an",
            "with", "on", "is", "are", "was", "be", "to", "at",
            "trials", "studies", "study", "trial", "patients",
        }
        words = [
            w.lower()
            for w in re.split(r"\W+", query)
            if len(w) > 3 and w.lower() not in stopwords
        ]
        return cls(
            broad_terms=words[:2],
            condition_terms=words[2:4],
            intervention_terms=words[4:6],
            synonym_terms=[],
        )


# ---------------------------------------------------------------------------
# Term extraction prompt
# ---------------------------------------------------------------------------

_TERM_EXTRACTION_PROMPT = """\
You are a medical search specialist. Extract search concepts from the query below.

Return ONLY a JSON object with exactly these four keys. No markdown, no explanation.

{
  "broad_terms": ["<2-3 broad medical terms>"],
  "condition_terms": ["<2-3 specific conditions or diseases>"],
  "intervention_terms": ["<2-3 drug names, treatments, or procedures>"],
  "synonym_terms": ["<2-3 synonyms or related concepts>"]
}

Rules:
- Each list must contain plain strings only.
- No URLs, no API parameters, no field names.
- Use common medical terminology.
- If a category has no relevant terms, return an empty list [].

Query: {query}
"""


# ---------------------------------------------------------------------------
# Main fetcher class
# ---------------------------------------------------------------------------

class ClinicalTrialsFetcher:
    """Fetches clinical trials data from ClinicalTrials.gov API v2.

    Architecture
    ------------
    1. LLM extracts medical concepts from the user query (SearchTerms).
    2. Deterministic URL builder constructs up to 5 valid API URLs.
    3. HTTP fetcher retrieves studies, skipping invalid/empty responses.
    4. On empty result set, retry with progressively broader terms.
    5. Results are collated and deduplicated by NCT ID.

    Parameters
    ----------
    llm_client : LLMClient, optional
        Pre-existing LLMClient instance. Lazy-loaded on first use if None.
    max_urls : int
        Maximum number of URLs to construct per query (default 5).
    page_size : int
        Number of studies to request per URL (default 50).
    """

    BASE_URL = "https://clinicaltrials.gov/api/v2"

    def __init__(
        self,
        llm_client=None,
        max_urls: int = 5,
        page_size: int = 50,
    ) -> None:
        self._llm = llm_client
        self.max_urls = max_urls
        self.page_size = page_size

    # ------------------------------------------------------------------ #
    # Lazy LLMClient accessor
    # ------------------------------------------------------------------ #

    @property
    def llm(self):
        if self._llm is None:
            from llm_client import LLMClient
            self._llm = LLMClient()
        return self._llm

    # ------------------------------------------------------------------ #
    # Step 1 — LLM concept extraction
    # ------------------------------------------------------------------ #

    # def extract_search_terms(
    #     self,
    #     user_query: str,
    #     max_retries: int = 3,
    #     wait_seconds: int = 1,
    # ) -> SearchTerms:
    #     """Extract medical search concepts from user query via LLM.

    #     Falls back to naive keyword extraction if LLM output cannot be
    #     parsed after max_retries attempts.

    #     Parameters
    #     ----------
    #     user_query : str
    #         Natural language medical query.
    #     max_retries : int
    #         Number of LLM retry attempts before falling back.
    #     wait_seconds : int
    #         Delay between retries.

    #     Returns
    #     -------
    #     SearchTerms
    #         Validated search concept model.
    #     """
    #     prompt = _TERM_EXTRACTION_PROMPT.format(query=user_query)

    #     for attempt in range(max_retries):
    #         try:
    #             raw = self.llm.chat(
    #                 messages=[{"role": "user", "content": prompt}],
    #                 temperature=0.1,      # low temp = more deterministic JSON
    #                 max_tokens=300,       # term lists are short
    #             )

    #             # Extract JSON from response (handles markdown fences)
    #             # Replace this block:
    #             # json_start = raw.find("{")
    #             # json_end = raw.rfind("}") + 1
    #             # if json_start == -1 or json_end <= json_start:
    #             #     raise ValueError("No JSON object found in LLM response")
    #             # data = json.loads(raw[json_start:json_end])

    #             # With this:
    #             # Strip markdown fences
    #             raw = re.sub(r"```(?:json)?", "", raw).strip()

    #             json_start = raw.find("{")
    #             json_end = raw.rfind("}") + 1

    #             if json_start == -1 or json_end <= json_start:
    #                 # Llama3 sometimes returns just values without wrapper
    #                 # Build dict manually from key: [list] pattern
    #                 data = {}
    #                 for key in ["broad_terms", "condition_terms", "intervention_terms", "synonym_terms"]:
    #                     match = re.search(
    #                         rf'"{key}"\s*:\s*\[([^\]]*)\]', raw, re.DOTALL
    #                     )
    #                     if match:
    #                         items = re.findall(r'"([^"]+)"', match.group(1))
    #                         data[key] = items
    #                 if not data:
    #                     raise ValueError("Could not extract any terms from LLM response")
    #             else:
    #                 data = json.loads(raw[json_start:json_end])
    #             terms = SearchTerms(**data)

    #             if terms.all_terms:
    #                 logger.info(
    #                     "Extracted %d concepts from query (attempt %d)",
    #                     len(terms.all_terms),
    #                     attempt + 1,
    #                 )
    #                 return terms

    #             logger.warning(
    #                 "Attempt %d: LLM returned empty term lists. Retrying...",
    #                 attempt + 1,
    #             )

    #         except (json.JSONDecodeError, ValueError, TypeError) as e:
    #             logger.warning(
    #                 "Attempt %d: Could not parse LLM output: %s. Retrying...",
    #                 attempt + 1,
    #                 e,
    #             )
    #         except Exception as e:
    #             logger.error(
    #                 "Attempt %d: Unexpected error during term extraction: %s",
    #                 attempt + 1,
    #                 e,
    #             )

    #         if attempt < max_retries - 1:
    #             time.sleep(wait_seconds)

    #     logger.warning(
    #         "LLM term extraction failed after %d attempts. "
    #         "Using keyword fallback.",
    #         max_retries,
    #     )
    #     return SearchTerms.from_query(user_query)

    def extract_search_terms(
        self,
        user_query: str,
        max_retries: int = 2,
        wait_seconds: int = 1,
    ) -> SearchTerms:
        """Extract search terms - tries LLM first, falls back to keyword extraction."""
        
        # Try LLM with a simpler prompt that Llama3 can handle
        simple_prompt = (
            f"List medical search terms for: {user_query}\n"
            f"Reply with ONLY this format, no other text:\n"
            f"broad: diabetes, hyperglycemia\n"
            f"condition: type 2 diabetes\n"
            f"intervention: SGLT2 inhibitor, empagliflozin\n"
            f"synonym: antidiabetic"
        )
        
        for attempt in range(max_retries):
            try:
                raw = self.llm.chat(
                    messages=[{"role": "user", "content": simple_prompt}],
                    temperature=0.1,
                    max_tokens=150,
                )
                
                # Parse key: value1, value2 format
                data = {
                    "broad_terms": [],
                    "condition_terms": [],
                    "intervention_terms": [],
                    "synonym_terms": [],
                }
                
                for line in raw.splitlines():
                    line = line.strip().lower()
                    if line.startswith("broad:"):
                        data["broad_terms"] = [t.strip() for t in line[6:].split(",") if t.strip()]
                    elif line.startswith("condition:"):
                        data["condition_terms"] = [t.strip() for t in line[10:].split(",") if t.strip()]
                    elif line.startswith("intervention:"):
                        data["intervention_terms"] = [t.strip() for t in line[13:].split(",") if t.strip()]
                    elif line.startswith("synonym:"):
                        data["synonym_terms"] = [t.strip() for t in line[8:].split(",") if t.strip()]
                
                terms = SearchTerms(**data)
                if terms.all_terms:
                    logger.info("LLM extracted %d terms", len(terms.all_terms))
                    return terms
                    
            except Exception as e:
                logger.warning("LLM extraction attempt %d failed: %s", attempt + 1, e)
                if attempt < max_retries - 1:
                    time.sleep(wait_seconds)
        
        # Always-reliable fallback
        logger.warning("Using keyword fallback for term extraction")
        return SearchTerms.from_query(user_query)

    # ------------------------------------------------------------------ #
    # Step 2 — Deterministic URL builder
    # ------------------------------------------------------------------ #

    def build_urls(self, terms: SearchTerms, broad_mode: bool = False) -> List[str]:
        """Construct valid ClinicalTrials.gov v2 API URLs from search terms.

        Uses a set to deduplicate URLs naturally. Constructs URLs across
        multiple search fields (term, cond, intr) and term combinations
        to maximise recall.

        Parameters
        ----------
        terms : SearchTerms
            Validated search concepts.
        broad_mode : bool
            When True, uses only the first broad term (retry widening).

        Returns
        -------
        list[str]
            Up to self.max_urls unique, validated API URLs.
        """
        base = f"{self.BASE_URL}/studies"
        ps = self.page_size
        urls: set = set()

        def enc(t: str) -> str:
            return quote_plus(t)

        if broad_mode:
            # Widened retry — single broad term, large page size
            fallback = terms.broad_terms[0] if terms.broad_terms else terms.all_terms[0]
            return [f"{base}?query.term={enc(fallback)}&pageSize=100&countTotal=true"]

        # --- Broad sweep ---
        for t in terms.broad_terms[:2]:
            urls.add(f"{base}?query.term={enc(t)}&pageSize={ps}&countTotal=true")

        # --- Condition focused ---
        for t in terms.condition_terms[:2]:
            urls.add(f"{base}?query.cond={enc(t)}&pageSize={ps}&countTotal=true")

        # --- Intervention focused ---
        for t in terms.intervention_terms[:2]:
            urls.add(f"{base}?query.intr={enc(t)}&pageSize={ps}&countTotal=true")

        # --- AND combination (condition + intervention) ---
        if terms.condition_terms and terms.intervention_terms:
            c = enc(terms.condition_terms[0])
            i = enc(terms.intervention_terms[0])
            urls.add(
                f"{base}?query.cond={c}&query.intr={i}"
                f"&pageSize={min(ps, 30)}&countTotal=true"
            )

        # --- Synonym sweep ---
        for t in terms.synonym_terms[:1]:
            urls.add(f"{base}?query.term={enc(t)}&pageSize={ps}&countTotal=true")

        # --- Multi-term OR sweep (join broad + condition) ---
        combined = terms.broad_terms[:1] + terms.condition_terms[:1]
        if len(combined) > 1:
            joined = enc(" ".join(combined))
            urls.add(f"{base}?query.term={joined}&pageSize={ps}&countTotal=true")

        result = list(urls)[: self.max_urls]
        logger.info("Built %d unique URLs from search terms", len(result))
        return result

    # ------------------------------------------------------------------ #
    # Step 3 — HTTP fetch
    # ------------------------------------------------------------------ #

    def fetch_clinical_trials_data(
        self, urls: List[str]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Fetch studies from a list of ClinicalTrials.gov API URLs.

        Skips URLs that return empty result sets or non-JSON responses.

        Parameters
        ----------
        urls : list[str]
            API URLs to fetch.

        Returns
        -------
        tuple[dict, list[str]]
            ``(accessible_content, failed_urls)``
        """
        accessible: Dict[str, Any] = {}
        failed: List[str] = []

        for url in urls:
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()

                if "application/json" not in resp.headers.get("Content-Type", ""):
                    logger.warning("Non-JSON response from %s", url)
                    failed.append(url)
                    continue

                data = resp.json()
                total = data.get("totalCount", 0)
                studies = data.get("studies", [])

                if total > 0 and studies:
                    accessible[url] = data
                    logger.info("Fetched %d studies from %s", total, url)
                else:
                    logger.warning("Empty result set from %s (totalCount=%d)", url, total)
                    failed.append(url)

            except requests.exceptions.RequestException as e:
                logger.error("Could not access URL %s: %s", url, e)
                failed.append(url)
            except Exception as e:
                logger.error("Unexpected error for URL %s: %s", url, e)
                failed.append(url)

        return accessible, failed

    # ------------------------------------------------------------------ #
    # Step 4 — Collation + deduplication
    # ------------------------------------------------------------------ #

    def collate_studies_data(self, accessible: Dict[str, Any]) -> Dict[str, Any]:
        """Collate and deduplicate studies from multiple URL responses.

        Deduplication is by NCT ID — highest-count source wins on conflict.

        Parameters
        ----------
        accessible : dict
            URL -> JSON response mapping from fetch_clinical_trials_data.

        Returns
        -------
        dict
            Collated data with keys: studies, totalCount,
            originalTotalCount, sourceUrls.
        """
        unique: Dict[str, Any] = {}
        total_raw = 0
        source_urls: List[str] = []

        for url, content in accessible.items():
            source_urls.append(url)
            total_raw += content.get("totalCount", 0)
            for study in content.get("studies", []):
                try:
                    nct_id = (
                        study.get("protocolSection", {})
                        .get("identificationModule", {})
                        .get("nctId")
                    )
                    if nct_id and nct_id not in unique:
                        unique[nct_id] = study
                except Exception as e:
                    logger.warning("Error deduplicating study: %s", e)

        logger.info(
            "Collated %d unique studies from %d sources",
            len(unique),
            len(source_urls),
        )
        return {
            "studies": list(unique.values()),
            "totalCount": len(unique),
            "originalTotalCount": total_raw,
            "sourceUrls": source_urls,
        }

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def analyze_user_query(
        self,
        user_input: str,
        retry_broad: bool = True,
    ) -> Dict[str, Any]:
        """Analyse a natural language query and fetch matching clinical trials.

        Pipeline
        --------
        1. Extract search concepts via LLM (SearchTerms).
        2. Build deterministic API URLs from concepts.
        3. Fetch studies from each URL.
        4. If all URLs return empty and retry_broad=True, widen and retry once.
        5. Collate and return results.

        Parameters
        ----------
        user_input : str
            Natural language clinical trials query.
        retry_broad : bool
            Whether to retry with broader terms on full empty result.

        Returns
        -------
        dict
            Keys: success, data, total_count, studies_returned,
            source_url, all_source_urls, failed_urls, attempted_urls,
            query_analysis.
        """
        logger.info("Analyzing query: %s", user_input)

        try:
            # Step 1: Extract concepts
            terms = self.extract_search_terms(user_input)
            logger.info("Search terms: %s", terms.model_dump())

            # Step 2: Build URLs
            urls = self.build_urls(terms)

            # Step 3: Fetch
            accessible, failed = self.fetch_clinical_trials_data(urls)

            # Step 4: Retry with broad mode on total failure
            if not accessible and retry_broad and terms.all_terms:
                logger.warning(
                    "All URLs returned empty. Retrying in broad mode..."
                )
                broad_urls = self.build_urls(terms, broad_mode=True)
                accessible, failed_broad = self.fetch_clinical_trials_data(broad_urls)
                failed += failed_broad
                urls += broad_urls

            if not accessible:
                return {
                    "success": False,
                    "error": "No studies found from any generated URLs",
                    "data": None,
                    "total_count": 0,
                    "failed_urls": failed,
                    "attempted_urls": urls,
                    "source_url": "",
                }

            # Step 5: Collate
            collated = self.collate_studies_data(accessible)

            return {
                "success": True,
                "data": collated,
                "total_count": collated["totalCount"],
                "studies_returned": len(collated["studies"]),
                "source_url": collated["sourceUrls"][0] if collated["sourceUrls"] else "",
                "all_source_urls": collated["sourceUrls"],
                "failed_urls": failed,
                "attempted_urls": urls,
                "query_analysis": {
                    "original_query": user_input,
                    "extracted_terms": terms.model_dump(),
                    "urls_attempted": len(urls),
                    "urls_successful": len(accessible),
                    "unique_studies_found": collated["totalCount"],
                    "url_strategy": "hybrid_llm_terms_deterministic_builder",
                },
            }

        except Exception as e:
            logger.error("Error in analyze_user_query: %s", e)
            return {
                "success": False,
                "error": str(e),
                "data": None,
                "total_count": 0,
                "source_url": "",
            }