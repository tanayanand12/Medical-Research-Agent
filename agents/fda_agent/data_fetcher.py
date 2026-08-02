"""
data_fetcher.py — Phase 7: openFDA regulatory data fetcher.

Architecture mirrors ClinicalTrialsFetcher exactly:
- LLM extracts medical/regulatory *concepts only* (drug names, conditions,
  safety terms, recall terms) — no URL schema knowledge required.
- Deterministic URL builder constructs valid openFDA API v1 URLs from
  those concepts, spanning multiple relevant endpoints.
- Pydantic FDASearchTerms validates and sanitises LLM output before URL
  construction, preventing malformed queries.
- Parallel ThreadPoolExecutor fetch (openFDA supports concurrent requests;
  retained from legacy FdaFetcherAgent performance optimisation).
- Retry widens terms automatically on full empty result set.
- All LLM calls route through LLMClient (LiteLLM — provider-agnostic).

Changes from legacy FdaFetcherAgent
------------------------------------
* LLM no longer generates URLs directly — it only extracts concepts.
  (Legacy URL generation was unreliable with local models: Llama3 produced
  syntactically invalid openFDA URLs ~40% of the time.)
* URL construction is deterministic: always valid, no LLM hallucination risk.
* Pydantic FDASearchTerms model validates and sanitises LLM output.
* Broad-mode retry added: automatically widens search on total empty result.
* All LLM calls through LLMClient (was: hardcoded OpenAI client).
* ThreadPoolExecutor parallel fetch retained from legacy for latency.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import requests  # type: ignore
from pydantic import BaseModel, field_validator  # type: ignore
from evaluation_core import safe_error_type, stable_query_fingerprint
from runtime_verification.deadline import (
    RuntimeDeadlineExceeded,
    ensure_deadline,
    remaining_seconds,
    sleep_with_deadline,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic model — validates LLM concept extraction output
# ---------------------------------------------------------------------------


class FDASearchTerms(BaseModel):
    """Validated search concepts extracted from a user FDA query.

    The LLM fills four concept lists only — no URL or API endpoint knowledge
    is required from the model.  Each list is capped at 4 items and
    sanitised (lowercased, punctuation stripped) before URL construction.

    Attributes
    ----------
    drug_names : list[str]
        Brand names, generic names, active ingredients, substance names.
    conditions : list[str]
        Medical conditions, disease states, indications.
    safety_terms : list[str]
        Adverse event, reaction, toxicity, or pharmacovigilance terms.
    recall_terms : list[str]
        Recall, quality defect, enforcement, or contamination terms.

    Properties
    ----------
    all_terms : list[str]
        Flat deduplicated list across all four categories.

    Class Methods
    -------------
    from_query(query)
        Naive keyword fallback when LLM extraction fails.

    Examples
    --------
    >>> terms = FDASearchTerms(
    ...     drug_names=["metformin", "glucophage"],
    ...     conditions=["type 2 diabetes"],
    ...     safety_terms=["lactic acidosis"],
    ...     recall_terms=[],
    ... )
    >>> terms.all_terms
    ['metformin', 'glucophage', 'type 2 diabetes', 'lactic acidosis']
    """

    drug_names: List[str] = []
    conditions: List[str] = []
    safety_terms: List[str] = []
    recall_terms: List[str] = []

    @field_validator(
        "drug_names", "conditions", "safety_terms", "recall_terms",
        mode="before",
    )
    @classmethod
    def clean_and_cap(cls, v: Any) -> List[str]:
        """Strip punctuation, lowercase, remove short tokens, cap at 4 items."""
        if not isinstance(v, list):
            return []
        cleaned: List[str] = []
        for item in v:
            if isinstance(item, str):
                item = re.sub(r"[^\w\s\-]", "", item).strip().lower()
                if item and len(item) >= 3:
                    cleaned.append(item)
        return cleaned[:4]

    @property
    def all_terms(self) -> List[str]:
        """Flat list of all unique terms across all four categories."""
        seen: set = set()
        out: List[str] = []
        for t in (
            self.drug_names
            + self.conditions
            + self.safety_terms
            + self.recall_terms
        ):
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    @classmethod
    def from_query(cls, query: str) -> "FDASearchTerms":
        """Naive keyword fallback when LLM extraction fails after all retries.

        Strips stopwords and JSON field-name artifacts (``drug_names``,
        ``safety_terms``, ``json``, ``here``, etc.) that appear when the
        expanded_query blob is accidentally passed instead of the raw query.

        Parameters
        ----------
        query : str
            Raw user query string.

        Returns
        -------
        FDASearchTerms
            Best-effort term model from keyword extraction.
        """
        stopwords = {
            # English function words
            "the", "for", "and", "or", "in", "of", "a", "an",
            "with", "on", "is", "are", "was", "be", "to", "at",
            "what", "how", "does", "its", "this", "that", "these", "those",
            "which", "when", "where", "who", "why", "about", "from",
            # Domain stopwords
            "fda", "drug", "drugs", "medication", "medicines",
            "patients", "patient", "reports", "report", "data",
            "including", "according", "based", "related", "associated",
            # JSON artifact words that appear when expanded_query is a JSON blob
            "drug_names", "conditions", "safety_terms", "recall_terms",
            "medical", "concepts", "mesh", "terms", "synonyms",
            "json", "here", "object", "keys", "list", "null", "true", "false",
            "return", "only", "output", "format", "example",
        }
        words = [
            w.lower()
            for w in re.split(r"\W+", query)
            if len(w) > 3 and w.lower() not in stopwords
        ]
        return cls(
            drug_names=words[:2],
            conditions=words[2:4],
            safety_terms=words[4:6],
            recall_terms=[],
        )


# ---------------------------------------------------------------------------
# Term extraction prompt (line-based format — reliable across all LLM sizes)
# ---------------------------------------------------------------------------

_TERM_EXTRACTION_PROMPT = """\
Extract FDA regulatory search terms from the query below.
Output exactly 4 lines. Each line starts with the label and a colon.
Do not add any other text before or after the 4 lines.

drug: <comma-separated drug or substance names>
condition: <comma-separated medical conditions or disease states>
safety: <comma-separated adverse event or reaction terms>
recall: <comma-separated recall reason or quality defect terms>

Example output for "aspirin gastrointestinal bleeding":
drug: aspirin, acetylsalicylic acid
condition: gastrointestinal disease
safety: gastrointestinal bleeding, haemorrhage
recall:

Query: {query}
Output:"""


# ---------------------------------------------------------------------------
# Main fetcher class
# ---------------------------------------------------------------------------


class FDAFetcher:
    """Fetches regulatory records from the openFDA API.

    Architecture
    ------------
    1. LLM extracts regulatory/medical concepts → FDASearchTerms.
    2. Deterministic URL builder constructs up to ``max_urls`` valid openFDA
       API URLs, spanning drug/label, drug/event, device/recall, and
       food/enforcement endpoints based on concept categories.
    3. Parallel HTTP fetch retrieves records (ThreadPoolExecutor).
    4. On full empty result set, retry once in broad mode (single-term sweep).
    5. Records deduplicated by stable record key and returned collated.

    Parameters
    ----------
    llm_client : LLMClient, optional
        Pre-existing LLMClient instance. Lazy-loaded on first use if None.
    max_urls : int
        Maximum number of URLs constructed per query (default 6).
    page_size : int
        Records per URL request, capped at openFDA's 200-record limit.
    max_workers : int
        ThreadPool size for parallel fetching (default 8).

    Examples
    --------
    >>> fetcher = FDAFetcher()
    >>> result = fetcher.analyze_user_query("adverse events for metformin")
    >>> result["success"]
    True
    >>> len(result["data"]["records"])  # up to 200 per URL * 6 URLs, deduplicated
    ...
    """

    BASE_URL = "https://api.fda.gov"
    MAX_LIMIT = 200  # openFDA hard cap on records per request

    def __init__(
        self,
        llm_client=None,
        max_urls: int = 6,
        page_size: int = 200,
        max_workers: int = 8,
    ) -> None:
        self._llm = llm_client
        self.max_urls = max_urls
        self.page_size = min(page_size, self.MAX_LIMIT)
        self.max_workers = max_workers
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "MedicalResearchAgent/1.0"})
        logger.info(
            "FDAFetcher init (max_urls=%d, page_size=%d, workers=%d)",
            max_urls, self.page_size, max_workers,
        )

    # ------------------------------------------------------------------ #
    # Lazy LLMClient accessor (mirrors ClinicalTrialsFetcher.llm property)
    # ------------------------------------------------------------------ #

    @property
    def llm(self):
        """Lazy-loaded LLMClient instance."""
        if self._llm is None:
            from llm_client import LLMClient
            self._llm = LLMClient()
        return self._llm

    # ------------------------------------------------------------------ #
    # Step 1 — LLM concept extraction
    # ------------------------------------------------------------------ #

    def extract_search_terms(
        self,
        user_query: str,
        max_retries: int = 2,
        wait_seconds: int = 1,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        deadline_at: Optional[float] = None,
    ) -> FDASearchTerms:
        """Extract FDA regulatory search concepts from a user query via LLM.

        Uses a line-based prompt format that is reliable across model sizes,
        including local Llama3 (which struggles with JSON schema compliance).
        Falls back to naive keyword extraction if LLM output cannot be
        parsed after max_retries attempts.

        Parameters
        ----------
        user_query : str
            Natural language FDA regulatory query.
        max_retries : int
            Number of LLM retry attempts before keyword fallback (default 2).
        wait_seconds : int
            Delay between retry attempts in seconds.

        Returns
        -------
        FDASearchTerms
            Validated and sanitised search concept model.
        """
        prompt = _TERM_EXTRACTION_PROMPT.format(query=user_query)

        llm_kwargs = dict(llm_kwargs or {})
        deadline_at = deadline_at or llm_kwargs.get("deadline_at")
        attempts = 1 if llm_kwargs else max_retries
        for attempt in range(attempts):
            ensure_deadline(deadline_at)
            try:
                raw = self.llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,   # deterministic concept extraction
                    max_tokens=150,    # term lists are short
                    **llm_kwargs,
                )

                data: Dict[str, List[str]] = {
                    "drug_names": [],
                    "conditions": [],
                    "safety_terms": [],
                    "recall_terms": [],
                }

                for line in raw.splitlines():
                    line = line.strip().lower()
                    if line.startswith("drug:"):
                        data["drug_names"] = [
                            t.strip() for t in line[5:].split(",") if t.strip()
                        ]
                    elif line.startswith("condition:"):
                        data["conditions"] = [
                            t.strip() for t in line[10:].split(",") if t.strip()
                        ]
                    elif line.startswith("safety:"):
                        data["safety_terms"] = [
                            t.strip() for t in line[7:].split(",") if t.strip()
                        ]
                    elif line.startswith("recall:"):
                        data["recall_terms"] = [
                            t.strip() for t in line[7:].split(",") if t.strip()
                        ]

                terms = FDASearchTerms(**data)
                if terms.all_terms:
                    logger.info(
                        "Extracted %d FDA concepts (attempt %d)",
                        len(terms.all_terms),
                        attempt + 1,
                    )
                    return terms

                logger.warning(
                    "Attempt %d: LLM returned empty concept lists", attempt + 1
                )

            except RuntimeDeadlineExceeded:
                raise
            except Exception as exc:
                logger.warning(
                    "Term extraction attempt %d failed error_type=%s",
                    attempt + 1,
                    safe_error_type(exc),
                )

            if attempt < attempts - 1:
                sleep_with_deadline(wait_seconds, deadline_at)

        logger.warning(
            "LLM term extraction failed after %d attempts; using keyword fallback",
            attempts,
        )
        return FDASearchTerms.from_query(user_query)

    # ------------------------------------------------------------------ #
    # Step 2 — Deterministic URL builder
    # ------------------------------------------------------------------ #

    def build_urls(
        self, terms: FDASearchTerms, broad_mode: bool = False
    ) -> List[str]:
        """Construct valid openFDA API URLs from search concepts.

        Maps concept categories to appropriate openFDA endpoints:
        - ``drug_names``   → drug/label.json (brand_name, generic_name fields)
        - ``drug_names``   → drug/event.json (patient.drug.openfda.generic_name)
        - ``conditions``   → drug/label.json (indications_and_usage field)
        - ``safety_terms`` → drug/event.json (patient.reaction.reactionmeddrapt)
        - ``recall_terms`` → device/recall.json and food/enforcement.json

        Uses a set to deduplicate URLs naturally, then caps to max_urls.

        Parameters
        ----------
        terms : FDASearchTerms
            Validated search concepts.
        broad_mode : bool
            When True, constructs a single broad drug/event + drug/label
            search using the first available term (retry widening strategy).

        Returns
        -------
        list[str]
            Up to self.max_urls unique, valid openFDA API URLs.
        """
        ps = self.page_size
        urls: set = set()

        def enc(t: str) -> str:
            return quote_plus(t)

        # Endpoint helpers — always include limit
        def drug_label(search: str) -> str:
            return f"{self.BASE_URL}/drug/label.json?search={search}&limit={ps}"

        def drug_event(search: str) -> str:
            return f"{self.BASE_URL}/drug/event.json?search={search}&limit={ps}"

        def device_recall(search: str) -> str:
            return f"{self.BASE_URL}/device/recall.json?search={search}&limit={ps}"

        def food_enforcement(search: str) -> str:
            return f"{self.BASE_URL}/food/enforcement.json?search={search}&limit={ps}"

        # ---- Broad-mode retry: single term, two endpoints ----
        if broad_mode:
            fallback = (terms.drug_names or terms.all_terms or ["drug"])[0]
            return [
                drug_event(enc(fallback)),
                drug_label(enc(fallback)),
            ]

        # ---- Drug label queries ----
        for name in terms.drug_names[:2]:
            urls.add(drug_label(f"openfda.brand_name:{enc(name)}"))
            urls.add(drug_label(f"openfda.generic_name:{enc(name)}"))

        for cond in terms.conditions[:1]:
            urls.add(drug_label(f"indications_and_usage:{enc(cond)}"))

        # ---- Adverse event queries ----
        for name in terms.drug_names[:2]:
            urls.add(
                drug_event(
                    f"patient.drug.openfda.generic_name:{enc(name)}"
                )
            )

        for term in terms.safety_terms[:1]:
            urls.add(
                drug_event(
                    f"patient.reaction.reactionmeddrapt:{enc(term)}"
                )
            )

        # ---- Recall / enforcement queries ----
        for term in terms.recall_terms[:1]:
            urls.add(device_recall(f"product_description:{enc(term)}"))
            urls.add(food_enforcement(f"product_description:{enc(term)}"))

        # ---- Broad fallback if nothing built ----
        if not urls and terms.all_terms:
            urls.add(drug_event(enc(terms.all_terms[0])))

        result = list(urls)[: self.max_urls]
        logger.info("Built %d unique openFDA URLs", len(result))
        return result

    # ------------------------------------------------------------------ #
    # Step 3 — Parallel HTTP fetch (retained from legacy FdaFetcherAgent)
    # ------------------------------------------------------------------ #

    def _fetch_single(
        self,
        url: str,
        timeout: int = 30,
        deadline_at: Optional[float] = None,
    ) -> Tuple[str, Optional[Any], Optional[str]]:
        """Thread worker: fetch one openFDA URL and validate the response.

        Validates that the response is JSON, non-empty (meta.results.total > 0),
        and contains at least one record in the results list.

        Parameters
        ----------
        url : str
            openFDA API URL to fetch.
        timeout : int
            HTTP request timeout in seconds.

        Returns
        -------
        tuple[str, dict | None, str | None]
            ``(url, json_data, error_message)``
            json_data is None on any failure; error_message is None on success.
        """
        try:
            ensure_deadline(deadline_at)
            r = self._session.get(
                url,
                timeout=remaining_seconds(deadline_at, default=timeout),
            )
            r.raise_for_status()

            if "application/json" not in r.headers.get("Content-Type", ""):
                return url, None, "non-json response"

            data = r.json()
            total = data.get("meta", {}).get("results", {}).get("total", 0)
            records = data.get("results", [])

            if total > 0 and records:
                logger.info(
                    "Fetched %d records url_sha256=%s",
                    total,
                    stable_query_fingerprint(url),
                )
                return url, data, None

            return url, None, f"empty (total={total})"

        except RuntimeDeadlineExceeded:
            raise
        except requests.exceptions.RequestException as exc:
            return url, None, f"HTTP error: {exc}"
        except Exception as exc:
            return url, None, str(exc)

    def fetch_fda_data(
        self,
        urls: List[str],
        *,
        deadline_at: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Fetch records from multiple openFDA URLs in parallel.

        Spawns max_workers threads, one future per URL.  Collects results
        as they complete; failed or empty URLs are logged and collected
        separately.

        Parameters
        ----------
        urls : list[str]
            openFDA API URLs to fetch.

        Returns
        -------
        tuple[dict, list[str]]
            ``(accessible_content, failed_urls)``
            accessible_content maps URL → JSON response for all successful fetches.
        """
        accessible: Dict[str, Any] = {}
        failed: List[str] = []
        ensure_deadline(deadline_at)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(
                    self._fetch_single,
                    u,
                    deadline_at=deadline_at,
                ): u
                for u in urls
            }
            for fut in as_completed(futures):
                url, data, err = fut.result()
                if data is not None:
                    accessible[url] = data
                else:
                    logger.warning(
                        "FDA fetch failed url_sha256=%s error_type=%s",
                        stable_query_fingerprint(url),
                        "provider_response_error",
                    )
                    failed.append(url)

        logger.info(
            "Fetch complete: %d success, %d failed",
            len(accessible),
            len(failed),
        )
        return accessible, failed

    # ------------------------------------------------------------------ #
    # Step 4 — Collation + deduplication
    # ------------------------------------------------------------------ #

    @staticmethod
    def _record_key(rec: Dict[str, Any]) -> str:
        """Compute a stable unique key for an openFDA record.

        Tries known FDA identifier fields in priority order; falls back to
        a SHA-256 hash of the full JSON-serialised record.

        Parameters
        ----------
        rec : dict
            Raw openFDA record.

        Returns
        -------
        str
            Stable unique key string.
        """
        for k in (
            "id",
            "event_key",
            "report_number",
            "recall_number",
            "enforcement_report_number",
            "safetyreportid",
            "lot_number",
        ):
            if k in rec and rec[k]:
                return f"{k}:{rec[k]}"
        spl = rec.get("openfda", {}).get("spl_id", [])
        if spl:
            return f"spl_id:{spl[0]}"
        return hashlib.sha256(
            json.dumps(rec, sort_keys=True).encode()
        ).hexdigest()[:16]

    def collate_records_data(
        self, accessible: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collate and deduplicate records from multiple URL responses.

        Merges all records from all successful URL responses into a single
        list, deduplicating by stable record key (first occurrence wins).

        Parameters
        ----------
        accessible : dict
            URL → JSON response mapping from fetch_fda_data.

        Returns
        -------
        dict
            Keys: records, totalCount, originalTotalCount, sourceUrls.
        """
        unique: Dict[str, Any] = {}
        total_raw = 0
        source_urls: List[str] = []

        for url, payload in accessible.items():
            source_urls.append(url)
            total_raw += (
                payload.get("meta", {}).get("results", {}).get("total", 0)
            )
            for rec in payload.get("results", []):
                key = self._record_key(rec)
                unique.setdefault(key, rec)

        logger.info(
            "Collated %d unique FDA records from %d sources (original total: %d)",
            len(unique),
            len(source_urls),
            total_raw,
        )
        return {
            "records": list(unique.values()),
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
        llm_kwargs: Optional[Dict[str, Any]] = None,
        deadline_at: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Analyse a natural language query and fetch matching FDA records.

        Full pipeline:
        1. Extract regulatory concepts via LLM → FDASearchTerms.
        2. Build deterministic openFDA API URLs from concepts.
        3. Fetch records from each URL in parallel.
        4. If all URLs return empty and retry_broad=True, widen and retry once.
        5. Collate, deduplicate, and return results.

        Parameters
        ----------
        user_input : str
            Natural language FDA regulatory query.
        retry_broad : bool
            Whether to retry with a single broad term on full empty result.

        Returns
        -------
        dict
            success, data (records/totalCount/sourceUrls),
            total_count, records_returned, source_url, all_source_urls,
            failed_urls, attempted_urls, query_analysis.

        Examples
        --------
        >>> result = fetcher.analyze_user_query("recalls for contaminated insulin")
        >>> result["success"]
        True
        >>> result["query_analysis"]["url_strategy"]
        'hybrid_llm_terms_deterministic_builder'
        """
        deadline_at = deadline_at or dict(llm_kwargs or {}).get(
            "deadline_at"
        )
        try:
            ensure_deadline(deadline_at)
            # Step 1 — Extract concepts
            terms = self.extract_search_terms(
                user_input,
                llm_kwargs=llm_kwargs,
                deadline_at=deadline_at,
            )
            logger.info(
                "FDA search term extraction complete term_count=%d",
                len(terms.all_terms),
            )

            # Step 2 — Build URLs
            urls = self.build_urls(terms)

            # Step 3 — Parallel fetch
            ensure_deadline(deadline_at)
            accessible, failed = self.fetch_fda_data(
                urls, deadline_at=deadline_at
            )

            # Step 4 — Broad retry on total failure
            if not accessible and retry_broad and terms.all_terms:
                ensure_deadline(deadline_at)
                logger.warning(
                    "All %d URLs returned empty. Retrying in broad mode...",
                    len(urls),
                )
                broad_urls = self.build_urls(terms, broad_mode=True)
                accessible, failed_broad = self.fetch_fda_data(
                    broad_urls, deadline_at=deadline_at
                )
                failed += failed_broad
                urls += broad_urls

            if not accessible:
                return {
                    "success": False,
                    "error": "No FDA records found from any generated URLs",
                    "data": None,
                    "total_count": 0,
                    "failed_urls": failed,
                    "attempted_urls": urls,
                    "source_url": "",
                }

            # Step 5 — Collate
            collated = self.collate_records_data(accessible)

            return {
                "success": True,
                "data": collated,
                "total_count": collated["totalCount"],
                "records_returned": len(collated["records"]),
                "source_url": (
                    collated["sourceUrls"][0] if collated["sourceUrls"] else ""
                ),
                "all_source_urls": collated["sourceUrls"],
                "failed_urls": failed,
                "attempted_urls": urls,
                "query_analysis": {
                    "original_query": user_input,
                    "extracted_terms": terms.model_dump(),
                    "urls_attempted": len(urls),
                    "urls_successful": len(accessible),
                    "unique_records_found": collated["totalCount"],
                    "url_strategy": "hybrid_llm_terms_deterministic_builder",
                },
            }

        except RuntimeDeadlineExceeded as exc:
            return {
                "success": False,
                "error": "runtime_deadline_exhausted",
                "error_type": "runtime_deadline_exhausted",
                "data": None,
                "total_count": 0,
                "source_url": "",
            }
        except Exception as exc:
            logger.error(
                "FDA query analysis failed error_type=%s",
                safe_error_type(exc),
            )
            return {
                "success": False,
                "error": f"fda_query_failed:{safe_error_type(exc)}",
                "error_type": safe_error_type(exc),
                "data": None,
                "total_count": 0,
                "source_url": "",
            }