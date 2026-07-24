"""
data_fetcher.py — Phase 7: PubMed data fetcher.

Merges legacy QueryProcessor (query_processor.py) and PubMedClient
(pubmed_client.py) into a single PubMedFetcher, mirroring the
ClinicalTrialsFetcher architecture.

Architecture
------------
1. LLM extracts medical concepts only (PubMedSearchTerms).
2. Deterministic URL builder constructs valid NCBI ESearch URLs.
3. Pydantic validates term lists before URL construction.
4. Parallel EFetch retrieves paper metadata + abstracts.
5. Optional PMC full-text extraction.

Changes from legacy
-------------------
* LLM no longer generates raw search strings — only extracts concepts.
* All LLM calls route through LLMClient (LiteLLM, provider-agnostic).
* Thread-safe rate limiter (threading.Lock) replaces asyncio.Lock.
* URL construction is deterministic and always NCBI-compliant.
"""
from __future__ import annotations

import json
import logging
import re
import time
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus
import os
from dotenv import load_dotenv  # type: ignore
load_dotenv()

import requests # type: ignore
from requests.adapters import HTTPAdapter # type: ignore
from urllib3.util.retry import Retry # type: ignore
from pydantic import BaseModel, field_validator # type: ignore
from tenacity import retry, stop_after_attempt, wait_exponential # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NCBI_API_KEY= os.getenv("NCBI_API_KEY", default=None)
PUBMED_BASE  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL  = f"{PUBMED_BASE}/esearch.fcgi"
EFETCH_URL   = f"{PUBMED_BASE}/efetch.fcgi"
ESUMMARY_URL = f"{PUBMED_BASE}/esummary.fcgi"

MAX_PAPERS_PER_BATCH = 200   # NCBI ESummary limit
USER_AGENT = "MedicalResearchAgent/7.0 (research; contact: research@example.com)"

_TERM_EXTRACTION_PROMPT = """\
You are a medical librarian expert in PubMed search strategies.
Analyse this medical research question and extract search terms.

Return ONLY a JSON object with exactly these four keys. No markdown, no explanation.

{{
  "medical_concepts": ["2-4 core disease or treatment concepts"],
  "mesh_terms": ["2-4 MeSH Medical Subject Headings"],
  "synonyms": ["2-4 abbreviations, brand names, or alternative names"],
  "related_terms": ["2-4 broader or narrower related terms"]
}}

Rules:
- Plain strings only — no URLs, no field qualifiers, no ICD codes.
- Use terminology researchers use in paper titles and abstracts.
- If a category has no relevant terms, return an empty list.

Query: {query}
"""


# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------

class PubMedSearchTerms(BaseModel):
    """Validated search concepts extracted from a user query."""

    medical_concepts: List[str] = []
    mesh_terms: List[str] = []
    synonyms: List[str] = []
    related_terms: List[str] = []

    @field_validator(
        "medical_concepts", "mesh_terms", "synonyms", "related_terms",
        mode="before",
    )
    @classmethod
    def clean_and_cap(cls, v: Any) -> List[str]:
        if not isinstance(v, list):
            return []
        cleaned = []
        for item in v:
            if isinstance(item, str):
                item = re.sub(r"[^\w\s\-]", "", item).strip().lower()
                item = re.sub(r"\b[a-z]\d+\b", "", item).strip()
                if len(item) >= 3:
                    cleaned.append(item)
        return cleaned[:6]

    @property
    def all_unique_terms(self) -> List[str]:
        """Flat deduplicated list — concepts first."""
        seen: set = set()
        out: List[str] = []
        for t in (
            self.medical_concepts
            + self.mesh_terms
            + self.synonyms
            + self.related_terms
        ):
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    @classmethod
    def from_query(cls, query: str) -> "PubMedSearchTerms":
        """Naive keyword fallback when LLM extraction fails."""
        stopwords = {
            "the", "for", "and", "or", "in", "of", "a", "an", "with",
            "on", "is", "are", "was", "be", "to", "at", "what", "how",
        }
        words = [
            w.lower()
            for w in re.split(r"\W+", query)
            if len(w) > 3 and w.lower() not in stopwords
        ]
        return cls(
            medical_concepts=words[:3],
            mesh_terms=[],
            synonyms=words[3:5],
            related_terms=[],
        )


# ---------------------------------------------------------------------------
# Thread-safe rate limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    def __init__(self, calls_per_second: float) -> None:
        self._min_interval = 1.0 / calls_per_second
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.time()


# ---------------------------------------------------------------------------
# Paper dataclass (lightweight — avoids importing models from Phase 4)
# ---------------------------------------------------------------------------

class Paper:
    """Minimal paper container. model_dump() returns JSON-serializable dict."""

    __slots__ = (
        "pmid", "title", "authors", "journal", "year",
        "volume", "issue", "pages", "doi", "abstract", "full_text",
    )

    def __init__(
        self,
        pmid: str,
        title: str,
        authors: List[str],
        journal: str,
        year: str,
        volume: Optional[str] = None,
        issue: Optional[str] = None,
        pages: Optional[str] = None,
        doi: Optional[str] = None,
        abstract: str = "",
        full_text: Optional[str] = None,
    ) -> None:
        self.pmid = pmid
        self.title = title
        self.authors = authors
        self.journal = journal
        self.year = year
        self.volume = volume
        self.issue = issue
        self.pages = pages
        self.doi = doi
        self.abstract = abstract
        self.full_text = full_text

    def model_dump(self) -> Dict[str, Any]:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "authors": self.authors,
            "journal": self.journal,
            "year": self.year,
            "volume": self.volume,
            "issue": self.issue,
            "pages": self.pages,
            "doi": self.doi,
            "abstract": self.abstract,
            "full_text": self.full_text,
        }


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------

def _build_session() -> requests.Session:
    retry_strategy = Retry(
        total=4,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
    )
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ---------------------------------------------------------------------------
# Main fetcher
# ---------------------------------------------------------------------------

class PubMedFetcher:
    """Fetches papers from PubMed/PMC using NCBI EUtils API.

    Parameters
    ----------
    llm_client : LLMClient, optional
        Lazy-loaded on first use if None.
    max_workers : int
        Thread pool size for parallel paper fetching.
    ncbi_api_key : str, optional
        Enables 10 req/s instead of 3 req/s.
    include_fulltext : bool
        Whether to attempt PMC full-text extraction.
    """

    def __init__(
        self,
        llm_client=None,
        max_workers: int = 8,
        ncbi_api_key: Optional[str] = None,
        include_fulltext: bool = False,
    ) -> None:
        import os
        self._llm = llm_client
        self.max_workers = max_workers
        self.include_fulltext = include_fulltext
        self._api_key = ncbi_api_key or os.getenv("NCBI_API_KEY")
        request_delay = 0.1 if self._api_key else 0.34
        self._rate_limiter = _RateLimiter(calls_per_second=1.0 / request_delay)
        self._session = _build_session()

    @property
    def llm(self):
        if self._llm is None:
            from llm_client import LLMClient
            self._llm = LLMClient()
        return self._llm

    # ------------------------------------------------------------------
    # Step 1 — concept extraction
    # ------------------------------------------------------------------

    def extract_search_terms(
        self,
        user_query: str,
        max_retries: int = 3,
    ) -> PubMedSearchTerms:
        """Extract medical search concepts from user query via LLM.

        Falls back to naive keyword extraction on repeated failure.
        """
        prompt = _TERM_EXTRACTION_PROMPT.format(query=user_query)

        for attempt in range(max_retries):
            try:
                raw = self.llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=300,
                )
                # Strip markdown fences if present
                clean = re.sub(r"```(?:json)?", "", raw).strip()
                j_start = clean.find("{")
                j_end = clean.rfind("}") + 1
                if j_start == -1:
                    raise ValueError("No JSON object in response")
                data = json.loads(clean[j_start:j_end])
                terms = PubMedSearchTerms.model_validate(data)
                logger.info(
                    "Extracted %d concepts, %d MeSH, %d synonyms",
                    len(terms.medical_concepts),
                    len(terms.mesh_terms),
                    len(terms.synonyms),
                )
                return terms
            except Exception as exc:
                logger.warning(
                    "Term extraction attempt %d/%d failed: %s",
                    attempt + 1, max_retries, exc,
                )
                if attempt < max_retries - 1:
                    time.sleep(1)

        logger.warning("All LLM attempts failed — using keyword fallback")
        return PubMedSearchTerms.from_query(user_query)

    # ------------------------------------------------------------------
    # Step 2 — URL construction
    # ------------------------------------------------------------------

    def build_urls(
        self,
        terms: PubMedSearchTerms,
        recent_years: int = 2,
        max_results_per_url: int = 50,
    ) -> List[str]:
        """Build deterministic NCBI ESearch URLs from extracted terms.

        Constructs up to 4 URLs covering:
        - Recent papers (last N years, relevance-sorted)
        - Recent MeSH-qualified papers
        - Foundational reviews (2010–cutoff)
        - Broad sweep (all terms, all dates)
        """
        urls: List[str] = []
        today = datetime.now()
        recent_min = (today - timedelta(days=recent_years * 365)).strftime("%Y/%m/%d")
        recent_max = today.strftime("%Y/%m/%d")
        found_min = "2010/01/01"

        def _url(
            term_clause: str,
            min_date: str,
            max_date: str,
            sort: str = "relevance",
            retmax: int = max_results_per_url,
        ) -> str:
            return (
                f"{ESEARCH_URL}?db=pubmed"
                f"&term={term_clause}"
                f"&retmode=json"
                f"&datetype=pdat"
                f"&mindate={min_date}"
                f"&maxdate={max_date}"
                f"&retmax={retmax}"
                f"&sort={sort}"
            )

        all_terms = terms.all_unique_terms

        # 1. Recent — broad concept search
        if all_terms:
            clause = "+OR+".join(quote_plus(t) for t in all_terms[:12])
            urls.append(_url(f"({clause})", recent_min, recent_max, sort="relevance"))

        # 2. Recent — MeSH-qualified (higher precision)
        if terms.mesh_terms:
            mesh_clause = "+OR+".join(
                f"{quote_plus(t)}[MeSH]" for t in terms.mesh_terms[:5]
            )
            urls.append(_url(f"({mesh_clause})", recent_min, recent_max, sort="pub_date"))

        # 3. Foundational reviews
        if terms.medical_concepts:
            concept_clause = "+OR+".join(
                quote_plus(t) for t in terms.medical_concepts[:5]
            )
            type_filter = (
                quote_plus("Review") + "[pt]"
                + "+OR+" + quote_plus("Meta-Analysis") + "[pt]"
                + "+OR+" + quote_plus("Systematic Review") + "[pt]"
            )
            urls.append(
                _url(
                    f"({concept_clause})+AND+({type_filter})",
                    found_min,
                    recent_min,
                    sort="relevance",
                    retmax=max_results_per_url // 2,
                )
            )

        # 4. Broad sweep — full date range
        if all_terms and len(urls) < 4:
            clause = "+OR+".join(quote_plus(t) for t in all_terms[:8])
            urls.append(
                _url(f"({clause})", found_min, recent_max, sort="relevance",
                     retmax=max_results_per_url // 2)
            )

        logger.info("Built %d ESearch URLs", len(urls))
        return urls

    # ------------------------------------------------------------------
    # Step 3 — ESearch for PMIDs
    # ------------------------------------------------------------------

    def search_pmids(
        self,
        urls: List[str],
        max_results: int = 100,
    ) -> Tuple[List[str], List[str]]:
        """Execute ESearch URLs and return deduplicated PMIDs.

        Returns
        -------
        tuple[list[str], list[str]]
            (pmids, urls_successfully_used)
        """
        seen: dict = {}  # pmid -> order of first appearance
        urls_used: List[str] = []

        for url in urls:
            full_url = url
            if self._api_key:
                full_url += f"&api_key={self._api_key}"

            self._rate_limiter.wait()
            try:
                resp = self._session.get(full_url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                pmids = data.get("esearchresult", {}).get("idlist", [])
                if pmids:
                    for p in pmids:
                        if p not in seen:
                            seen[p] = len(seen)
                    urls_used.append(url)
                    logger.info("ESearch returned %d PMIDs from %s", len(pmids), url[:80])
            except Exception as exc:
                logger.warning("ESearch failed for URL %s: %s", url[:80], exc)

        ordered = sorted(seen.keys(), key=lambda p: seen[p])
        result = ordered[:max_results]
        logger.info("Total unique PMIDs: %d (capped at %d)", len(seen), max_results)
        return result, urls_used

    # ------------------------------------------------------------------
    # Step 4 — Parallel paper fetching
    # ------------------------------------------------------------------

    def fetch_papers(
        self,
        pmids: List[str],
        include_fulltext: Optional[bool] = None,
    ) -> Dict[str, Paper]:
        """Fetch paper metadata + abstracts in parallel.

        Parameters
        ----------
        pmids : list[str]
            PubMed IDs to fetch.
        include_fulltext : bool, optional
            Overrides instance default.
        """
        if not pmids:
            return {}

        ft = include_fulltext if include_fulltext is not None else self.include_fulltext
        unique = list(dict.fromkeys(pmids))
        logger.info("Fetching %d papers (fulltext=%s)...", len(unique), ft)

        metadata = self._bulk_fetch_metadata(unique)

        papers: Dict[str, Paper] = {}

        def _fetch_one(pmid: str) -> Optional[Paper]:
            self._rate_limiter.wait()
            return self._build_paper(pmid, metadata.get(pmid, {}), ft)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_fetch_one, pmid): pmid for pmid in unique}
            for future in as_completed(futures):
                paper = future.result()
                if paper:
                    papers[paper.pmid] = paper

        logger.info("Fetched %d / %d papers successfully", len(papers), len(unique))
        return papers

    # ------------------------------------------------------------------
    # Internal helpers — ported from PubMedClient
    # ------------------------------------------------------------------

    def _bulk_fetch_metadata(self, pmids: List[str]) -> Dict[str, Dict]:
        """Batch-fetch ESummary metadata for all PMIDs."""
        metadata: Dict[str, Dict] = {}

        for i in range(0, len(pmids), MAX_PAPERS_PER_BATCH):
            batch = pmids[i : i + MAX_PAPERS_PER_BATCH]
            ids_param = ",".join(batch)
            url = f"{ESUMMARY_URL}?db=pubmed&id={ids_param}&retmode=json"
            if self._api_key:
                url += f"&api_key={self._api_key}"

            self._rate_limiter.wait()
            try:
                resp = self._session.get(url, timeout=60)
                resp.raise_for_status()
                data = resp.json().get("result", {})
                for pmid in batch:
                    if pmid in data and pmid != "uids":
                        metadata[pmid] = data[pmid]
            except Exception as exc:
                logger.warning("ESummary batch failed: %s", exc)

        return metadata

    def _build_paper(
        self,
        pmid: str,
        meta: Dict,
        include_fulltext: bool,
    ) -> Optional[Paper]:
        """Build a Paper from ESummary metadata + fetched abstract."""
        try:
            title   = meta.get("title", "No title")
            journal = meta.get("fulljournalname", "Unknown journal")
            pub_date = meta.get("pubdate", "")
            year    = pub_date.split()[0] if pub_date else "n.d."
            volume  = meta.get("volume") or None
            issue   = meta.get("issue") or None
            pages   = meta.get("pages") or None

            doi = None
            for aid in meta.get("articleids", []):
                if aid.get("idtype") == "doi":
                    doi = aid.get("value")
                    break

            authors = [
                a.get("name", "") for a in meta.get("authors", [])[:20]
                if a.get("name")
            ]

            abstract = self._fetch_abstract(pmid)
            full_text = self._fetch_fulltext(meta) if include_fulltext else None

            return Paper(
                pmid=pmid,
                title=title,
                authors=authors,
                journal=journal,
                year=year,
                volume=volume,
                issue=issue,
                pages=pages,
                doi=doi,
                abstract=abstract,
                full_text=full_text,
            )
        except Exception as exc:
            logger.warning("_build_paper failed for PMID %s: %s", pmid, exc)
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=5))
    def _fetch_abstract(self, pmid: str) -> str:
        """Fetch and parse abstract XML for a single PMID."""
        url = f"{EFETCH_URL}?db=pubmed&id={pmid}&retmode=xml&rettype=abstract"
        if self._api_key:
            url += f"&api_key={self._api_key}"
        try:
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            parts = []
            for elem in root.findall(".//AbstractText"):
                label = elem.get("Label", "")
                text  = (elem.text or "").strip()
                if text:
                    parts.append(f"{label}: {text}" if label else text)
            return " ".join(parts) if parts else "No abstract available."
        except Exception as exc:
            logger.debug("Abstract fetch failed for PMID %s: %s", pmid, exc)
            return "No abstract available."

    def _fetch_fulltext(self, meta: Dict) -> Optional[str]:
        """Fetch PMC full text if a PMC ID exists in article metadata."""
        pmc_id = None
        for aid in meta.get("articleids", []):
            if aid.get("idtype") == "pmc":
                pmc_id = aid.get("value")
                break
        if not pmc_id:
            return None

        url = f"{EFETCH_URL}?db=pmc&id={pmc_id}&retmode=xml"
        if self._api_key:
            url += f"&api_key={self._api_key}"
        try:
            resp = self._session.get(url, timeout=60)
            resp.raise_for_status()
            return self._extract_pmc_text(resp.text)
        except Exception as exc:
            logger.debug("Full-text fetch failed for PMC %s: %s", pmc_id, exc)
            return None

    @staticmethod
    def _extract_pmc_text(xml_text: str) -> str:
        """Walk PMC XML body and extract readable text."""
        try:
            root = ET.fromstring(xml_text)
            body = root.find(".//body")
            if body is None:
                return ""
            parts: List[str] = []

            def _walk(elem):
                if elem.text:
                    parts.append(elem.text.strip())
                for child in elem:
                    _walk(child)
                    if child.tail:
                        parts.append(child.tail.strip())

            _walk(body)
            return " ".join(p for p in parts if p)
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def analyze_user_query(
        self,
        user_input: str,
        max_papers: int = 50,
        include_fulltext: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Full pipeline: concept extraction → URL build → search → fetch.

        Returns
        -------
        dict
            Keys: success, papers (Dict[str,Paper]), total_count,
            pmids_fetched, urls_used, query_analysis, error.
        """
        logger.info("PubMedFetcher.analyze_user_query: %s", user_input[:80])
        try:
            terms = self.extract_search_terms(user_input)
            urls  = self.build_urls(terms)
            pmids, urls_used = self.search_pmids(urls, max_results=max_papers)

            ft = include_fulltext if include_fulltext is not None else self.include_fulltext
            papers = self.fetch_papers(pmids, include_fulltext=ft)

            return {
                "success": True,
                "papers": papers,
                "total_count": len(papers),
                "pmids_fetched": pmids,
                "urls_used": urls_used,
                "query_analysis": {
                    "original_query": user_input,
                    "extracted_terms": terms.model_dump(),
                    "urls_attempted": len(urls),
                    "urls_successful": len(urls_used),
                    "unique_papers_fetched": len(papers),
                },
                "error": None,
            }
        except Exception as exc:
            logger.error("analyze_user_query failed: %s", exc, exc_info=True)
            return {
                "success": False,
                "papers": {},
                "total_count": 0,
                "pmids_fetched": [],
                "urls_used": [],
                "query_analysis": {},
                "error": str(exc),
            }