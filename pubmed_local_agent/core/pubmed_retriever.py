"""
core.pubmed_retriever v2
~~~~~~~~~~~~~~~~~~~~~~~~

Scalable PubMed/PMC crawler that can collect up to 2e5 papers per run
while respecting NCBI rate limits.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import math
import os
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------- #
# Constants & endpoints
# ---------------------------------------------------------------------- #
USER_AGENT = "pubmed-rag/1.1 (+https://github.com/your-org/pubmed-rag)"
BASE_EUTIL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH = f"{BASE_EUTIL}/esearch.fcgi"
ESUMMARY = f"{BASE_EUTIL}/esummary.fcgi"
EFETCH = f"{BASE_EUTIL}/efetch.fcgi"

PAGE_SIZE = 10_000   # max retmax NCBI allows
BATCH_PMIDS = 200  # process in RAM‑friendly slices


def _build_session(total_retries: int = 15, backoff: float = 0.4) -> requests.Session:
    retry_cfg = Retry(
        total=total_retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"})
    )
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    s.mount("https://", HTTPAdapter(max_retries=retry_cfg))
    s.mount("http://", HTTPAdapter(max_retries=retry_cfg))
    return s


class PubMedRetriever:
    """
    Fetches up to `max_papers` PubMed articles + optional PMC full texts.
    """

    def __init__(self, request_delay: float = 0.34):
        self.session = _build_session()
        self.delay = request_delay

    # ------------------------------------------------------------------ #
    # Public entry
    # ------------------------------------------------------------------ #
    def create_corpus(
        self,
        search_url: str,
        max_papers: int = 25,
        include_fulltext: bool = False,
    ) -> Dict[str, Dict]:
        logger.info("Starting PubMed corpus creation")
        pmids = self._collect_pmids(search_url, max_papers)
        logger.info("Total PMIDs collected: %d", len(pmids))

        corpus: Dict[str, Dict] = {}
        for slice_start in range(0, len(pmids), BATCH_PMIDS):
            slice_pmids = pmids[slice_start: slice_start + BATCH_PMIDS]
            logger.info("Processing slice %d‑%d", slice_start, slice_start + len(slice_pmids))

            # Metadata fetch in bulk (ESummary supports 10 000 IDs)
            meta_map = self._bulk_metadata(slice_pmids)

            # Parallel abstract / full‑text fetching
            with cf.ThreadPoolExecutor(max_workers=8) as pool:
                futures = {
                    pool.submit(self._build_paper_entry, pmid, meta_map.get(pmid, {}), include_fulltext): pmid
                    for pmid in slice_pmids
                }
                for fut in cf.as_completed(futures):
                    paper = fut.result()
                    if paper:
                        corpus[paper["paper_id"]] = paper

        return corpus

    # ------------------------------------------------------------------ #
    # Step 1 · Collect PMIDs with paging
    # ------------------------------------------------------------------ #
    def _collect_pmids(self, base_url: str, max_papers: int) -> List[str]:
        all_pmids: List[str] = []
        page = 0
        while len(all_pmids) < max_papers:
            # check if retmax is set in the URL
            if "retmax" not in base_url:
                paged_url = f"{base_url}&retstart={page * PAGE_SIZE}&retmax={PAGE_SIZE}"
            else:
                paged_url = f"{base_url}&retstart={page * PAGE_SIZE}"
            logger.info(f"Paging URL: {paged_url}")
            ids = self._get_json(paged_url).get("esearchresult", {}).get("idlist", [])
            logger.info("PMIDs collected: %s", ids)
            if not ids:
                break
            all_pmids.extend(ids)
            if len(ids) < PAGE_SIZE:  # last page
                break
            page += 1
            time.sleep(self.delay)
            if len(all_pmids) >= 100_000:  # API ceiling
                break
        return all_pmids[:max_papers]

    # ------------------------------------------------------------------ #
    # Step 2 · Bulk metadata
    # ------------------------------------------------------------------ #
    def _bulk_metadata(self, pmid_list: List[str]) -> Dict[str, Dict]:
        ids = ",".join(pmid_list)
        url = f"{ESUMMARY}?db=pubmed&id={ids}&retmode=json"
        data = self._get_json(url).get("result", {})
        return {pid: meta for pid, meta in data.items() if pid != "uids"}

    # ------------------------------------------------------------------ #
    # Step 3 · Per‑paper assembly
    # ------------------------------------------------------------------ #
    def _build_paper_entry(
        self,
        pmid: str,
        meta: Dict,
        include_fulltext: bool,
    ) -> Optional[Dict]:
        try:
            title = meta.get("title", "No title")
            journal = meta.get("fulljournalname", "Unknown journal")
            year = meta.get("pubdate", "n.d.").split()[0]
            authors = ", ".join(a.get("name") for a in meta.get("authors", [])[:20])

            abstract = self._fetch_abstract(pmid)

            content_parts = [
                f"TITLE: {title}",
                f"AUTHORS: {authors}",
                f"JOURNAL: {journal} ({year})",
                f"ABSTRACT: {abstract}",
            ]

            full_text = "Full text not retrieved."
            if include_fulltext:
                pmc_xml = self._fetch_fulltext_xml(meta)
                if pmc_xml:
                    full_text = self._xml_to_text(pmc_xml) or full_text
                    content_parts.append(f"FULL TEXT: {full_text}")

            return {
                "paper_id": pmid,
                "paper_title": title,
                "paper_authors": authors,
                "paper_year": year,
                "paper_journal": journal,
                "paper_abstract": abstract,
                "paper_full_text": full_text,
                "content": "\n\n".join(content_parts),
            }
        except Exception as exc:
            logger.exception("Failed to parse PMID %s: %s", pmid, exc)
            return None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _fetch_abstract(self, pmid: str) -> str:
        url = f"{EFETCH}?db=pubmed&id={pmid}&retmode=xml&rettype=abstract"
        xml_text = self._get_text(url)
        root = ET.fromstring(xml_text)
        abstracts = [
            (n.get("Label") or "", (n.text or "").strip())
            for n in root.findall(".//AbstractText")
        ]
        return (
            " ".join(f"{lbl}: {txt}" if lbl else txt for lbl, txt in abstracts if txt)
            or "No abstract available."
        )

    def _fetch_fulltext_xml(self, meta: Dict) -> Optional[str]:
        pmcid = next(
            (x.get("value") for x in meta.get("articleids", []) if x.get("idtype") == "pmc"),
            None,
        )
        if not pmcid:
            return None
        url = f"{EFETCH}?db=pmc&id={pmcid}&retmode=xml"
        return self._get_text(url)

    @staticmethod
    def _xml_to_text(xml_text: str) -> str:
        root = ET.fromstring(xml_text)
        body = root.find(".//body")
        if body is None:
            return ""
        parts: List[str] = []

        def walk(elem: ET.Element):
            if elem.text:
                parts.append(elem.text.strip())
            for child in elem:
                walk(child)
                if child.tail:
                    parts.append(child.tail.strip())

        walk(body)
        return " ".join(parts)

    # HTTP helpers
    # ------------

    def _get_json(self, url: str) -> Dict:
        resp = self.session.get(url, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def _get_text(self, url: str) -> str:
        resp = self.session.get(url, timeout=60)
        resp.raise_for_status()
        return resp.text
