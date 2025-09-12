"""
fda_fetcher.py  – Parallel, limit-capped fetcher for openFDA

Major optimisations
-------------------
1.  ThreadPool (I/O-bound) for concurrent HTTP GETs.
2.  Automatic limit-capping to 200 records per URL.
3.  Connection reuse via a single `requests.Session`.

NOTE: Requires only the stdlib + requests.
"""
from __future__ import annotations

import json
import time
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse, quote_plus

import requests  # type: ignore

# ------------------------------------------------------------------------------
# Logger configuration
# ------------------------------------------------------------------------------
logger = logging.getLogger("fda_fetcher")
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    _h = logging.StreamHandler()
    _h.setLevel(logging.DEBUG)
    _h.setFormatter(
        logging.Formatter(
            "[%(levelname)s] %(asctime)s - %(filename)s:%(lineno)d - %(message)s"
        )
    )
    logger.addHandler(_h)

# ------------------------------------------------------------------------------
# Fetcher Agent
# ------------------------------------------------------------------------------
class FdaFetcherAgent:
    """
    Translates natural language → openFDA URLs (via LLM) and fetches them in
    parallel, capped to `limit=200` to reduce payload size and latency.
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(self, openai_client=None, model: str = "gpt-4o", max_workers: int = 8):
        self.base_url = "https://api.fda.gov"
        self.client = openai_client
        self.model = model
        self.max_workers = max_workers
        self.system_prompt = self._get_system_prompt()

        if not self.client:
            logger.warning("OpenAI client not provided. URL generation will be skipped.")

        # Maintain a single Session for connection-pool reuse
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "TrendForge-FDA/1.0"})

        logger.info("FDA Fetcher Agent initialised (model=%s, workers=%d)", model, max_workers)

    # ------------------------------------------------------------------ #
    # Prompt
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_system_prompt() -> str:
        return r'''
# openFDA API URL-Builder Prompt

You are an expert agent that converts user questions into **working** openFDA
API v1 URLs. **Every returned URL must yield non-empty JSON** where
`meta.results.total > 0` and `results` contains at least one record.

## Base
`https://api.fda.gov`

## Common Endpoints
| Dataset | Path | Typical Use |
|---------|------|-------------|
| Drug Labeling           | `/drug/label.json`      | product labels         |
| Drug Adverse Events     | `/drug/event.json`      | safety reports         |
| Drug NDC Directory      | `/drug/ndc.json`        | product codes          |
| Device Events           | `/device/event.json`    | device adverse events  |
| Device Recalls          | `/device/recall.json`   | recalls                |
| Food Enforcement        | `/food/enforcement.json`| food recalls           |

## Core Parameters
* `search`   – Elasticsearch-style field search  
* `limit`    – 1-1000 (default = 200) 
* `skip`     – offset for pagination  
* `count`    – aggregation on a field (optional)

**Always prefer simple, broad `search=` terms first** (e.g. `search=brand_name:aspirin`)
then add refinements only if results would still be plentiful.

## Success Rule
A URL is valid **only if**  
`meta.results.total > 0` **AND** `results` list is non-empty.

## Diversified URL Strategy (return exactly 5)  
1. Broad free-text search across all fields.  
2. Field-specific search (e.g. `openfda.substance_name`).  
3. Alternative spelling / synonym.  
4. Related concept or effect.  
5. Combined or filtered query that still returns results.
6. `limit=200` IMPORTANT – to maximize top relevant records.

## Output
Return **only**:
```json
{
  "urls": ["url1", "url2", "url3", "url4", "url5"]
}
```
No extra text, no markdown.
'''

    # ------------------------------------------------------------------ #
    # URL generation via LLM
    # ------------------------------------------------------------------ #
    def generate_api_urls(
        self,
        user_query: str,
        max_retries: int = 3,
        wait_seconds: int = 2,
    ) -> Optional[Dict[str, List[str]]]:
        if not self.client:
            raise ValueError("OpenAI client not provided; cannot generate URLs.")

        logger.info("Generating URLs for query: %s", user_query)

        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_query},
                    ],
                )
                raw = resp.choices[0].message.content
                json_start, json_end = raw.find("{"), raw.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    data = json.loads(raw[json_start:json_end])
                    urls = data.get("urls", [])
                    if isinstance(urls, list) and urls:
                        logger.info("✅ Generated %d URL(s)", len(urls))
                        # enforce limit cap
                        capped = [self._ensure_limit_200(u) for u in urls]
                        return {"urls": capped}
            except Exception as exc:
                logger.warning("Attempt %d failed: %s", attempt + 1, exc)

            if attempt < max_retries - 1:
                time.sleep(wait_seconds)

        logger.error("🚫 Failed to obtain URLs after %d attempts", max_retries)
        return None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ensure_limit_200(url: str) -> str:
        """
        Ensure every URL contains limit<=200. If limit absent or >200, set to 200.
        """
        parsed = urlparse(url)
        qs = parse_qs(parsed.query, keep_blank_values=True)

        limit_val = int(qs.get("limit", ["200"])[0] or 0)
        if limit_val == 0 or limit_val > 200:
            qs["limit"] = ["200"]

        # Re-encode (parse_qs gives lists)
        new_qs = urlencode(qs, doseq=True, quote_via=quote_plus)
        capped_url = urlunparse(parsed._replace(query=new_qs))
        return capped_url

    def _fetch_single(self, url: str, timeout: int = 30) -> Tuple[str, Optional[Any], Optional[str]]:
        """
        Thread worker: fetch URL, return (url, json|None, error|None)
        """
        try:
            logger.debug("→ Requesting %s", url)
            r = self._session.get(url, timeout=timeout)
            r.raise_for_status()

            if "application/json" not in r.headers.get("Content-Type", ""):
                return url, None, "non-json"
            data = r.json()
            total = data.get("meta", {}).get("results", {}).get("total", 0)
            results = data.get("results", [])
            if total > 0 and results:
                logger.info("✅ %d records from %s", total, url)
                return url, data, None
            return url, None, "empty"
        except Exception as exc:
            return url, None, str(exc)

    # ------------------------------------------------------------------ #
    # Parallel data fetch
    # ------------------------------------------------------------------ #
    def fetch_fda_data(self, urls: List[str]) -> Tuple[Dict[str, Any], List[str]]:
        logger.info("Fetching %d URL(s) in parallel…", len(urls))
        accessible: Dict[str, Any] = {}
        failed: List[str] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._fetch_single, u): u for u in urls}
            for fut in as_completed(futures):
                url, data, err = fut.result()
                if data:
                    accessible[url] = data
                else:
                    logger.warning("✗ %s – %s", url, err)
                    failed.append(url)

        logger.info("Fetch complete: %d success, %d failed", len(accessible), len(failed))
        return accessible, failed

    # ------------------------------------------------------------------ #
    # Record collation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _record_key(rec: Dict[str, Any]) -> str:
        for k in ("id", "report_id", "safetyreportid", "lot_number"):
            if k in rec:
                return f"{k}:{rec[k]}"
        return hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()

    def collate_records_data(self, payloads: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Collating %d response payload(s)…", len(payloads))
        merged: Dict[str, Dict[str, Any]] = {}
        orig_total = 0

        for url, data in payloads.items():
            orig_total += data.get("meta", {}).get("results", {}).get("total", 0)
            for rec in data.get("results", []):
                key = self._record_key(rec)
                merged.setdefault(key, rec)

        logger.info("✅ Collated %d unique records (from %d original)", len(merged), orig_total)
        return {
            "records": list(merged.values()),
            "totalCount": len(merged),
            "originalTotalCount": orig_total,
            "sourceUrls": list(payloads.keys()),
        }

    # ------------------------------------------------------------------ #
    # Public: analyse query end-to-end
    # ------------------------------------------------------------------ #
    def analyze_user_query(self, user_input: str) -> Dict[str, Any]:
        logger.info("🔎 User query: %s", user_input)

        url_payload = self.generate_api_urls(user_input)
        if not url_payload:
            return {"success": False, "error": "URL generation failed", "data": None}

        urls = url_payload["urls"]
        success_payloads, failed = self.fetch_fda_data(urls)

        if not success_payloads:
            return {
                "success": False,
                "error": "All URLs failed or returned empty data",
                "failed_urls": failed,
                "attempted_urls": urls,
            }

        collated = self.collate_records_data(success_payloads)
        return {
            "success": True,
            "data": collated,
            "total_count": collated["totalCount"],
            "records_returned": len(collated["records"]),
            "source_url": collated["sourceUrls"][0],
            "all_source_urls": collated["sourceUrls"],
            "failed_urls": failed,
            "attempted_urls": urls,
        }


# ------------------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------------------
def create_fda_agent(openai_client=None, model: str = "gpt-4o", max_workers: int = 8) -> FdaFetcherAgent:
    return FdaFetcherAgent(openai_client=openai_client, model=model, max_workers=max_workers)
# ------------------------------------------------------------------------------











###################################################
# import json
# import time
# import hashlib
# import requests #type: ignore
# from typing import Dict, List, Optional, Tuple, Any
# import logging

# # ------------------------------------------------------------------------------
# # Logger Configuration
# # ------------------------------------------------------------------------------
# logger = logging.getLogger("fda_fetcher")
# logger.setLevel(logging.DEBUG)

# if not logger.hasHandlers():
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.DEBUG)
#     formatter = logging.Formatter(
#         "[%(levelname)s] %(asctime)s - %(filename)s:%(lineno)d - %(message)s"
#     )
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)

# class FdaFetcherAgent:
#     """
#     Fetcher agent for querying the openFDA API using an LLM to translate
#     natural-language queries into concrete REST URLs.
#     """

#     def __init__(self, openai_client=None, model: str = "gpt-4o"):
#         """
#         Args:
#             openai_client: Initialized OpenAI client (needed for URL generation).
#             model:         LLM model name.
#         """
#         self.base_url = "https://api.fda.gov"
#         self.client = openai_client
#         self.model = model
#         self.system_prompt = self._get_system_prompt()

#         if not self.client:
#             logger.warning(
#                 "OpenAI client not provided. URL generation will be skipped."
#             )

#         logger.info("FDA Fetcher Agent initialized using model: %s", model)

#     # --------------------------------------------------------------------- #
#     #                           PROMPT GENERATION                            #
#     # --------------------------------------------------------------------- #
#     @staticmethod
#     def _get_system_prompt() -> str:
#         """
#         System prompt instructing the LLM how to build openFDA URLs.
#         """
#         return r"""
# # openFDA API URL-Builder Prompt

# You are an expert agent that converts user questions into **working** openFDA
# API v1 URLs. **Every returned URL must yield non-empty JSON** where
# `meta.results.total > 0` and `results` contains at least one record.

# ## Base
# `https://api.fda.gov`

# ## Common Endpoints
# | Dataset | Path | Typical Use |
# |---------|------|-------------|
# | Drug Labeling           | `/drug/label.json`      | product labels         |
# | Drug Adverse Events     | `/drug/event.json`      | safety reports         |
# | Drug NDC Directory      | `/drug/ndc.json`        | product codes          |
# | Device Events           | `/device/event.json`    | device adverse events  |
# | Device Recalls          | `/device/recall.json`   | recalls                |
# | Food Enforcement        | `/food/enforcement.json`| food recalls           |

# ## Core Parameters
# * `search`   – Elasticsearch-style field search  
# * `limit`    – 1-1000 (default = 1) 
# * `skip`     – offset for pagination  
# * `count`    – aggregation on a field (optional)

# **Always prefer simple, broad `search=` terms first** (e.g. `search=brand_name:aspirin`)
# then add refinements only if results would still be plentiful.

# ## Success Rule
# A URL is valid **only if**  
# `meta.results.total > 0` **AND** `results` list is non-empty.

# ## Diversified URL Strategy (return exactly 5)  
# 1. Broad free-text search across all fields.  
# 2. Field-specific search (e.g. `openfda.substance_name`).  
# 3. Alternative spelling / synonym.  
# 4. Related concept or effect.  
# 5. Combined or filtered query that still returns results.
# 6. `limit=1000` IMPORTANT – to maximize retrieving top records.

# ## Output
# Return **only**:
# ```json
# {
#   "urls": ["url1", "url2", "url3", "url4", "url5"]
# }
# ```
# No extra text, no markdown.
# """

#     # --------------------------------------------------------------------- #
#     #                         URL GENERATION METHOD                          #
#     # --------------------------------------------------------------------- #
#     def generate_api_urls(
#         self,
#         user_query: str,
#         max_retries: int = 5,
#         wait_seconds: int = 2,
#     ) -> Optional[Dict[str, List[str]]]:
#         """
#         Ask the LLM to turn a natural-language query into 5 openFDA URLs.
#         """
#         logger.info("Generating FDA API URLs for query: '%s'", user_query)

#         if not self.client:
#             raise ValueError("OpenAI client not provided; cannot generate URLs.")

#         for attempt in range(max_retries):
#             try:
#                 logger.debug("Attempt %d: Sending request to OpenAI", attempt + 1)
#                 response = self.client.chat.completions.create(
#                     model=self.model,
#                     messages=[
#                         {"role": "system", "content": self.system_prompt},
#                         {"role": "user", "content": user_query},
#                     ],
#                 )
#                 content = response.choices[0].message.content
#                 json_start = content.find("{")
#                 json_end = content.rfind("}") + 1
#                 if json_start != -1 and json_end > json_start:
#                     payload = json.loads(content[json_start:json_end])
#                     if isinstance(payload.get("urls"), list) and payload["urls"]:
#                         logger.info(
#                             "✅ Generated %d URLs from LLM", len(payload["urls"])
#                         )
#                         return payload
#             except json.JSONDecodeError as e:
#                 logger.warning("❌ Attempt %d: JSON parse error: %s", attempt + 1, e)
#             except Exception as e:
#                 logger.error("❌ Attempt %d: Unexpected error: %s", attempt + 1, e)

#             if attempt < max_retries - 1:
#                 logger.debug("Retrying in %d seconds...", wait_seconds)
#                 time.sleep(wait_seconds)

#         logger.error("🚫 Failed to generate any URLs after %d attempts", max_retries)
#         return None

#     # --------------------------------------------------------------------- #
#     #                            DATA FETCHING                               #
#     # --------------------------------------------------------------------- #
#     def fetch_fda_data(
#         self, urls: List[str]
#     ) -> Tuple[Dict[str, Any], List[str]]:
#         """
#         Execute the generated URLs and return only those with records.
#         """
#         logger.info("Fetching data from %d URLs...", len(urls))
#         accessible: Dict[str, Any] = {}
#         failed: List[str] = []

#         for url in urls:
#             try:
#                 logger.debug("→ Requesting: %s", url)
#                 r = requests.get(url, timeout=30)
#                 r.raise_for_status()

#                 if "application/json" not in r.headers.get("Content-Type", ""):
#                     logger.warning("⚠️ Skipped non-JSON response from %s", url)
#                     failed.append(url)
#                     continue

#                 data = r.json()
#                 total = (
#                     data.get("meta", {})
#                     .get("results", {})
#                     .get("total", 0)
#                 )
#                 records = data.get("results", [])

#                 if total > 0 and records:
#                     logger.info("✅ %d records from: %s", total, url)
#                     accessible[url] = data
#                 else:
#                     logger.warning("🚫 Empty or zero-record response: %s", url)
#                     failed.append(url)

#             except requests.exceptions.RequestException as e:
#                 logger.error("🔌 Network error for %s: %s", url, e)
#                 failed.append(url)
#             except Exception as e:
#                 logger.error("🔥 Unexpected error for %s: %s", url, e)
#                 failed.append(url)

#         logger.info("Fetch complete: %d accessible, %d failed", len(accessible), len(failed))
#         return accessible, failed

#     # --------------------------------------------------------------------- #
#     #                         DATA COLLATION / MERGE                         #
#     # --------------------------------------------------------------------- #
#     @staticmethod
#     def _record_key(record: Dict[str, Any]) -> str:
#         """
#         Best-effort unique key finder for a generic openFDA record.
#         Falls back to a stable hash if no obvious ID exists.
#         """
#         for candidate in ("id", "report_id", "safetyreportid", "lot_number"):
#             if candidate in record:
#                 return f"{candidate}:{record[candidate]}"
#         # fallback: hash entire record
#         return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()

#     def collate_records_data(self, accessible: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Combine records from multiple sources, removing duplicates.
#         """
#         logger.info("Collating records from %d URLs...", len(accessible))
#         total_original = 0
#         merged: Dict[str, Dict[str, Any]] = {}
#         source_urls = list(accessible.keys())

#         for url, payload in accessible.items():
#             total_original += (
#                 payload.get("meta", {})
#                 .get("results", {})
#                 .get("total", 0)
#             )
#             for rec in payload.get("results", []):
#                 key = self._record_key(rec)
#                 if key not in merged:
#                     merged[key] = rec

#         logger.info("✅ Collated %d unique records from %d original", len(merged), total_original)
#         return {
#             "records": list(merged.values()),
#             "totalCount": len(merged),
#             "originalTotalCount": total_original,
#             "sourceUrls": source_urls,
#         }

#     # --------------------------------------------------------------------- #
#     #                       TOP-LEVEL ANALYZE METHOD                         #
#     # --------------------------------------------------------------------- #
#     def analyze_user_query(self, user_input: str) -> Dict[str, Any]:
#         """
#         End-to-end:
#         1. Generate URLs,
#         2. Fetch data,
#         3. Collate,
#         4. Return structured response.
#         """
#         logger.info("User query: %s", user_input)

#         url_payload = self.generate_api_urls(user_input)
#         if not url_payload:
#             return {
#                 "success": False,
#                 "error": "URL generation failed",
#                 "data": None,
#                 "total_count": 0,
#             }

#         urls = url_payload["urls"]
#         accessible, failed = self.fetch_fda_data(urls)

#         if not accessible:
#             logger.warning("❌ No accessible URLs returned data.")
#             return {
#                 "success": False,
#                 "error": "No accessible URLs returned data",
#                 "data": None,
#                 "failed_urls": failed,
#                 "attempted_urls": urls,
#             }

#         collated = self.collate_records_data(accessible)
#         logger.info("🎯 Query analysis complete.")
#         return {
#             "success": True,
#             "data": collated,
#             "total_count": collated["totalCount"],
#             "records_returned": len(collated["records"]),
#             "source_url": collated["sourceUrls"][0],
#             "all_source_urls": collated["sourceUrls"],
#             "failed_urls": failed,
#             "attempted_urls": urls,
#             "query_analysis": {
#                 "original_query": user_input,
#                 "urls_attempted": len(urls),
#                 "urls_successful": len(accessible),
#                 "unique_records_found": collated["totalCount"],
#             },
#         }

# # ------------------------------------------------------------------------------
# # Factory Function
# # ------------------------------------------------------------------------------
# def create_fda_agent(openai_client=None, model: str = "gpt-4o") -> FdaFetcherAgent:
#     return FdaFetcherAgent(openai_client=openai_client, model=model)