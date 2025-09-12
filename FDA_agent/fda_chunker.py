# fda_chunker.py
"""
FDA Chunker – outputs simple dict chunks compatible with ClinicalTrialsVectorizer
"""

from __future__ import annotations
import hashlib, json, logging, re, textwrap
from typing import List, Dict, Any

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logger = logging.getLogger("fda_chunker")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s:%(lineno)d - %(message)s")
    )
    logger.addHandler(_h)


class FdaChunker:
    """
    Split raw openFDA records into text chunks ready for embedding.

    Each chunk is a *dict* with at least:
        {"content": str, "record_id": str, "chunk_type": str, "section": str}
    so the vectorizer can consume it with `c["content"]`.
    """

    # keys we try in order to get a stable ID
    CANDIDATE_IDS = (
        "id",
        "event_key",
        "report_number",
        "recall_number",
        "enforcement_report_number",
        "safetyreportid",
        "lot_number",
    )

    def __init__(self, max_chunk_size: int = 30_000, overlap: int = 800):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        logger.info("FdaChunker init (chunk_size=%d, overlap=%d)", max_chunk_size, overlap)

    # ------------------------------------------------------------------ #
    # ID helper
    # ------------------------------------------------------------------ #
    def _extract_record_id(self, rec: Dict[str, Any]) -> str:
        for k in self.CANDIDATE_IDS:
            if k in rec and rec[k]:
                return str(rec[k])
        spl = rec.get("openfda", {}).get("spl_id", [])
        if spl:
            return str(spl[0])
        # fallback – short hash
        return hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()[:12]

    # ------------------------------------------------------------------ #
    # Text helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clean(txt: str) -> str:
        txt = re.sub(r"\s+", " ", txt or "")
        txt = re.sub(r"[^\w\s\\-\\.,;:()\\[\\]/%]", " ", txt)
        return txt.strip()

    def _split_large(self, text: str, record_id: str, section: str) -> List[Dict[str, Any]]:
        stride = self.max_chunk_size - self.overlap
        chunks: List[Dict[str, Any]] = []
        for start in range(0, len(text), stride):
            part = text[start : start + self.max_chunk_size]
            chunks.append(
                {
                    "content": part,
                    "chunk_type": f"{section}_{start//stride}",
                    "record_id": record_id,
                    "section": section,
                    "similarity_score": 0.0,
                }
            )
            if start + self.max_chunk_size >= len(text):
                break
        return chunks

    # ------------------------------------------------------------------ #
    # Single record → chunks
    # ------------------------------------------------------------------ #
    def chunk_record(self, rec: Dict[str, Any]) -> List[Dict[str, Any]]:
        rid = self._extract_record_id(rec)
        chunks: List[Dict[str, Any]] = []

        # Always add trimmed raw JSON
        raw = json.dumps(rec, ensure_ascii=False)[: self.max_chunk_size]
        chunks.append(
            {
                "content": raw,
                "chunk_type": "raw_json",
                "record_id": rid,
                "section": "raw",
                "similarity_score": 0.0,
                "raw_json": rec,  # keep for fallback ID recovery
            }
        )

        # Drug label
        if "indications_and_usage" in rec:
            sections = {
                "indications": rec.get("indications_and_usage", [""])[0],
                "dosage": rec.get("dosage_and_administration", [""])[0],
                "warnings": rec.get("warnings_and_cautions", [""])[0] or rec.get("warnings", [""])[0],
                "adverse_reactions": rec.get("adverse_reactions", [""])[0],
            }
            for sec, txt in sections.items():
                txt = self._clean(txt)
                if not txt:
                    continue
                if len(txt) > self.max_chunk_size:
                    chunks.extend(self._split_large(txt, rid, sec))
                else:
                    chunks.append(
                        {
                            "content": txt,
                            "chunk_type": sec,
                            "record_id": rid,
                            "section": sec,
                            "similarity_score": 0.0,
                        }
                    )

        # Device event
        elif "event_type" in rec:
            summary = self._clean(rec.get("summary_report", ""))
            if summary:
                chunks.append(
                    {
                        "content": summary,
                        "chunk_type": "event_summary",
                        "record_id": rid,
                        "section": "summary",
                        "similarity_score": 0.0,
                    }
                )

        # Recall
        elif "reason_for_recall" in rec:
            txt = textwrap.dedent(
                f"""
                Product: {rec.get('product_description','')}
                Reason: {rec.get('reason_for_recall','')}
                Action: {rec.get('corrective_action','')}
                Classification: {rec.get('classification','')}
                """
            ).strip()
            cleaned = self._clean(txt)
            if cleaned:
                chunks.append(
                    {
                        "content": cleaned,
                        "chunk_type": "recall_overview",
                        "record_id": rid,
                        "section": "overview",
                        "similarity_score": 0.0,
                    }
                )

        return chunks

    # ------------------------------------------------------------------ #
    # Bulk helper
    # ------------------------------------------------------------------ #
    def chunk_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info("Chunking %d FDA records…", len(records))
        all_chunks: List[Dict[str, Any]] = []
        for rec in records:
            try:
                all_chunks.extend(self.chunk_record(rec))
            except Exception as exc:
                logger.warning("Chunker error: %s", exc)
        logger.info("✅ Generated %d chunks total", len(all_chunks))
        return all_chunks


# --------------------------------------------------------------------------- #
# Factory (optional)
# --------------------------------------------------------------------------- #
def create_fda_chunker(max_chunk_size: int = 30_000, overlap: int = 800) -> FdaChunker:
    return FdaChunker(max_chunk_size=max_chunk_size, overlap=overlap)



# ------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

# fda_chunker.py
# """
# Chunker for openFDA records (drug label, adverse event, recall, etc.).

# Optimisation 2025-06-07
# -----------------------
# * max_chunk_size raised 10 000 → 30 000
# * overlap raised   400   → 800
#   → cuts typical chunk-count by ~3×, reducing embedding calls
# """
# from __future__ import annotations

# import logging
# import re
# import json
# import textwrap
# from typing import Dict, List, Any

# # ------------------------------------------------------------------------------
# # Logger
# # ------------------------------------------------------------------------------
# logger = logging.getLogger("fda_chunker")
# logger.setLevel(logging.INFO)
# if not logger.hasHandlers():
#     _h = logging.StreamHandler()
#     _h.setLevel(logging.INFO)
#     _h.setFormatter(
#         logging.Formatter(
#             "[%(levelname)s] %(asctime)s - %(filename)s:%(lineno)d - %(message)s"
#         )
#     )
#     logger.addHandler(_h)


# # ------------------------------------------------------------------------------
# # Chunker
# # ------------------------------------------------------------------------------
# class FdaChunker:
#     """
#     Create semantic-ish chunks from openFDA records suitable for embedding.

#     Args
#     ----
#     max_chunk_size : int
#         Maximum chars per chunk before splitting (default 30 000).
#     overlap        : int
#         Characters of overlap when splitting very long text (default 800).
#     """

#     def __init__(self, max_chunk_size: int = 30_000, overlap: int = 800):
#         self.max_chunk_size = max_chunk_size
#         self.overlap = overlap
#         logger.info(
#             "FdaChunker init (chunk_size=%d, overlap=%d)", max_chunk_size, overlap
#         )

#     # ------------------------------------------------------------------ #
#     # Helpers
#     # ------------------------------------------------------------------ #
#     @staticmethod
#     def _clean(text: str) -> str:
#         if not text:
#             return ""
#         text = re.sub(r"\s+", " ", text)  # collapse whitespace
#         text = re.sub(r"[^\w\s\\-\\.,;:()\\[\\]/%]", " ", text)
#         return text.strip()

#     def _split_large(
#         self, text: str, record_id: str, section: str
#     ) -> List[Dict[str, Any]]:
#         chunks: List[Dict[str, Any]] = []
#         stride = self.max_chunk_size - self.overlap
#         for start in range(0, len(text), stride):
#             part = text[start : start + self.max_chunk_size]
#             chunks.append(
#                 {
#                     "content": part,
#                     "chunk_type": f"{section}_{start//stride}",
#                     "record_id": record_id,
#                     "section": section,
#                 }
#             )
#             if start + self.max_chunk_size >= len(text):
#                 break
#         return chunks

#     # ------------------------------------------------------------------ #
#     # Single-record → chunks
#     # ------------------------------------------------------------------ #
#     def chunk_record(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
#         record_id = (
#             record.get("id")
#             or record.get("openfda", {}).get("spl_id", ["unknown"])[0]
#             or "unknown"
#         )
#         chunks: List[Dict[str, Any]] = []

#         # Raw JSON snapshot (first max_chunk_size chars)
#         raw = json.dumps(record, ensure_ascii=False)[: self.max_chunk_size]
#         chunks.append(
#             {
#                 "content": raw,
#                 "chunk_type": "raw_json",
#                 "record_id": record_id,
#                 "section": "raw",
#             }
#         )

#         # ----------------- Dataset-specific parsing ------------------ #
#         if "indications_and_usage" in record:  # Drug label
#             sections = {
#                 "indications": record.get("indications_and_usage", [""])[0],
#                 "dosage": record.get("dosage_and_administration", [""])[0],
#                 "warnings": record.get("warnings_and_cautions", [""])[0]
#                 or record.get("warnings", [""])[0],
#                 "adverse_reactions": record.get("adverse_reactions", [""])[0],
#             }
#             for sec, txt in sections.items():
#                 txt = self._clean(txt)
#                 if not txt:
#                     continue
#                 if len(txt) > self.max_chunk_size:
#                     chunks.extend(self._split_large(txt, record_id, sec))
#                 else:
#                     chunks.append(
#                         {
#                             "content": txt,
#                             "chunk_type": sec,
#                             "record_id": record_id,
#                             "section": sec,
#                         }
#                     )

#         elif "event_type" in record:  # Device event
#             summary = self._clean(record.get("summary_report", ""))
#             if summary:
#                 chunks.append(
#                     {
#                         "content": summary,
#                         "chunk_type": "event_summary",
#                         "record_id": record_id,
#                         "section": "summary",
#                     }
#                 )

#         elif "reason_for_recall" in record:  # Recall (food/device)
#             recall_text = textwrap.dedent(
#                 f"""
#                 Product: {record.get('product_description', '')}
#                 Reason: {record.get('reason_for_recall', '')}
#                 Action: {record.get('corrective_action', '')}
#                 Classification: {record.get('classification', '')}
#                 Initial Firm Notification: {record.get('initial_firm_notification', '')}
#                 """.strip()
#             )
#             cleaned = self._clean(recall_text)
#             if cleaned:
#                 chunks.append(
#                     {
#                         "content": cleaned,
#                         "chunk_type": "recall_overview",
#                         "record_id": record_id,
#                         "section": "overview",
#                     }
#                 )

#         logger.debug("Record %s → %d chunk(s)", record_id, len(chunks))
#         return chunks

#     # ------------------------------------------------------------------ #
#     # Bulk helper
#     # ------------------------------------------------------------------ #
#     def chunk_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         all_chunks: List[Dict[str, Any]] = []
#         for rec in records:
#             try:
#                 all_chunks.extend(self.chunk_record(rec))
#             except Exception as exc:
#                 logger.error("Chunking error on record %s: %s", rec.get("id"), exc)
#         logger.info("Generated %d chunks total", len(all_chunks))
#         return all_chunks



#-----------------------------------------------------------------------------




# # fda_chunker.py

# """
# Chunker for openFDA records (drug label, adverse event, recall, etc.).
# Creates semantic, <= max_chunk_size snippets suitable for embedding.
# """
# import logging
# import re
# import json
# import textwrap
# from typing import Dict, List, Any

# logger = logging.getLogger("fda_chunker")
# logger.setLevel(logging.DEBUG)
# if not logger.hasHandlers():
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.DEBUG)
#     formatter = logging.Formatter(
#         "[%(levelname)s] %(asctime)s - %(filename)s:%(lineno)d - %(message)s"
#     )
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)


# class FdaChunker:
#     def __init__(self, max_chunk_size: int = 10_000, overlap: int = 400):
#         """
#         Args:
#             max_chunk_size: Maximum number of characters per chunk.
#             overlap:       Number of characters to overlap between successive chunks
#                            when splitting very large sections.
#         """
#         self.max_chunk_size = max_chunk_size
#         self.overlap = overlap
#         logger.info(
#             "Initialized FdaChunker (max_chunk_size=%d, overlap=%d)",
#             max_chunk_size,
#             overlap,
#         )

#     @staticmethod
#     def _clean(text: str) -> str:
#         """
#         Basic cleaning: collapse whitespace, remove unwanted characters.
#         """
#         if not text:
#             return ""
#         # Collapse multiple whitespace to single space
#         text = re.sub(r"\s+", " ", text)
#         # Remove any characters except letters, numbers, punctuation, or common symbols
#         text = re.sub(r"[^\w\s\-\.\,\;\:\(\)\[\]\/\%]", " ", text)
#         return text.strip()

#     def _split_large(self, text: str, record_id: str, section: str) -> List[Dict[str, Any]]:
#         """
#         Split a very large text block into multiple chunks with overlap.
#         """
#         chunks: List[Dict[str, Any]] = []
#         text_length = len(text)
#         stride = self.max_chunk_size - self.overlap
#         for start in range(0, text_length, stride):
#             end = min(start + self.max_chunk_size, text_length)
#             part = text[start:end]
#             chunk_metadata = {
#                 "content": part,
#                 "chunk_type": f"{section}_{start // self.max_chunk_size}",
#                 "record_id": record_id,
#                 "section": section,
#             }
#             chunks.append(chunk_metadata)
#             logger.debug(
#                 "Created overlapping chunk for record %s: section '%s' (%d chars)",
#                 record_id,
#                 section,
#                 len(part),
#             )
#             if end == text_length:
#                 break
#         return chunks

#     def chunk_record(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
#         """
#         Turn one openFDA record (drug label, device recall, etc.) into a list of chunks.
#         Supports the 3 most common datasets out-of-box; otherwise, includes raw JSON.

#         Returns:
#             List of dicts, each with keys:
#                 - content: str
#                 - chunk_type: str
#                 - record_id: str
#                 - section: str
#         """
#         record_id = (
#             record.get("id")
#             or record.get("openfda", {}).get("spl_id", ["unknown"])[0]
#             or "unknown"
#         )
#         chunks: List[Dict[str, Any]] = []

#         # 1. Fallback: include raw JSON up to max_chunk_size
#         raw_json_str = json.dumps(record, ensure_ascii=False)
#         raw_snippet = raw_json_str[: self.max_chunk_size]
#         chunks.append(
#             {
#                 "content": raw_snippet,
#                 "chunk_type": "raw_json",
#                 "record_id": record_id,
#                 "section": "raw",
#             }
#         )
#         logger.debug(
#             "Added raw_json chunk for record %s (%d chars)",
#             record_id,
#             len(raw_snippet),
#         )

#         # 2. Dataset-specific parsing
#         # 2a. Drug Label (has 'indications_and_usage')
#         if "indications_and_usage" in record:
#             sections = {
#                 "indications": record.get("indications_and_usage", [""])[0],
#                 "dosage": record.get("dosage_and_administration", [""])[0],
#                 "warnings": record.get("warnings_and_cautions", [""])[0]
#                 or record.get("warnings", [""])[0],
#                 "adverse_reactions": record.get("adverse_reactions", [""])[0],
#             }
#             for sec, txt in sections.items():
#                 cleaned = self._clean(txt or "")
#                 if not cleaned:
#                     continue
#                 if len(cleaned) > self.max_chunk_size:
#                     logger.debug(
#                         "Section '%s' for record %s is large (%d chars). Splitting.",
#                         sec,
#                         record_id,
#                         len(cleaned),
#                     )
#                     chunks.extend(self._split_large(cleaned, record_id, sec))
#                 else:
#                     chunks.append(
#                         {
#                             "content": cleaned,
#                             "chunk_type": sec,
#                             "record_id": record_id,
#                             "section": sec,
#                         }
#                     )
#                     logger.debug(
#                         "Added chunk for record %s: section '%s' (%d chars)",
#                         record_id,
#                         sec,
#                         len(cleaned),
#                     )

#         # 2b. Device Event (has 'event_type')
#         elif "event_type" in record:
#             summary = self._clean(record.get("summary_report", ""))
#             if summary:
#                 chunks.append(
#                     {
#                         "content": summary,
#                         "chunk_type": "event_summary",
#                         "record_id": record_id,
#                         "section": "summary",
#                     }
#                 )
#                 logger.debug(
#                     "Added event_summary chunk for record %s (%d chars)",
#                     record_id,
#                     len(summary),
#                 )

#         # 2c. Recall (has 'reason_for_recall')
#         elif "reason_for_recall" in record:
#             recall_text = textwrap.dedent(
#                 f"""
#                 Product: {record.get('product_description', '')}
#                 Reason: {record.get('reason_for_recall', '')}
#                 Action: {record.get('corrective_action', '')}
#                 Classification: {record.get('classification', '')}
#                 Initial Firm Notification: {record.get('initial_firm_notification', '')}
#                 """
#             ).strip()
#             cleaned = self._clean(recall_text)
#             if cleaned:
#                 chunks.append(
#                     {
#                         "content": cleaned,
#                         "chunk_type": "recall_overview",
#                         "record_id": record_id,
#                         "section": "overview",
#                     }
#                 )
#                 logger.debug(
#                     "Added recall_overview chunk for record %s (%d chars)",
#                     record_id,
#                     len(cleaned),
#                 )

#         # If none of the above, only raw JSON is returned (already added)

#         logger.info(
#             "Record %s chunked into %d chunks", record_id, len(chunks)
#         )
#         return chunks

#     def chunk_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """
#         Batch-chunk a list of openFDA records. Returns a flat list of chunks.
#         """
#         all_chunks: List[Dict[str, Any]] = []
#         for record in records:
#             try:
#                 rec_chunks = self.chunk_record(record)
#                 all_chunks.extend(rec_chunks)
#             except Exception as e:
#                 logger.error(
#                     "Error chunking record %s: %s", record.get("id", "<unknown>"), e
#                 )
#         logger.info("Total chunks created from all records: %d", len(all_chunks))
#         return all_chunks
