"""
skill_router.py — Phase 3: Skill discovery and routing.

Loads YAML skill manifests from ./skills/ and scores them against a user
query using keyword matching, domain matching, and (optionally) embedding-
based semantic similarity via LLMClient.

Scoring weights:
    With embeddings:    40% semantic + 35% keyword + 25% domain
    Without embeddings: 60% keyword + 40% domain
"""

import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from evaluation_core import (
    RuntimeDeadlineExceeded,
    stable_query_fingerprint,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain keyword mapping — used to infer query domains
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "pharmacology": [
        "drug", "medication", "pharmaceutical", "pharmacology", "dose",
        "dosage", "treatment", "therapy", "inhibitor", "receptor",
        "mechanism", "agonist", "antagonist", "metformin", "statin",
        "antibiotic",
    ],
    "epidemiology": [
        "prevalence", "incidence", "population", "cohort", "risk factor",
        "mortality", "morbidity", "outbreak", "epidemic", "pandemic",
        "surveillance",
    ],
    "genetics": [
        "gene", "genetic", "mutation", "genome", "hereditary", "chromosome",
        "allele", "polymorphism", "dna", "rna", "brca", "crispr",
    ],
    "oncology": [
        "cancer", "tumor", "tumour", "malignant", "chemotherapy", "radiation",
        "oncology", "carcinoma", "lymphoma", "leukemia", "metastasis",
    ],
    "cardiology": [
        "heart", "cardiac", "cardiovascular", "hypertension", "arrhythmia",
        "coronary", "myocardial", "atrial", "ventricular", "heart failure",
        "sglt2",
    ],
    "neurology": [
        "brain", "neural", "neurological", "alzheimer", "parkinson",
        "epilepsy", "stroke", "dementia", "cognitive", "neuropathy",
    ],
    "endocrinology": [
        "diabetes", "insulin", "glucose", "thyroid", "endocrine", "hormonal",
        "hba1c", "type 2", "type 1", "glycemic", "metabolic",
    ],
    "clinical_trials": [
        "trial", "clinical trial", "phase", "randomized", "placebo",
        "enrollment", "endpoint", "nct",
    ],
    "drug_safety": [
        "adverse", "side effect", "safety", "toxicity", "recall", "warning",
        "contraindication",
    ],
    "regulatory": [
        "fda", "approval", "regulation", "compliance", "label", "ema",
    ],
    "pharmacovigilance": [
        "adverse event", "pharmacovigilance", "post-market", "surveillance",
        "safety report", "maude",
    ],
    "interventional": [
        "intervention", "procedure", "surgery", "surgical", "implant",
        "device",
    ],
    "general": [],
}

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
    "for", "of", "and", "or", "with", "by", "from", "that", "this", "what",
    "which", "who", "how", "can", "has", "had", "its", "but", "not", "all",
    "been", "will", "would", "may", "about", "into", "than", "most",
})


class SkillRouter:
    """Skill discovery engine.

    Loads YAML manifests from ``./skills/`` and ranks tools against a query
    using a weighted combination of keyword matching, domain matching, and
    optional embedding-based semantic similarity.

    Usage::

        router = SkillRouter()
        tools, scores = router.rank_tools("metformin type 2 diabetes", top_k=3)
        # tools  = ["search_pubmed", "search_clinical_trials", "search_local_index"]
        # scores = [0.82, 0.71, 0.65]
    """

    def __init__(self, skills_dir: Optional[str] = None) -> None:
        self._skills_dir = (
            Path(skills_dir) if skills_dir
            else Path(__file__).parent / "skills"
        )
        self._manifests: Dict[str, Dict[str, Any]] = {}
        self._embedding_cache: Dict[str, List[float]] = {}
        self._query_embedding_cache: Dict[str, List[float]] = {}
        self._embedding_available: Optional[bool] = None
        self._llm_client: Any = None
        self._load_manifests()

    # ------------------------------------------------------------------
    # Manifest loading
    # ------------------------------------------------------------------

    def _load_manifests(self) -> None:
        """Load all YAML skill manifests from the skills directory."""
        if not self._skills_dir.exists():
            logger.warning("Skills directory not found: %s", self._skills_dir)
            return

        for yaml_file in sorted(self._skills_dir.glob("*.yaml")):
            try:
                with open(yaml_file, "r", encoding="utf-8") as fh:
                    manifest = yaml.safe_load(fh)
                if manifest and "tool_name" in manifest:
                    tool_name = manifest["tool_name"]
                    self._manifests[tool_name] = manifest
                    logger.info(
                        "SkillRouter: loaded manifest %r from %s",
                        tool_name, yaml_file.name,
                    )
                else:
                    logger.warning(
                        "SkillRouter: skipping %s (missing tool_name)",
                        yaml_file.name,
                    )
            except Exception as exc:
                logger.warning(
                    "SkillRouter: failed to load %s: %s",
                    yaml_file.name, exc,
                )

        logger.info(
            "SkillRouter: %d skill manifests loaded: %s",
            len(self._manifests), list(self._manifests.keys()),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank_tools(
        self,
        query: str,
        top_k: int = 3,
        min_threshold: float = 0.1,
        deadline_at: Optional[float] = None,
    ) -> Tuple[List[str], List[float]]:
        """Rank all loaded tools by relevance to *query*.

        Parameters
        ----------
        query : str
            The user's medical research question.
        top_k : int
            Maximum number of tools to return.
        min_threshold : float
            Minimum score to be considered (tools below this are dropped
            unless **all** tools are below, in which case all are returned).

        Returns
        -------
        tuple[list[str], list[float]]
            ``(tool_names, scores)`` sorted by descending score.
        """
        if not self._manifests:
            logger.warning("SkillRouter: no manifests loaded — returning empty")
            return [], []

        scored: List[Tuple[str, float]] = []
        for tool_name, manifest in self._manifests.items():
            score = self._score(query, manifest, deadline_at=deadline_at)
            scored.append((tool_name, score))

        # Sort descending by score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Apply threshold
        above = [(n, s) for n, s in scored if s >= min_threshold]
        if not above:
            # Graceful degradation: return all tools
            above = scored
            logger.info(
                "SkillRouter: no tools above threshold %.2f — returning all",
                min_threshold,
            )

        selected = above[:top_k]

        names = [n for n, _ in selected]
        scores = [round(s, 4) for _, s in selected]

        logger.info(
            "SkillRouter: query_sha256=%s query_length=%d tools=%s",
            stable_query_fingerprint(query),
            len(query),
            list(zip(names, scores)),
        )
        return names, scores

    def select_skills(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[str]:
        """Return tool names ranked by relevance (convenience wrapper).

        This is the simple interface used for quick skill selection.
        """
        names, _ = self.rank_tools(query, top_k=top_k)
        return names

    @property
    def available_skills(self) -> List[str]:
        """Return all loaded tool names."""
        return sorted(self._manifests.keys())

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(
        self,
        query: str,
        manifest: Dict[str, Any],
        *,
        deadline_at: Optional[float] = None,
    ) -> float:
        """Compute composite relevance score for a query–manifest pair."""
        kw_score = self._keyword_score(query, manifest)
        dom_score = self._domain_score(query, manifest)
        sem_score = (
            self._semantic_score(query, manifest)
            if deadline_at is None
            else self._semantic_score(
                query, manifest, deadline_at=deadline_at
            )
        )
        priority_score = self._priority_keyword_score(query, manifest)

        if sem_score is not None:
            base = 0.40 * sem_score + 0.35 * kw_score + 0.25 * dom_score
        else:
            base = 0.60 * kw_score + 0.40 * dom_score
        return min(1.0, base + 0.30 * priority_score)

    # -- keyword scoring ------------------------------------------------

    def _keyword_score(self, query: str, manifest: Dict[str, Any]) -> float:
        """Score based on word overlap between query and skill text pool.

        Combines trigger keywords and description into a single text pool.
        Measures what fraction of meaningful query tokens find a match
        (exact or 4-char prefix).
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return 0.0

        # Build text pool from triggers + description
        trigger_kws = manifest.get("triggers", {}).get("keywords", [])
        description = manifest.get("description", "")
        pool_text = " ".join(trigger_kws) + " " + description
        pool_tokens = self._tokenize(pool_text)

        if not pool_tokens:
            return 0.0

        matches = 0
        for qt in query_tokens:
            for pt in pool_tokens:
                if self._token_match(qt, pt):
                    matches += 1
                    break

        return matches / len(query_tokens)

    @staticmethod
    def _priority_keyword_score(query: str, manifest: Dict[str, Any]) -> float:
        """Return a source-intent boost for explicit, curated routing cues."""
        query_lower = query.lower()
        priority_keywords = (
            manifest.get("triggers", {}).get("priority_keywords", [])
        )
        matches = sum(
            1 for keyword in priority_keywords if keyword.lower() in query_lower
        )
        return min(1.0, matches / 2.0)

    # -- domain scoring -------------------------------------------------

    def _domain_score(self, query: str, manifest: Dict[str, Any]) -> float:
        """Score based on overlap between query-inferred domains and skill domains."""
        query_domains = self._detect_domains(query)
        skill_domains = set(manifest.get("domains", []))

        if not skill_domains:
            return 0.0

        if not query_domains:
            # No domains detected; give partial credit to "general"
            return 0.3 if "general" in skill_domains else 0.0

        overlap = query_domains & skill_domains
        # Score = fraction of detected query domains covered by this skill
        return len(overlap) / len(query_domains)

    def _detect_domains(self, query: str) -> set:
        """Infer medical domains from query text."""
        query_lower = query.lower()
        detected = set()

        for domain, keywords in _DOMAIN_KEYWORDS.items():
            if domain == "general":
                continue
            for kw in keywords:
                if kw in query_lower:
                    detected.add(domain)
                    break

        return detected

    # -- semantic scoring -----------------------------------------------

    def _semantic_score(
        self,
        query: str,
        manifest: Dict[str, Any],
        *,
        deadline_at: Optional[float] = None,
    ) -> Optional[float]:
        """Compute cosine similarity between query and skill description embeddings.

        Returns ``None`` if embedding is unavailable (no API key, model
        error, etc.), causing the caller to fall back to keyword+domain only.
        """
        embedding_available = (
            self._is_embedding_available()
            if deadline_at is None
            else self._is_embedding_available(deadline_at=deadline_at)
        )
        if not embedding_available:
            return None

        try:
            client = self._get_llm_client()
            if query not in self._query_embedding_cache:
                if len(self._query_embedding_cache) >= 128:
                    self._query_embedding_cache.clear()
                self._query_embedding_cache[query] = (
                    client.embed(query)
                    if deadline_at is None
                    else client.embed(
                        query,
                        deadline_at=deadline_at,
                        client_max_attempts=1,
                    )
                )
            query_vec = self._query_embedding_cache[query]

            desc = manifest.get("description", "")
            if desc not in self._embedding_cache:
                self._embedding_cache[desc] = (
                    client.embed(desc)
                    if deadline_at is None
                    else client.embed(
                        desc,
                        deadline_at=deadline_at,
                        client_max_attempts=1,
                    )
                )
            desc_vec = self._embedding_cache[desc]

            return self._cosine_similarity(query_vec, desc_vec)
        except RuntimeDeadlineExceeded:
            raise
        except Exception as exc:
            logger.debug(
                "SkillRouter: semantic scoring failed error_type=%s",
                type(exc).__name__,
            )
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Extract meaningful tokens (>= 3 chars, no stop words)."""
        return [
            t for t in re.findall(r"\b\w+\b", text.lower())
            if len(t) >= 3 and t not in _STOP_WORDS
        ]

    @staticmethod
    def _token_match(a: str, b: str) -> bool:
        """Check whether two tokens match (exact or shared 4-char prefix)."""
        if a == b:
            return True
        min_prefix = min(4, len(a), len(b))
        return a[:min_prefix] == b[:min_prefix] and min_prefix >= 4

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _get_llm_client(self) -> Any:
        """Lazily obtain the LLMClient singleton."""
        if self._llm_client is None:
            from llm_client import LLMClient
            self._llm_client = LLMClient()
        return self._llm_client

    def _is_embedding_available(
        self, *, deadline_at: Optional[float] = None
    ) -> bool:
        """Check once whether embedding calls succeed."""
        if self._embedding_available is not None:
            return self._embedding_available

        try:
            client = self._get_llm_client()
            if deadline_at is None:
                client.embed("test")
            else:
                client.embed(
                    "test",
                    deadline_at=deadline_at,
                    client_max_attempts=1,
                )
            self._embedding_available = True
        except RuntimeDeadlineExceeded:
            raise
        except Exception:
            self._embedding_available = False
            logger.info(
                "SkillRouter: embedding unavailable — "
                "falling back to keyword + domain scoring"
            )
        return self._embedding_available
