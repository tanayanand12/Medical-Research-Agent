"""Lightweight biomedical entity normalization for runtime attribution checks."""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, Literal, Set


_PUNCTUATION_RE = re.compile(r"[^a-z0-9]+")
_DOSAGE_FORMS = {
    "capsule",
    "capsules",
    "cream",
    "injection",
    "intravenous",
    "iv",
    "oral",
    "solution",
    "tablet",
    "tablets",
    "topical",
}

# Deliberately bounded: these aliases cover common high-salience substitutions
# without adding a heavyweight clinical NER dependency to the request path.
_ALIASES = {
    "drug:aspirin": {
        "aspirin",
        "acetylsalicylic acid",
        "asa",
    },
    "drug:ibuprofen": {"ibuprofen", "advil", "motrin"},
    "drug:metformin": {"metformin", "glucophage"},
    "drug:insulin": {"insulin", "human insulin", "insulin therapy"},
    "drug:warfarin": {"warfarin", "coumadin"},
    "drug:apixaban": {"apixaban", "eliquis"},
    "intervention:glp-1": {
        "glp 1",
        "glp 1 receptor agonist",
        "glp 1 receptor agonists",
        "glucagon like peptide 1",
    },
    "disease:myocardial_infarction": {
        "myocardial infarction",
        "heart attack",
        "mi",
    },
    "disease:stroke": {
        "stroke",
        "cerebrovascular accident",
        "cva",
    },
    "organization:fda": {
        "fda",
        "food and drug administration",
        "us food and drug administration",
        "united states food and drug administration",
    },
    "organization:ema": {
        "ema",
        "european medicines agency",
    },
    "population:adult": {"adult", "adults", "elderly", "older adults"},
    "population:child": {
        "child",
        "children",
        "adolescent",
        "adolescents",
        "pediatric",
        "paediatric",
    },
    "population:pregnant": {
        "pregnancy",
        "pregnant",
        "pregnant women",
    },
    "population:female": {"female", "females", "woman", "women"},
    "population:male": {"male", "males", "man", "men"},
}

_DRUG_SUFFIXES = (
    "cillin",
    "farin",
    "formin",
    "gliptin",
    "glutide",
    "mab",
    "nib",
    "olol",
    "pril",
    "sartan",
    "statin",
    "vir",
    "xaban",
)
_BIOMEDICAL_CODE_RE = re.compile(
    r"\b[A-Z]{2,8}(?:-[A-Z0-9]+)*-?\d{2,}\b",
    re.IGNORECASE,
)

_ATTRIBUTION_CONTEXT_TOKENS = {
    "adults",
    "adverse",
    "associated",
    "answer",
    "boys",
    "children",
    "cited",
    "cohort",
    "compared",
    "cardiovascular",
    "decreased",
    "disease",
    "effect",
    "events",
    "evidence",
    "exact",
    "function",
    "girls",
    "improve",
    "improved",
    "inconclusive",
    "increase",
    "increased",
    "increases",
    "lower",
    "lowered",
    "lowers",
    "intervention",
    "mortality",
    "motor",
    "outcome",
    "outcomes",
    "patients",
    "people",
    "reduced",
    "reduce",
    "reduces",
    "relapses",
    "reported",
    "showed",
    "study",
    "supported",
    "therapy",
    "treatment",
    "trial",
    "women",
}

_ATTRIBUTION_PREDICATES = {
    "associate",
    "compare",
    "decrease",
    "improve",
    "increase",
    "lower",
    "reduce",
    "report",
    "show",
    "support",
}


def _predicate_stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            candidate = token[: -len(suffix)]
            if candidate.endswith("i"):
                candidate = candidate[:-1] + "y"
            if candidate in _ATTRIBUTION_PREDICATES:
                return candidate
            if candidate + "e" in _ATTRIBUTION_PREDICATES:
                return candidate + "e"
    return token


def normalize_biomedical_text(text: str) -> str:
    """Normalize case, punctuation, Unicode, and non-attributive dosage forms."""
    normalized = unicodedata.normalize("NFKD", str(text)).encode(
        "ascii", "ignore"
    ).decode("ascii")
    tokens = [
        token
        for token in _PUNCTUATION_RE.sub(" ", normalized.lower()).split()
        if token not in _DOSAGE_FORMS
    ]
    return " ".join(tokens)


def extract_salient_entities(text: str) -> Dict[str, Set[str]]:
    """Return bounded canonical entities grouped by attribution category."""
    normalized = normalize_biomedical_text(text)
    padded = f" {normalized} "
    found: Dict[str, Set[str]] = {}

    for canonical, aliases in _ALIASES.items():
        category, _ = canonical.split(":", 1)
        if any(f" {normalize_biomedical_text(alias)} " in padded for alias in aliases):
            found.setdefault(category, set()).add(canonical)

    for token in normalized.split():
        if len(token) >= 6 and token.endswith(_DRUG_SUFFIXES):
            found.setdefault("drug", set()).add(f"drug:{token}")

    for code in _BIOMEDICAL_CODE_RE.findall(str(text)):
        normalized_code = normalize_biomedical_text(code).replace(" ", "_")
        found.setdefault("identifier", set()).add(
            f"identifier:{normalized_code}"
        )

    # Preserve unknown all-caps organizations without treating sentence-initial
    # title-case words as entities.
    for acronym in re.findall(r"\b[A-Z]{2,8}\b", str(text)):
        normalized_acronym = normalize_biomedical_text(acronym)
        if normalized_acronym in {"rct", "pmc", "pmid"}:
            continue
        alias_entities = {
            canonical
            for canonical, aliases in _ALIASES.items()
            if any(
                (
                    normalized_alias == normalized_acronym
                    or normalized_alias.startswith(
                        f"{normalized_acronym} "
                    )
                )
                and f" {normalized_alias} " in padded
                for normalized_alias in (
                    normalize_biomedical_text(alias)
                    for alias in aliases
                )
            )
        }
        already_classified = any(
            entity in alias_entities
            for entities in found.values()
            for entity in entities
        )
        if not already_classified:
            found.setdefault("organization", set()).add(
                f"organization:{normalized_acronym}"
            )
    return found


def _canonical_attribution_tokens(text: str) -> Set[str]:
    """Canonicalize known aliases, retaining unknown biomedical terms."""
    normalized = normalize_biomedical_text(text)
    for canonical, aliases in _ALIASES.items():
        placeholder = "entity" + re.sub(r"[^a-z0-9]", "", canonical)
        for alias in sorted(aliases, key=len, reverse=True):
            normalized_alias = normalize_biomedical_text(alias)
            normalized = re.sub(
                rf"\b{re.escape(normalized_alias)}\b",
                placeholder,
                normalized,
            )
    return set(normalized.split())


def _is_known_biomedical_token(token: str) -> bool:
    normalized = normalize_biomedical_text(token)
    return any(
        normalize_biomedical_text(alias) == normalized
        for aliases in _ALIASES.values()
        for alias in aliases
    )


def _unknown_attribution_candidates(text: str) -> Set[str]:
    """Find unexplained content tokens without pretending they are entities.

    These candidates are a triage signal only.  They deliberately route
    asymmetric attribution to semantic verification rather than classifying
    an unknown token as a drug or disease.
    """
    candidates = {
        token
        for token in _canonical_attribution_tokens(text)
        if (
            not token.startswith("entity")
            and token.isalpha()
            and len(token) >= 5
            and not token.endswith(_DRUG_SUFFIXES)
            and token not in _ATTRIBUTION_CONTEXT_TOKENS
            and _predicate_stem(token) not in _ATTRIBUTION_PREDICATES
        )
    }
    for token in re.findall(r"\b[A-Z][a-z]{2,7}\b", str(text)):
        normalized = normalize_biomedical_text(token)
        if normalized in _ATTRIBUTION_CONTEXT_TOKENS:
            continue
        if _is_known_biomedical_token(token):
            continue
        candidates.add(normalized)
    for code in re.findall(r"\b[A-Z]\d+\b", str(text)):
        normalized = normalize_biomedical_text(code)
        if not _is_known_biomedical_token(normalized):
            candidates.add(normalized)
    return candidates


EntityAttributionStatus = Literal[
    "supported", "mismatch", "uncertain", "no_salient_entity"
]


def entity_attribution_status(
    claim: str, cited_evidence: Iterable[str]
) -> EntityAttributionStatus:
    """Classify salient claim entities against the cited context only."""
    evidence_items = list(cited_evidence)
    claim_entities = extract_salient_entities(claim)
    evidence_text = " ".join(evidence_items)
    evidence_entities = extract_salient_entities(evidence_text)
    has_salient_claim_entity = False
    has_unresolved_entity = False
    for category, claimed in claim_entities.items():
        if not claimed:
            continue
        has_salient_claim_entity = True
        evidenced = evidence_entities.get(category, set())
        if not evidenced:
            has_unresolved_entity = True
        elif not claimed.issubset(evidenced):
            return "mismatch"
        if any(entity not in _ALIASES for entity in claimed):
            # Suffix-inferred drugs, biomedical identifiers, and unknown
            # acronyms are triage signals, not deterministic attribution.
            # Even exact lexical overlap must be checked semantically.
            has_unresolved_entity = True
    if has_unresolved_entity:
        return "uncertain"
    claim_candidates = _unknown_attribution_candidates(claim)
    if claim_candidates:
        # Unknown biomedical-looking terms always require the cited-context
        # semantic verifier. Exact token overlap alone is not attribution.
        return "uncertain"
    if has_salient_claim_entity:
        return "supported"
    return "no_salient_entity"


def entity_attribution_mismatch(
    claim: str, cited_evidence: Iterable[str]
) -> bool:
    """Backward-compatible mismatch predicate."""
    return entity_attribution_status(claim, cited_evidence) == "mismatch"
