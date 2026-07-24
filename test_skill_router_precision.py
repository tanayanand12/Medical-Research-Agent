from skill_router import SkillRouter


def _keyword_router() -> SkillRouter:
    router = SkillRouter()
    router._semantic_score = lambda query, manifest: None
    return router


def test_institutional_protocol_routes_to_local_index():
    names, _ = _keyword_router().rank_tools(
        "What threshold is stated in this institutional protocol?", top_k=1
    )

    assert names == ["search_local_index"]


def test_general_treatment_evidence_routes_to_pubmed():
    names, _ = _keyword_router().rank_tools(
        "What is the evidence for SGLT2 inhibitors in HFpEF?", top_k=1
    )

    assert names == ["search_pubmed"]


def test_explicit_deep_research_routes_to_deep_pubmed():
    names, _ = _keyword_router().rank_tools(
        "Perform a comprehensive deep research literature review", top_k=1
    )

    assert names == ["search_pubmed_deep"]


def test_semantic_router_embeds_each_query_once():
    calls = []

    class FakeLLM:
        def embed(self, text):
            calls.append(text)
            return [1.0, 0.0]

    router = SkillRouter()
    router._embedding_available = True
    router._llm_client = FakeLLM()

    router.rank_tools("single cached query", top_k=3)

    assert calls.count("single cached query") == 1
