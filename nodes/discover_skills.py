"""
Node 2: discover_skills — Tool selection via semantic matching.

Uses skill router to rank available MCP tools by relevance to the query.
Selects top-K tools based on semantic similarity and keyword matching.
"""

from agent_state import AgentState
from skill_router import SkillRouter
import logging

logger = logging.getLogger(__name__)


def discover_skills(state: AgentState) -> AgentState:
    """
    Discover and rank relevant MCP tools for the query.

    Uses SkillRouter to perform semantic similarity + keyword matching.
    Respects explicit agent_to_use from context if provided (overrides discovery).

    Args:
        state: Current agent state

    Returns:
        Updated state with discovered_skills and skill_scores
    """
    trace_id = state.get("trace_id", "unknown")

    try:
        # Check if user explicitly specified agents (overrides discovery)
        if "agents_to_use" in state.get("context", {}) and state["context"]["agents_to_use"]:
            explicit_agents = state["context"]["agents_to_use"]
            logger.info(
                f"[{trace_id}] discover_skills: Using explicit agents from context: "
                f"{explicit_agents}"
            )
            state["discovered_skills"] = explicit_agents
            # Assign equal scores to explicit agents
            state["skill_scores"] = {agent: 1.0 for agent in explicit_agents}
            return state

        # Use SkillRouter for semantic discovery
        router = SkillRouter()
        discovered_tools, scores = router.rank_tools(
            query=state["input_query"],
            top_k=3,  # Select top 3 tools by default
        )

        state["discovered_skills"] = discovered_tools
        state["skill_scores"] = dict(zip(discovered_tools, scores))

        logger.info(
            f"[{trace_id}] discover_skills: Discovered {len(discovered_tools)} tools: "
            f"{[(t, f'{s:.2f}') for t, s in zip(discovered_tools, scores)]}"
        )

    except Exception as e:
        logger.error(
            f"[{trace_id}] discover_skills: Discovery failed: {str(e)}", exc_info=True
        )
        # Fallback: use all available tools
        all_tools = ["local", "pubmed", "pubmed_deep_research", "clinical_trials", "fda"]
        state["discovered_skills"] = all_tools
        state["skill_scores"] = {tool: 0.5 for tool in all_tools}
        state["error_occurred"] = True
        state["error_messages"].append(f"Skill discovery error, using all tools: {str(e)}")
        logger.warning(
            f"[{trace_id}] discover_skills: Fallback to all tools due to error"
        )

    return state
