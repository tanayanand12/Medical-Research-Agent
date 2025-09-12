from typing import List, Dict, Any
import logging
import openai
from pathlib import Path

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/aggregator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Aggregator:
    """
    Aggregator for synthesizing information from multiple agent responses,
    with integrated fallback handling and domain filtering support.
    """
    
    def __init__(self, model_id: str):
        """Initialize the aggregator with system prompts."""
        self.model_id = model_id


        self.system_prompt = """
        User Persona: {{PERSONA}}

        You are an expert at synthesizing information from multiple sources.
        Analyze the provided answers and create a comprehensive response.
        Include relevant citations and maintain high accuracy.
        
        GUIDELINES FOR YOUR RESPONSE:
        1. Prioritize information directly present in the provided papers
        2. Extract and synthesize findings across multiple sources when available
        3. Present a nuanced analysis that acknowledges:
           - Strength and consistency of evidence
           - Methodology quality (study design, sample size, controls)
           - Statistical significance of findings (p-values, confidence intervals)
           - Clinical relevance versus statistical significance
        4. When evaluating evidence quality:
           - Clearly identify study designs (RCT, meta-analysis, cohort, case-control, etc.)
           - Note sample characteristics (size, demographics, inclusion/exclusion criteria)
           - Address potential limitations or biases in methodology
           - Indicate levels of evidence using recognized frameworks (e.g., GRADE)
        5. For numerical data:
           - Include specific effect sizes, risk ratios, hazard ratios, or odds ratios
           - Provide confidence intervals and p-values when available
           - Contextualize percentages with absolute numbers
           - Compare findings across studies when possible
        6. For comparative analyses:
           - Present data from each intervention/group side-by-side
           - Highlight statistical and clinical significance of differences
           - Note heterogeneity in methods or populations that might affect comparability

        FORMAT YOUR RESPONSE AS:
        1. Executive Summary 
        2. Key Findings 
        3. Supporting Evidence 
        4. Clinical Implications 
        5. Evidence Quality Assessment
        6. Clinical Trials Data (if applicable) with Trial ID, Trial Name and all
        7. References (use citation to create the references section with pdf name, page number and title/topic *DO NOT INCLUDE TEXT*)

        Remember to:
        - Use bracketed numbers [1], [2], etc. for citations
        - Include all relevant sources when multiple papers address the same point
        - Focus on evidence-supported findings
        - Maintain scientific objectivity
        - Provide actionable insights based on available evidence
        - Make sure to adhere to persona if populated


        IMPORTANT:-
        - If the query is classified as non-medical research, do not use this system prompt.
        - If the query is classified as medical research, and is tipically made for cllinical trials, use the clinical trials agent responce aggregation with the trials number and spicific data point.
        - Check the `orchestrator_result` for domain filtering and fallback results.
        - You need to prioritize the aggregation of the query belongs too with specific domain.
        - If the query is classified as medical research, and is not tipically made for cllinical trials, use the primary agent aggregation with the query and agent responses.
        """
        
    # def _aggregate_primary_responses(self, query: str, agent_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    #     """
    #     Aggregate responses from primary agents.
        
    #     Parameters
    #     ----------
    #     query : str
    #         User query
    #     agent_responses : List[Dict[str, Any]]
    #         Responses from all agents
            
    #     Returns
    #     -------
    #     Dict[str, Any]
    #         Aggregated response
    #     """
    #     formatted_responses = []
    #     all_citations = []
        
    #     for resp in agent_responses:
    #         agent_name = resp["agent"]
    #         response = resp["response"]
    #         formatted_responses.append(f"Agent {agent_name}: {response['answer']}")
    #         all_citations.extend(response.get("citations", []))
            
    #     try:
    #         logger.info("Aggregating primary agent responses")
    #         completion = openai.chat.completions.create(
    #             model="gpt-4",
    #             messages=[
    #                 {"role": "system", "content": self.system_prompt},
    #                 {"role": "user", "content": f"Question: {query}\n\nResponses:\n" + "\n".join(formatted_responses)}
    #             ]
    #         )
            
    #         final_answer = completion.choices[0].message.content
            
    #         return {
    #             "answer": final_answer,
    #             "citations": all_citations,
    #             "agent_responses": agent_responses,
    #             "fallback_used": False
    #         }
            
    #     except Exception as e:
    #         logger.error(f"Error aggregating primary responses: {str(e)}", exc_info=True)
    #         return {
    #             "answer": f"Error aggregating responses: {str(e)}",
    #             "citations": all_citations,
    #             "agent_responses": agent_responses,
    #             "fallback_used": False
    #         }

    def _fetch_client_persona(self) -> str:
        """
        Fetch the client persona from the configured source.
        
        Returns
        -------
        str
            The client persona string to be used in the system prompt.
        """
        # Placeholder for actual implementation to fetch client persona
        # This could be a database call, API request, etc.
        import requests
        from client_persona_model import ClientPersonaTrackerData, SummaryPersonaRequest, SummaryType

        request_url = f"{ClientPersonaTrackerData().base_url}{ClientPersonaTrackerData().generate_summary_endpoint}"
        summary_request = SummaryPersonaRequest(
            uid=self.model_id,
            summary_type=SummaryType.HISTORY_ONLY,
        )
        logger.info(f"Fetching client persona from {request_url} with data: {summary_request.model_dump()}")
        response = requests.post(
            request_url,
            json=summary_request.model_dump(),
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            persona_data = response.json()
            logger.info(f"Successfully fetched client persona for UID: {self.model_id}")
            return persona_data["data"]["generated_summary"]
        else:
            logger.error(f"Failed to fetch client persona: {response.status_code} - {response.text}")
            # Return a default persona if fetching fails
        return "No Persona Specified"

    def _aggregate_primary_responses(self, query: str, agent_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate responses from primary agents.
        """
        formatted_responses = []
        all_citations = []
        
        for resp in agent_responses:
            try:
                agent_name = resp["agent"]
                response = resp["response"]
                
                # Handle different response formats safely
                answer = response.get("answer", "")
                if answer:
                    formatted_responses.append(f"Agent {agent_name}: {answer}")
                
                # Handle citations safely
                citations = response.get("citations", [])
                if isinstance(citations, list):
                    all_citations.extend(citations)
                elif citations:  # Handle string citations
                    all_citations.append(str(citations))
                    
            except Exception as e:
                logger.warning(f"Error processing response from agent {resp.get('agent', 'unknown')}: {str(e)}")
                continue
            
        try:
            logger.info("Aggregating primary agent responses")
            # Fetch client persona to include in the system prompt
            client_persona = self._fetch_client_persona()
            self.system_prompt = self.system_prompt.replace("{{PERSONA}}", client_persona)
            completion = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Question: {query}\n\nResponses:\n" + "\n".join(formatted_responses)}
                ]
            )
            
            final_answer = completion.choices[0].message.content
            
            return {
                "answer": final_answer,
                "citations": all_citations,
                "agent_responses": agent_responses,
                "fallback_used": False
            }
            
        except Exception as e:
            logger.error(f"Error aggregating primary responses: {str(e)}", exc_info=True)
            return {
                "answer": f"Error aggregating responses: {str(e)}",
                "citations": all_citations,
                "agent_responses": agent_responses,
                "fallback_used": False
            }
    
    def aggregate(self, orchestrator_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate results, handling domain filtering and fallback mechanisms.
        
        Parameters
        ----------
        orchestrator_result : Dict[str, Any]
            Results from the orchestrator including agent responses, fallback results,
            and domain filtering information
            
        Returns
        -------
        Dict[str, Any]
            Final aggregated response
        """
        query = orchestrator_result["query"]
        agent_responses = orchestrator_result["agent_responses"]
        fallback_result = orchestrator_result["fallback_result"]
        
        # Check if domain filtering was applied (non-medical query)
        domain_filtered = orchestrator_result.get("domain_filtered", False)
        if domain_filtered:
            logger.info("Query was filtered as non-medical, passing through domain filter response")
            classification = orchestrator_result.get("query_classification", {})
            domain = classification.get("domain", "non-medical")
            
            return {
                "answer": fallback_result["answer"],
                "citations": [],
                "agent_responses": [],
                "fallback_used": True,
                "fallback_reason": f"Query classified as {domain} (non-medical research)"
            }
        
        # If not domain filtered, proceed with normal aggregation or fallback
        if fallback_result["fallback_activated"]:
            logger.info("Using fallback response instead of primary aggregation")
            return {
                "answer": fallback_result["answer"],
                "citations": fallback_result["citations"],
                "agent_responses": agent_responses,
                "fallback_used": True,
                "fallback_reason": fallback_result.get("fallback_reason", "Unknown reason")
            }
        else:
            logger.info("Using primary aggregation (no fallback needed)")
            return self._aggregate_primary_responses(query, agent_responses)
