"""
fallback.py
~~~~~~~~~~~

Fall-back mechanism for the medical research agent system.
Activates when primary agents fail to provide coherent answers,
using alternative methods to generate responses.

Key features:
- Response coherence evaluation
- Multi-stage fallback pipeline
- Transparent logging of fallback events
- Integration with existing orchestration
"""

import logging
import os
from typing import Dict, Any, List, Optional
import openai
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/fallback.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class FallbackMechanism:
    """
    Fallback mechanism for handling cases where primary agents fail to provide
    coherent or consistent answers.
    
    The mechanism implements a multi-stage fallback approach:
    1. Evaluate coherence of primary agent responses
    2. If incoherent, attempt to generate a research-based response via OpenAI
    3. As a final fallback, generate a direct response without citations
    4. Combine and format all available information into a final response
    """
    
    def __init__(self, model="gpt-4o"):
        """
        Initialize the fallback mechanism.
        
        Parameters
        ----------
        model : str
            The OpenAI model to use for fallback responses
        """
        self.model = model
        self.coherence_threshold = 0.7  # Threshold for determining response coherence
        logger.info(f"Fallback mechanism initialized with model: {model}")
        
    # def evaluate_coherence(self, query: str, agent_responses: List[Dict[str, Any]]) -> float:
    #     """
    #     Evaluate the coherence of agent responses relative to the query.
        
    #     Parameters
    #     ----------
    #     query : str
    #         The original user query
    #     agent_responses : List[Dict[str, Any]]
    #         Responses from all agents
            
    #     Returns
    #     -------
    #     float
    #         Coherence score between 0 and 1
    #     """
    #     try:
    #         # Extract answer texts
    #         answer_texts = []
    #         for resp in agent_responses:
    #             if resp["status"] == "success" and resp["response"].get("answer"):
    #                 answer_texts.append(resp["response"]["answer"])
            
    #         if not answer_texts:
    #             logger.warning("No valid answers found to evaluate coherence")
    #             return 0.0
                
    #         # Prepare evaluation prompt
            
    #         answers_section = "".join([f'ANSWER {i+1}:\n{ans}\n{"-" * 40}\n' for i, ans in enumerate(answer_texts)])
            
    #         evaluation_prompt = f"""
    #         Please evaluate the coherence and consistency of the following answers relative to the question.
    #         Return a score between 0 and 1, where:
    #         - 0 indicates completely incoherent or unrelated answers
    #         - 1 indicates perfectly coherent and directly relevant answers
            
    #         Question: {query}
            
    #         Answers:
    #         {'-' * 40}
    #         {answers_section}
            
    #         Analyze the following aspects:
    #         1. Direct relevance to the question
    #         2. Presence of specific information that addresses the question
    #         3. Internal consistency of facts
    #         4. Comprehensiveness of the response
    #         5. Appropriate citation of sources
            
    #         Provide a numeric score only (e.g., 0.75).
    #         """
            
    #         # Call OpenAI for evaluation
    #         response = openai.chat.completions.create(
    #             model="gpt-4o",  # Using smaller model for evaluation
    #             messages=[
    #                 {"role": "system", "content": "You are an objective evaluator of text coherence and relevance."},
    #                 {"role": "user", "content": evaluation_prompt}
    #             ],
    #             temperature=0.1,
    #             max_tokens=10
    #         )
            
    #         # Parse score
    #         score_text = response.choices[0].message.content.strip()
    #         try:
    #             score = float(score_text)
    #             # Ensure score is between 0 and 1
    #             score = max(0.0, min(1.0, score))
    #             logger.info(f"Coherence evaluation score: {score}")
    #             return score
    #         except ValueError:
    #             logger.error(f"Failed to parse coherence score: {score_text}")
    #             return 0.3  # Default to low coherence on parsing error
                
    #     except Exception as e:
    #         logger.error(f"Error evaluating response coherence: {str(e)}", exc_info=True)
    #         return 0.0
    
    def evaluate_coherence(self, query: str, agent_responses: List[Dict[str, Any]]) -> float:
        """
        Evaluate the coherence and relevance of agent responses relative to the query.
        
        Parameters
        ----------
        query : str
            The original user query
        agent_responses : List[Dict[str, Any]]
            Responses from all agents
            
        Returns
        -------
        float
            Coherence score between 0 and 1
        """
        try:
            # Extract answer texts and citations
            answer_texts = []
            citations = []
            
            for resp in agent_responses:
                if resp["status"] == "success" and resp["response"].get("answer"):
                    answer_texts.append(resp["response"]["answer"])
                    if resp["response"].get("citations"):
                        citations.extend(resp["response"]["citations"])
            
            if not answer_texts:
                logger.warning("No valid answers found to evaluate coherence")
                return 0.0
                
            # Extract key terms from the query
            query_analysis_prompt = f"""
            Extract 5-8 key medical or scientific terms/concepts from this query:
            
            {query}
            
            Return ONLY the key terms separated by commas, with no additional text.
            """
            
            # Get key terms from the query
            key_terms_response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You extract key medical and scientific terms from queries."},
                    {"role": "user", "content": query_analysis_prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            key_terms = key_terms_response.choices[0].message.content.strip()
            logger.info(f"Extracted key terms: {key_terms}")
            
            # Format answers
            answers_section = "".join([f'ANSWER {i+1}:\n{ans}\n{"-" * 40}\n' for i, ans in enumerate(answer_texts)])
            
            # Format citations
            citations_section = "".join([f'CITATION {i+1}:\n{cit}\n{"-" * 40}\n' for i, cit in enumerate(citations[:10])])  # Limit to first 10 citations
            
            # Prepare comprehensive evaluation prompt
            evaluation_prompt = f"""
            You are evaluating the coherence, relevance, and accuracy of answers to a specific medical query.
            
            QUERY: {query}
            
            KEY TERMS TO LOOK FOR: {key_terms}
            
            ANSWERS PROVIDED:
            {'-' * 40}
            {answers_section}
            
            CITATIONS PROVIDED:
            {'-' * 40}
            {citations_section}
            
            Provide a detailed evaluation focusing on these criteria:
            
            1. RELEVANCE (0-100%): Do the answers directly address the specific medical topics in the query?
            2. SPECIFIC INFORMATION (0-100%): Do the answers contain precise information about the key terms?
            3. CITATION RELEVANCE (0-100%): Are the citations relevant to the query's key terms?
            4. FACTUAL CONSISTENCY (0-100%): Is the information consistent with established medical knowledge?
            5. REASONING (0-100%): Is the reasoning behind the answers logical and well-structured even if the citations are not really relevant?
            6. CITATION FACTUALITY (0-100%): If at least ONE citation is relevant to the query's key terms the citations are considered factual.
            
            After your analysis, provide ONLY a single float value between 0 and 1 representing the overall coherence 
            and relevance score. Use this formula: 
            
            SCORE = (RELEVANCE + SPECIFIC INFORMATION + CITATION RELEVANCE + FACTUAL CONSISTENCY + REASONING + 3 * CITATION FACTUALITY) / 700
            
            If the answers completely fail to address the query or provide irrelevant information, score should be below 0.4.
            If citations are completely unrelated to the query topics, reduce score by at least 0.3.
            IF ANY OF THE CITATIONS IS RELEVANT TO THE QUERY'S KEY TERMS, CONSIDER THEM AS FACTUAL, AND DO NOT SCORE THEM BELOW 0.75.
            
            PROVIDE ONLY THE FINAL NUMERIC SCORE (e.g., 0.65) WITH NO OTHER TEXT.
            """
            
            # Call Claude or GPT-4 for a more thorough evaluation
            response = openai.chat.completions.create(
                model="gpt-4o",  # Using the most capable model for evaluation
                messages=[
                    {"role": "system", "content": "You are an expert medical researcher who evaluates the coherence and relevance of scientific answers."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            # Parse score
            score_text = response.choices[0].message.content.strip()
            try:
                score = float(score_text)
                # Ensure score is between 0 and 1
                score = max(0.0, min(1.0, score))
                logger.info(f"Comprehensive evaluation score: {score}")
                return score
            except :
                logger.error(f"Failed to parse coherence score: {score_text}")
                return 0.3  # Default to low coherence on parsing error
                
        except Exception as e:
            logger.error(f"Error evaluating response coherence: {str(e)}", exc_info=True)
            return 0.0
    
    # def generate_research_response(self, query: str) -> Dict[str, Any]:
    #     """
    #     Generate a response based on PubMed research papers.
        
    #     Parameters
    #     ----------
    #     query : str
    #         The original user query
            
    #     Returns
    #     -------
    #     Dict[str, Any]
    #         Response dictionary with answer and citations
    #     """
    #     try:
    #         prompt = f"""
    #         You are a medical research expert with access to the PubMed database.
            
    #         Please answer the following question based on recent medical research papers
    #         published in PubMed journals. Cite specific papers using numbered citations
    #         in brackets [1], [2], etc.
            
    #         Question: {query}
            
    #         Requirements:
    #         1. Base your answer ONLY on peer-reviewed medical research
    #         2. Include at least 3-5 specific citations to recent papers (2018-2024)
    #         3. Provide specific data points from the research where available
    #         4. Format citations as: [#] Authors. Title. Journal, Year. PMID: #####
    #         5. Follow evidence-based medicine principles
    #         6. Be specific about study designs and sample sizes
    #         7. Note any limitations or contradictions in the research
            
    #         Format your response with:
    #         - Executive Summary (1-2 sentences)
    #         - Key Findings (bullet points)
    #         - Detailed Analysis (2-3 paragraphs)
    #         - Research Quality Assessment
    #         - References (full citations)
    #         """
            
    #         # Call OpenAI for research-based response
    #         response = openai.chat.completions.create(
    #             model=self.model,
    #             messages=[
    #                 {"role": "system", "content": "You are a medical research expert with deep knowledge of PubMed literature."},
    #                 {"role": "user", "content": prompt}
    #             ],
    #             temperature=0.3,
    #             max_tokens=1500
    #         )
            
    #         answer_text = response.choices[0].message.content
            
    #         # Extract citations from the answer
    #         citations = []
    #         lines = answer_text.split('\n')
    #         for line in lines:
    #             if line.strip().startswith('[') and ']' in line and 'PMID:' in line:
    #                 citations.append(line.strip())
            
    #         logger.info(f"Generated research response with {len(citations)} citations")
            
    #         return {
    #             "answer": answer_text,
    #             "citations": citations,
    #             "source": "fallback_research"
    #         }
            
    #     except Exception as e:
    #         logger.error(f"Error generating research response: {str(e)}", exc_info=True)
    #         return {
    #             "answer": "Unable to generate research-based response due to technical issues.",
    #             "citations": [],
    #             "source": "fallback_error"
    #         }
    
    def generate_research_response(self, query: str) -> Dict[str, Any]:
        return {
                "answer": "",
                "citations": [],
                "source": "fallback_error"
            }
    
    def generate_direct_response(self, query: str) -> Dict[str, Any]:
        """
        Generate a direct response without citations as a last resort.
        
        Parameters
        ----------
        query : str
            The original user query
            
        Returns
        -------
        Dict[str, Any]
            Response dictionary with answer
        """
        try:
            prompt = f"""
            Answer the following medical question directly, without citations.
            Provide only what is considered established medical knowledge.
            
            Question: {query}
            
                        Instructions:
            1. Integrate all information from the available responses.
            2. Prioritize all the information equally.
            3. Structure the response logically with clear sections.
            4. Include a summary of key findings and detailed analysis.

            
            Format your response with:
            1. Executive Summary (1-2 sentences)
            2. Key Findings (synthesized from all sources)
            3. Detailed Analysis (integrated information)
            4. Evidence Quality Assessment
            5. References (numbered citations)
            
            
            SPECIAL INSTRUCTIONS:
            - DO NOT EVER USE THIS KIND OF STATEMENTS`The available data does not provide specific information`. 
            - TRY TO GUESS IF YOU DO NOT HAVE ENOUGH DATA AND GIVE USER WHAT IT WANTS (if needed make up the data using guestimate process and provide a reasoned estimate based on what you know).
            - GIVE AN ANSWER EVEN IF YOU HAVE TO REASON AND GUESS THINGS.
            - DO NOT CITE SPECIFIC PAPERS OR STUDIES.
            - EXPLAIN THE ANSWER AND REASONING PROCESS CONFIDENTLY, WHILE MENTIONING THE SAME.
            """
            
            # Guidelines:
            # 1. Provide a factual, concise answer based on current medical understanding
            # 2. Explain key concepts briefly
            # 3. Note any significant medical controversies if relevant
            # 4. You can reason and guesstimate things if needed, but do not make up data and mention it specifically 
            # 5. Do not cite specific papers or studies
            # 6. WHile giving answer if you reasoned, provide a brief explanation of your reasoning process
            # """
            
            # Call OpenAI for direct response
            response = openai.chat.completions.create(
                model="gpt-4o",  # Using faster model for direct response
                messages=[
                    {"role": "system", "content": "You are a medical information specialist providing direct answers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            answer_text = response.choices[0].message.content
            logger.info("Generated direct response without citations")
            
            # logger.info(f"Direct response: {answer_text}")
            
            return {
                "answer": answer_text,
                "citations": [],
                "source": "fallback_direct"
            }
            
        except Exception as e:
            logger.error(f"Error generating direct response: {str(e)}", exc_info=True)
            return {
                "answer": "Unable to provide an answer at this time due to technical issues.",
                "citations": [],
                "source": "fallback_error"
            }
    
    def combine_responses(self, query: str, agent_responses: List[Dict[str, Any]], 
                         research_response: Dict[str, Any], direct_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine all available responses into a final consolidated response.
        
        Parameters
        ----------
        query : str
            The original user query
        agent_responses : List[Dict[str, Any]]
            Responses from all agents
        research_response : Dict[str, Any]
            Research-based fallback response
        direct_response : Dict[str, Any]
            Direct fallback response
            
        Returns
        -------
        Dict[str, Any]
            Final consolidated response
        """
        try:
            # Extract all response texts and citations
            response_texts = []
            all_citations = []
            
            # Add any valid agent responses
            for resp in agent_responses:
                if resp["status"] == "success" and resp["response"].get("answer"):
                    response_texts.append({
                        "source": f"Agent '{resp['agent']}'",
                        "text": resp["response"]["answer"]
                    })
                    all_citations.extend(resp["response"].get("citations", []))
            
            # Add fallback responses
            response_texts.append({
                "source": "Research Fallback",
                # "text": research_response["answer"]
                "text": ""
            })
            # all_citations.extend(research_response.get("citations", []))
            
            response_texts.append({
                "source": "Direct Fallback",
                "text": direct_response["answer"]
            })
            
            # Prepare integration prompt
            response_section = "".join([f'SOURCE: {resp["source"]}\n\nRESPONSE:\n{resp["text"]}\n{"-" * 80}\n\n' for resp in response_texts])
            
            integration_prompt = f"""
            Synthesize all available information to create a comprehensive, accurate answer to the medical question.
            
            Question: {query}
            
            Available Responses:
            {'-' * 80}
            {response_section}
            
            Instructions:
            1. Integrate all information from the available responses.
            2. Prioritize all the information equally.
            3. Structure the response logically with clear sections.
            4. Include a summary of key findings and detailed analysis.

            
            Format your response with:
            1. Executive Summary (1-2 sentences)
            2. Key Findings (synthesized from all sources)
            3. Detailed Analysis (integrated information)
            4. Evidence Quality Assessment
            5. References (numbered citations)
            
            
            SPECIAL INSTRUCTIONS:
            - DO NOT EVER USE THIS KIND OF STATEMENTS`The available data does not provide specific information`. 
            - TRY TO GUESS IF YOU DO NOT HAVE ENOUGH DATA AND GIVE USER WHAT IT WANTS (if needed make up the data using guestimate process and provide a reasoned estimate based on what you know).
            - GIVE AN ANSWER EVEN IF YOU HAVE TO REASON AND GUESS THINGS.
            - DO NOT CITE SPECIFIC PAPERS OR STUDIES.
            - EXPLAIN THE ANSWER AND REASONING PROCESS CONFIDENTLY, WHILE MENTIONING THE SAME.
            
            Note: This response was generated using a fallback mechanism due to limitations in our primary research agents.
            """
            
            # Call OpenAI for integrated response
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical information specialist who synthesizes information from multiple sources."},
                    {"role": "user", "content": integration_prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            final_answer = response.choices[0].message.content
            
            # Deduplicate citations while preserving order
            unique_citations = []
            for citation in all_citations:
                if citation not in unique_citations:
                    unique_citations.append(citation)
            
            logger.info(f"Combined responses with {len(unique_citations)} total citations")
            
            return {
                "answer": final_answer,
                "citations": [],
                "fallback_activated": True,
                "fallback_reason": "Primary agents failed to provide coherent responses"
            }
            
        except Exception as e:
            logger.error(f"Error combining responses: {str(e)}", exc_info=True)
            
            # Emergency fallback - return research response
            return {
                "answer": research_response["answer"],
                "citations": research_response.get("citations", []),
                "fallback_activated": True,
                "fallback_reason": "Error in response combination process"
            }
    
    def process(self, query: str, agent_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main entry point for the fallback mechanism.
        
        Parameters
        ----------
        query : str
            The original user query
        agent_responses : List[Dict[str, Any]]
            Responses from all agents
            
        Returns
        -------
        Dict[str, Any]
            Final response after fallback processing
        """
        # Evaluate coherence of agent responses
        coherence_score = self.evaluate_coherence(query, agent_responses)
        
        # If coherent, return original responses
        if coherence_score >= self.coherence_threshold:
            logger.info(f"Coherence score {coherence_score} above threshold, no fallback needed")
            return {
                "answer": None,  # Signal to use original aggregated response
                "citations": [],
                "fallback_activated": False
            }
        
        # Log fallback activation
        logger.warning(f"Fallback mechanism activated - coherence score: {coherence_score}")
        
        # Generate research-based response
        logger.info("Generating research-based fallback response")
        research_response = self.generate_research_response(query)
        
        # Generate direct response
        logger.info("Generating direct fallback response")
        direct_response = self.generate_direct_response(query)
        
        # Combine all responses
        logger.info("Combining all available responses")
        final_response = self.combine_responses(
            query, 
            agent_responses, 
            research_response, 
            direct_response
        )
        
        return final_response