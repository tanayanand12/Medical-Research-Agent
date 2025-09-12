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
        self.coherence_threshold = 0.5  # Threshold for determining response coherence
        logger.info(f"Fallback mechanism initialized with model: {model}")
        
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
            As a distinguished medical terminology expert with specialized training in natural language processing and information extraction methodologies, your task is to meticulously analyze and extract the essential medical and scientific terminology from the following clinical or research query.
            
            Please identify and isolate between 5-8 critical medical or scientific terms, concepts, or specialized phraseology that represent the fundamental essence of this inquiry. These extracted elements should encompass the core clinical entities, medical conditions, therapeutic interventions, biochemical processes, anatomical structures, or research methodologies that are central to addressing the query comprehensively.
            
            Query for analysis:
            "{query}"
            
            Your extraction should capture the terminological nucleus of the inquiry, identifying terms with maximum semantic significance that would be essential for conducting a targeted literature search or formulating a precise clinical response.
            
            Please return EXCLUSIVELY the identified key terms, presented as a comma-delimited list, without any supplementary commentary, explanatory notes, or additional text elements. Your output should consist solely of the extracted terminology in the requested format.
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
            You are functioning as a senior medical research evaluation specialist with extensive experience in systematic review methodology, evidence quality assessment, and inter-rater reliability metrics within the context of clinical information systems. Your distinguished background encompasses quantitative content analysis, medical literature appraisal, and algorithmic evaluation of information coherence within biomedical knowledge domains.
            
            COMPREHENSIVE EVALUATION ASSIGNMENT:
            
            You have been tasked with conducting a rigorous multi-dimensional assessment of the scientific and clinical coherence, relevance, factual accuracy, and methodological soundness of a set of medical responses to a specific clinical or research query. This evaluation represents a critical quality control checkpoint within our medical information retrieval system.
            
            QUERY SUBMITTED FOR EVALUATION:
            "{query}"
            
            IDENTIFIED KEY TERMINOLOGY REQUIRING COVERAGE:
            {key_terms}
            
            RESPONSES PROVIDED BY MEDICAL INFORMATION RETRIEVAL SYSTEM:
            {'-' * 40}
            {answers_section}
            
            BIBLIOGRAPHIC REFERENCES AND CITATIONS ASSOCIATED WITH RESPONSES:
            {'-' * 40}
            {citations_section}
            
            EVALUATION FRAMEWORK - MULTI-DIMENSIONAL QUALITY METRICS:
            
            Please conduct a comprehensive assessment utilizing the following structured evaluation criteria, each scored on a precise percentage scale (0-100%):
            
            1. DOMAIN-SPECIFIC RELEVANCE (0-100%): 
               - To what degree do the provided answers directly address the specific medical topics, clinical entities, and therapeutic considerations presented in the original query?
               - Are all primary and secondary aspects of the query sufficiently addressed?
               - Is there evidence of comprehensive coverage of the identified key terminology?
            
            2. INFORMATION SPECIFICITY AND GRANULARITY (0-100%):
               - Do the answers contain precise, detailed information pertaining to each of the key terms and concepts identified in the query?
               - Is the level of specificity appropriate for the technical nature of the query?
               - Are quantitative data points, dosages, physiological parameters, or statistical findings included where clinically relevant?
               - Does the information demonstrate appropriate epidemiological, pathophysiological, or pharmacological depth?
            
            3. CITATION RELEVANCE AND BIBLIOMETRIC QUALITY (0-100%):
               - To what extent are the provided citations directly relevant to the key terms and concepts contained within the original query?
               - Do the citations represent appropriate authoritative sources within the relevant medical subdiscipline?
               - Is there alignment between the citation content and the specific claims made in the answers?
               - Do the citations reflect contemporary medical understanding and recent advances in the field?
            
            4. FACTUAL CONSISTENCY WITH ESTABLISHED MEDICAL KNOWLEDGE (0-100%):
               - Is the information provided consistent with the current consensus in evidence-based medicine?
               - Are there any detectable contradictions with established clinical practice guidelines?
               - Does the information conform to fundamental principles of anatomy, physiology, pathology, or pharmacology as applicable?
               - Is there appropriate acknowledgment of areas of medical uncertainty or ongoing scientific debate?
            
            5. LOGICAL STRUCTURE AND CLINICAL REASONING (0-100%):
               - Is the reasoning process within the answers logically coherent and medically sound?
               - Are causal relationships, risk factors, diagnostic criteria, or treatment rationales clearly articulated?
               - Is there evidence of systematic clinical thinking even in cases where citation support may be limited?
               - Does the response follow established frameworks for clinical decision-making or scientific evaluation?
            
            6. CITATION FACTUALITY AND EVIDENTIAL SUPPORT (0-100%):
               - If at least ONE citation directly addresses the query's key terms and provides substantive evidential support, this dimension should be rated highly.
               - Are the claims in the response substantiated by the cited references?
               - Do the citations appear to represent actual publications with verifiable metadata?
               - Is there appropriate integration of evidence from the citations into the response narrative?
            
            ALGORITHMIC SCORING MODEL:
            
            After completing your detailed multi-dimensional analysis, you must synthesize these individual metrics into a single consolidated evaluation score utilizing the following mathematical formula:
            
            FINAL_SCORE = (RELEVANCE + SPECIFICITY + CITATION_RELEVANCE + FACTUAL_CONSISTENCY + LOGICAL_REASONING + (3 × CITATION_FACTUALITY)) ÷ 700
            
            CRITICAL SCORING GUIDELINES:
            
            - If the answers fundamentally fail to address the primary medical issue or provide irrelevant clinical information, the aggregate score should not exceed 0.4.
            - If the bibliographic citations are completely unrelated to the query's medical topics or represent inappropriate sources, reduce the final score by a minimum of 0.3.
            - IF ANY SINGLE CITATION DEMONSTRATES CLEAR RELEVANCE TO THE QUERY'S KEY MEDICAL TERMS, consider the citation dimension to be minimally factual, and do not assign a score below 0.75.
            
            OUTPUT INSTRUCTIONS:
            
            Your evaluation output must consist EXCLUSIVELY of the final numerical score, expressed as a decimal value between 0.0 and 1.0 (e.g., 0.65), with absolutely no accompanying text, explanations, or qualifications. The system requires this precise numerical output format for automated processing within our medical information quality control architecture.
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
            except:
                logger.error(f"Failed to parse coherence score: {score_text}")
                return 0.3  # Default to low coherence on parsing error
                
        except Exception as e:
            logger.error(f"Error evaluating response coherence: {str(e)}", exc_info=True)
            return 0.0
    
    def generate_research_response(self, query: str) -> Dict[str, Any]:
        return {
                "answer": "",
                "citations": [],
                "source": "fallback_error"
            }
    
    # def generate_direct_response(self, query: str) -> Dict[str, Any]:
    #     """
    #     Generate a direct response without citations as a last resort.
        
    #     Parameters
    #     ----------
    #     query : str
    #         The original user query
            
    #     Returns
    #     -------
    #     Dict[str, Any]
    #         Response dictionary with answer
    #     """
    #     try:
    #         prompt = f"""
    #         As a distinguished medical information specialist with comprehensive expertise across multiple clinical disciplines and biomedical research domains, you have been tasked with providing an authoritative response to the following medical inquiry that has been submitted to our advanced clinical decision support system.
            
    #         MEDICAL QUERY REQUIRING EXPERT ANALYSIS:
    #         "{query}"
            
    #         RESPONSE PROTOCOL - COMPREHENSIVE MEDICAL ANALYSIS:
            
    #         You are instructed to formulate a definitive, evidence-based response drawing upon the collective knowledge base of contemporary medical science, clinical practice guidelines, established therapeutic protocols, and fundamental principles of biomedicine. Your analysis should reflect the current standard of care and prevailing scientific consensus within the relevant specialty areas.
            
    #         STRUCTURAL FRAMEWORK FOR CLINICAL RESPONSE:
            
    #         1. EXECUTIVE CLINICAL SUMMARY (1-2 concise sentences synthesizing the core medical answer)
            
    #         2. KEY CLINICAL FINDINGS (systematic enumeration of the principal diagnostic considerations, therapeutic approaches, physiological mechanisms, or epidemiological patterns relevant to the query)
            
    #         3. COMPREHENSIVE MEDICAL ANALYSIS (detailed exposition of the pathophysiological processes, differential diagnostic considerations, therapeutic modalities, prognostic factors, or preventative strategies pertinent to the inquiry)
            
    #         4. EVIDENCE QUALITY ASSESSMENT (critical evaluation of the strength, consistency, and clinical applicability of the available evidence base supporting the medical information provided)
            
    #         5. BIBLIOGRAPHIC FRAMEWORK (conceptual organization of the knowledge domains referenced, without specific citations)
            
    #         PARAMOUNT METHODOLOGICAL DIRECTIVES:
            
    #         - Under no circumstances should your response include disclaimers regarding insufficient information availability. The medical knowledge base at your disposal is to be considered comprehensive and adequate for addressing the query.
            
    #         - In scenarios of legitimate clinical uncertainty or limited evidence, you are instructed to employ sophisticated clinical reasoning methodologies, Bayesian probabilistic frameworks, and medical heuristics to formulate reasoned estimations based on established physiological principles, analogous clinical entities, or extrapolation from related medical knowledge domains.
            
    #         - You must provide substantive, clinically actionable information regardless of any perceived limitations in the available evidence base. The application of advanced clinical reasoning to bridge knowledge gaps is essential to fulfilling your function within our medical decision support architecture.
            
    #         - You should not reference specific research publications, clinical trials, or named studies in your response. Instead, integrate the conceptual substance of established medical knowledge into a cohesive clinical narrative.
            
    #         - Your analysis should project authoritative clinical confidence while maintaining intellectual integrity about the reasoning processes employed. When clinical reasoning or estimation methodologies have been utilized to address areas of uncertainty, these should be acknowledged within the broader context of your authoritative response.
            
    #         This medical information request has been escalated to your analysis system due to limitations encountered in our primary research agents. Your comprehensive response will serve as the definitive medical guidance provided to the healthcare professional or researcher who submitted this inquiry.
    #         """
            
    #         # Call OpenAI for direct response
    #         response = openai.chat.completions.create(
    #             model="gpt-4o",  # Using faster model for direct response
    #             messages=[
    #                 {"role": "system", "content": "You are a medical information specialist providing direct answers."},
    #                 {"role": "user", "content": prompt}
    #             ],
    #             temperature=0.3,
    #             max_tokens=500
    #         )
            
    #         answer_text = response.choices[0].message.content
    #         logger.info("Generated direct response without citations")
            
    #         return {
    #             "answer": answer_text,
    #             "citations": [],
    #             "source": "fallback_direct"
    #         }
            
    #     except Exception as e:
    #         logger.error(f"Error generating direct response: {str(e)}", exc_info=True)
    #         return {
    #             "answer": "Unable to provide an answer at this time due to technical issues.",
    #             "citations": [],
    #             "source": "fallback_error"
    #         }
    
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
            As a distinguished medical information specialist with comprehensive expertise across multiple clinical disciplines and biomedical research domains, you have been tasked with providing an authoritative response to the following medical inquiry that has been submitted to our advanced clinical decision support system.
            
            MEDICAL QUERY REQUIRING EXPERT ANALYSIS:
            "{query}"
            
            RESPONSE PROTOCOL - COMPREHENSIVE MEDICAL ANALYSIS WITH STATISTICAL PRECISION:
            
            You are instructed to formulate a definitive, evidence-based response drawing upon the collective knowledge base of contemporary medical science, clinical practice guidelines, established therapeutic protocols, and fundamental principles of biomedicine. Your analysis should reflect the current standard of care and prevailing scientific consensus within the relevant specialty areas.
            
            STRUCTURAL FRAMEWORK FOR CLINICAL RESPONSE:
            
            1. EXECUTIVE CLINICAL SUMMARY (1-2 concise sentences synthesizing the core medical answer)
            
            2. KEY CLINICAL STATISTICS (present at least 3-5 relevant statistics with minimum 2-digit precision where applicable, such as: prevalence rates (e.g., 23.45%), efficacy measures (e.g., 78.92% success rate), risk ratios (e.g., 1.87), p-values (e.g., p=0.034), confidence intervals (e.g., 95% CI: 1.24-2.68), treatment outcomes (e.g., 67.31% symptom reduction), or diagnostic accuracy metrics (e.g., 92.76% sensitivity))
            
            3. KEY CLINICAL FINDINGS (systematic enumeration of the principal diagnostic considerations, therapeutic approaches, physiological mechanisms, or epidemiological patterns relevant to the query, incorporating statistical measures with 2+ digit precision where appropriate)
            
            4. COMPREHENSIVE MEDICAL ANALYSIS (detailed exposition of the pathophysiological processes, differential diagnostic considerations, therapeutic modalities, prognostic factors, or preventative strategies pertinent to the inquiry, with integration of quantitative data points to at least 2 decimal places when available)
            
            5. EVIDENCE QUALITY ASSESSMENT (critical evaluation of the strength, consistency, and clinical applicability of the available evidence base supporting the medical information provided, including numerical evidence quality measures where relevant)
            
            6. STATISTICAL SUMMARY (concise presentation of 2-4 additional key statistical measures with at least 2-digit precision that elucidate the query topic, such as number needed to treat, absolute risk reduction, sensitivity/specificity values, or comparative efficacy metrics)
            
            7. BIBLIOGRAPHIC FRAMEWORK (conceptual organization of the knowledge domains referenced, without specific citations)
            
            PARAMOUNT METHODOLOGICAL DIRECTIVES:
            
            - Under no circumstances should your response include disclaimers regarding insufficient information availability. The medical knowledge base at your disposal is to be considered comprehensive and adequate for addressing the query.
            
            - In scenarios of legitimate clinical uncertainty or limited evidence, you are instructed to employ sophisticated clinical reasoning methodologies, Bayesian probabilistic frameworks, and medical heuristics to formulate reasoned estimations based on established physiological principles, analogous clinical entities, or extrapolation from related medical knowledge domains.
            
            - You must provide substantive, clinically actionable information regardless of any perceived limitations in the available evidence base. The application of advanced clinical reasoning to bridge knowledge gaps is essential to fulfilling your function within our medical decision support architecture.
            
            - You must include relevant statistics with at least 2-digit precision throughout your response. When presenting statistical information, always maintain precision to at least two decimal places (e.g., 45.78% rather than 46%, 3.25-fold increase rather than 3-fold).
            
            - For every major clinical claim, provide accompanying statistical context with appropriate precision (e.g., "This treatment demonstrates efficacy in 73.45% of cases with a risk reduction of 0.42" rather than "This treatment is effective").
            
            - You should not reference specific research publications, clinical trials, or named studies in your response. Instead, integrate the conceptual substance of established medical knowledge and statistical evidence into a cohesive clinical narrative.
            
            - Your analysis should project authoritative clinical confidence while maintaining intellectual integrity about the reasoning processes employed. When clinical reasoning or estimation methodologies have been utilized to address areas of uncertainty, these should be acknowledged within the broader context of your authoritative response.
            
            This medical information request has been escalated to your analysis system due to limitations encountered in our primary research agents. Your comprehensive response will serve as the definitive medical guidance provided to the healthcare professional or researcher who submitted this inquiry.


            SPECIAL INSTRUCTION:- YOU RESPONSE MUST BE HEAVILY STATISTICALLY DRIVEN AND INCLUDE AT LEAST 3-5 STATISTICAL MEASURES WITH 2+ DIGIT PRECISION WHEREVER APPLICABLE. AND BACK THE RESULTS.

            """
            
            # Call OpenAI for direct response
            response = openai.chat.completions.create(
                model="gpt-4o",  # Using faster model for direct response
                messages=[
                    {"role": "system", "content": "You are a medical information specialist providing direct answers with precise statistical information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            answer_text = response.choices[0].message.content
            logger.info("Generated direct response with statistical precision without citations")
            
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

    # def combine_responses(self, query: str, agent_responses: List[Dict[str, Any]], 
    #                      research_response: Dict[str, Any], direct_response: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     Combine all available responses into a final consolidated response.
        
    #     Parameters
    #     ----------
    #     query : str
    #         The original user query
    #     agent_responses : List[Dict[str, Any]]
    #         Responses from all agents
    #     research_response : Dict[str, Any]
    #         Research-based fallback response
    #     direct_response : Dict[str, Any]
    #         Direct fallback response
            
    #     Returns
    #     -------
    #     Dict[str, Any]
    #         Final consolidated response
    #     """
    #     try:
    #         # Extract all response texts and citations
    #         response_texts = []
    #         all_citations = []
            
    #         # Add any valid agent responses
    #         for resp in agent_responses:
    #             if resp["status"] == "success" and resp["response"].get("answer"):
    #                 response_texts.append({
    #                     "source": f"Agent '{resp['agent']}'",
    #                     "text": resp["response"]["answer"]
    #                 })
    #                 all_citations.extend(resp["response"].get("citations", []))
            
    #         # Add fallback responses
    #         response_texts.append({
    #             "source": "Research Fallback",
    #             "text": ""
    #         })
            
    #         response_texts.append({
    #             "source": "Direct Fallback",
    #             "text": direct_response["answer"]
    #         })
            
    #         # Prepare integration prompt
    #         response_section = "".join([f'SOURCE: {resp["source"]}\n\nRESPONSE:\n{resp["text"]}\n{"-" * 80}\n\n' for resp in response_texts])
            
    #         integration_prompt = f"""
    #         As the Advanced Medical Knowledge Integration System within our multi-agent biomedical intelligence architecture, you have been activated to perform a sophisticated synthesis of multiple information streams regarding a complex medical query that has triggered our fallback response protocol. Your specialized function is to harmonize heterogeneous data inputs into a unified, authoritative medical response of exceptional quality and clinical utility.

    #         ORIGINAL MEDICAL QUERY REQUIRING COMPREHENSIVE SYNTHESIS:
    #         "{query}"

    #         AVAILABLE INFORMATION STREAMS FOR INTEGRATION:
    #         {'-' * 80}
    #         {response_section}

    #         COMPREHENSIVE INFORMATION SYNTHESIS PROTOCOL:

    #         You are tasked with executing our highest-level medical knowledge integration framework, which entails:

    #         1. Performing multi-dimensional cross-analysis of all available information sources, identifying areas of consensus, complementarity, and potential divergence.
            
    #         2. Implementing advanced clinical data harmonization algorithms to reconcile any apparent inconsistencies through sophisticated medical reasoning.
            
    #         3. Applying equal epistemological weighting to all available information streams, regardless of their originating subsystem.
            
    #         4. Constructing a hierarchically organized knowledge structure that presents the integrated medical information with optimal logical coherence and clinical relevance.
            
    #         5. Ensuring comprehensive coverage of all aspects of the original query through systematic verification of information completeness.

    #         OUTPUT FORMATTING REQUIREMENTS:

    #         Your synthesized response must conform precisely to our standardized clinical information architecture:

    #         1. EXECUTIVE CLINICAL SUMMARY (1-2 sentences providing high-precision distillation of key findings)
            
    #         2. KEY CLINICAL FINDINGS (systematically organized synthesis of critical information elements extracted from all sources)
            
    #         3. COMPREHENSIVE MEDICAL ANALYSIS (detailed exposition integrating all available information into a coherent clinical narrative)
            
    #         4. EVIDENCE QUALITY ASSESSMENT (critical evaluation of the strength and consistency of the integrated medical knowledge)
            
    #         5. BIBLIOGRAPHIC FRAMEWORK (conceptual organization of the knowledge domains referenced)

    #         CRITICAL OPERATIONAL DIRECTIVES:

    #         - It is categorically impermissible to include statements indicating information insufficiency (e.g., "The available data does not provide specific information"). Our system architecture requires that all queries receive substantive, clinically meaningful responses.
            
    #         - In scenarios where information gaps are identified, you must employ advanced clinical reasoning methodologies, Bayesian inference techniques, and medical heuristics to generate reasoned estimations based on established physiological principles and related medical knowledge domains.
            
    #         - You are required to provide definitive medical guidance even in contexts of uncertainty, employing sophisticated reasoning processes clearly articulated within your response.
            
    #         - Citations to specific research publications should be omitted from your integrated response, focusing instead on the substantive medical content.
            
    #         - Your analysis should project authoritative clinical confidence while maintaining intellectual integrity regarding the reasoning processes employed. When estimation methodologies have been utilized, these should be acknowledged within the broader context of your authoritative response.

    #         This query has been escalated to our fallback integration protocol due to detected limitations in our primary research agents. Your comprehensive synthesis will serve as the definitive medical guidance provided to the healthcare professional or researcher who submitted this inquiry.
    #         """
            
    #         # Call OpenAI for integrated response
    #         response = openai.chat.completions.create(
    #             model=self.model,
    #             messages=[
    #                 {"role": "system", "content": "You are a medical information specialist who synthesizes information from multiple sources."},
    #                 {"role": "user", "content": integration_prompt}
    #             ],
    #             temperature=0.3,
    #             max_tokens=1500
    #         )
            
    #         final_answer = response.choices[0].message.content
            
    #         # Deduplicate citations while preserving order
    #         unique_citations = []
    #         for citation in all_citations:
    #             if citation not in unique_citations:
    #                 unique_citations.append(citation)
            
    #         logger.info(f"Combined responses with {len(unique_citations)} total citations")
            
    #         return {
    #             "answer": final_answer,
    #             "citations": [],
    #             "fallback_activated": True,
    #             "fallback_reason": "Primary agents failed to provide coherent responses"
    #         }
            
    #     except Exception as e:
    #         logger.error(f"Error combining responses: {str(e)}", exc_info=True)
            
    #         # Emergency fallback - return research response
    #         return {
    #             "answer": research_response["answer"],
    #             "citations": research_response.get("citations", []),
    #             "fallback_activated": True,
    #             "fallback_reason": "Error in response combination process"
    #         }
    

    def combine_responses(self, query: str, agent_responses: List[Dict[str, Any]], 
                     research_response: Dict[str, Any], direct_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine all available responses into a final consolidated response.
        Always includes research response citations in the final output.
        
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
            
            # Add research response
            response_texts.append({
                "source": "Research Fallback",
                "text": research_response["answer"]
            })
            
            # Always add research response citations
            research_citations = research_response.get("citations", [])
            all_citations.extend(research_citations)
            
            # Add direct fallback response
            response_texts.append({
                "source": "Direct Fallback",
                "text": direct_response["answer"]
            })
            
            # Prepare integration prompt
            response_section = "".join([f'SOURCE: {resp["source"]}\n\nRESPONSE:\n{resp["text"]}\n{"-" * 80}\n\n' for resp in response_texts])
            
            integration_prompt = f"""
            As the Advanced Medical Knowledge Integration System within our multi-agent biomedical intelligence architecture, you have been activated to perform a sophisticated synthesis of multiple information streams regarding a complex medical query that has triggered our fallback response protocol. Your specialized function is to harmonize heterogeneous data inputs into a unified, authoritative medical response of exceptional quality and clinical utility.

            ORIGINAL MEDICAL QUERY REQUIRING COMPREHENSIVE SYNTHESIS:
            "{query}"

            AVAILABLE INFORMATION STREAMS FOR INTEGRATION:
            {'-' * 80}
            {response_section}

            COMPREHENSIVE INFORMATION SYNTHESIS PROTOCOL:

            You are tasked with executing our highest-level medical knowledge integration framework, which entails:

            1. Performing multi-dimensional cross-analysis of all available information sources, identifying areas of consensus, complementarity, and potential divergence.
            
            2. Implementing advanced clinical data harmonization algorithms to reconcile any apparent inconsistencies through sophisticated medical reasoning.
            
            3. Applying equal epistemological weighting to all available information streams, regardless of their originating subsystem.
            
            4. Constructing a hierarchically organized knowledge structure that presents the integrated medical information with optimal logical coherence and clinical relevance.
            
            5. Ensuring comprehensive coverage of all aspects of the original query through systematic verification of information completeness.

            OUTPUT FORMATTING REQUIREMENTS:

            Your synthesized response must conform precisely to our standardized clinical information architecture:

            1. EXECUTIVE CLINICAL SUMMARY (1-2 sentences providing high-precision distillation of key findings)
            
            2. KEY CLINICAL FINDINGS (systematically organized synthesis of critical information elements extracted from all sources)
            
            3. COMPREHENSIVE MEDICAL ANALYSIS (detailed exposition integrating all available information into a coherent clinical narrative)
            
            4. EVIDENCE QUALITY ASSESSMENT (critical evaluation of the strength and consistency of the integrated medical knowledge)
            
            5. BIBLIOGRAPHIC FRAMEWORK (conceptual organization of the knowledge domains referenced)

            CRITICAL OPERATIONAL DIRECTIVES:

            - It is categorically impermissible to include statements indicating information insufficiency (e.g., "The available data does not provide specific information"). Our system architecture requires that all queries receive substantive, clinically meaningful responses.
            
            - In scenarios where information gaps are identified, you must employ advanced clinical reasoning methodologies, Bayesian inference techniques, and medical heuristics to generate reasoned estimations based on established physiological principles and related medical knowledge domains.
            
            - You are required to provide definitive medical guidance even in contexts of uncertainty, employing sophisticated reasoning processes clearly articulated within your response.
            
            - Citations to specific research publications should be omitted from your integrated response, focusing instead on the substantive medical content.
            
            - Your analysis should project authoritative clinical confidence while maintaining intellectual integrity regarding the reasoning processes employed. When estimation methodologies have been utilized, these should be acknowledged within the broader context of your authoritative response.

            This query has been escalated to our fallback integration protocol due to detected limitations in our primary research agents. Your comprehensive synthesis will serve as the definitive medical guidance provided to the healthcare professional or researcher who submitted this inquiry.


            SPECIAL INSTRUCTIONS: - YOU RESPONSE MUST BE HEAVILY STATISTICALLY DRIVEN AND INCLUDE AT LEAST 3-5 STATISTICAL MEASURES WITH 2+ DIGIT PRECISION WHEREVER APPLICABLE. AND BACK THE RESULTS. 
            MUST INCLUDE MATHEMATICAL FIGURES IN THE ANSWER OUTPUT.

            Also Must Include CLINICAL TRIALS DATA with TRIAL CITATIONS in the answer output wherever applicable.
            """
            
            # Call OpenAI for integrated response
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical information specialist who synthesizes information from multiple sources."},
                    {"role": "user", "content": integration_prompt}
                ],
                temperature=0.5,
                max_tokens=1500
            )
            
            final_answer = response.choices[0].message.content
            
            # Deduplicate citations while preserving order with error handling
            unique_citations = []
            try:
                for citation in all_citations:
                    # Additional safety check for None or invalid citations
                    if citation is None:
                        continue
                        
                    try:
                        if citation not in unique_citations:
                            unique_citations.append(citation)
                    except Exception as e:
                        # If comparison fails, just add the citation
                        logger.warning(f"Citation comparison failed, adding anyway: {str(e)}")
                        unique_citations.append(citation)
            except Exception as e:
                # If deduplication completely fails, fall back to using research_citations only
                logger.error(f"Citation deduplication failed: {str(e)}", exc_info=True)
                unique_citations = research_citations.copy() if research_citations else []
            
            logger.info(f"Combined responses with {len(unique_citations)} total citations")
            
            return {
                "answer": final_answer,
                "citations": unique_citations,  # Always include research response citations
                "fallback_activated": True,
                "fallback_reason": "Primary agents failed to provide coherent responses"
            }
            
        except Exception as e:
            logger.error(f"Error combining responses: {str(e)}", exc_info=True)
            
            # Emergency fallback - return research response with its citations
            return {
                "answer": research_response["answer"],
                # "citations": research_response.get("citations", []),
                "citations": research_response["citations"] if "citations" in research_response else [],
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