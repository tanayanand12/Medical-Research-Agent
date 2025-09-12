"""
guardrails.py
~~~~~~~~~~~~

Guardrail system for medical research RAG agents.
Implements content filtering, scientific integrity checks,
response quality evaluation, and ethical compliance verification.

Key features:
- Input query filtering and validation
- Response content safety checks
- Scientific integrity validation
- Citation and source verification
- Ethical compliance monitoring
- Response quality assessment
"""

import os
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
import json
from pathlib import Path
import openai

# Configure logging
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/guardrails.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load dictionary of medical terms if available
MEDICAL_TERMS_PATH = "resources/medical_terms.json"
try:
    if os.path.exists(MEDICAL_TERMS_PATH):
        with open(MEDICAL_TERMS_PATH, 'r') as f:
            MEDICAL_TERMS = json.load(f)
    else:
        MEDICAL_TERMS = {}
        logger.warning(f"Medical terms dictionary not found at {MEDICAL_TERMS_PATH}")
except Exception as e:
    MEDICAL_TERMS = {}
    logger.error(f"Error loading medical terms: {str(e)}")


class GuardrailSystem:
    """
    Comprehensive guardrail system for medical research agent.
    
    Implements multiple layers of safety checks:
    1. Input query validation and filtering
    2. Response content safety verification
    3. Scientific integrity checks
    4. Citation and source validation
    5. Response quality assessment
    6. Ethical compliance monitoring
    """
    
    def __init__(self, model="gpt-4o"):
        """
        Initialize the guardrail system.
        
        Parameters
        ----------
        model : str
            OpenAI model to use for evaluations
        """
        self.model = model
        
        # Configure thresholds
        self.flagged_terms = self._load_flagged_terms()
        self.sensitive_topics = self._load_sensitive_topics()
        self.citation_threshold = 3  # Minimum citations expected for complex queries
        
        # Response quality thresholds
        self.quality_threshold = 0.7  # Minimum quality score for responses
        self.coherence_threshold = 0.65  # Minimum coherence score
        
        logger.info(f"Guardrail system initialized with model: {model}")
    
    def _load_flagged_terms(self) -> Dict[str, List[str]]:
        """Load dictionary of flagged terms by category."""
        # In production, this would load from a maintained file
        return {
            "harmful_procedures": [
                "DIY surgery", "self-surgery", "home amputation", "self-medication overdose",
                "dangerous drug combination", "lethal dosage", "suicide method",
                "home abortion technique", "illegal drug synthesis"
            ],
            "illegal_substances": [
                "how to make meth", "synthesize fentanyl", "produce illegal drugs",
                "manufacture controlled substances", "home drug lab"
            ],
            "misinformation": [
                "vaccine causes autism", "COVID is a hoax", "5G causes cancer",
                "bleach injection cure", "cancer conspiracy"
            ]
        }
    
    def _load_sensitive_topics(self) -> List[str]:
        """Load list of sensitive medical topics requiring special handling."""
        return [
            "euthanasia", "assisted suicide", "abortion", "gender affirmation",
            "controversial treatments", "experimental procedures", "terminal illness prognosis",
            "genetic testing ethics", "reproductive technology", "medical marijuana"
        ]
    
    def validate_input_query(self, query: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate the input query for safety and appropriateness.
        
        Parameters
        ----------
        query : str
            The user's query
            
        Returns
        -------
        Tuple[bool, str, Dict[str, Any]]
            (is_valid, message, metadata)
        """
        query_lower = query.lower()
        metadata = {
            "flagged_categories": [],
            "sensitive_topics": [],
            "query_type": "general",
            "requires_disclaimer": False,
            "medical_terms_detected": []
        }
        
        # Check for medical terms to determine query type
        for term in list(MEDICAL_TERMS.keys())[:1000]:  # Limit check to first 1000 terms for performance
            if term.lower() in query_lower:
                metadata["medical_terms_detected"].append(term)
        
        if len(metadata["medical_terms_detected"]) > 2:
            metadata["query_type"] = "medical"
        
        # Check for flagged terms
        for category, terms in self.flagged_terms.items():
            for term in terms:
                if term.lower() in query_lower:
                    metadata["flagged_categories"].append(category)
        
        # Check for sensitive topics
        for topic in self.sensitive_topics:
            if topic.lower() in query_lower:
                metadata["sensitive_topics"].append(topic)
                metadata["requires_disclaimer"] = True
        
        # Determine if query is valid
        if "harmful_procedures" in metadata["flagged_categories"] or "illegal_substances" in metadata["flagged_categories"]:
            return False, "This query appears to request information about harmful or illegal activities which we cannot provide.", metadata
        
        if "misinformation" in metadata["flagged_categories"]:
            # Allow query but flag for special handling
            metadata["requires_disclaimer"] = True
        
        return True, "", metadata
    
    def evaluate_scientific_integrity(self, response: Dict[str, Any], query_metadata: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate the scientific integrity of a response.
        
        Parameters
        ----------
        response : Dict[str, Any]
            Agent response to evaluate
        query_metadata : Dict[str, Any]
            Metadata from query validation
            
        Returns
        -------
        Tuple[float, Dict[str, Any]]
            (integrity_score, integrity_metadata)
        """
        answer = response.get("answer", "")
        citations = response.get("citations", [])
        
        integrity_metadata = {
            "citations_count": len(citations),
            "has_sufficient_citations": len(citations) >= self.citation_threshold,
            "citation_quality": 0.0,
            "contains_hedge_statements": False,
            "contains_certainty_statements": False,
            "recognizes_limitations": False,
        }
        
        # Check for hedging language (indicates proper scientific caution)
        hedge_phrases = ["may", "might", "could", "suggests", "indicates", "appears to", 
                         "is associated with", "evidence points to", "studies show",
                         "research suggests", "is linked to", "correlation between"]
        
        hedge_count = sum(1 for phrase in hedge_phrases if phrase in answer.lower())
        integrity_metadata["contains_hedge_statements"] = hedge_count > 0
        
        # Check for inappropriate certainty
        certainty_phrases = ["always", "never", "proven", "definitely", "certainly", 
                             "absolutely", "undoubtedly", "100% effective", "completely safe",
                             "guaranteed", "perfect", "miracle", "revolutionary", "breakthrough"]
        
        certainty_count = sum(1 for phrase in certainty_phrases if phrase in answer.lower())
        integrity_metadata["contains_certainty_statements"] = certainty_count > 2  # Allow some use
        
        # Check for recognition of limitations
        limitation_phrases = ["limitation", "limited evidence", "further research", 
                              "small sample size", "preliminary", "more studies needed",
                              "conflicting results", "inconclusive", "mixed findings"]
        
        for phrase in limitation_phrases:
            if phrase in answer.lower():
                integrity_metadata["recognizes_limitations"] = True
                break
        
        # Calculate integrity score based on these factors
        score_components = [
            0.7 if integrity_metadata["has_sufficient_citations"] else 0.3,
            0.2 if integrity_metadata["contains_hedge_statements"] else 0.0,
            0.0 if integrity_metadata["contains_certainty_statements"] else 0.2,
            0.1 if integrity_metadata["recognizes_limitations"] else 0.0
        ]
        
        integrity_score = sum(score_components)
        integrity_metadata["integrity_score"] = integrity_score
        
        return integrity_score, integrity_metadata
    
    def verify_citations(self, citations: List[str]) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify that citations appear valid and appropriate.
        
        Parameters
        ----------
        citations : List[str]
            List of citation strings
            
        Returns
        -------
        Tuple[bool, Dict[str, Any]]
            (citations_valid, citation_metadata)
        """
        citation_metadata = {
            "count": len(citations),
            "has_journal_references": False,
            "has_pmid": False,
            "has_doi": False,
            "has_recent_references": False,
            "years_detected": []
        }
        
        if not citations:
            return True, citation_metadata
        
        # Check for journal references
        for citation in citations:
            # Look for journal indicators
            if any(j in citation for j in [" J ", "Journal", "Med", "Lancet", " Ann ", "Proceedings"]):
                citation_metadata["has_journal_references"] = True
            
            # Check for PMID
            if "PMID" in citation or "pmid" in citation.lower():
                citation_metadata["has_pmid"] = True
            
            # Check for DOI
            if "doi" in citation.lower() or "DOI" in citation or "10." in citation:
                citation_metadata["has_doi"] = True
            
            # Extract years to check recency
            year_matches = re.findall(r"(19|20)\d{2}", citation)
            if year_matches:
                for year_str in year_matches:
                    try:
                        year = int(year_str)
                        if 1900 <= year <= 2025:  # Valid publication year range
                            citation_metadata["years_detected"].append(year)
                    except:
                        pass
        
        # Check if any citations are recent (within last 5 years)
        if citation_metadata["years_detected"]:
            current_year = 2025  # Hard-coded current year
            citation_metadata["has_recent_references"] = any(year >= (current_year - 5) for year in citation_metadata["years_detected"])
        
        # Basic validation passed if we have either PMID, DOI, or journal references
        citations_valid = (citation_metadata["has_pmid"] or 
                           citation_metadata["has_doi"] or 
                           citation_metadata["has_journal_references"])
        
        return citations_valid, citation_metadata
    
    def evaluate_response_quality(self, response: Dict[str, Any], query: str) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate overall response quality using LLM assessment.
        
        Parameters
        ----------
        response : Dict[str, Any]
            The response to evaluate
        query : str
            The original query
            
        Returns
        -------
        Tuple[float, Dict[str, Any]]
            (quality_score, quality_metadata)
        """
        try:
            answer = response.get("answer", "")
            citations = response.get("citations", [])
            
            # Prepare evaluation prompt
            evaluation_prompt = f"""
            Evaluate the quality of this medical research answer based on the following criteria.
            For each criterion, provide a score from 0-100.
            
            QUERY: {query}
            
            ANSWER:
            {answer}
            
            EVALUATION CRITERIA:
            1. RELEVANCE (0-100): How directly does the answer address the specific query?
            2. COMPREHENSIVENESS (0-100): How thorough is the answer in covering relevant aspects?
            3. ACCURACY (0-100): Are statements consistent with established medical knowledge?
            4. EVIDENCE-BASED (0-100): How well is the answer supported by research evidence?
            5. CLARITY (0-100): How clear and understandable is the explanation?
            6. OBJECTIVITY (0-100): Does the answer avoid bias and present balanced information?
            
            After scoring each criterion, calculate a WEIGHTED OVERALL SCORE using this formula:
            OVERALL = (RELEVANCE×0.25 + COMPREHENSIVENESS×0.15 + ACCURACY×0.25 + EVIDENCE-BASED×0.20 + CLARITY×0.10 + OBJECTIVITY×0.05)
            
            Return only the numerical OVERALL score as a value between 0 and 1 (e.g., 0.78).
            """
            
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert medical research evaluator who assesses the quality of answers to medical questions."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            
            try:
                quality_score = float(score_text)
                # Ensure score is between 0 and 1
                quality_score = max(0.0, min(1.0, quality_score))
            except ValueError:
                logger.error(f"Failed to parse quality score: {score_text}")
                quality_score = 0.5  # Default score on parsing error
            
            quality_metadata = {
                "quality_score": quality_score,
                "meets_threshold": quality_score >= self.quality_threshold
            }
            
            return quality_score, quality_metadata
            
        except Exception as e:
            logger.error(f"Error evaluating response quality: {str(e)}", exc_info=True)
            return 0.5, {"quality_score": 0.5, "meets_threshold": False, "error": str(e)}
    
    def check_ethical_compliance(self, response: Dict[str, Any], query_metadata: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if response complies with ethical guidelines.
        
        Parameters
        ----------
        response : Dict[str, Any]
            The response to check
        query_metadata : Dict[str, Any]
            Metadata from query validation
            
        Returns
        -------
        Tuple[bool, Dict[str, Any]]
            (is_compliant, compliance_metadata)
        """
        answer = response.get("answer", "")
        answer_lower = answer.lower()
        
        compliance_metadata = {
            "has_medical_disclaimer": False,
            "has_consultation_advice": False,
            "avoids_direct_medical_advice": True,
            "handles_sensitive_topics_appropriately": True,
            "issues": []
        }
        
        # Check for medical disclaimer
        disclaimer_phrases = [
            "consult with a healthcare professional",
            "consult your doctor",
            "seek medical advice",
            "not a substitute for professional medical advice",
            "speak with your physician",
            "talk to your healthcare provider"
        ]
        
        compliance_metadata["has_medical_disclaimer"] = any(phrase in answer_lower for phrase in disclaimer_phrases)
        
        # Check for direct medical advice (problematic)
        direct_advice_phrases = [
            "you should take",
            "you need to take",
            "you must take",
            "stop taking",
            "increase your dosage",
            "decrease your dosage",
            "you should immediately",
            "I recommend you",
            "you should discontinue"
        ]
        
        for phrase in direct_advice_phrases:
            if phrase in answer_lower:
                compliance_metadata["avoids_direct_medical_advice"] = False
                compliance_metadata["issues"].append(f"Contains direct medical advice: '{phrase}'")
                break
        
        # Check handling of sensitive topics if relevant
        if query_metadata.get("sensitive_topics", []):
            sensitive_phrases = {
                "euthanasia": ["how to perform", "methods for", "protocols for"],
                "assisted suicide": ["how to assist", "methods for", "instructions"],
                "abortion": ["how to perform at home", "DIY", "self-induced"],
            }
            
            for topic in query_metadata["sensitive_topics"]:
                if topic in sensitive_phrases:
                    for phrase in sensitive_phrases[topic]:
                        if phrase in answer_lower:
                            compliance_metadata["handles_sensitive_topics_appropriately"] = False
                            compliance_metadata["issues"].append(f"Inappropriate handling of {topic}: '{phrase}'")
        
        # Determine overall compliance
        is_compliant = (
            (compliance_metadata["has_medical_disclaimer"] or not query_metadata.get("requires_disclaimer", False)) and
            compliance_metadata["avoids_direct_medical_advice"] and
            compliance_metadata["handles_sensitive_topics_appropriately"]
        )
        
        return is_compliant, compliance_metadata
    
    def apply_guardrails(self, query: str, response: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Apply all guardrails to a query-response pair and modify if needed.
        
        Parameters
        ----------
        query : str
            The original user query
        response : Dict[str, Any]
            The generated response
            
        Returns
        -------
        Tuple[Dict[str, Any], Dict[str, Any]]
            (modified_response, guardrail_metadata)
        """
        guardrail_metadata = {
            "input_validation": {},
            "scientific_integrity": {},
            "citation_verification": {},
            "response_quality": {},
            "ethical_compliance": {},
            "modifications_applied": [],
            "passed_all_checks": True
        }
        
        # Step 1: Input validation
        is_valid, validation_message, query_metadata = self.validate_input_query(query)
        guardrail_metadata["input_validation"] = query_metadata
        
        if not is_valid:
            # Return rejection response
            modified_response = {
                "answer": f"I'm unable to provide information on this topic. {validation_message}",
                "citations": [],
                "confidence": 0.0,
                "guardrail_blocked": True
            }
            guardrail_metadata["passed_all_checks"] = False
            guardrail_metadata["modifications_applied"].append("blocked_query")
            return modified_response, guardrail_metadata
        
        # Step 2: Scientific integrity evaluation
        integrity_score, integrity_metadata = self.evaluate_scientific_integrity(response, query_metadata)
        guardrail_metadata["scientific_integrity"] = integrity_metadata
        
        # Step 3: Citation verification
        citations_valid, citation_metadata = self.verify_citations(response.get("citations", []))
        guardrail_metadata["citation_verification"] = citation_metadata
        
        # Step 4: Response quality assessment
        quality_score, quality_metadata = self.evaluate_response_quality(response, query)
        guardrail_metadata["response_quality"] = quality_metadata
        
        # Step 5: Ethical compliance check
        is_compliant, compliance_metadata = self.check_ethical_compliance(response, query_metadata)
        guardrail_metadata["ethical_compliance"] = compliance_metadata
        
        # Apply modifications based on checks
        modified_response = response.copy()
        
        # Handle scientific integrity issues
        if integrity_score < 0.7:
            guardrail_metadata["passed_all_checks"] = False
            guardrail_metadata["modifications_applied"].append("added_scientific_disclaimer")
            
            disclaimer = ("\n\nNOTE: This response provides preliminary information based on available research. "
                         "The scientific evidence in this area may be limited or evolving.")
            
            if "answer" in modified_response:
                modified_response["answer"] += disclaimer
        
        # Handle citation issues
        if not citations_valid and query_metadata.get("query_type") == "medical":
            guardrail_metadata["passed_all_checks"] = False
            guardrail_metadata["modifications_applied"].append("added_citation_disclaimer")
            
            disclaimer = ("\n\nNOTE: The citations provided may not represent the complete body of research "
                         "on this topic. Please consult recent medical literature for additional information.")
            
            if "answer" in modified_response:
                modified_response["answer"] += disclaimer
        
        # Handle quality issues
        if not quality_metadata.get("meets_threshold", True):
            guardrail_metadata["passed_all_checks"] = False
            guardrail_metadata["modifications_applied"].append("reduced_confidence")
            
            # Reduce confidence score
            if "confidence" in modified_response:
                modified_response["confidence"] = min(modified_response["confidence"], 0.5)
        
        # Handle ethical compliance issues
        if not is_compliant:
            guardrail_metadata["passed_all_checks"] = False
            guardrail_metadata["modifications_applied"].append("added_medical_disclaimer")
            
            disclaimer = ("\n\nIMPORTANT: This information is provided for educational purposes only and is not a substitute "
                         "for professional medical advice, diagnosis, or treatment. Always seek the advice of your "
                         "physician or other qualified health provider with any questions you may have regarding a "
                         "medical condition.")
            
            if "answer" in modified_response:
                modified_response["answer"] += disclaimer
        
        # Add general disclaimer for sensitive topics
        if query_metadata.get("requires_disclaimer", False):
            guardrail_metadata["modifications_applied"].append("added_sensitive_topic_disclaimer")
            
            disclaimer = ("\n\nNOTE: This topic involves complex medical, ethical, or legal considerations. "
                         "Information provided here aims to be objective and educational.")
            
            if "answer" in modified_response:
                modified_response["answer"] += disclaimer
        
        return modified_response, guardrail_metadata
    
    def process_response(self, query: str, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a response through the guardrail system.
        
        Parameters
        ----------
        query : str
            Original user query
        response : Dict[str, Any]
            Response from agent
            
        Returns
        -------
        Dict[str, Any]
            Modified response with applied guardrails
        """
        try:
            modified_response, guardrail_metadata = self.apply_guardrails(query, response)
            
            # Add guardrail metadata to response for debugging/logging
            modified_response["_guardrail_metadata"] = guardrail_metadata
            
            logger.info(f"Processed response through guardrails: {len(guardrail_metadata['modifications_applied'])} modifications applied")
            
            return modified_response
            
        except Exception as e:
            logger.error(f"Error in guardrail processing: {str(e)}", exc_info=True)
            
            # Return original response with error note
            response["_guardrail_error"] = str(e)
            return response