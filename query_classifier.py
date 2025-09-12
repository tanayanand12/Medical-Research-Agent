"""
query_classifier.py
~~~~~~~~~~~~~~~~~~

Medical query classification module for filtering out non-medical research queries
before passing them to the agent pipeline.
"""

import logging
import os
from typing import Dict, Tuple
import openai
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/query_classifier.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class QueryClassifier:
    """
    Classifier for determining if a query is related to medical research.
    
    Uses OpenAI's chat completion API to filter out non-medical research
    queries before they reach the agent pipeline.
    """
    
    def __init__(self):
        """Initialize the query classifier with LLM configuration."""
        # Load LLM system prompt for classification
        self.classification_prompt = """
        You are a query classifier specializing in distinguishing medical research questions from other domains.
        
        TASK:
        Determine if the input query is related to medical research including:
        - Clinical studies and trials
        - Medical treatments and interventions
        - Disease mechanisms and pathology
        - Drug development and pharmacology
        - Medical devices and diagnostics
        - Public health research
        - Medical literature and publication analysis
        - Health systems and policy research
        - Medical education and training research
        - Patient-centered outcomes research
        
        IMPORTANT: Medical device queries (like TR Band, VasoStat, hemostasis techniques) 
        should be classified as medical research queries, even if they involve markets, 
        business aspects, or usage statistics.
        
        Respond ONLY with a JSON-formatted classification:
        {
            "is_medical_research": true/false,
            "confidence": 0.0-1.0,
            "domain": "medical_research" OR [specific non-medical domain],
            "reason": "Brief explanation of classification"
        }
        """
    
    def is_medical_research_query(self, query: str) -> Tuple[bool, Dict]:
        """
        Determine if a query is related to medical research using OpenAI's chat completion.
        
        Parameters
        ----------
        query : str
            The user query to classify
            
        Returns
        -------
        Tuple[bool, Dict]
            Boolean indicating if query is related to medical research,
            and a dictionary with classification details
        """
        try:
            classification_request = f"""
            USER QUERY: {query}
            
            Carefully analyze whether this query is related to medical research.
            Consider questions about medical devices, procedures, or healthcare systems
            as medical research queries.
            
            Respond ONLY with a JSON-formatted classification.
            """
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",  # Using a smaller, faster model for classification
                messages=[
                    {"role": "system", "content": self.classification_prompt},
                    {"role": "user", "content": classification_request}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parse classification result
            classification = response.choices[0].message.content
            
            # Convert string to dict if needed
            if isinstance(classification, str):
                import json
                try:
                    classification = json.loads(classification)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse LLM classification response: {classification}")
                    # Default to medical research to avoid false negatives
                    return True, {
                        "is_medical_research": True,
                        "confidence": 0.5,
                        "domain": "medical_research",
                        "reason": "Classification parsing error, defaulting to medical research"
                    }
            
            is_medical = classification.get("is_medical_research", True)
            logger.info(f"Query classified by LLM: {is_medical}, query: {query}")
            
            return is_medical, classification
            
        except Exception as e:
            logger.error(f"Error in LLM classification: {str(e)}", exc_info=True)
            # Default to medical research in case of errors to avoid false negatives
            return True, {
                "is_medical_research": True,
                "confidence": 0.99,
                "domain": "medical_research",
                "reason": f"Error in classification process: {str(e)}"
            }
    
    def get_non_medical_response(self, query: str, classification: Dict) -> Dict:
        """
        Generate a generic response for non-medical research queries.
        
        Parameters
        ----------
        query : str
            The original query
        classification : Dict
            Classification details from is_medical_research_query
        
        Returns
        -------
        Dict
            Response containing generic message about domain limitations
        """
        domain = classification.get("domain", "non-medical")
        confidence = classification.get("confidence", 0.0)
        reason = classification.get("reason", "This query is outside my medical research focus")
        
        response = {
            "answer": (
                "I'm a medical research assistant specialized in answering questions about medical academic "
                "research, clinical studies, and scientific literature. "
                f"Your question appears to be about {domain}, which is outside my area of expertise. "
                "I can help with questions about medical research papers, clinical trials, treatment efficacy, "
                "disease mechanisms, drug development, and other topics in the medical research domain."
            ),
            "citations": [],
            "confidence": confidence,
            "classification": classification
        }
        
        logger.info(f"Generated non-medical response for query: {query}")
        return response