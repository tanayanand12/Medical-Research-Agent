# endpoint_prediction_integration.py
import requests
import logging
import openai
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import os

logger = logging.getLogger(__name__)

class EndpointPredictionAPIIntegration:
    """Integration module for endpoint prediction API."""
    
    def __init__(self, 
                 api_base_url: str = "https://clinical-trials-endpoint-prediction-508047128875.europe-west1.run.app",
                 docs_model_id: str = "ct_epa_1", 
                 csv_model_id: str = "ct_endpoints1"):
        """
        Initialize the integration.
        
        Args:
            api_base_url: Base URL of the endpoint prediction API
            docs_model_id: Model ID for document embeddings
            csv_model_id: Model ID for CSV embeddings
        """
        load_dotenv()
        self.api_base_url = api_base_url
        self.docs_model_id = docs_model_id
        self.csv_model_id = csv_model_id
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def requires_endpoint_prediction(self, query: str) -> bool:
        """
        Classify if query requires endpoint prediction using LLM.
        
        Args:
            query: User query
            
        Returns:
            True if endpoint prediction needed
        """
        try:
            prompt = f"""Classify if this clinical trial query requires endpoint timing prediction.

Query: "{query}"

Answer "YES" if the query asks about:
- When to measure endpoints/outcomes
- Timing of assessments
- Follow-up schedules
- Duration of studies
- When results will be observed
- Optimal timepoints for measurements

Answer "NO" for other types of queries about:
- General trial information
- Eligibility criteria
- Study procedures
- Background information
- Safety data
- Efficacy results (without timing)

Classification:"""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1
            )
            
            classification = response.choices[0].message.content.strip().upper()
            return "YES" in classification
            
        except Exception as e:
            logger.warning(f"Classification failed, defaulting to False: {e}")
            return False
    
    def get_endpoint_prediction(self, query: str) -> Dict[str, Any]:
        """
        Get endpoint prediction from API.
        
        Args:
            query: User query
            
        Returns:
            Prediction results or empty dict
        """
        try:
            url = f"{self.api_base_url}/predict"
            payload = {
                "query": query,
                "docs_model_id": self.docs_model_id,
                "csv_model_id": self.csv_model_id,
                "top_k_docs": 6,
                "top_k_csv": 10,
                "return_trials": 5
            }
            
            response = requests.post(url, json=payload, timeout=2000)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Endpoint prediction successful")
                return result
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"Endpoint prediction failed: {e}")
            return {}
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Main processing method: classify and predict if needed.
        
        Args:
            query: User query
            
        Returns:
            Endpoint prediction results or empty dict
        """
        # Check if endpoint prediction is needed
        if self.requires_endpoint_prediction(query):
            logger.info(f"Endpoint prediction required for query")
            return self.get_endpoint_prediction(query)
        else:
            logger.info(f"No endpoint prediction needed")
            return {}
    
    def format_prediction_for_pipeline(self, prediction: Dict[str, Any]) -> str:
        """
        Format prediction results for concatenation with pipeline results.
        
        Args:
            prediction: Prediction results from API
            
        Returns:
            Formatted string for concatenation
        """
        if not prediction:
            return ""
        
        lines = ["\n\n=== ENDPOINT TIMING PREDICTION ==="]
        
        primary_days = prediction.get('predicted_primary_time_days')
        secondary_days = prediction.get('predicted_secondary_time_days')
        confidence = prediction.get('confidence_score', 0)
        
        if primary_days:
            lines.append(f"Predicted Primary Endpoint: {primary_days} days")
        if secondary_days:
            lines.append(f"Predicted Secondary Endpoint: {secondary_days} days")
        
        if confidence:
            lines.append(f"Confidence: {confidence:.2f}")
        
        rationale = prediction.get('rationale', '')
        if rationale:
            lines.append(f"\nRationale: {rationale[:200]}...")
        
        supporting_trials = prediction.get('supporting_trials', [])
        if supporting_trials:
            trial_str = ", ".join(supporting_trials[:3])
            lines.append(f"Supporting Trials: {trial_str}")
        
        return "\n".join(lines)
