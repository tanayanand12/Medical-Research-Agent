# clinical_trials_rag_module.py
import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv # type: ignore
import time

try:
    from openai import OpenAI # type: ignore
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

class ClinicalTrialsRAGModule:
    """Module for generating answers using clinical trial context via RAG."""
    
    def __init__(self, model_name: str = "gpt-4-turbo"):
        """
        Initialize the RAG module.
        
        Args:
            model_name: Name of the OpenAI model to use
        """
        load_dotenv()
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Please install with: pip install openai")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        logger.info(f"Initialized ClinicalTrialsRAGModule with model: {self.model_name}")
    
    def create_system_prompt(self) -> str:
        """
        Create the system prompt for clinical trials analysis.
        
        Returns:
            System prompt string
        """
        return """You are an expert clinical trials research assistant specializing in analyzing and interpreting clinical trial data from ClinicalTrials.gov. Your role is to provide comprehensive, evidence-based insights using the most current clinical trial information.

CORE CAPABILITIES:
- Analyze clinical trial protocols, designs, and methodologies
- Interpret eligibility criteria, inclusion/exclusion requirements
- Evaluate study endpoints, outcome measures, and statistical plans
- Assess study status, recruitment, and timeline information
- Compare interventions, treatments, and study designs across trials
- Identify patterns in sponsor types, geographic distribution, and study characteristics

RESPONSE GUIDELINES:
1. Always base responses on the provided clinical trial data
2. Clearly distinguish between different types of clinical trial information:
   - Study design and methodology details
   - Eligibility criteria and patient populations
   - Intervention protocols and treatment arms
   - Primary and secondary endpoints
   - Study status and recruitment information
   - Sponsor and location data

3. When analyzing multiple studies:
   - Compare and contrast key study characteristics
   - Highlight similarities and differences in design approaches
   - Note variations in patient populations and inclusion criteria
   - Identify trends in intervention strategies

4. For statistical and methodological information:
   - Reference specific study designs (RCT, observational, etc.)
   - Note sample sizes and statistical power considerations
   - Describe randomization and blinding strategies when available
   - Mention study phases for interventional trials

5. Present information in a structured, accessible format that includes:
   - Clear section headers for different aspects
   - Specific citations to relevant studies using NCT IDs
   - Quantitative details when available (enrollment numbers, timeframes)
   - Practical implications for patients and researchers

IMPORTANT LIMITATIONS:
- Extensively interpret reason and provide information that is stated in the clinical trial data
- Reason with the Documented records and metadata from ClinicalTrials.gov
- Clearly indicate your reasoning process and how you arrived at conclusions
- Focus on factual trial characteristics rather than clinical advice

Remember that clinical trial data represents research protocols and should not be interpreted as clinical recommendations.

HARD LIMITATIONS:
- Do NOT use phrases like "None of the identified clinical trials explicitly mention" or "No specific studies were found or "not directly available" or "did not explicitly provide" or anything that reduces you response CONFIDENCE and RELEVANCE.
- Try to extract and synthesize all relevant information from the provided clinical trial data.
- Answer confidently and comprehensively based on the available clinical trial records.
- State your reasoning clearly and provide detailed information from the clinical trial data.
- Avoid making low confidence statements or vague assertions.
- Your task is to provide a thorough analysis of the clinical trial data, not to provide answers to personal questions other than the like `how is the weather today?` OR `What your name?` OR `who made you?` etc.
"""

    def create_user_prompt(self, query: str, context: str, studies: List[Dict[str, Any]]) -> str:
        """
        Create the user prompt with query and context.
        
        Args:
            query: User query
            context: Extracted context from clinical trials
            studies: List of relevant studies metadata
            
        Returns:
            Formatted user prompt
        """
        # Format study citations
        citations = []
        for i, study in enumerate(studies):
            citation = f"[{i+1}] {study['title']} (NCT ID: {study['study_id']}, Relevance: {study['similarity_score']:.3f})"
            citations.append(citation)
        
        citations_text = "\n".join(citations) if citations else "No specific studies identified."
        
        prompt = f"""
USER QUESTION: {query}

CLINICAL TRIAL DATA CONTEXT:
{context}

RELEVANT STUDIES IDENTIFIED:
{citations_text}

Please provide a comprehensive analysis based on the clinical trial information above. Structure your response as follows:

1. **Executive Summary** (50-75 words)
   - Direct answer to the user's question
   - Key findings from the clinical trial data

2. **Detailed Analysis** 
   - Comprehensive synthesis of relevant trial information
   - Specific details from individual studies with NCT ID references
   - Comparison across multiple trials when applicable

3. **Study Characteristics**
   - Overview of study designs, phases, and methodologies
   - Patient populations and eligibility criteria
   - Intervention details and treatment protocols

4. **Key Findings and Patterns**
   - Notable trends or patterns across the identified trials
   - Variations in approach or methodology
   - Geographic distribution and sponsor types if relevant

5. **Practical Implications**
   - What this means for patients seeking relevant trials
   - Considerations for researchers in this field
   - Current state of research in this area

6. **Data Limitations**
   - What information was not available in the trial records
   - Areas where additional research or data might be needed

Use NCT IDs [NCT########] when referencing specific trials. If multiple studies address the same point, cite all relevant NCT IDs.

Focus on factual information from the trial records and avoid clinical recommendations or medical advice.
"""
        return prompt
    
    def generate_answer(self, query: str, context: str, studies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate an answer using OpenAI with clinical trial context.
        
        Args:
            query: User query
            context: Extracted context from clinical trials
            studies: List of relevant studies
            
        Returns:
            Dictionary containing answer and metadata
        """
        try:
            system_prompt = self.create_system_prompt()
            user_prompt = self.create_user_prompt(query, context, studies)
            
            # Generate response using OpenAI
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,  # Lower temperature for more consistent, factual responses
                max_tokens=2000,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            generation_time = time.time() - start_time
            answer = response.choices[0].message.content
            
            # Format citations for display
            formatted_citations = []
            for i, study in enumerate(studies):
                citation = f"[{i+1}] {study['title']} - NCT ID: {study['study_id']} (Relevance Score: {study['similarity_score']:.3f})"
                formatted_citations.append(citation)
            
            # Calculate usage statistics
            usage = response.usage
            total_tokens = usage.total_tokens if usage else 0
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            
            return {
                "answer": answer,
                "citations": formatted_citations,
                "studies": studies,
                "metadata": {
                    "model_used": self.model_name,
                    "generation_time": generation_time,
                    "total_tokens": total_tokens,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "studies_analyzed": len(studies),
                    "context_length": len(context)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"I encountered an error while analyzing the clinical trial data: {str(e)}. Please try rephrasing your question or check if the clinical trial data is available.",
                "citations": [],
                "studies": studies,
                "metadata": {
                    "error": str(e),
                    "model_used": self.model_name,
                    "studies_analyzed": len(studies)
                }
            }
    
    def validate_response_quality(self, response: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Validate the quality of the generated response.
        
        Args:
            response: Generated response dictionary
            query: Original user query
            
        Returns:
            Response with quality assessment
        """
        quality_metrics = {
            "has_answer": bool(response.get("answer")),
            "has_citations": len(response.get("citations", [])) > 0,
            "answer_length": len(response.get("answer", "")),
            "studies_referenced": len(response.get("studies", [])),
            "contains_nct_ids": "NCT" in response.get("answer", ""),
            "structured_response": any(marker in response.get("answer", "") for marker in ["**", "##", "1.", "2.", "3."])
        }
        
        # Calculate overall quality score
        quality_score = sum([
            quality_metrics["has_answer"] * 0.3,
            quality_metrics["has_citations"] * 0.2,
            (quality_metrics["answer_length"] > 200) * 0.2,
            (quality_metrics["studies_referenced"] > 0) * 0.15,
            quality_metrics["contains_nct_ids"] * 0.1,
            quality_metrics["structured_response"] * 0.05
        ])
        
        response["quality_assessment"] = {
            "metrics": quality_metrics,
            "overall_score": quality_score,
            "quality_level": "high" if quality_score > 0.8 else "medium" if quality_score > 0.5 else "low"
        }
        
        logger.info(f"Response quality assessment: {quality_score:.2f} ({response['quality_assessment']['quality_level']})")
        return response