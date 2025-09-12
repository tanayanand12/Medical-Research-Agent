from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum

class ClientPersonaTrackerData(BaseModel):
    base_url: str = Field(default="https://client-persona-tracker-508047128875.europe-west1.run.app")
    update_endpoint: str = Field(default="/update")
    generate_summary_endpoint: str = Field(default="/generate-summary")

class UpdatePersonaRequest(BaseModel):
    """Request model for updating persona data"""
    
    uid: str = Field(..., min_length=1, max_length=100, description="User ID (mandatory)")
    question: Optional[str] = Field(None, description="User query/question")
    answer: Optional[str] = Field(None, description="Response/answer")
    
    # Persona fields
    tone_preference: Optional[str] = Field(None, max_length=50, description="User's tone preference")
    topics_of_interest: Optional[List[str]] = Field(None, description="List of topics user is interested in")
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "uid": "user-1234",
                "question": "How can I invest in stocks?",
                "answer": "One way to start is through mutual funds or ETFs.",
                "tone_preference": "formal",
                "topics_of_interest": ["finance", "education"],
                "metadata": {"source": "chat", "confidence": 0.95}
            }
        }

class SummaryType(str, Enum):
    PERSONA_ONLY = "persona_only"
    HISTORY_ONLY = "history_only"
    FULL_SUMMARY = "full_summary"
    INSIGHTS = "insights"

class SummaryPersonaRequest(BaseModel):
    """Request model for generating persona summary"""
    
    uid: str = Field(..., min_length=1, max_length=15000, description="User ID")
    summary_type: SummaryType = Field(
        default=SummaryType.FULL_SUMMARY, 
        description="Type of summary to generate"
    )
    custom_prompt: Optional[str] = Field(
        None, 
        max_length=5000, 
        description="Custom prompt to include in the summary generation"
    )
    max_tokens: Optional[int] = Field(
        None, 
        ge=50, 
        le=15000, 
        description="Maximum tokens for the generated summary"
    )
    
    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "uid": "user-1234",
                "summary_type": "full_summary",
                "custom_prompt": "Focus on the user's learning preferences and communication style",
                "max_tokens": 3000
            }
        }
