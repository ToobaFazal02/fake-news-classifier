"""
PYDANTIC SCHEMAS FOR FAKE NEWS API
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional

class ArticleRequest(BaseModel):
    """Request model for article classification"""
    text: str = Field(
        ...,
        min_length=50,
        description="News article text (minimum 50 characters)",
        example="Scientists at MIT have announced a breakthrough in renewable energy technology that could revolutionize solar panel efficiency."
    )
    
    @validator('text')
    def validate_text(cls, v):
        if len(v.strip()) < 50:
            raise ValueError('Article text must be at least 50 characters')
        return v.strip()

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: str = Field(..., description="Classification: REAL or FAKE")
    fake_probability: float = Field(..., description="Probability of being fake (0-1)")
    real_probability: float = Field(..., description="Probability of being real (0-1)")
    confidence: float = Field(..., description="Confidence percentage (0-100)")
    label: int = Field(..., description="Numeric label: 0=REAL, 1=FAKE")
    trigger_words: List[str] = Field(..., description="Key words influencing prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "FAKE",
                "fake_probability": 0.92,
                "real_probability": 0.08,
                "confidence": 92.0,
                "label": 1,
                "trigger_words": ["shocking", "miracle", "click", "secret", "exposed"]
            }
        }

class BatchRequest(BaseModel):
    """Request model for batch predictions"""
    articles: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of article texts (max 100)"
    )

class BatchResponse(BaseModel):
    """Response model for batch predictions"""
    total: int
    results: List[PredictionResponse]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    message: str
    accuracy: Optional[float] = None