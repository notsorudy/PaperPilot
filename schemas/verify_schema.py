from pydantic import BaseModel, Field
from typing import List

class ExtractedClaims(BaseModel):
    """List of atomic claims extracted from a synthesis document."""
    claims: List[str] = Field(
        ...,
        description="5 to 10 atomic, factual claims extracted from the literature review text."
    )

class VerificationResult(BaseModel):
    """Structured output for the self-check verification stage."""
    claims: List[str] = Field(
        ...,
        description="List of 5 to 10 atomic, factual claims extracted from the literature review."
    )
    supported_claims: List[str] = Field(
        default_factory=list,
        description="List of claims directly supported by the source excerpts."
    )
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="List of claims that could not be adequately verified or lack sufficient grounding in the source excerpts."
    )
    faithfulness_score: float = Field(
        ...,
        description="Ratio of supported claims to total claims (between 0.0 and 1.0)."
    )
