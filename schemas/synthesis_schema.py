from pydantic import BaseModel, Field
from typing import List

class SynthesisOutput(BaseModel):
    """Structured output for the literature review synthesis step."""
    literature_review: str = Field(
        ...,
        description="The comprehensive literature review answering the user's query with relevant academic details."
    )
    unanswered_aspects: List[str] = Field(
        default_factory=list,
        description="Aspects of the user's query that could NOT be adequately answered from the retrieved chunks. Empty list if fully covered."
    )
    confidence_notes: str = Field(
        default="",
        description="Any caveats or remarks about the confidence or completeness of the review based on retrieved evidence."
    )
