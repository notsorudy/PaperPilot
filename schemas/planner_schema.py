from pydantic import BaseModel , Field 
from typing import Literal, List

class SearchPlan(BaseModel):
    """return a list of relevant search queries that can be used to search on internet."""
    search_queries : List[str] = Field(..., description="list of relevant search queries that can be used to search on internet.")
    intent: str = Field(..., description="A short label describing the user's objective.")
    reasoning: str = Field(..., description="Explanation of why these search queries were chosen.")
    expected_sources: List[str] = Field(..., description="List of sources that are most likely to contain useful information.")
