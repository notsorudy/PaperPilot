from typing import List
from pydantic import BaseModel, Field

class OptimizeQuerySchema(BaseModel):
    optimized_query: str = Field(
        ...,
        description="A concise keyword-based query optimized for arXiv search."
    )
    keywords: List[str] = Field(
        ...,
        description="The key technical concepts extracted from the user's query."
    )

class BroadenedQueriesPlan(BaseModel):
    queries: List[OptimizeQuerySchema] = Field(
        ...,
        description="List of broadened search queries with their associated technical keywords."
    )