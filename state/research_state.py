from typing import TypedDict, List, Any, Optional
from schemas.planner_schema import SearchPlan
from schemas.paper_schema import PaperMetadata
from schemas.optimize_query_schema import OptimizeQuerySchema

class ResearchState(TypedDict, total=False):
    query: str
    plan: SearchPlan
    optimized_queries: List[OptimizeQuerySchema]
    retry_count: int
    # Retrieved papers
    papers: List[PaperMetadata]
    # Raw extracted text chunks
    chunks: List[Any]
    # Final top reranked chunks
    relevant_chunks: List[Any]
    # Final synthesized answer
    final_answer: str
    # Insufficient evidence aspects
    unanswered_aspects: List[str]
    # Faithfulness score from verification
    faithfulness_score: float
    # Generated BibTeX citations
    citations: str