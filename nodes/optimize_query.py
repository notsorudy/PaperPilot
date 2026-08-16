from prompts.optimize_query_prompt import OPTIMIZE_QUERY_ARXIV_PROMPT
from schemas.optimize_query_schema import OptimizeQuerySchema
from app.llm import llm_cerebras, llm_groq
from state.research_state import ResearchState

async def optimize_query(state: ResearchState):
    """Optimize the search queries using the LLM."""
    plan = state.get('plan')
    optimized_queries = []
    
    if plan and plan.search_queries:
        for query in plan.search_queries:
            try:
                llm_optimizer = llm_cerebras.with_structured_output(OptimizeQuerySchema)
                chain = OPTIMIZE_QUERY_ARXIV_PROMPT | llm_optimizer
                result = await chain.ainvoke({"query": query})
            except Exception as e:
                print(f"Cerebras LLM failed in optimize_query for '{query}': {e}")
                print("Falling back to Groq...")
                llm_optimizer_fallback = llm_groq.with_structured_output(OptimizeQuerySchema)
                chain_fallback = OPTIMIZE_QUERY_ARXIV_PROMPT | llm_optimizer_fallback
                result = await chain_fallback.ainvoke({"query": query})
                
            optimized_queries.append(result)
            
    return {"optimized_queries": optimized_queries}
