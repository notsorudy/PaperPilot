from app.llm import llm_cerebras, llm_groq
from prompts.broaden_query_prompt import BROADEN_QUERY_PROMPT
from schemas.optimize_query_schema import BroadenedQueriesPlan
from state.research_state import ResearchState

async def broaden_query(state: ResearchState):
    """Broadens and reformulates queries when search results are insufficient."""
    query = state.get('query', '')
    previous_optimized = state.get('optimized_queries', [])
    retry_count = state.get('retry_count', 0)
    
    prev_query_strings = "\n".join([f"- {q.optimized_query}" for q in previous_optimized])
    print(f"\n⚠️ Search returned insufficient papers. Triggering query broadening (Retry #{retry_count + 1})...")
    
    try:
        llm_structured = llm_cerebras.with_structured_output(BroadenedQueriesPlan)
        chain = BROADEN_QUERY_PROMPT | llm_structured
        result = await chain.ainvoke({
            "query": query,
            "previous_queries": prev_query_strings or "None"
        })
    except Exception as e:
        print(f"Cerebras LLM failed in broaden_query: {e}. Falling back to Groq...")
        llm_structured_fallback = llm_groq.with_structured_output(BroadenedQueriesPlan)
        chain_fallback = BROADEN_QUERY_PROMPT | llm_structured_fallback
        result = await chain_fallback.ainvoke({
            "query": query,
            "previous_queries": prev_query_strings or "None"
        })
        
    new_queries = result.queries if result and result.queries else previous_optimized
    print(f"Generated {len(new_queries)} broadened search queries:")
    for q in new_queries:
        print(f"  → {q.optimized_query}")
        
    return {
        "optimized_queries": new_queries,
        "retry_count": retry_count + 1
    }
