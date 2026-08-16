from app.llm import llm_cerebras, llm_groq
from prompts.planner_prompts import SEARCH_PLAN_PROMPT
from schemas.planner_schema import SearchPlan
from state.research_state import ResearchState
import traceback

async def generate_search_queries(state: ResearchState):
    """return a list of relevant search queries that can be used to search on internet along with other planning information."""
    query=state['query']
    chat_prompt=SEARCH_PLAN_PROMPT
    
    try:
        # Primary: Try Cerebras first
        llm_search_queries = llm_cerebras.with_structured_output(SearchPlan)
        chain = chat_prompt | llm_search_queries
        result = await chain.ainvoke({"query": query})
    except Exception as e:
        print(f"Cerebras LLM failed in planner: {e}")
        print("Falling back to Groq...")
        # Fallback: Use Groq
        llm_search_queries_fallback = llm_groq.with_structured_output(SearchPlan)
        chain_fallback = chat_prompt | llm_search_queries_fallback
        result = await chain_fallback.ainvoke({"query": query})
        
    return {
        "plan": result
    }
