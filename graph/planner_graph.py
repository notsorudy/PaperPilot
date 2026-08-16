from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from nodes.planner import generate_search_queries
from nodes.optimize_query import optimize_query
from nodes.search_arxiv_node import search_papers
from nodes.broaden_query_node import broaden_query
from nodes.rank_papers_node import rank_papers
from nodes.extract_text_node import extract_text
from nodes.retrieve_node import retrieve_chunks
from nodes.generate_answer_node import generate_answer
from nodes.verify_answer_node import verify_answer
from nodes.generate_citations_node import generate_citations
from state.research_state import ResearchState

def should_retry_search(state: ResearchState) -> str:
    """Conditional router after paper search: broadens query if yield < 3 papers and retry < 1."""
    papers = state.get('papers', [])
    retry_count = state.get('retry_count', 0)
    if len(papers) < 3 and retry_count < 1:
        return 'broaden_query'
    return 'rank_papers'

def should_retry_extraction(state: ResearchState) -> str:
    """Conditional router after PDF extraction: broadens query if yield < 5 chunks and retry < 1."""
    chunks = state.get('chunks', [])
    retry_count = state.get('retry_count', 0)
    if len(chunks) < 5 and retry_count < 1:
        return 'broaden_query'
    return 'retrieve_chunks'

def PlannerAgent(enable_hitl: bool = True, checkpointer = None):
    """Compiles the PaperPilot research agent graph with optional Human-in-the-Loop interrupts."""
    graph = StateGraph(ResearchState)
    
    # 1. Register all nodes
    graph.add_node('generate_search_queries', generate_search_queries)
    graph.add_node('optimize_query', optimize_query)
    graph.add_node('search_papers', search_papers)
    graph.add_node('broaden_query', broaden_query)
    graph.add_node('rank_papers', rank_papers)
    graph.add_node('extract_text', extract_text)
    graph.add_node('retrieve_chunks', retrieve_chunks)
    graph.add_node('generate_answer', generate_answer)
    graph.add_node('verify_answer', verify_answer)
    graph.add_node('generate_citations', generate_citations)
    
    # 2. Sequential & Conditional Edges
    graph.add_edge(START, 'generate_search_queries')
    graph.add_edge('generate_search_queries', 'optimize_query')
    graph.add_edge('optimize_query', 'search_papers')
    
    # Retry loop 1: search_papers -> (too few?) -> broaden_query -> search_papers
    graph.add_conditional_edges('search_papers', should_retry_search, {
        'broaden_query': 'broaden_query',
        'rank_papers': 'rank_papers'
    })
    graph.add_edge('broaden_query', 'search_papers')
    
    graph.add_edge('rank_papers', 'extract_text')
    
    # Retry loop 2: extract_text -> (too few chunks?) -> broaden_query -> search_papers
    graph.add_conditional_edges('extract_text', should_retry_extraction, {
        'broaden_query': 'broaden_query',
        'retrieve_chunks': 'retrieve_chunks'
    })
    
    graph.add_edge('retrieve_chunks', 'generate_answer')
    graph.add_edge('generate_answer', 'verify_answer')
    graph.add_edge('verify_answer', 'generate_citations')
    graph.add_edge('generate_citations', END)

    # 3. Compilation with checkpointer and HITL interrupt points
    if checkpointer is None:
        checkpointer = MemorySaver()
        
    interrupts = ['search_papers', 'extract_text'] if enable_hitl else None
    
    compiled_agent = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupts
    )
    
    return compiled_agent