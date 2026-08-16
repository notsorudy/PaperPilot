import string
from state.research_state import ResearchState
from tools.arxiv import search_arxiv

def normalize_title(title: str) -> str:
    """Normalize title for deduplication: lowercase and remove punctuation/spaces."""
    if not title:
        return ""
    title = title.lower()
    return title.translate(str.maketrans('', '', string.punctuation)).replace(" ", "")

import asyncio

async def search_papers(state: ResearchState):
    """Search arxiv using the optimized queries in parallel."""
    optimized_queries = state.get('optimized_queries', [])
    papers = state.get('papers', [])
    
    # We will use a dictionary to deduplicate papers by normalized title
    papers_dict = {normalize_title(p.title): p for p in papers} if papers else {}
    
    async def fetch_arxiv(query):
        return await asyncio.to_thread(search_arxiv, query, 5)
        
    tasks = [fetch_arxiv(query_schema.optimized_query) for query_schema in optimized_queries]
    results = await asyncio.gather(*tasks)
    
    for arxiv_results in results:
        for paper in arxiv_results:
            norm_title = normalize_title(paper.title)
            if norm_title not in papers_dict:
                papers_dict[norm_title] = paper
            
    return {"papers": list(papers_dict.values())}
