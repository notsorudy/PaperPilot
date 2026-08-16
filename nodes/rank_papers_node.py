from state.research_state import ResearchState

from app.models import reranker

async def rank_papers(state: ResearchState):
    """Rank the retrieved papers using a Hugging Face Cross-Encoder and keep the top 10."""
    papers = state.get('papers', [])
    query = state.get('query', '')
    
    if not papers:
        print("No papers to rank.")
        return {"papers": []}
        
    if not reranker:
        print("Warning: sentence-transformers not installed. Skipping ranking.")
        return {"papers": papers[:10]}
        
    print(f"Ranking {len(papers)} papers using Hugging Face CrossEncoder...")
        
    try:
        # Create pairs of (query, document)
        pairs = []
        for p in papers:
            # We combine title and abstract for the document representation
            doc_text = f"{p.title}. {p.abstract}"
            pairs.append((query, doc_text))
            
        # Predict scores for all pairs simultaneously
        scores = reranker.predict(pairs)
        
        # Zip original papers with their scores
        scored_papers = list(zip(papers, scores))
        
        # Sort descending by score
        scored_papers.sort(key=lambda x: x[1], reverse=True)
        
        # Debug print
        for p, score in scored_papers[:3]:
            print(f"Top Match [Score {score:.2f}]: {p.title}")
            
        # Keep top 10
        top_papers = [p[0] for p in scored_papers[:10]]
        return {"papers": top_papers}
        
    except Exception as e:
        print(f"Error ranking papers: {e}")
        # Fallback to returning the first 10
        return {"papers": papers[:10]}
