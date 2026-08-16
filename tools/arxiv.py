# tools/arxiv.py

import json
import os
import arxiv
from schemas.paper_schema import PaperMetadata

DATA_PATH = os.path.join("data", "arxiv-metadata-oai-snapshot.json")

def search_arxiv(query: str, max_results: int = 5) -> list[PaperMetadata]:
    """
    Search arXiv web API and return a list of PaperMetadata objects.
    """
    client = arxiv.Client(
        num_retries=2,
        delay_seconds=3.0
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers: dict[str, PaperMetadata] = {}
    
    try:
        for paper in client.results(search):
            metadata = PaperMetadata(
                paper_id=paper.entry_id,
                title=paper.title.strip(),
                authors=[author.name for author in paper.authors],
                abstract=paper.summary.strip(),
                pdf_url=paper.pdf_url,
                published=paper.published.isoformat(),
                source="arXiv API",
            )
            papers[metadata.paper_id] = metadata
    except Exception as e:
        print(f"Error fetching from ArXiv API: {e}")

    return list(papers.values())

def search_arxiv_local(query: str, max_results: int = 5) -> list[PaperMetadata]:
    """
    Search the local arXiv JSON dataset and return a list of PaperMetadata objects.
    """
    papers = []
    query_lower = query.lower()
    keywords = query_lower.split()
    
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}")
        return papers
        
    try:
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line_lower = line.lower()
                if all(kw in line_lower for kw in keywords):
                    try:
                        paper = json.loads(line)
                        title = paper.get('title', '').lower()
                        abstract = paper.get('abstract', '').lower()
                        
                        if all(kw in title or kw in abstract for kw in keywords):
                            authors_parsed = paper.get('authors_parsed', [])
                            authors = []
                            for author in authors_parsed:
                                name = " ".join([part for part in reversed(author) if part]).strip()
                                if not name:
                                    name = paper.get('authors', '')
                                authors.append(name)
                                
                            metadata = PaperMetadata(
                                paper_id=paper.get('id', ''),
                                title=paper.get('title', '').strip(),
                                authors=authors if authors else [paper.get('authors', '')],
                                abstract=paper.get('abstract', '').strip(),
                                pdf_url=f"https://arxiv.org/pdf/{paper.get('id')}",
                                published=paper.get('update_date', ''),
                                source="arXiv Local"
                            )
                            papers.append(metadata)
                            
                            if len(papers) >= max_results:
                                break
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Error reading local arxiv dataset: {e}")
        
    return papers