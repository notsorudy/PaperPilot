import re
from schemas.paper_schema import PaperMetadata
from state.research_state import ResearchState

def format_bibtex(paper: PaperMetadata) -> str:
    """Format a single PaperMetadata object into a standard BibTeX @article entry."""
    # Generate clean citation key: firstauthor + year + short keyword
    if paper.authors:
        first_author = paper.authors[0].strip()
        last_name_match = re.findall(r'[A-Za-z]+', first_author)
        first_author_last = last_name_match[-1].lower() if last_name_match else "author"
    else:
        first_author_last = "author"
        
    year_match = re.search(r'\d{4}', paper.published or "")
    year = year_match.group(0) if year_match else "2024"
    
    # Clean title words for key
    title_words = re.findall(r'[A-Za-z]+', paper.title or "")
    keyword = title_words[0].lower() if title_words else "paper"
    cite_key = f"{first_author_last}{year}{keyword}"
    
    # Format author list with ' and '
    authors_str = " and ".join(paper.authors) if paper.authors else "Unknown Authors"
    
    # Extract clean journal / source note
    source = paper.source or "arXiv"
    clean_id = paper.paper_id.replace("http://arxiv.org/abs/", "").replace("https://arxiv.org/abs/", "")
    
    return f"""@article{{{cite_key},
  title     = {{{paper.title}}},
  author    = {{{authors_str}}},
  year      = {{{year}}},
  journal   = {{{source}}},
  url       = {{{paper.pdf_url}}},
  eprint    = {{{clean_id}}}
}}"""

async def generate_citations(state: ResearchState):
    """Generates a complete BibTeX bibliography from the top ranked research papers."""
    papers = state.get('papers', [])
    if not papers:
        return {"citations": "% No papers available to generate citations."}
        
    print(f"\nFormatting BibTeX citations for {len(papers)} source papers...")
    bibtex_entries = [format_bibtex(p) for p in papers]
    full_bibtex = "\n\n".join(bibtex_entries)
    
    return {"citations": full_bibtex}
