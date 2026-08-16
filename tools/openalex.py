import requests
from schemas.paper_schema import PaperMetadata

EMAIL = "anshuyadav1223334444@gmail.com"

def reconstruct_abstract(inverted_index: dict) -> str:
    """Reconstructs the abstract from OpenAlex's abstract_inverted_index format."""
    if not inverted_index:
        return ""
    
    max_pos = -1
    for positions in inverted_index.values():
        for pos in positions:
            if pos > max_pos:
                max_pos = pos
                
    if max_pos == -1:
        return ""
        
    words = [""] * (max_pos + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
            
    return " ".join(words)

def search_openalex(query: str, max_results: int = 5) -> list[PaperMetadata]:
    """Search OpenAlex and return a list of PaperMetadata objects."""
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": max_results,
        "mailto": EMAIL
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error fetching from OpenAlex: {e}")
        return []
        
    papers = []
    for item in data.get("results", []):
        authors = []
        for authorship in item.get("authorships", []):
            author = authorship.get("author", {})
            if author.get("display_name"):
                authors.append(author["display_name"])
                
        abstract = reconstruct_abstract(item.get("abstract_inverted_index"))
        
        pdf_url = ""
        open_access = item.get("open_access", {})
        if open_access and open_access.get("is_oa") and open_access.get("oa_url"):
            pdf_url = open_access["oa_url"]
            
        # Fallback if no Open Access URL is available
        if not pdf_url:
            primary_location = item.get("primary_location", {})
            if primary_location:
                # We only want ACTUAL pdf URLs, not landing pages.
                loc_pdf = primary_location.get("pdf_url", "")
                if loc_pdf and loc_pdf.endswith(".pdf"):
                    pdf_url = loc_pdf
            
        paper_id = item.get("id", "").replace("https://openalex.org/", "")
        
        metadata = PaperMetadata(
            paper_id=paper_id,
            title=item.get("title") or "No Title",
            authors=authors,
            abstract=abstract,
            pdf_url=pdf_url,
            published=item.get("publication_date") or "",
            source="OpenAlex"
        )
        papers.append(metadata)
        
    return papers
