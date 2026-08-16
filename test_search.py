import sys
sys.stdout.reconfigure(encoding='utf-8')
from tools.arxiv import search_arxiv, search_arxiv_local
from tools.openalex import search_openalex

def main():
    query = "attention is all you need"
    print(f"Testing with query: '{query}'")
    
    print("\n--- Testing OpenAlex API ---")
    oa_papers = search_openalex(query, max_results=2)
    if not oa_papers:
        print("No papers found in OpenAlex.")
    for paper in oa_papers:
        print(f"- [OpenAlex] {paper.title} ({paper.published}) by {', '.join(paper.authors)}")
        
    print("\n--- Testing ArXiv API ---")
    ar_papers = search_arxiv(query, max_results=2)
    if not ar_papers:
        print("No papers found in ArXiv API.")
    for paper in ar_papers:
        print(f"- [ArXiv API] {paper.title} ({paper.published}) by {', '.join(paper.authors)}")

    print("\n--- Testing Local ArXiv Dataset ---")
    local_papers = search_arxiv_local(query, max_results=2)
    if not local_papers:
        print("No papers found in Local ArXiv.")
    for paper in local_papers:
        print(f"- [ArXiv Local] {paper.title} ({paper.published}) by {', '.join(paper.authors)}")

if __name__ == "__main__":
    main()
