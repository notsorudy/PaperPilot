from tools.arxiv import search_arxiv
import sys

def main():
    keyword = "hyperspectral imaging drought stress"
    print(f"Testing arXiv search with query: '{keyword}'")
    papers = search_arxiv(keyword, max_results=2)
    
    if not papers:
        print("No papers found.")
        sys.exit(1)
        
    print(f"Found {len(papers)} papers:")
    for paper in papers:
        print(f"- Title: {paper.title}")
        print(f"  ID: {paper.paper_id}")
        print(f"  Authors: {', '.join(paper.authors)}")
        print(f"  URL: {paper.pdf_url}")
        print("---")

if __name__ == "__main__":
    main()
