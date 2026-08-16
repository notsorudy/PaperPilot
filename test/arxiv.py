import requests
import feedparser

query = "Self-RAG"

url = (
    "http://export.arxiv.org/api/query?"
    f"search_query=all:{query}"
    "&start=0"
    "&max_results=3"
)

response = requests.get(url)

feed = feedparser.parse(response.text)

for paper in feed.entries:
    print(paper.title)
    # print(paper.summary)
    # print(paper.pdf_url)