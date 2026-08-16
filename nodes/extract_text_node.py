import os
import requests
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from state.research_state import ResearchState

import asyncio

async def extract_text(state: ResearchState):
    """Downloads PDFs (or injects abstracts) and chunks the text concurrently."""
    papers = state.get('papers', [])
    
    if not papers:
        print("No papers available to extract text from.")
        return {"chunks": []}

    print(f"\n--- Initiating Text Extraction on {len(papers)} papers ---")
    documents = []
    
    def download_and_parse(p):
        pdf_url = p.pdf_url
        success = False
        local_docs = []
        
        if pdf_url:
            try:
                print(f"Downloading: {p.title[:50]}...")
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
                    'Accept': 'application/pdf, application/octet-stream, */*'
                }
                response = requests.get(pdf_url, headers=headers, timeout=15)
                if response.status_code == 200 and ('application/pdf' in response.headers.get('Content-Type', '') or 'application/octet-stream' in response.headers.get('Content-Type', '')):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(response.content)
                        tmp_path = tmp_file.name
                        
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    
                    for d in docs:
                        d.metadata['title'] = p.title
                        d.metadata['paper_id'] = p.paper_id
                        
                    local_docs.extend(docs)
                    os.remove(tmp_path)
                    success = True
                else:
                    print(f"Warning: URL did not return a valid PDF: {pdf_url}")
            except Exception as e:
                print(f"Warning: Failed to download PDF for '{p.title}': {e}")
                
        if not success:
            print(f"Fallback triggered: Injecting abstract for '{p.title[:50]}...'")
            abstract_doc = Document(
                page_content=f"Title: {p.title}\n\nAbstract: {p.abstract}",
                metadata={'title': p.title, 'paper_id': p.paper_id, 'is_fallback': True}
            )
            local_docs.append(abstract_doc)
            
        return local_docs

    async def process_paper(p):
        return await asyncio.to_thread(download_and_parse, p)

    # 1. Download and Parse PDFs (concurrently)
    tasks = [process_paper(p) for p in papers]
    results = await asyncio.gather(*tasks)
    
    for docs in results:
        documents.extend(docs)

    if not documents:
        print("Failed to extract any text.")
        return {"chunks": []}

    print(f"Extracted {len(documents)} pages/abstracts. Chunking text...")

    # 3. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")
    
    return {"chunks": chunks}
