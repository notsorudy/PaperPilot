import asyncio
import os
import sys
import time
import requests
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from tools.arxiv import search_arxiv
from schemas.optimize_query_schema import OptimizeQuerySchema
from schemas.paper_schema import PaperMetadata

TEST_QUERIES = [
    [
        OptimizeQuerySchema(optimized_query="Self-RAG architecture", keywords=["Self-RAG", "architecture"]),
        OptimizeQuerySchema(optimized_query="critique reflection tokens", keywords=["critique", "tokens"]),
        OptimizeQuerySchema(optimized_query="Self-RAG RAG comparison", keywords=["Self-RAG", "comparison"]),
    ],
    [
        OptimizeQuerySchema(optimized_query="LoRA fine-tuning", keywords=["LoRA", "fine-tuning"]),
        OptimizeQuerySchema(optimized_query="QLoRA 4-bit quantization", keywords=["QLoRA", "quantization"]),
        OptimizeQuerySchema(optimized_query="parameter efficient adapter", keywords=["adapter", "PEFT"]),
    ],
    [
        OptimizeQuerySchema(optimized_query="Corrective RAG", keywords=["CRAG", "Corrective RAG"]),
        OptimizeQuerySchema(optimized_query="retrieval evaluator RAG", keywords=["evaluator", "RAG"]),
        OptimizeQuerySchema(optimized_query="active retrieval generation", keywords=["retrieval", "generation"]),
    ]
]

def download_and_parse_pdf_sync(p: PaperMetadata):
    if not p.pdf_url:
        return [Document(page_content=p.abstract, metadata={'title': p.title})]
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/115.0.0.0 Safari/537.36',
            'Accept': 'application/pdf, application/octet-stream, */*'
        }
        res = requests.get(p.pdf_url, headers=headers, timeout=12)
        if res.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(res.content)
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            os.remove(tmp_path)
            return docs
    except Exception:
        pass
    return [Document(page_content=p.abstract, metadata={'title': p.title, 'is_fallback': True})]

async def run_sequential(queries: list[OptimizeQuerySchema]):
    """Execute search and PDF extraction strictly sequentially (one-by-one)."""
    start = time.perf_counter()
    
    # 1. Sequential search
    papers = []
    for q in queries:
        res = search_arxiv(q.optimized_query, max_results=3)
        papers.extend(res)
        
    # Deduplicate
    unique_papers = {p.paper_id: p for p in papers}.values()
    
    # 2. Sequential PDF downloads
    docs = []
    for p in list(unique_papers)[:4]:
        extracted = download_and_parse_pdf_sync(p)
        docs.extend(extracted)
        
    elapsed = time.perf_counter() - start
    return elapsed, len(unique_papers), len(docs)

async def run_concurrent(queries: list[OptimizeQuerySchema]):
    """Execute search and PDF extraction concurrently using asyncio.gather()."""
    start = time.perf_counter()
    
    # 1. Concurrent search
    async def fetch_async(q_str):
        return await asyncio.to_thread(search_arxiv, q_str, 3)
        
    tasks = [fetch_async(q.optimized_query) for q in queries]
    search_results = await asyncio.gather(*tasks)
    
    papers = []
    for res in search_results:
        papers.extend(res)
    unique_papers = {p.paper_id: p for p in papers}.values()
    
    # 2. Concurrent PDF downloads
    async def download_async(p):
        return await asyncio.to_thread(download_and_parse_pdf_sync, p)
        
    download_tasks = [download_async(p) for p in list(unique_papers)[:4]]
    pdf_results = await asyncio.gather(*download_tasks)
    
    docs = []
    for extracted in pdf_results:
        docs.extend(extracted)
        
    elapsed = time.perf_counter() - start
    return elapsed, len(unique_papers), len(docs)

async def main():
    print("=" * 70)
    print("   ⚡ PaperPilot I/O Concurrency Benchmark: Async vs. Sequential")
    print("=" * 70)
    
    seq_times = []
    conc_times = []
    
    for idx, query_set in enumerate(TEST_QUERIES, 1):
        print(f"\n[Test Set {idx}/3] Benchmarking 3 academic queries with PDF downloads...")
        
        # Sequential Run
        print("  • Running Sequential Execution...")
        seq_time, seq_p_cnt, seq_d_cnt = await run_sequential(query_set)
        seq_times.append(seq_time)
        print(f"    ↳ Sequential Time: {seq_time:.2f}s ({seq_p_cnt} papers, {seq_d_cnt} pages)")
        
        # Concurrent Run
        print("  • Running Concurrent Execution (asyncio.gather)...")
        conc_time, conc_p_cnt, conc_d_cnt = await run_concurrent(query_set)
        conc_times.append(conc_time)
        print(f"    ↳ Concurrent Time: {conc_time:.2f}s ({conc_p_cnt} papers, {conc_d_cnt} pages)")
        
        speedup = ((seq_time - conc_time) / seq_time) * 100
        print(f"  ✨ Speedup for Set {idx}: {speedup:.1f}% faster ({seq_time / conc_time:.2f}x speedup)")

    avg_seq = sum(seq_times) / len(seq_times)
    avg_conc = sum(conc_times) / len(conc_times)
    overall_speedup = ((avg_seq - avg_conc) / avg_seq) * 100
    
    print("\n" + "=" * 70)
    print("                     📊 FINAL SPEEDUP RESULTS")
    print("=" * 70)
    print(f"Average Sequential Wall-Clock Time:  {avg_seq:.2f} seconds")
    print(f"Average Concurrent Wall-Clock Time:  {avg_conc:.2f} seconds")
    print(f"Overall Concurrency Speedup:         {overall_speedup:.1f}% reduction in I/O wall-clock time")
    print(f"Throughput Multiplier:               {avg_seq / avg_conc:.2f}x faster")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
