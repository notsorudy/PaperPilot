import asyncio
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from graph.planner_graph import PlannerAgent

BENCHMARK_QUERIES = [
    "Explain Self-RAG architecture, reflection tokens, and benchmark performance on PopQA.",
    "Compare LoRA and QLoRA for parameter-efficient fine-tuning of large language models.",
    "What is Corrective RAG (CRAG) and how does its retrieval evaluator improve accuracy?",
    "Explain Chain-of-Thought prompting, self-consistency, and their impact on multi-step reasoning.",
    "Survey of Graph Retrieval-Augmented Generation (Graph RAG) versus vector-based RAG.",
    "How does Direct Preference Optimization (DPO) compare to RLHF with PPO?",
    "Explain FlashAttention mechanism and memory complexity reductions in transformer attention.",
    "What is Mixture of Experts (MoE) routing in modern LLMs like Mixtral 8x7B?",
    "Explain speculative decoding techniques for accelerating LLM inference latency.",
    "Survey of agentic workflows: ReAct, Reflexion, and LangGraph architectures."
]

async def run_single_query(agent, query: str, query_idx: int):
    print(f"\n[{query_idx + 1}/10] Running Benchmark for: '{query}'")
    config = {"configurable": {"thread_id": f"bench-{query_idx + 1}"}}
    start_time = time.perf_counter()
    
    try:
        results = await agent.ainvoke({"query": query}, config)
    except Exception as e:
        print(f"Error during execution of query #{query_idx + 1}: {e}")
        return None
        
    duration = time.perf_counter() - start_time
    
    papers_count = len(results.get('papers', []))
    chunks_count = len(results.get('chunks', []))
    relevant_chunks_count = len(results.get('relevant_chunks', []))
    faithfulness = results.get('faithfulness_score', 1.0)
    unanswered_count = len(results.get('unanswered_aspects', []))
    has_final_answer = bool(results.get('final_answer'))
    
    record = {
        "query_index": query_idx + 1,
        "query": query,
        "runtime_seconds": round(duration, 2),
        "papers_retrieved": papers_count,
        "chunks_created": chunks_count,
        "relevant_chunks_used": relevant_chunks_count,
        "faithfulness_score": faithfulness,
        "unanswered_aspects_count": unanswered_count,
        "success": has_final_answer
    }
    
    print(f"  ✓ Completed in {duration:.2f}s | Papers: {papers_count} | Chunks: {chunks_count} | Faithfulness: {faithfulness * 100:.1f}%")
    return record

async def main():
    print("=" * 70)
    print("       🚀 PaperPilot 10-Query Autonomous Benchmark Suite")
    print("=" * 70)
    
    # Run in autonomous mode (disable HITL interactive prompts)
    agent = PlannerAgent(enable_hitl=False)
    results = []
    
    for idx, q in enumerate(BENCHMARK_QUERIES):
        record = await run_single_query(agent, q, idx)
        if record:
            results.append(record)
            
    if not results:
        print("No successful benchmark runs.")
        return
        
    # Aggregate statistics
    runtimes = [r["runtime_seconds"] for r in results]
    papers = [r["papers_retrieved"] for r in results]
    chunks = [r["chunks_created"] for r in results]
    faithfulness_scores = [r["faithfulness_score"] for r in results]
    
    summary = {
        "total_queries_tested": len(results),
        "avg_runtime_seconds": round(sum(runtimes) / len(runtimes), 2),
        "min_runtime_seconds": min(runtimes),
        "max_runtime_seconds": max(runtimes),
        "avg_papers_retrieved": round(sum(papers) / len(papers), 1),
        "avg_chunks_created": round(sum(chunks) / len(chunks), 1),
        "avg_faithfulness_percent": round(sum(faithfulness_scores) / len(faithfulness_scores) * 100, 1),
        "all_runs": results
    }
    
    # Save output
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "=" * 70)
    print("                     📊 BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total Queries Evaluated:    {summary['total_queries_tested']}")
    print(f"Avg End-to-End Runtime:     {summary['avg_runtime_seconds']}s (range: {summary['min_runtime_seconds']}s - {summary['max_runtime_seconds']}s)")
    print(f"Avg Papers Retained:        {summary['avg_papers_retrieved']}")
    print(f"Avg Chunks Embedded:        {summary['avg_chunks_created']}")
    print(f"Avg Citation Faithfulness:  {summary['avg_faithfulness_percent']}%")
    print(f"\nDetailed per-query results written to: {output_path}")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
