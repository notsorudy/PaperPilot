import asyncio
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from graph.planner_graph import PlannerAgent
from app.llm import llm_cerebras, llm_groq
from app.models import reranker
from prompts.verify_prompt import EXTRACT_CLAIMS_PROMPT
from schemas.verify_schema import ExtractedClaims

AUDIT_QUERIES = [
    "Can you find the official Self-RAG paper and explain its architecture, how the critique tokens work, and compare its performance to standard RAG models?",
    "Explain Low-Rank Adaptation (LoRA) and QLoRA for LLMs, detailing rank decomposition and 4-bit NormalFloat quantization.",
    "What is Corrective Retrieval Augmented Generation (CRAG), and how does its retrieval evaluator trigger web search fallbacks?"
]

async def audit_query(agent, query: str, idx: int):
    print(f"\n[Audit {idx}/3] Processing: \"{query}\"")
    config = {"configurable": {"thread_id": f"faith-bench-{idx}"}}
    
    result = await agent.ainvoke({"query": query}, config)
    final_answer = result.get('final_answer', '')
    relevant_chunks = result.get('relevant_chunks', [])
    
    if not final_answer or not relevant_chunks:
        print("  ❌ Missing final answer or chunks.")
        return None
        
    print(f"  • Extracting claims from generated review ({len(relevant_chunks)} source chunks available)...")
    
    try:
        chain = EXTRACT_CLAIMS_PROMPT | llm_cerebras.with_structured_output(ExtractedClaims)
        claim_obj = await chain.ainvoke({"query": query, "literature_review": final_answer})
        claims = claim_obj.claims
    except Exception:
        chain = EXTRACT_CLAIMS_PROMPT | llm_groq.with_structured_output(ExtractedClaims)
        claim_obj = await chain.ainvoke({"query": query, "literature_review": final_answer})
        claims = claim_obj.claims
        
    print(f"  • Extracted {len(claims)} atomic claims. Scoring grounding with Cross-Encoder...")
    
    supported = []
    unsupported = []
    
    for c in claims:
        pairs = [(c, doc.page_content) for doc in relevant_chunks]
        scores = reranker.predict(pairs)
        max_score = float(max(scores)) if len(scores) > 0 else -999.0
        
        if max_score >= -0.5:
            supported.append({"claim": c, "score": round(max_score, 2)})
        else:
            unsupported.append({"claim": c, "score": round(max_score, 2)})
            
    total_claims = len(claims)
    accuracy = round(len(supported) / total_claims * 100, 1) if total_claims > 0 else 100.0
    
    print(f"  ✓ Faithfulness Accuracy: {accuracy}% ({len(supported)}/{total_claims} supported)")
    
    return {
        "query": query,
        "total_claims": total_claims,
        "supported_count": len(supported),
        "unsupported_count": len(unsupported),
        "faithfulness_percent": accuracy,
        "supported_claims": supported,
        "unsupported_claims": unsupported
    }

async def main():
    print("=" * 70)
    print("      🛡️ PaperPilot Citation & Claim Faithfulness Benchmark")
    print("=" * 70)
    
    agent = PlannerAgent(enable_hitl=False)
    audit_results = []
    
    for idx, q in enumerate(AUDIT_QUERIES, 1):
        res = await audit_query(agent, q, idx)
        if res:
            audit_results.append(res)
            
    if not audit_results:
        print("No audit runs completed.")
        return
        
    all_total = sum(r["total_claims"] for r in audit_results)
    all_supp = sum(r["supported_count"] for r in audit_results)
    avg_acc = round(all_supp / all_total * 100, 1) if all_total > 0 else 0.0
    
    output_data = {
        "overall_citation_faithfulness_percent": avg_acc,
        "total_claims_audited": all_total,
        "total_supported_claims": all_supp,
        "per_query_audits": audit_results
    }
    
    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faithfulness_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
        
    print("\n" + "=" * 70)
    print("                   📊 OVERALL FAITHFULNESS AUDIT RESULT")
    print("=" * 70)
    print(f"Total Atomic Claims Audited:          {all_total}")
    print(f"Grounded Claims in Paper Excerpts:    {all_supp}")
    print(f"Overall Citation / Claim Faithfulness: {avg_acc}%")
    print(f"Detailed claim-by-claim audit saved to: {out_file}")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
