import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

from graph.planner_graph import PlannerAgent

async def main():
    agent = PlannerAgent(enable_hitl=True)
    config = {"configurable": {"thread_id": "session-main-1"}}
    
    initial_state = {
        'query': "Can you find the official Self-RAG paper and explain its architecture, how the critique tokens work, and compare its performance to standard RAG models?"
    }
    
    print("=========================================================")
    print("   📄 PaperPilot — Autonomous Literature Review Agent   ")
    print("=========================================================\n")
    print(f"Research Query: \"{initial_state['query']}\"\n")
    
    # -------------------------------------------------------------------
    # STAGE 1: Query Planning & Optimization
    # -------------------------------------------------------------------
    print("Executing Phase 1: Planning and Query Optimization...")
    await agent.ainvoke(initial_state, config)
    
    # Check interrupt before search_papers
    snapshot = await agent.aget_state(config)
    state = snapshot.values
    
    print("\n" + "=" * 60)
    print("🔍 [HITL Gate 1] RESEARCH PLAN REVIEW")
    print("=" * 60)
    plan = state.get('plan')
    if plan:
        print(f"  • Intent:    {plan.intent}")
        print(f"  • Reasoning: {plan.reasoning}")
        print(f"  • Expected Sources: {', '.join(plan.expected_sources)}")
        
    print("\n  • Optimized Search Queries for arXiv:")
    for i, q in enumerate(state.get('optimized_queries', []), 1):
        print(f"    {i}. \"{q.optimized_query}\"  (keywords: {', '.join(q.keywords)})")
    print("=" * 60)
    
    user_choice = input("\n[HITL] Approve research plan and proceed to search arXiv? (Y/n): ").strip().lower()
    if user_choice in ['n', 'no', 'abort']:
        print("Pipeline aborted by user.")
        return
        
    # -------------------------------------------------------------------
    # STAGE 2: Search & Ranking
    # -------------------------------------------------------------------
    print("\nExecuting Phase 2: Searching arXiv & Neural Paper Ranking...")
    await agent.ainvoke(None, config)
    
    # Check interrupt before extract_text
    snapshot = await agent.aget_state(config)
    state = snapshot.values
    
    print("\n" + "=" * 60)
    print("📚 [HITL Gate 2] RETRIEVED PAPERS REVIEW")
    print("=" * 60)
    papers = state.get('papers', [])
    print(f"Total Candidate Papers Ranked: {len(papers)}")
    for idx, p in enumerate(papers[:10], 1):
        print(f"\n  {idx}. [{p.source}] {p.title}")
        print(f"     Authors: {', '.join(p.authors[:4])}")
        print(f"     PDF URL: {p.pdf_url}")
    print("=" * 60)
    
    user_choice = input(f"\n[HITL] Approve downloading full PDFs and building Qdrant RAG index for top {len(papers[:10])} papers? (Y/n): ").strip().lower()
    if user_choice in ['n', 'no', 'abort']:
        print("Pipeline aborted by user.")
        return
        
    # -------------------------------------------------------------------
    # STAGE 3: PDF Extraction -> Qdrant RAG -> Synthesis -> Audit -> BibTeX
    # -------------------------------------------------------------------
    print("\nExecuting Phase 3: Text Extraction, Qdrant Retrieval, Synthesis & Self-Check...")
    await agent.ainvoke(None, config)
    
    # Retrieve final state
    snapshot = await agent.aget_state(config)
    final_state = snapshot.values
    
    if final_state.get('final_answer'):
        print("\n" + "=" * 65)
        print("           📑 FINAL SYNTHESIZED LITERATURE REVIEW")
        print("=" * 65 + "\n")
        print(final_state['final_answer'])
        print("\n" + "=" * 65 + "\n")
        
    if final_state.get('citations'):
        print("=" * 65)
        print("                 📚 BIBTEX CITATIONS")
        print("=" * 65 + "\n")
        print(final_state['citations'])
        print("\n" + "=" * 65 + "\n")

if __name__ == "__main__":
    asyncio.run(main())