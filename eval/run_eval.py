import asyncio
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

from graph.planner_graph import PlannerAgent

GOLDEN_SET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_set.json")
EVAL_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.json")

async def evaluate_case(agent, test_case: dict, index: int, total: int):
    query = test_case["query"]
    expected_mentions = test_case.get("expected_mentions", [])
    
    print(f"\n[{index}/{total}] Evaluating: \"{query[:65]}...\"")
    config = {"configurable": {"thread_id": f"eval-{test_case['id']}"}}
    
    start_time = time.perf_counter()
    try:
        results = await agent.ainvoke({"query": query}, config)
        runtime = time.perf_counter() - start_time
    except Exception as e:
        print(f"  ❌ Execution failed: {e}")
        return None
        
    final_answer = results.get('final_answer', '')
    answer_lower = final_answer.lower()
    
    found_mentions = []
    missing_mentions = []
    
    for item in expected_mentions:
        if item.lower() in answer_lower:
            found_mentions.append(item)
        else:
            missing_mentions.append(item)
            
    hit_rate = len(found_mentions) / len(expected_mentions) if expected_mentions else 1.0
    faithfulness = results.get('faithfulness_score', 1.0)
    unanswered = results.get('unanswered_aspects', [])
    papers_count = len(results.get('papers', []))
    
    print(f"  ✓ Key Concept Recall: {hit_rate * 100:.1f}% ({len(found_mentions)}/{len(expected_mentions)} found)")
    print(f"  ✓ Faithfulness Score: {faithfulness * 100:.1f}% | Runtime: {runtime:.2f}s | Papers: {papers_count}")
    if missing_mentions:
        print(f"    ↳ Missing concepts: {', '.join(missing_mentions)}")
        
    return {
        "id": test_case["id"],
        "query": query,
        "intent": test_case.get("intent", ""),
        "expected_mentions": expected_mentions,
        "found_mentions": found_mentions,
        "missing_mentions": missing_mentions,
        "concept_recall_percent": round(hit_rate * 100, 1),
        "faithfulness_score_percent": round(faithfulness * 100, 1),
        "runtime_seconds": round(runtime, 2),
        "papers_retrieved": papers_count,
        "unanswered_aspects": unanswered
    }

async def main():
    print("=" * 70)
    print("      🧪 PaperPilot Held-Out Golden Set Evaluation Harness")
    print("=" * 70)
    
    if not os.path.exists(GOLDEN_SET_PATH):
        print(f"Golden set not found at: {GOLDEN_SET_PATH}")
        return
        
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)
        
    print(f"Loaded {len(cases)} gold test queries from: {GOLDEN_SET_PATH}")
    agent = PlannerAgent(enable_hitl=False)
    
    eval_records = []
    for idx, tc in enumerate(cases, 1):
        rec = await evaluate_case(agent, tc, idx, len(cases))
        if rec:
            eval_records.append(rec)
            
    if not eval_records:
        print("No evaluation records completed.")
        return
        
    # Aggregate stats
    avg_recall = sum(r["concept_recall_percent"] for r in eval_records) / len(eval_records)
    avg_faith = sum(r["faithfulness_score_percent"] for r in eval_records) / len(eval_records)
    avg_time = sum(r["runtime_seconds"] for r in eval_records) / len(eval_records)
    
    summary_report = {
        "total_eval_cases": len(eval_records),
        "mean_concept_recall_percent": round(avg_recall, 1),
        "mean_faithfulness_percent": round(avg_faith, 1),
        "mean_runtime_seconds": round(avg_time, 2),
        "records": eval_records
    }
    
    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
        
    print("\n" + "=" * 70)
    print("                 🏆 GOLDEN SET EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Test Cases Evaluated:   {len(eval_records)} / {len(cases)}")
    print(f"Mean Concept Recall:          {summary_report['mean_concept_recall_percent']}%")
    print(f"Mean Neural Faithfulness:     {summary_report['mean_faithfulness_percent']}%")
    print(f"Mean End-to-End Latency:      {summary_report['mean_runtime_seconds']} seconds")
    print(f"\nDetailed evaluation results written to: {EVAL_RESULTS_PATH}")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
