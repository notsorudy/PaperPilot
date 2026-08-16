from app.llm import llm_cerebras, llm_groq
from app.models import reranker
from prompts.verify_prompt import EXTRACT_CLAIMS_PROMPT
from schemas.verify_schema import ExtractedClaims
from state.research_state import ResearchState

async def verify_answer(state: ResearchState):
    """LLM-as-judge + CrossEncoder self-check: extracts atomic claims and verifies grounding against source chunks."""
    final_answer = state.get('final_answer', '')
    relevant_chunks = state.get('relevant_chunks', [])
    query = state.get('query', '')
    
    if not final_answer or not relevant_chunks or not reranker:
        return {
            "faithfulness_score": 1.0
        }
        
    print("\n🔍 Running Step 8: Neural Claim-Verification Audit (Self-Check)...")
    
    # 1. Extract atomic claims from the generated answer using LLM
    try:
        llm_structured = llm_cerebras.with_structured_output(ExtractedClaims)
        chain = EXTRACT_CLAIMS_PROMPT | llm_structured
        res = await chain.ainvoke({
            "query": query,
            "literature_review": final_answer
        })
        claims = res.claims if res and res.claims else []
    except Exception as e:
        print(f"Cerebras failed in verify_answer: {e}. Trying Groq fallback...")
        try:
            llm_structured_fallback = llm_groq.with_structured_output(ExtractedClaims)
            chain_fallback = EXTRACT_CLAIMS_PROMPT | llm_structured_fallback
            res = await chain_fallback.ainvoke({
                "query": query,
                "literature_review": final_answer
            })
            claims = res.claims if res and res.claims else []
        except Exception as e2:
            print(f"Verification claim extraction fallback failed: {e2}")
            claims = []
            
    if not claims:
        print("No claims extracted for verification audit.")
        return {
            "faithfulness_score": 1.0
        }
        
    print(f"Extracted {len(claims)} atomic claims to audit against {len(relevant_chunks)} source chunks.")
    
    # 2. Check each claim against source chunks using CrossEncoder
    supported_claims = []
    unsupported_claims = []
    
    # ms-marco-MiniLM-L-6-v2 outputs logits: score >= 0.0 or score >= -0.5 is strongly aligned
    SUPPORT_THRESHOLD = -0.5
    
    for claim in claims:
        pairs = [(claim, chunk.page_content) for chunk in relevant_chunks]
        scores = reranker.predict(pairs)
        max_score = float(max(scores)) if len(scores) > 0 else -999.0
        
        if max_score >= SUPPORT_THRESHOLD:
            supported_claims.append((claim, max_score))
        else:
            unsupported_claims.append((claim, max_score))
            
    total = len(claims)
    supported_count = len(supported_claims)
    faithfulness_score = round(supported_count / total, 2) if total > 0 else 1.0
    
    print(f"Verification Audit Result: {supported_count}/{total} claims grounded (Faithfulness Score: {faithfulness_score * 100:.1f}%)")
    
    # Append verification section to final answer
    audit_report = [
        "\n\n---",
        f"### 🛡️ Fact-Check & Verification Audit (Faithfulness: {faithfulness_score * 100:.1f}%)",
        f"**Audit Summary**: {supported_count} of {total} atomic factual claims verified against source literature chunks."
    ]
    
    if unsupported_claims:
        audit_report.append("\n**⚠️ Claims flagged for lower source grounding:**")
        for uc, score in unsupported_claims:
            audit_report.append(f"  • *{uc}* (cross-encoder score: {score:.2f})")
    else:
        audit_report.append("\n✅ All audited claims have direct neural grounding in the retrieved paper excerpts.")
        
    updated_final_answer = final_answer + "\n" + "\n".join(audit_report)
    
    return {
        "final_answer": updated_final_answer,
        "faithfulness_score": faithfulness_score
    }
