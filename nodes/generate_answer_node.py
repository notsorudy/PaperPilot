from state.research_state import ResearchState
from app.llm import llm_cerebras, llm_groq
from prompts.synthesis_prompts import SYNTHESIS_PROMPT
from schemas.synthesis_schema import SynthesisOutput

async def generate_answer(state: ResearchState):
    """Takes the top relevant chunks and writes the structured literature review using the LLM."""
    relevant_chunks = state.get('relevant_chunks', [])
    query = state.get('query', '')
    
    if not relevant_chunks:
        return {
            "final_answer": "No relevant chunks found to answer the query.",
            "unanswered_aspects": [query]
        }
        
    # Format chunks for the final prompt
    context_chunks_text = ""
    for idx, doc in enumerate(relevant_chunks):
        title = doc.metadata.get('title', 'Unknown Paper')
        context_chunks_text += f"--- EXCERPT {idx+1} (Source: {title}) ---\n{doc.page_content}\n\n"
        
    # Synthesize Final Answer with Structured Output
    print(f"Feeding Top {len(relevant_chunks)} chunks to LLM for final synthesis...")
    
    try:
        llm_structured = llm_cerebras.with_structured_output(SynthesisOutput)
        chain = SYNTHESIS_PROMPT | llm_structured
        result = await chain.ainvoke({
            "query": query,
            "context_chunks": context_chunks_text
        })
    except Exception as e:
        print(f"Cerebras LLM failed in generate_answer: {e}. Falling back to Groq...")
        llm_structured_fallback = llm_groq.with_structured_output(SynthesisOutput)
        chain_fallback = SYNTHESIS_PROMPT | llm_structured_fallback
        result = await chain_fallback.ainvoke({
            "query": query,
            "context_chunks": context_chunks_text
        })
        
    if isinstance(result, SynthesisOutput):
        review_text = result.literature_review
        unanswered = result.unanswered_aspects or []
        notes = result.confidence_notes
        
        # Build complete formatted response string
        full_text_blocks = [review_text]
        if unanswered:
            unanswered_str = "\n".join([f"  • {item}" for item in unanswered])
            full_text_blocks.append(f"\n### ⚠️ Unanswered Aspects (Insufficient Evidence in Chunks):\n{unanswered_str}")
        if notes:
            full_text_blocks.append(f"\n### 📝 Confidence Notes:\n{notes}")
            
        final_answer = "\n".join(full_text_blocks)
    else:
        final_answer = str(result)
        unanswered = []
        
    print("\n--- FINAL ANSWER GENERATED ---")
    return {
        "final_answer": final_answer,
        "unanswered_aspects": unanswered
    }
