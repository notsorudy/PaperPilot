from langchain_core.prompts import ChatPromptTemplate

EXTRACT_CLAIMS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a meticulous scientific peer-reviewer and factual verification auditor.
Your job is to extract 5 to 10 key atomic factual claims from the provided literature review.

Rules:
- Each claim must be a standalone, self-contained, concrete factual proposition (e.g., "Self-RAG uses reflection tokens called [Critique] and [Retrieve] to evaluate output quality during inference" or "The model achieves 54.9 accuracy on PopQA").
- Focus on key architecture claims, benchmark numbers, methodology details, and comparative assertions.
- Do NOT include generic filler phrases, transitions, or meta-comments.
- Return ONLY the list of atomic claims.
"""
    ),
    (
        "human",
        """Original Query: {query}

Literature Review to audit:
{literature_review}

Please extract 5 to 10 atomic factual claims."""
    )
])
