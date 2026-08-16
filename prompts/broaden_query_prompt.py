from langchain_core.prompts import ChatPromptTemplate

BROADEN_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert research query expansion specialist.

The previous search queries for the user's research topic returned too few results or failed to find adequate literature on arXiv.
Your goal is to broaden and reformulate the search strategy to cast a wider net without losing the core subject matter.

Guidelines:
- Broaden terms by using widely accepted academic terminology, synonyms, or parent domain concepts (e.g., if 'self-adaptive RAG critique' returned 0 results, expand to 'retrieval augmented generation' or 'critique language models').
- Drop overly narrow constraints, specific benchmark names, or multi-word compound adjectives.
- Produce 3 to 4 concise keyword queries (2-3 words each) tailored for arXiv search.
- Return ONLY the list of rewritten queries.
"""
    ),
    (
        "human",
        """Original User Query: {query}
Previous Search Queries that yielded insufficient results:
{previous_queries}

Please generate broadened, high-recall search queries for arXiv."""
    )
])
