from langchain_core.prompts import ChatPromptTemplate

SEARCH_PAPER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "Find academic papers related to {query}.")
])