# app/models.py

try:
    from sentence_transformers import CrossEncoder
    print("Loading CrossEncoder model globally...")
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
except ImportError:
    reranker = None

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("Loading Embedding model globally...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except ImportError:
    embedding_model = None
