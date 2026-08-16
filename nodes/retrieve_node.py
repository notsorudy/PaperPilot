import hashlib
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import VectorParams, Distance
from state.research_state import ResearchState
from app.models import embedding_model, reranker
from app.vectorstore import get_qdrant_client

async def retrieve_chunks(state: ResearchState):
    """Embeds chunks into persistent Qdrant collection, performs vector search, and reranks results."""
    chunks = state.get('chunks', [])
    query = state.get('query', '')
    
    if not chunks:
        print("No chunks available for retrieval.")
        return {"relevant_chunks": []}
        
    if not embedding_model or not reranker:
        print("Missing required RAG models (embedding_model or reranker).")
        return {"relevant_chunks": []}

    # 1. Generate unique collection name from paper IDs for persistence & caching
    paper_ids = sorted(list(set(
        str(doc.metadata.get('paper_id', '')) for doc in chunks if doc.metadata.get('paper_id')
    )))
    if paper_ids:
        raw_key = "_".join(paper_ids)
    else:
        raw_key = "_".join([c.page_content[:50] for c in chunks[:5]])
    
    collection_hash = hashlib.md5(raw_key.encode('utf-8')).hexdigest()[:12]
    collection_name = f"papers_{collection_hash}"
    
    client = get_qdrant_client()
    
    # 2. Check if collection exists or create and populate it
    if client.collection_exists(collection_name):
        print(f"Loading existing Qdrant collection: '{collection_name}' (cached)...")
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding_model,
        )
    else:
        print(f"Creating new Qdrant collection: '{collection_name}' and persisting chunks...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding_model,
        )
        vectorstore.add_documents(chunks)
        print(f"Persisted {len(chunks)} chunks into Qdrant collection '{collection_name}'.")

    # 3. Stage 1 Retrieval (Dense Vector Search via Qdrant)
    k_retrieve = min(30, len(chunks))
    print(f"Searching Qdrant for top {k_retrieve} most relevant chunks for query: '{query}'")
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_retrieve})
    stage_1_docs = retriever.invoke(query)
    
    if not stage_1_docs:
        print("No matching documents found in vector search.")
        return {"relevant_chunks": []}
    
    # 4. Stage 2 Retrieval (Neural Re-Ranking with CrossEncoder)
    print("Re-ranking retrieved chunks using Hugging Face CrossEncoder...")
    pairs = [(query, doc.page_content) for doc in stage_1_docs]
    scores = reranker.predict(pairs)
    
    # Zip docs with scores and sort descending
    scored_docs = list(zip(stage_1_docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Keep top 15
    top_15_docs = [doc for doc, score in scored_docs[:15]]
    print(f"Selected top {len(top_15_docs)} chunks after CrossEncoder re-ranking.")
    
    return {"relevant_chunks": top_15_docs}
