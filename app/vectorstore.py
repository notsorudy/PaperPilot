import os
from qdrant_client import QdrantClient

QDRANT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".qdrant_data")

_client_instance = None

def get_qdrant_client() -> QdrantClient:
    """Returns a singleton Qdrant client configured for on-disk persistence."""
    global _client_instance
    if _client_instance is None:
        os.makedirs(QDRANT_PATH, exist_ok=True)
        _client_instance = QdrantClient(path=QDRANT_PATH)
    return _client_instance
