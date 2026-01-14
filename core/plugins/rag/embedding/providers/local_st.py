# core/rag/embedding/providers/local_st.py
from sentence_transformers import SentenceTransformer
from core.rag.embedding.base import EmbeddingProvider

class LocalSTEmbeddingProvider(EmbeddingProvider):

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()
