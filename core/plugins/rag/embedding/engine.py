# core/rag/embedding/engine.py
from core.rag.embedding.factory import create_embedding_provider

class EmbeddingEngine:

    def __init__(self, provider):
        self.provider = provider

    @classmethod
    def from_profile(cls, profile):
        return cls(create_embedding_provider(profile))

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.provider.embed(texts)
