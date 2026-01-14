from openai import OpenAI
from core.rag.embedding.base import EmbeddingProvider

class OpenAIEmbeddingProvider(EmbeddingProvider):

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        res = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in res.data]
