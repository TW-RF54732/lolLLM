# core/rag/embedding/factory.py
from core.rag.embedding.providers.local_st import LocalSTEmbeddingProvider
from core.rag.embedding.providers.openai import OpenAIEmbeddingProvider

def create_embedding_provider(profile):
    if profile.type == "local_st":
        return LocalSTEmbeddingProvider(profile.model_name)

    if profile.type == "openai":
        return OpenAIEmbeddingProvider(
            api_key=profile.api_key,
            model=profile.model_name
        )

    raise ValueError(f"Unknown embedding provider: {profile.type}")
