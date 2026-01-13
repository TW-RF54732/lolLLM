# core/llm/factory.py
from core.llm.providers.llama import LlamaCppProvider
from core.llm.providers.openai import OpenAIProvider

def create_provider(profile):
    if profile.type == "llama_cpp":
        return LlamaCppProvider(
            model_path=profile.model_path,
            n_ctx=profile.n_ctx,
            n_threads=profile.n_threads,
            n_batch=profile.n_batch,
        )

    if profile.type == "openai":
        return OpenAIProvider(
            api_key=profile.api_key,
            model_name=profile.model_name,
        )

    raise ValueError(f"Unknown provider type: {profile.type}")
