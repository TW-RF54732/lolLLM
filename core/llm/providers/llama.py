from llama_cpp import Llama
from core.llm.base import LLMProvider

class LlamaCppProvider(LLMProvider):

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int = 8,
        n_batch: int = 512,
    ):
        self.model_path = model_path

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch,
        )

    def chat(self, messages):
        res = self.llm.create_chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )
        return res["choices"][0]["message"]["content"]
