# core/llm/providers/llama.py
from llama_cpp import Llama
from core.llm.contract.provider import BaseLLMProvider
from core.llm.contract.message import Message
from core.llm.contract.result import TextResult

class LlamaCppProvider(BaseLLMProvider):

    def __init__(
        self,
        model_path: str,
        n_ctx=4096,
        n_threads=8,
        n_batch=512,
    ):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch,
        )


    def generate(self, messages: list[Message]) -> TextResult:
        prompt = self._to_prompt(messages)

        output = self.llm(
            prompt,
            max_tokens=512,
            stop=["</s>"],
        )

        text = output["choices"][0]["text"]
        return TextResult(content=text)

    def _to_prompt(self, messages: list[Message]) -> str:
        lines = []
        for m in messages:
            lines.append(f"{m['role']}: {m['content']}")
        lines.append("assistant:")
        return "\n".join(lines)
