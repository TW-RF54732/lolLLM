# core/llm/providers/openai.py
from openai import OpenAI

from core.llm.contract.provider import BaseLLMProvider
from core.llm.contract.message import Message
from core.llm.contract.result import TextResult, LLMResult


class OpenAIProvider(BaseLLMProvider):

    def __init__(self, api_key: str, model_name: str):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, messages: list[Message]) -> LLMResult:
        # 1. convert normalized Message → OpenAI format
        oa_messages = [
            {
                "role": m["role"],
                "content": m["content"]
            }
            for m in messages
        ]

        # 2. call OpenAI
        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=oa_messages,
            temperature=0.7,
        )

        # 3. normalize output → LLMResult
        content = res.choices[0].message.content
        return TextResult(content=content)
