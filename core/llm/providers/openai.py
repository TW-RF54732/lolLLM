# core/llm/providers/openai.py
from openai import OpenAI
from core.llm.base import LLMProvider

class OpenAIProvider(LLMProvider):

    def __init__(self, api_key: str, model_name: str):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def chat(self, messages):
        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
        )
        return res.choices[0].message.content
