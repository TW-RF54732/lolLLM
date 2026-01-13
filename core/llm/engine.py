# core\llm\engine.py
from core.llm.contract.result import BaseResult,TextResult
from core.llm.contract.provider import BaseLLMProvider

class LLMEngine:
    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider

    def chat(self, session) -> BaseResult:
        result = self.provider.generate(session.get_messages())

        if isinstance(result, TextResult):
            session.assistant(result.content)

        return result