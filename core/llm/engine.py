# core\llm\engine.py
from core.llm.contract.result import TextResult
from core.llm.contract.provider import BaseLLMProvider

class LLMEngine:
    def __init__(self, provider:BaseLLMProvider):
        self.provider = provider

    def chat(self, session) -> TextResult:
        result = self.provider.generate(session.get_messages())

        # 基礎對話階段，只接受 TextResult
        if isinstance(result, TextResult):
            session.assistant(result.content)
            return result

        raise RuntimeError("Unexpected LLMResult type in base chat mode")
