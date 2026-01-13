# core/llm/contract/provider.py
from abc import ABC, abstractmethod
from .message import Message
from .result import BaseResult

class BaseLLMProvider(ABC):

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        config=None,
    ) -> BaseResult:
        pass