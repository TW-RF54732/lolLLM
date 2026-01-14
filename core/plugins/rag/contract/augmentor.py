# core/rag/contract/augmentor.py
from abc import ABC, abstractmethod
from core.llm.contract.message import Message

class BaseRAGAugmentor(ABC):

    @abstractmethod
    def augment(self, messages: list[Message]) -> list[Message]:
        """
        接收標準化 messages，回傳 augmented messages
        """
        pass
