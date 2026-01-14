# core/rag/context/base.py
from abc import ABC, abstractmethod

class ContextAugmentor(ABC):

    @abstractmethod
    def augment(
        self,
        messages: list[dict],
        query: str
    ) -> list[dict]:
        """
        回傳一份「新的 messages」，不可修改原本 messages
        """
        pass
