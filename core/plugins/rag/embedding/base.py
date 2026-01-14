from abc import ABC, abstractmethod

class EmbeddingProvider(ABC):

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Input: list of strings
        Output: list of embedding vectors
        """
        pass
