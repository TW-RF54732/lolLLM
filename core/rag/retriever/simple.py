from core.rag.utils.similarity import cosine_similarity

class SimpleRetriever:

    def __init__(self, embeddings: list[list[float]], texts: list[str]):
        self.embeddings = embeddings
        self.texts = texts

    def retrieve(self, query_embedding, top_k: int = 3) -> list[str]:
        scores = [
            (cosine_similarity(query_embedding, emb), text)
            for emb, text in zip(self.embeddings, self.texts)
        ]
        scores.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scores[:top_k]]
