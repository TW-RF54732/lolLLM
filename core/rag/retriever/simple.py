# core/rag/retriever/simple.py
class SimpleRetriever:

    def __init__(self, embeddings, documents):
        self.embeddings = embeddings
        self.documents = documents

    def retrieve(self, query_embedding, top_k=3):
        scores = []
        for emb, text in zip(self.embeddings, self.documents):
            score = cosine_similarity(query_embedding, emb)
            scores.append((score, text))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]
