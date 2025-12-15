class SimpleRAGContextAugmentor:
    def __init__(self, retriever, embedding_provider, top_k=3):
        self.retriever = retriever
        self.embedding_provider = embedding_provider
        self.top_k = top_k

    def augment(self, messages, query):
        query_embedding = self.embedding_provider.embed([query])[0]
        contexts = self.retriever.retrieve(query_embedding, top_k=self.top_k)

        context_text = "\n".join(text for _, text in contexts)

        augmented = messages.copy()
        augmented.insert(
            1,
            {
                "role": "system",
                "content": f"以下是相關背景資訊：\n{context_text}"
            }
        )
        return augmented
