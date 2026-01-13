# core/rag/context/simple.py
from core.rag.contract.augmentor import BaseRAGAugmentor
from core.llm.contract.message import Message

class SimpleRAGContextAugmentor(BaseRAGAugmentor):

    def __init__(self, retriever, embedding_provider, top_k=3):
        self.retriever = retriever
        self.embedding_provider = embedding_provider
        self.top_k = top_k

    def augment(self, messages: list[Message]) -> list[Message]:
        # 1. 找最後一個 user 訊息
        user_message = next(
            (m for m in reversed(messages) if m["role"] == "user"),
            None
        )

        if user_message is None:
            return messages

        # 2. Embed user query
        query = user_message["content"]
        query_embedding = self.embedding_provider.embed([query])[0]

        # 3. Retrieve contexts
        hits = self.retriever.retrieve(
            query_embedding,
            top_k=self.top_k
        )

        if not hits:
            return messages

        context_text = "\n".join(text for _, text in hits)

        # 4. 插入 system context（固定位置）
        augmented = messages.copy()
        augmented.insert(
            1,
            {
                "role": "system",
                "content": f"以下是相關背景資訊，請在回答時參考：\n{context_text}"
            }
        )

        return augmented
