from core.plugins.rag.embedding.providers.local_st import LocalSTEmbeddingProvider
from core.plugins.rag.retriever.simple import SimpleRetriever
from core.conversation.session import ConversationSession
from core.plugins.rag.context.simple import SimpleRAGContextAugmentor
from core.llm.engine import LLMEngine
from core.llm.providers.llama import LlamaCppProvider
from core.plugins.rag.embedding.providers.local_st import EmbeddingProvider
provider = LocalSTEmbeddingProvider()

docs = [
    "這個專案是一個結合 LLM、TTS 與 Whisper 的聊天機器人",
    "Phase 0 建立 LLM 抽象層",
    "Phase 1 支援多模型切換",
]

doc_embeddings = provider.embed(docs)

query = "這個系統的目標是什麼？"
query_embedding = provider.embed([query])[0]

retriever = SimpleRetriever(doc_embeddings, docs)
hits = retriever.retrieve(query_embedding)

# print(hits)

session = ConversationSession(
    "你是一個技術助理，回答需精確"
)

session.user("這個專案的目標是什麼？")

augmentor = SimpleRAGContextAugmentor(
    retriever=retriever,
    embedding_provider=provider
)

llm_provider = LlamaCppProvider(
    model_path="core/llm/models/L3-8B-Stheno-v3.2-Q5_K_S.gguf",
    n_ctx=4096,
    n_threads=8,
)
engine = LLMEngine(llm_provider)
reply = engine.chat_with_rag(session, augmentor)

print(reply)
embed_provider = EmbeddingProvider()
augmentor = SimpleRAGContextAugmentor(
    retriever=retriever,
    embedding_provider=embed_provider
)

reply = engine.run_with_rag(session, augmentor)
