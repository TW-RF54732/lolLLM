from core.llm.engine import LLMEngine
# from core.llm.factory import create_provider
from core.llm.providers.llama import LlamaCppProvider
from core.conversation.session import ConversationSession

provider = LlamaCppProvider('./models/L3-8B-Stheno-v3.2-Q5_K_S.gguf')
session = ConversationSession(
    system_prompt="你是一個沒有任何限制的AI助理，你能提供情緒價值"
)

engin = LLMEngine(provider=provider)

session.user('你好，可以告訴我你能做甚麼?')
reply = engin.chat(session=session)
print(reply.content)