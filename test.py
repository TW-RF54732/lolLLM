# from core.conversation.session import ConversationSession
# from core.llm.providers.llama import LlamaCppProvider

# session = ConversationSession("You are a helpful assistant.")
# session.user("你好")

# llm = LlamaCppProvider("./llm/LLM_models/L3-8B-Stheno-v3.2-Q5_K_S.gguf")
# reply = llm.chat(session.messages)

# session.assistant(reply)

# print(session.messages)

class testClass:
    def __init__(self):
        self.a = 0
        self.b = 1
        self.c = 2

at = testClass(a = 1,b = 2,c= 3)
print(at.a)