from core.llm.factory import create_provider

class LLMEngine:

    def __init__(self, provider):
        self.provider = provider

    def chat(self, messages):
        return self.provider.chat(messages)

    def chat_with_rag(self, session, augmentor):
        user_query = session.messages[-1]["content"]

        augmented = augmentor.augment(
            messages=session.messages,
            query=user_query
        )

        return self.provider.chat(augmented)
