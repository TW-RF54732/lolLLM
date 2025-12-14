from core.llm.factory import create_provider

class LLMEngine:

    def __init__(self, provider):
        self.provider = provider

    @classmethod
    def from_profile(cls, profile):
        provider = create_provider(profile)
        return cls(provider)

    def chat(self, messages):
        return self.provider.chat(messages)
