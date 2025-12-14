class ConversationSession:
    def __init__(self, system_prompt: str):
        self.messages = [
            {"role": "system", "content": system_prompt}
        ]

    def user(self, text: str):
        self.messages.append({"role": "user", "content": text})

    def assistant(self, text: str):
        self.messages.append({"role": "assistant", "content": text})
