#core\conversation\session.py
from core.llm.contract.message import Message

class ConversationSession:
    def __init__(self, system_prompt: str | None = None):
        self.messages: list[Message] = []

        if system_prompt:
            self.messages.append({
                "role": "system",
                "content": system_prompt
            })

    def user(self, content: str) -> None:
        self.messages.append({
            "role": "user",
            "content": content
        })

    def assistant(self, content: str) -> None:
        self.messages.append({
            "role": "assistant",
            "content": content
        })

    def get_messages(self) -> list[Message]:
        return self.messages
    
    def last(self) -> Message | None:
        return self.messages[-1] if self.messages else None

