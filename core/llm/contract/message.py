# core/llm/contract/message.py
from typing import Literal, TypedDict

Role = Literal["system", "user", "assistant", "tool"]

class Message(TypedDict):
    role: Role
    content: str
