# core/llm/contract/result.py
from dataclasses import dataclass
from typing import Union
    
@dataclass
class TextResult:
    content: str

@dataclass
class FunctionCallResult:
    name: str
    arguments: dict

LLMResult = Union[TextResult, FunctionCallResult]
