# core/llm/contract/result.py
from dataclasses import dataclass
from typing import Union
    
class BaseResult:
    """Marker base class for all LLM results."""
    pass

@dataclass
class TextResult(BaseResult):
    content: str

@dataclass
class FunctionCallResult(BaseResult):
    name: str
    arguments: dict

LLMResult = BaseResult
