# core/llm/contract/function.py
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class FunctionSpec:
    name: str
    description: str
    parameters: Dict[str, Any]
