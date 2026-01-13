# core/config.py
from dataclasses import dataclass

@dataclass
class LLMProfile:
    type: str
    model_name: str
    model_path: str | None = None
    api_key: str | None = None
    n_ctx: int = 4096
    n_threads: int = 8
    n_batch: int = 512

