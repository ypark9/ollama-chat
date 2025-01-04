from pydantic import BaseModel
from typing import Optional

class OllamaConfig(BaseModel):
    """Configuration for Ollama LLM"""
    model: str = "llama2"  # default model
    temperature: float = 0.0
    format: str = "json"
    base_url: str = "http://localhost:11434"
    timeout: Optional[int] = None
    context_window: Optional[int] = None 