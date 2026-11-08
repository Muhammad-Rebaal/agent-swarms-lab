"""Groq Cloud language model client wrapper.

Preserved for backward compatibility, wrapping the unified LLMClient.
"""

from src.llm import LLMClient


class GroqLLM(LLMClient):
    """Groq-specific client subclass inheriting unified retry and rate pacing."""

    def __init__(self, model: str = "llama-3.1-8b-instant", delay: float = 2.0):
        super().__init__(provider="groq", model=model, delay=delay)
