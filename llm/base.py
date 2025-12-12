from abc import ABC, abstractmethod
from typing import Iterator
from .config import LLMConfig

class LLMClient(ABC):
    """
    Abstract Base Class for all LLM clients.
    Implementations must provide both:
      - generate(messages): full response
      - stream(messages): incremental chunks

    messages format:
    [
        {"role": "system", "content": "You are ..."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"}
    ]
    """

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def generate(self, messages: [dict[str, str]]) -> dict:
        """
        Send a list of messages to the model and return a full response.
        """
        raise NotImplementedError

    @abstractmethod
    def stream(self, messages: [dict[str, str]]) -> Iterator[dict]:
        """
        Stream both reasoning tokens and final assistant output.

        Yields events shaped like: {"type": "reasoning", "token": "..."}
        or
        { "type": "content", "token": "..."}
        """
        raise NotImplementedError
