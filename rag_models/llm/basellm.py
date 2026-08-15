from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Base class for all LLM implementations."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name

    @abstractmethod
    def ask(self, query: str) -> str:
        """Send a query and return the generated response."""
        raise NotImplementedError
