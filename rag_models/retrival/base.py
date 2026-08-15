import logging
from abc import ABC, abstractmethod
from typing import Any, Generic, List, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

TDocument = TypeVar("TDocument")


class BaseRetriever(ABC,Generic[TDocument]):
    """Base abstraction for document retrieval."""

    def __init__(self, vector_store: Optional[Any] = None):
        self.vector_store = vector_store

    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[TDocument]:
        """Retrieve relevant documents for a query."""

    def retrieve_with_scores(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[TDocument, float]]:
        """Retrieve documents together with similarity scores when available."""
        return []

    def bind(self, vector_store: Any) -> "BaseRetriever":
        """Attach a vector store instance to the retriever."""
        self.vector_store = vector_store
        return self
