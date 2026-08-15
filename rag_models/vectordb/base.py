import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class BaseVectorDB(ABC):
    """Base abstraction for vector database operations."""

    def __init__(self, collection_name: str, persist_directory: Optional[str] = None):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._store = None

    @abstractmethod
    def connect(self) -> Any:
        """Create or return a connection/store instance."""

    @abstractmethod
    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
        metadata: Optional[List[dict]] = None,
    ) -> int:
        """Add documents and embeddings into the vector DB."""

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Any]:
        """Search documents similar to the query."""

    def get_store(self) -> Any:
        if self._store is None:
            self._store = self.connect()
        return self._store
