import logging
from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class BaseChunking(ABC):
    """Base class for document chunking strategies."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize chunking strategy.

        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of overlapping characters between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.debug(
            f"Initialized {self.__class__.__name__} with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
        )

    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.

        Args:
            documents: List of langchain Documents to chunk.

        Returns:
            List of chunked Documents.
        """
        pass

    def _add_chunk_metadata(
        self, chunks: List[Document], source_doc_idx: int
    ) -> List[Document]:
        """
        Add chunk metadata to track source document and chunk index.

        Args:
            chunks: List of chunked documents.
            source_doc_idx: Index of the source document.

        Returns:
            Documents with added chunk metadata.
        """
        for chunk_idx, chunk in enumerate(chunks):
            if chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata["source_doc_index"] = source_doc_idx
            chunk.metadata["chunk_index"] = chunk_idx
        return chunks
