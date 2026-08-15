import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import BaseChunking

logger = logging.getLogger(__name__)


class TextChunker(BaseChunking):
    """Chunk documents using recursive character splitting strategy."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None,
    ):
        """
        Initialize TextChunker with recursive character splitting.

        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of overlapping characters between chunks.
            separators: List of separators to split on. Defaults to common text separators.
        """
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or ["\n\n", "\n", "." ," ", ""]
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
        )
        logger.debug(f"Initialized TextChunker with separators: {self.separators}")

    def chunk(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into text chunks using recursive character splitting.

        Args:
            documents: List of langchain Documents to chunk.

        Returns:
            List of chunked Documents with source and chunk index metadata.
        """
        logger.info(f"Chunking {len(documents)} document(s) with TextChunker")
        all_chunks: List[Document] = []

        for doc_idx, doc in enumerate(documents):
            try:
                logger.debug(
                    f"Chunking document {doc_idx}: {doc.metadata.get('source', 'unknown')}"
                )
                chunks = self.splitter.split_documents([doc])
                logger.debug(f"Document {doc_idx} split into {len(chunks)} chunk(s)")

                # Add chunk metadata
                chunks = self._add_chunk_metadata(chunks, doc_idx)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(
                    f"Error chunking document {doc_idx}: {type(e).__name__}: {e}"
                )
                raise

        logger.info(
            f"Completed chunking: {len(documents)} document(s) -> {len(all_chunks)} chunk(s)"
        )
        return all_chunks


def chunk_text(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Convenience function to chunk documents using TextChunker.

    Args:
        documents: List of langchain Documents to chunk.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of chunked Documents.
    """
    logger.info(f"chunk_text called: {len(documents)} document(s)")
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk(documents)
