import logging
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class BaseEmbedding(ABC):
    """Base class for document embedding strategies."""

    def __init__(self, model_name: str = None):
        """
        Initialize embedding strategy.

        Args:
            model_name: Name of the embedding model to use.
        """
        self.model_name = model_name
        logger.debug(f"Initialized {self.__class__.__name__} with model: {model_name}")

    @abstractmethod
    def embed_documents(self, documents: List[Document]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of documents.

        Args:
            documents: List of langchain Documents to embed.

        Returns:
            List of embedding vectors (numpy arrays).
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query string.

        Args:
            query: Query text to embed.

        Returns:
            Embedding vector (numpy array).
        """
        pass

    def add_embeddings_to_documents(
        self, documents: List[Document], embeddings: List[np.ndarray]
    ) -> List[Document]:
        """
        Add embedding vectors to document metadata.

        Args:
            documents: List of Documents.
            embeddings: List of corresponding embedding vectors.

        Returns:
            Documents with embeddings added to metadata.
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Number of documents ({len(documents)}) and embeddings ({len(embeddings)}) must match"
            )

        for doc, emb in zip(documents, embeddings):
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata["embedding"] = emb.tolist()

        logger.info(f"Added embeddings to {len(documents)} document(s)")
        return documents
