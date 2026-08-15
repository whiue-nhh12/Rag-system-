import logging
from typing import List

import numpy as np
from langchain_core.documents import Document

from .base import BaseEmbedding

logger = logging.getLogger(__name__)

# Singleton model instance
_model_instance = None


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Get or create singleton SentenceTransformer model instance.

    Args:
        model_name: HuggingFace model identifier for sentence-transformers.

    Returns:
        SentenceTransformer model instance.
    """
    global _model_instance
    if _model_instance is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install it with: pip install sentence-transformers"
            )

        logger.info(f"Loading SentenceTransformer model: {model_name}")
        _model_instance = SentenceTransformer(model_name)
        logger.info("Model loaded successfully")
    return _model_instance


class SentenceTransformerEmbedder(BaseEmbedding):
    """Embed documents and queries using sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize SentenceTransformerEmbedder.

        Args:
            model_name: HuggingFace model identifier (e.g., "sentence-transformers/all-MiniLM-L6-v2").
        """
        super().__init__(model_name)
        self.model = get_embedding_model(model_name)
        logger.info(f"SentenceTransformerEmbedder initialized with model: {model_name}")

    def embed_documents(self, documents: List[Document]) -> List[np.ndarray]:
        """
        Generate embeddings for documents.

        Args:
            documents: List of Documents to embed.

        Returns:
            List of embedding vectors.
        """
        logger.info(f"Embedding {len(documents)} document(s)")
        try:
            texts = [doc.page_content for doc in documents]
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            logger.debug(f"Generated embeddings with shape: {embeddings.shape}")
            return [np.array(emb) for emb in embeddings]
        except Exception as e:
            logger.error(f"Error embedding documents: {type(e).__name__}: {e}")
            raise

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query.

        Args:
            query: Query text to embed.

        Returns:
            Embedding vector.
        """
        logger.debug(
            f"Embedding query: {query[:50]}..."
            if len(query) > 50
            else f"Embedding query: {query}"
        )
        try:
            embedding = self.model.encode(query, convert_to_numpy=True)
            logger.debug(f"Query embedding shape: {embedding.shape}")
            return np.array(embedding)
        except Exception as e:
            logger.error(f"Error embedding query: {type(e).__name__}: {e}")
            raise


def embed_documents(
    documents: List[Document],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[Document]:
    """
    Convenience function to embed documents and add embeddings to metadata.

    Args:
        documents: List of Documents to embed.
        model_name: SentenceTransformer model identifier.

    Returns:
        Documents with embeddings added to metadata.
    """
    logger.info(f"embed_documents called: {len(documents)} document(s)")
    embedder = SentenceTransformerEmbedder(model_name)
    embeddings = embedder.embed_documents(documents)
    return embedder.add_embeddings_to_documents(documents, embeddings)


def embed_query(
    query: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Convenience function to embed a query.

    Args:
        query: Query text to embed.
        model_name: SentenceTransformer model identifier.

    Returns:
        Embedding vector.
    """
    logger.info("embed_query called")
    embedder = SentenceTransformerEmbedder(model_name)
    return embedder.embed_query(query)
