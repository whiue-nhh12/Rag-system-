"""
Core exception classes for RAG system
"""


class RAGException(Exception):
    """Base exception for RAG system"""
    pass


class DocumentLoadException(RAGException):
    """Raised when document loading fails"""
    pass


class EmbeddingException(RAGException):
    """Raised when embedding fails"""
    pass


class RetrievalException(RAGException):
    """Raised when retrieval fails"""
    pass


class ChainExecutionException(RAGException):
    """Raised when chain execution fails"""
    pass


class ConfigurationException(RAGException):
    """Raised when configuration is invalid"""
    pass


class VectorStoreException(RAGException):
    """Raised when vector store operations fail"""
    pass
