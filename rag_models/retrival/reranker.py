import logging
from typing import List, Tuple, Optional
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class RerankerSystem:
    """
    Singleton-based reranker system using CrossEncoder models.
    Re-ranks retrieved documents based on relevance to the query.
    """
    
    _instance: Optional["RerankerSystem"] = None
    _initialized: bool = False
    
    def __new__(cls, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> "RerankerSystem":
        """Implement singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super(RerankerSystem, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the reranker system.
        
        Args:
            model_name: Name of the CrossEncoder model to use.
                       Defaults to multilingual mini model.
        """
        # Only initialize once due to singleton pattern
        if RerankerSystem._initialized:
            return
        
        self.model_name = model_name
        self.model: Optional[CrossEncoder] = None
        self._load_model()
        RerankerSystem._initialized = True
        logger.info(f"RerankerSystem initialized with model: {model_name}")
    
    def _load_model(self) -> None:
        """Load the CrossEncoder model."""
        try:
            logger.info(f"Loading CrossEncoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info("CrossEncoder model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model: {e}")
            raise
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        batch_size: int = 128
    ) -> List[Tuple[str, float]]:
        """
        Re-rank documents based on relevance to the query.
        
        Args:
            query: The search query string.
            documents: List of document texts to re-rank.
            top_k: Number of top results to return. If None, returns all.
            batch_size: Batch size for scoring (default: 128).
        
        Returns:
            List of tuples (document, relevance_score) sorted by score descending.
        
        Raises:
            ValueError: If documents list is empty or query is empty.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        try:
            # Prepare query-document pairs
            query_doc_pairs = [[query, doc] for doc in documents]
            
            # Score all documents
            logger.debug(f"Scoring {len(documents)} documents for query: {query[:50]}...")
            scores = self.model.predict(query_doc_pairs, batch_size=batch_size , show_progress_bar=True)
            
            # Combine documents with scores and sort
            ranked_results = list(zip(documents, scores))
            ranked_results.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k if specified
            if top_k is not None:
                ranked_results = ranked_results[:top_k]
            else:
                ranked_results = ranked_results[:10]
            logger.debug(f"Reranking complete. Returned {len(ranked_results)} results")
            return ranked_results
        
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            raise
    
    def rerank_with_metadata(
        self,
        query: str,
        documents: List[dict],
        doc_text_key: str = "content",
        top_k: Optional[int] = None,
        batch_size: int = 128
    ) -> List[Tuple[dict, float]]:
        """
        Re-rank documents with metadata preservation.
        
        Args:
            query: The search query string.
            documents: List of document dictionaries with metadata.
            doc_text_key: Key in document dict containing the text content.
            top_k: Number of top results to return. If None, returns all.
            batch_size: Batch size for scoring (default: 128).
        
        Returns:
            List of tuples (document_dict, relevance_score) sorted by score descending.
        
        Raises:
            ValueError: If documents list is empty or query is empty.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        try:
            # Extract text from documents
            doc_texts = [doc.get(doc_text_key, "") for doc in documents]
            
            # Prepare query-document pairs
            query_doc_pairs = [[query, text] for text in doc_texts]
            
            # Score all documents
            logger.debug(f"Scoring {len(documents)} documents for query: {query[:50]}...")
            scores = self.model.predict(query_doc_pairs, batch_size=batch_size)
            
            # Combine documents with scores and sort
            ranked_results = list(zip(documents, scores))
            ranked_results.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k if specified
            if top_k is not None:
                ranked_results = ranked_results[:top_k]
            else:
                ranked_results = ranked_results[:10]
            
            logger.debug(f"Reranking complete. Returned {len(ranked_results)} results")
            return ranked_results
        
        except Exception as e:
            logger.error(f"Error during reranking with metadata: {e}")
            raise
    
    @classmethod
    def reset_singleton(cls) -> None:
        """Reset singleton instance (useful for testing)."""
        cls._instance = None
        cls._initialized = False
        logger.info("RerankerSystem singleton has been reset")
    
    @classmethod
    def get_instance(cls, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> "RerankerSystem":
        """
        Get or create the singleton instance.
        
        Args:
            model_name: Model name (only used on first call).
        
        Returns:
            The singleton RerankerSystem instance.
        """
        if cls._instance is None:
            cls(model_name)
        return cls._instance
