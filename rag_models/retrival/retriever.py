import hashlib
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

from .base import BaseRetriever

TDocument = TypeVar("TDocument")


class Retriever(BaseRetriever, Generic[TDocument]):
    """Simple retriever wrapper around a vector store implementation."""

    def __init__(self, vector_store: Optional[Any] = None):
        super().__init__(vector_store=vector_store)

    def retrieve(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[TDocument]:
        if self.vector_store is None:
            raise ValueError("vector_store is not configured")

        if hasattr(self.vector_store, "similarity_search"):
            return self.vector_store.similarity_search(query=query, k=k, filter=filter)

        if hasattr(self.vector_store, "search"):
            return self.vector_store.search(query=query, k=k, filter=filter)

        raise AttributeError("vector_store does not expose a supported retrieve method")

    def retrieve_with_scores(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[TDocument, float]]:
        if self.vector_store is None:
            raise ValueError("vector_store is not configured")

        if hasattr(self.vector_store, "similarity_search_with_score"):
            return self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter,
            )

        if hasattr(self.vector_store, "search_with_score"):
            return self.vector_store.search_with_score(query=query, k=k, filter=filter)

        documents = self.retrieve(query=query, k=k, filter=filter)
        return [(document, 0.0) for document in documents]

    def get_relevant_documents(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[TDocument]:
        return self.retrieve(query=query, k=k, filter=filter)


class HybridRetriever(BaseRetriever, Generic[TDocument]):
    """Combine semantic and keyword retrieval results into a unified ranking."""

    def __init__(
        self,
        semantic_search_fn: Optional[
            Callable[[str, int, Optional[dict]], List[Tuple[TDocument, float]]]
        ] = None,
        keyword_search_fn: Optional[
            Callable[[str, int, Optional[dict]], List[Tuple[TDocument, float]]]
        ] = None,
        fusion_k: int = 60,
    ):
        super().__init__(vector_store=None)
        self.semantic_search_fn = semantic_search_fn
        self.keyword_search_fn = keyword_search_fn
        self.fusion_k = fusion_k

    def retrieve(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[TDocument]:
        return [
            item[0]
            for item in self.retrieve_with_scores(query=query, k=k, filter=filter)
        ]

    def retrieve_with_scores(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> List[Tuple[TDocument, float]]:
        semantic_results = []
        keyword_results = []

        if self.semantic_search_fn is not None:
            semantic_results = self.semantic_search_fn(query, k, filter)
        if self.keyword_search_fn is not None:
            keyword_results = self.keyword_search_fn(query, k, filter)

        combined: Dict[str, Tuple[TDocument, float]] = {}

        def normalize_id(document: TDocument) -> str:
            page_content = getattr(document, "page_content", None)
            if page_content is None:
                raise ValueError("Document missing page_content for fallback id generation")

            return hashlib.sha256(
                str(page_content).encode("utf-8", errors="ignore")
            ).hexdigest()

        def add_rrf_score(document: TDocument, rank: int) -> None:
            doc_id = normalize_id(document)
            current_score = combined.get(doc_id, (document, 0.0))[1]
            fused_score = 1.0 / (self.fusion_k + rank)
            combined[doc_id] = (document, current_score + fused_score)

        for rank, (document, _) in enumerate(semantic_results, start=1):
            add_rrf_score(document, rank)

        for rank, (document, _) in enumerate(keyword_results, start=1):
            add_rrf_score(document, rank)

        ranked = sorted(combined.values(), key=lambda item: item[1], reverse=True)
        print(
            f"-> HybridRetriever RRF: {len(ranked)} results (semantic: {len(semantic_results)}, keyword: {len(keyword_results)}, fusion_k: {self.fusion_k})"
        )
        return ranked[:k]
