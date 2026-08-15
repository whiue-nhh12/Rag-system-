import logging
from typing import Any, List, Optional

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from .base import BaseVectorDB

logger = logging.getLogger(__name__)


class QdrantVectorDB(BaseVectorDB):
    """Wrapper for interacting with a Qdrant vector database."""

    def __init__(
        self,
        collection_name: str,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding: Optional[Any] = None,
        vector_size: int = 384,
        host: Optional[str] = None,
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        collection_params: Optional[dict] = None,
        timeout: Optional[float] = None,
    ):
        super().__init__(collection_name=collection_name, persist_directory=None)
        self.url = url
        self.api_key = api_key
        self.embedding = embedding
        self.vector_size = vector_size
        self.host = host or "localhost"
        self.port = port
        self.grpc_port = grpc_port
        self.prefer_grpc = prefer_grpc
        self.collection_params = collection_params or {}
        self.timeout = timeout
        self._client = None

    def connect(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                host=self.host,
                port=self.port,
                grpc_port=self.grpc_port,
                prefer_grpc=self.prefer_grpc,
                timeout=self.timeout,
            )
        return self._client

    def _ensure_collection(self, client: QdrantClient) -> None:
        if not client.collection_exists(collection_name=self.collection_name):
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                **self.collection_params,
            )

    def _build_points(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
        metadata: Optional[List[dict]] = None,
    ) -> List[dict]:
        points = []
        for idx, document in enumerate(documents):
            embedding = None
            if embeddings is not None and idx < len(embeddings):
                embedding = embeddings[idx]
            elif self.embedding is not None:
                embedding = self.embedding.embed_documents([document.page_content])[0]

            if embedding is None:
                raise ValueError("Embedding is required for Qdrant insertion")

            doc_metadata = metadata[idx] if metadata and idx < len(metadata) else dict(document.metadata or {})
            doc_metadata.setdefault("source", document.metadata.get("source") if document.metadata else None)
            doc_metadata.setdefault("content", document.page_content)

            points.append(
                {
                    "id": idx,
                    "vector": embedding,
                    "payload": doc_metadata,
                }
            )
        return points

    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
        metadata: Optional[List[dict]] = None,
    ) -> int:
        client = self.get_store()
        self._ensure_collection(client)
        points = self._build_points(documents, embeddings=embeddings, metadata=metadata)
        if not points:
            return 0

        client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Any]:
        client = self.get_store()
        self._ensure_collection(client)

        if self.embedding is None:
            raise ValueError("Embedding function is required for similarity_search")

        query_vector = self.embedding.embed_query(query)
        search_result = client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k,
            query_filter=filter,
        )

        documents: List[Document] = []
        for hit in search_result:
            payload = getattr(hit, "payload", None) or {}
            content = payload.get("content") or payload.get("page_content") or ""
            metadata = dict(payload)
            metadata.pop("content", None)
            metadata.pop("page_content", None)
            documents.append(Document(page_content=content, metadata=metadata))
        return documents

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[tuple[Document, float]]:
        client = self.get_store()
        self._ensure_collection(client)

        if self.embedding is None:
            raise ValueError("Embedding function is required for similarity_search_with_score")

        query_vector = self.embedding.embed_query(query)
        search_result = client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k,
            query_filter=filter,
            with_payload=True,
            with_vectors=False,
        )

        results = []
        for hit in search_result:
            payload = getattr(hit, "payload", None) or {}
            content = payload.get("content") or payload.get("page_content") or ""
            metadata = dict(payload)
            metadata.pop("content", None)
            metadata.pop("page_content", None)
            results.append((Document(page_content=content, metadata=metadata), float(getattr(hit, "score", 0.0))))
        return results
