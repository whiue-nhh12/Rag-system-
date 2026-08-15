import logging
from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma

from .base import BaseVectorDB

logger = logging.getLogger(__name__)


class ChromaVectorDB(BaseVectorDB):
    """Wrapper for interacting with Chroma vector database."""

    def __init__(
        self,
        collection_name: str,
        persist_directory: Optional[str] = None,
        embedding: Optional[Any] = None,
        collection_metadata: Optional[dict] = None,
    ):
        super().__init__(
            collection_name=collection_name, persist_directory=persist_directory
        )
        self.embedding = embedding
        self.collection_metadata = collection_metadata or {"hnsw:space": "cosine"}

    def _get_collection(self) -> Any:
        store = self.get_store()
        return getattr(store, "_collection", None) or getattr(store, "collection", None)

    def _get_existing_document_ids(self, identity_key: str) -> dict:
        collection = self._get_collection()
        if collection is None or not hasattr(collection, "get"):
            return {}

        try:
            result = collection.get(include=["metadatas"])
        except Exception as exc:
            logger.warning(
                f"Unable to inspect collection for existing documents: {exc}"
            )
            return {}

        ids = result.get("ids", []) or []
        metadatas = result.get("metadatas", []) or []
        existing_map = {}

        for doc_id, metadata in zip(ids, metadatas):
            if not isinstance(metadata, dict):
                continue
            identity_value = metadata.get(identity_key)
            if identity_value is None:
                continue
            existing_map.setdefault(str(identity_value), []).append(str(doc_id))

        return existing_map

    def _get_document_identity(
        self, document: Document, identity_key: str
    ) -> Optional[str]:
        metadata = document.metadata or {}
        for key in (identity_key, "id", "source"):
            value = metadata.get(key)
            if value is not None:
                return str(value)
        return None

    def connect(self) -> Chroma:
        if self._store is None:
            self._store = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                embedding_function=self.embedding,
                collection_metadata=self.collection_metadata,
            )
        return self._store

    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
        metadata: Optional[List[dict]] = None,
    ) -> int:
        store = self.get_store()
        if hasattr(store, "from_documents") and self._store is None:
            store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                collection_metadata=self.collection_metadata,
            )
            self._store = store
            return len(documents)

        if embeddings is not None:
            return store.add_documents(
                documents=documents, embeddings=embeddings, metadata=metadata
            )

        return store.add_documents(documents=documents, metadata=metadata)

    def upsert_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
        metadata: Optional[List[dict]] = None,
        identity_key: str = "source",
        replace_existing: bool = False,
        reembed_if_missing: bool = True,
    ) -> List[dict]:
        """
        Add documents to the vector store or replace existing ones based on a stable identity field.

        Returns a list of decisions for each document: add, skip, or replace.
        """
        store = self.get_store()
        existing_map = self._get_existing_document_ids(identity_key)
        decisions = []

        for index, document in enumerate(documents):
            doc_metadata = (
                metadata[index]
                if metadata and index < len(metadata)
                else document.metadata or {}
            )
            identity_value = self._get_document_identity(document, identity_key)
            existing_ids = (
                existing_map.get(str(identity_value), []) if identity_value else []
            )

            if existing_ids:
                if replace_existing:
                    if hasattr(store, "delete"):
                        try:
                            store.delete(ids=existing_ids)
                        except Exception as exc:
                            logger.warning(
                                f"Failed to replace existing document(s): {exc}"
                            )
                    else:
                        logger.warning(
                            "Store does not support delete(); replace will be skipped"
                        )
                    action = "replace"
                    needs_embedding = True
                else:
                    action = "skip"
                    needs_embedding = False
            else:
                action = "add"
                needs_embedding = reembed_if_missing or embeddings is not None

            decisions.append(
                {
                    "action": action,
                    "identity": identity_value,
                    "needs_embedding": needs_embedding,
                    "document": document,
                }
            )

            if action == "skip":
                continue

            if embeddings is not None and index < len(embeddings):
                store.add_documents(
                    documents=[document],
                    embeddings=[embeddings[index]],
                    metadata=[doc_metadata],
                )
            else:
                store.add_documents(documents=[document], metadata=[doc_metadata])

        return decisions

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Any]:
        store = self.get_store()
        return store.similarity_search(query=query, k=k, filter=filter)
