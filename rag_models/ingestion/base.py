import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

from langchain_core.documents import Document

try:
    from ..core.exception import ContentTooLargeError, FileNotFound
except ImportError:  # pragma: no cover
    from ..core.exception import ContentTooLargeError, FileNotFound

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONTENT_SIZE = 20_000_000  # 20 MB


class BaseLoader:
    """Base class for file ingestion loaders."""

    supported_extensions: Set[str] = set()

    def __init__(
        self,
        source: str,
        metadata: Optional[Dict] = None,
        max_content_size: int = DEFAULT_MAX_CONTENT_SIZE,
    ):
        self.source = Path(source)
        self.metadata = metadata or {}
        self.max_content_size = max_content_size
        self._validate_source()

    def _validate_source(self) -> None:
        logger.debug(f"Validating source: {self.source}")
        if not self.source.exists():
            logger.error(f"File not found: {self.source}")
            raise FileNotFound(f"File not found: {self.source}")
        if self.source.is_dir():
            logger.error(f"Expected a file but got a directory: {self.source}")
            raise FileNotFound(f"Expected a file but got a directory: {self.source}")
        file_size = self.source.stat().st_size
        if file_size > self.max_content_size:
            logger.error(f"File size {file_size} exceeds limit {self.max_content_size}: {self.source}")
            raise ContentTooLargeError(
                f"File is larger than {self.max_content_size} bytes: {self.source}"
            )
        logger.info(f"Source validated successfully: {self.source} (size: {file_size} bytes)")

    def _assert_extension(self, supported: Optional[Set[str]] = None) -> str:
        supported = supported or self.supported_extensions
        ext = self.source.suffix.lower()
        logger.debug(f"Checking extension: {ext} against supported: {supported}")
        if ext not in supported:
            logger.error(f"Unsupported file extension: {ext}. Supported: {supported}")
            raise ValueError(
                f"Unsupported file extension: {ext}. Supported: {supported}"
            )
        return ext

    def _add_metadata(self, docs: List[Document]) -> List[Document]:
        logger.debug(f"Adding metadata to {len(docs)} document(s)")
        for doc in docs:
            combined = {**doc.metadata, **self.metadata, "source": str(self.source)}
            doc.metadata = combined
        logger.info(f"Metadata added successfully to {len(docs)} document(s)")
        return docs

    def load(self) -> List[Document]:
        raise NotImplementedError("Subclasses must implement load()")
