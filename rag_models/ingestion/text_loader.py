import logging
from typing import Dict, List, Optional

from langchain_core.documents import Document

from .base import BaseLoader

logger = logging.getLogger(__name__)


class TextLoader(BaseLoader):
    supported_extensions = {".txt", ".log", ".text"}

    def __init__(
        self, source: str, metadata: Optional[Dict] = None, encoding: str = "utf-8"
    ):
        super().__init__(source, metadata)
        self.encoding = encoding

    def load(self) -> List[Document]:
        logger.info(f"Loading text file: {self.source} with encoding: {self.encoding}")
        self._assert_extension()
        try:
            text = self.source.read_text(encoding=self.encoding, errors="replace")
            logger.debug(f"Successfully read text file: {self.source} ({len(text)} characters)")
        except Exception as e:
            logger.error(f"Error reading text file {self.source}: {e}")
            raise
        doc = Document(page_content=text, metadata={**self.metadata})
        logger.debug(f"Created document from {self.source}")
        return self._add_metadata([doc])


def load_text(
    source: str, metadata: Optional[Dict] = None, encoding: str = "utf-8"
) -> List[Document]:
    logger.info(f"load_text called with source: {source}")
    return TextLoader(source, metadata, encoding).load()
