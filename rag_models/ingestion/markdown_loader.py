import logging
from typing import Dict, List, Optional

from langchain_core.documents import Document

from .base import BaseLoader

logger = logging.getLogger(__name__)


class MarkdownLoader(BaseLoader):
    supported_extensions = {".md", ".markdown"}

    def __init__(
        self, source: str, metadata: Optional[Dict] = None, encoding: str = "utf-8"
    ):
        super().__init__(source, metadata)
        self.encoding = encoding

    def load(self) -> List[Document]:
        logger.info(
            f"Loading markdown file: {self.source} with encoding: {self.encoding}"
        )
        self._assert_extension()
        try:
            text = self.source.read_text(encoding=self.encoding, errors="replace")
            logger.debug(
                f"Successfully read markdown file: {self.source} ({len(text)} characters)"
            )
        except Exception as e:
            logger.error(
                f"Error reading markdown file {self.source}: {type(e).__name__}: {e}"
            )
            raise
        doc = Document(page_content=text, metadata={**self.metadata})
        return self._add_metadata([doc])


def load_markdown(
    source: str, metadata: Optional[Dict] = None, encoding: str = "utf-8"
) -> List[Document]:
    return MarkdownLoader(source, metadata, encoding).load()
