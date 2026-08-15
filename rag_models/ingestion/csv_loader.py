import csv
import logging
from typing import Dict, List, Optional

from langchain_core.documents import Document

from .base import BaseLoader

logger = logging.getLogger(__name__)


class CSVLoader(BaseLoader):
    supported_extensions = {".csv"}

    def __init__(
        self, source: str, metadata: Optional[Dict] = None, encoding: str = "utf-8"
    ):
        super().__init__(source, metadata)
        self.encoding = encoding

    def load(self) -> List[Document]:
        self._assert_extension()
        logger.info(f"Loading CSV file: {self.source} with encoding: {self.encoding}")
        try:
            with self.source.open(
                mode="r", encoding=self.encoding, errors="replace", newline=""
            ) as csv_file:
                reader = csv.DictReader(csv_file)
                if reader.fieldnames is None:
                    logger.error("CSV file missing header row")
                    raise ValueError("CSV file must contain a header row")
                rows = []
                for row_index, row in enumerate(reader, start=1):
                    row_text = f"Row {row_index}: " + ", ".join(
                        f"{key}={value}" for key, value in row.items()
                    )
                    rows.append(row_text)
        except Exception as e:
            logger.error(f"Error reading CSV {self.source}: {type(e).__name__}: {e}")
            raise

        content = "\n".join(rows)
        doc = Document(
            page_content=content,
            metadata={"columns": reader.fieldnames, **(self.metadata or {})},
        )
        logger.info(
            f"CSV loaded successfully: {self.source} ({len(rows)} rows, columns={reader.fieldnames})"
        )
        return self._add_metadata([doc])


def load_csv(
    source: str, metadata: Optional[Dict] = None, encoding: str = "utf-8"
) -> List[Document]:
    logger = logging.getLogger(__name__)
    logger.info(f"load_csv called with source: {source}")
    return CSVLoader(source, metadata, encoding).load()
