"""Ingestion package for LangChain RAG document loaders."""

from typing import Dict, List
from pathlib import Path
from .base import BaseLoader
from .csv_loader import CSVLoader, load_csv
from .image_loader import ImageOCRLoader, load_image
from .markdown_loader import MarkdownLoader, load_markdown
from .pdf_loader import PDFLoader, load_pdf
from .text_loader import TextLoader, load_text
from langchain_core.documents import Document

__all__ = [
    "BaseLoader",
    "CSVLoader",
    "ImageOCRLoader",
    "MarkdownLoader",
    "PDFLoader",
    "TextLoader",
    "load_csv",
    "load_image",
    "load_markdown",
    "load_pdf",
    "load_text",
    "load_document",
]

_EXTENSION_TO_LOADER = {
    ".pdf": load_pdf,
    ".txt": load_text,
    ".text": load_text,
    ".log": load_text,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".csv": load_csv,
    ".png": load_image,
    ".jpg": load_image,
    ".jpeg": load_image,
    ".bmp": load_image,
    ".tiff": load_image,
    ".tif": load_image,
    ".webp": load_image,
}


def load_document(source: str, metadata: Dict = None) -> List["Document"]:
    """Load a single document from a supported source file."""
    source_path = Path(source)
    extension = source_path.suffix.lower()
    loader = _EXTENSION_TO_LOADER.get(extension)
    if loader is None:
        supported = sorted(_EXTENSION_TO_LOADER.keys())
        raise ValueError(
            f"Unsupported file extension {extension}. Supported: {supported}"
        )
    return loader(source, metadata)
