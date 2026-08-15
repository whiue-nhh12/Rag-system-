import logging
from typing import Dict, List, Optional
from io import BytesIO

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

from ..pre_process.normalize_text import BasePreProcess
from .base import BaseLoader
from .image_loader import ocr_image

logger = logging.getLogger(__name__)


class PDFLoader(BaseLoader):
    supported_extensions = {".pdf"}

    def _ocr_pdf(self) -> List[Document]:
        # Render each PDF page to an image and reuse the image OCR helper
        try:
            import fitz
            from PIL import Image
        except Exception:
            logger.error("OCR dependencies not available (fitz/Pillow). Install pymupdf and pillow to enable OCR fallback.")
            raise RuntimeError("OCR dependencies missing")

        logger.info(f"Starting OCR fallback for PDF: {self.source}")
        docs: List[Document] = []
        try:
            doc = fitz.open(str(self.source))
            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.tobytes(output="png")
                img = Image.open(BytesIO(img_bytes))
                metadata = {**(self.metadata or {}), "source": str(self.source), "page": i}
                try:
                    doc_item = ocr_image(img, metadata=metadata)
                except Exception as e:
                    logger.error(f"OCR failed on page {i} of {self.source}: {e}")
                    doc_item = Document(page_content="", metadata=metadata)
                docs.append(doc_item)
            logger.info(f"OCR extracted text from {len(docs)} page(s) in PDF: {self.source}")
            return docs
        except Exception as e:
            logger.error(f"OCR fallback failed for {self.source}: {type(e).__name__}: {e}")
            raise

    def load(self) -> List[Document]:
        logger.info(f"Loading PDF file: {self.source}")
        self._assert_extension()
        try:
            loader = PyMuPDFLoader(file_path=str(self.source), mode="page")
            logger.debug(f"PyMuPDFLoader initialized for: {self.source}")
            docs = loader.load()
            logger.info(f"Successfully loaded {len(docs)} page(s) from PDF: {self.source}")

            combined = "".join((d.page_content or "") for d in docs).strip()
            if not combined:
                logger.warning(f"PDF {self.source} produced empty text from parser; attempting OCR fallback")
                try:
                    ocr_docs = self._ocr_pdf()
                    return self._add_metadata(ocr_docs)
                except Exception:
                    logger.exception("OCR fallback also failed; returning parsed (possibly empty) docs")

            cleaned_docs: list[Document] = docs

            return self._add_metadata(cleaned_docs)
        except Exception as e:
            logger.error(f"Error loading PDF file {self.source}: {type(e).__name__}: {e}")
            err_msg = str(e).lower()
            mapping_issue = isinstance(e, UnicodeDecodeError) or "mapping" in err_msg or "glyph" in err_msg
            if mapping_issue:
                logger.warning(f"Detected mapping/encoding-like error for {self.source}; attempting OCR fallback")
                try:
                    ocr_docs = self._ocr_pdf()
                    return self._add_metadata(ocr_docs)
                except Exception:
                    logger.exception("OCR fallback failed after mapping error")
            raise


def load_pdf(source: str, metadata: Optional[Dict] = None) -> List[Document]:
    logger.info(f"load_pdf called with source: {source}")
    return PDFLoader(source, metadata).load()
