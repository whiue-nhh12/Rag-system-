import logging
from typing import Dict, List, Optional

from langchain_core.documents import Document

from .base import BaseLoader

try:
    from ..core.exception import LoadImageError, OCRProcessingError
except ImportError:  # pragma: no cover
    from ..core.exception import LoadImageError, OCRProcessingError

logger = logging.getLogger(__name__)

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:  # pragma: no cover
    Image = None
    UnidentifiedImageError = Exception

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from paddleocr import PaddleOCR
except ImportError:  # pragma: no cover
    PaddleOCR = None

_ocr_engine = None


def get_ocr_engine(lang: str = "vi", use_angle_cls: bool = True, use_gpu: bool = False):
    global _ocr_engine
    if PaddleOCR is None:
        raise OCRProcessingError(
            "PaddleOCR is required for OCR. Install paddleocr and paddlepaddle to continue."
        )
    if _ocr_engine is None:
        logger.info("Initializing PaddleOCR engine")
        _ocr_engine = PaddleOCR(lang=lang, use_angle_cls=use_angle_cls, use_gpu=use_gpu)
    return _ocr_engine


def _flatten_paddle_result(result) -> str:
    lines: List[str] = []
    for page in result:
        for line in page:
            if len(line) >= 2 and isinstance(line[1], (list, tuple)):
                text = line[1][0]
            elif len(line) >= 2:
                text = str(line[1])
            else:
                text = str(line)
            lines.append(text)
    return "\n".join(lines).strip()


class ImageOCRLoader(BaseLoader):
    supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

    def load(self) -> List[Document]:
        self._assert_extension()

        if Image is None:
            raise LoadImageError(
                "Pillow is required to load image files. Install pillow to continue."
            )
        if np is None:
            raise LoadImageError(
                "NumPy is required for PaddleOCR. Install numpy to continue."
            )

        try:
            image = Image.open(self.source)
        except (UnidentifiedImageError, OSError) as exc:
            raise LoadImageError(f"Failed to open image: {self.source}. Error: {exc}")

        doc = ocr_image(image, metadata={**(self.metadata or {})})
        return self._add_metadata([doc])


def load_image(source: str, metadata: Optional[Dict] = None) -> List[Document]:
    return ImageOCRLoader(source, metadata).load()


def ocr_image(img, metadata: Optional[Dict] = None) -> Document:
    """Run OCR on a PIL Image object and return a Document."""
    if PaddleOCR is None:
        raise OCRProcessingError(
            "PaddleOCR is required for OCR. Install paddleocr and paddlepaddle to continue."
        )
    if np is None:
        raise OCRProcessingError(
            "NumPy is required for OCR. Install numpy to continue."
        )
    if Image is None:
        raise OCRProcessingError(
            "Pillow is required for OCR. Install pillow to continue."
        )

    try:
        img_arr = np.array(img)
        ocr_engine = get_ocr_engine()
        result = ocr_engine.ocr(img_arr, cls=True)
        text = _flatten_paddle_result(result)
    except Exception as exc:
        raise OCRProcessingError(f"OCR processing failed: {exc}")

    if text is None:
        text = ""

    doc = Document(page_content=text, metadata={**(metadata or {})})
    return doc
