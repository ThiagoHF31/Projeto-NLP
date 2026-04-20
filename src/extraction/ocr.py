"""
Fallback OCR para PDFs escaneados (sem texto nativo).

Hierarquia de tentativas por página:
  1. Tesseract (pytesseract) — rápido, requer instalação nativa
  2. EasyOCR — mais robusto, sem dependência nativa adicional
"""

from __future__ import annotations

import logging

from .config import IMAGE_DPI, OCR_LANG_EASYOCR, OCR_LANG_TESSERACT
from .models import ContentBlock
from .utils import is_valid_text, sanitize_text

logger = logging.getLogger(__name__)


def _ocr_page_tesseract(img) -> str:
    import pytesseract
    return pytesseract.image_to_string(img, lang=OCR_LANG_TESSERACT)


def _ocr_page_easyocr(img) -> str:
    import easyocr
    import numpy as np
    reader = easyocr.Reader(OCR_LANG_EASYOCR, gpu=False, verbose=False)
    result = reader.readtext(np.array(img), detail=0)
    return " ".join(result)


def _render_pages_to_images(file_path: str) -> list:
    from pdf2image import convert_from_path
    return convert_from_path(file_path, dpi=IMAGE_DPI)


def extract_blocks_ocr(file_path: str, page_num_offset: int = 1) -> list[ContentBlock]:
    """Extrai texto via OCR quando nenhum texto nativo foi encontrado."""
    blocks: list[ContentBlock] = []
    try:
        images = _render_pages_to_images(file_path)
    except Exception as exc:
        logger.warning("OCR — pdf2image falhou (Poppler instalado?): %s", exc)
        return blocks

    for i, img in enumerate(images):
        page_num = page_num_offset + i
        text = ""
        source = "ocr-failed"

        try:
            text = _ocr_page_tesseract(img)
            source = "tesseract-ocr"
        except Exception as exc:
            logger.debug("Tesseract pág %d falhou: %s", page_num, exc)

        if not is_valid_text(text):
            try:
                text = _ocr_page_easyocr(img)
                source = "easyocr"
            except Exception as exc:
                logger.debug("EasyOCR pág %d falhou: %s", page_num, exc)
                source = "ocr-failed"

        text = sanitize_text(text)
        if text:
            blocks.append(ContentBlock(
                type="text",
                page=page_num,
                content=text,
                source=source,
            ))
    return blocks
