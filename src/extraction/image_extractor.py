"""
Extração de imagens embutidas em PDFs.

Hierarquia de tentativas:
  1. PyMuPDF — extrai imagens embutidas individualmente (preferível)
  2. pdf2image — renderiza cada página inteira como PNG (fallback)
"""

from __future__ import annotations

import logging
from pathlib import Path

from .config import IMAGE_DPI, IMAGE_OUTPUT_DIR
from .models import ContentBlock
from .utils import ensure_dir

logger = logging.getLogger(__name__)


def _extract_images_pymupdf(file_path: str, output_dir: Path) -> list[ContentBlock]:
    import fitz
    blocks: list[ContentBlock] = []
    with fitz.open(file_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            for img_idx, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                try:
                    base_img = doc.extract_image(xref)
                    ext = base_img.get("ext", "png")
                    out_path = output_dir / f"p{page_num}_img{img_idx}.{ext}"
                    out_path.write_bytes(base_img["image"])
                    blocks.append(ContentBlock(
                        type="image",
                        page=page_num,
                        image_path=str(out_path),
                        source="pymupdf",
                    ))
                except Exception as exc:
                    logger.debug("Imagem xref=%s pág %d falhou: %s", xref, page_num, exc)
    return blocks


def _extract_images_pdf2image(file_path: str, output_dir: Path) -> list[ContentBlock]:
    from pdf2image import convert_from_path
    blocks: list[ContentBlock] = []
    pages = convert_from_path(file_path, dpi=IMAGE_DPI)
    for page_num, img in enumerate(pages, start=1):
        out_path = output_dir / f"page_{page_num}.png"
        img.save(str(out_path), "PNG")
        blocks.append(ContentBlock(
            type="image",
            page=page_num,
            image_path=str(out_path),
            source="pdf2image",
        ))
    return blocks


def extract_images(file_path: str, log: dict) -> list[ContentBlock]:
    stem = Path(file_path).stem
    output_dir = ensure_dir(Path(IMAGE_OUTPUT_DIR) / stem)

    try:
        blocks = _extract_images_pymupdf(file_path, output_dir)
        if blocks:
            log["images"] = f"PyMuPDF ({len(blocks)} imagens embutidas)"
            return blocks
        logger.info("PyMuPDF: sem imagens embutidas em %s", Path(file_path).name)
    except Exception as exc:
        logger.warning("Imagens — PyMuPDF falhou: %s", exc)

    try:
        blocks = _extract_images_pdf2image(file_path, output_dir)
        log["images"] = f"pdf2image ({len(blocks)} páginas renderizadas)"
        return blocks
    except Exception as exc:
        logger.warning("Imagens — pdf2image falhou (Poppler instalado?): %s", exc)

    log["images"] = "nenhuma imagem extraída"
    return []
