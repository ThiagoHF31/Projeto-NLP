"""Extração de metadados de arquivos PDF."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def extract_metadata(file_path: str) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "file_name": Path(file_path).name,
        "file_size_kb": round(Path(file_path).stat().st_size / 1024, 2),
        "num_pages": 0,
    }
    try:
        import fitz
        with fitz.open(file_path) as doc:
            meta["num_pages"] = doc.page_count
            raw = doc.metadata or {}
            meta.update({k: v for k, v in raw.items() if v})
    except Exception as exc:
        logger.warning("Metadados — PyMuPDF falhou: %s", exc)
    return meta
