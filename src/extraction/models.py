"""Estruturas de dados de retorno da extração."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContentBlock:
    """
    Unidade atômica de conteúdo extraído, em ordem de leitura.
    type: "text" | "table" | "image"
    """
    type: str
    page: int
    bbox: list[float] = field(default_factory=list)
    content: str = ""
    data: list[dict] = field(default_factory=list)
    image_path: str = ""
    source: str = ""

    def to_dict(self) -> dict:
        base = {"type": self.type, "page": self.page, "source": self.source}
        if self.bbox:
            base["bbox"] = self.bbox
        if self.type == "text":
            base["content"] = self.content
        elif self.type == "table":
            base["data"] = self.data
        elif self.type == "image":
            base["image_path"] = self.image_path
        return base


@dataclass
class ExtractionResult:
    """Resultado completo da extração de um PDF."""
    file_path: str
    metadata: dict[str, Any] = field(default_factory=dict)
    content_blocks: list[ContentBlock] = field(default_factory=list)
    images_paths: list[str] = field(default_factory=list)
    extraction_log: dict[str, str] = field(default_factory=dict)
    success: bool = True
    error: str = ""

    @property
    def full_text(self) -> str:
        """Concatenação dos blocos de texto em ordem de leitura (sem tabelas)."""
        return "\n\n".join(
            b.content for b in self.content_blocks if b.type == "text" and b.content
        )

    @property
    def tables(self) -> list[dict]:
        """Lista de tabelas no formato {page, bbox, data}."""
        return [
            {"page": b.page, "bbox": b.bbox, "source": b.source, "data": b.data}
            for b in self.content_blocks
            if b.type == "table"
        ]

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "metadata": self.metadata,
            "content_blocks": [b.to_dict() for b in self.content_blocks],
            "full_text": self.full_text,
            "tables": self.tables,
            "images_paths": self.images_paths,
            "extraction_log": self.extraction_log,
            "success": self.success,
            "error": self.error,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
