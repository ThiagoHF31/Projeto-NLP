"""
Extração de texto e tabelas página a página via pdfplumber.

Estratégia por página:
  1. Detecta tabelas com bordas (lattice) via find_tables().
  2. Descarta tabelas espúrias (>80% células vazias).
  3. Extrai texto FORA das bboxes de tabelas válidas usando page.filter().
  4. Ordena todos os blocos pelo topo (Y) para manter leitura natural.

Por que não usar stream mode aqui:
  Stream mode trata qualquer coluna de texto como tabela, causando falsos
  positivos em textos normais com recuo ou numeração. Só é ativado no
  fallback camelot/tabula quando lattice falha completamente.
"""

from __future__ import annotations

import logging

from .models import ContentBlock
from .utils import cells_to_records, obj_inside_bbox, sanitize_text, table_is_real

logger = logging.getLogger(__name__)


def extract_blocks_for_page(plumber_page, page_num: int) -> list[ContentBlock]:
    """Extrai blocos de texto e tabelas de uma única página pdfplumber."""
    blocks: list[ContentBlock] = []

    # ── Passo 1: detectar tabelas por linhas vetoriais (lattice)
    try:
        raw_tables = plumber_page.find_tables()
    except Exception as exc:
        logger.debug("find_tables pág %d falhou: %s", page_num, exc)
        raw_tables = []

    valid_table_bboxes: list[tuple] = []
    for tbl in raw_tables:
        try:
            cells = tbl.extract()
        except Exception:
            continue
        if not table_is_real(cells):
            logger.debug(
                "Pág %d: tabela bbox=%s descartada (>80%% vazia)", page_num, tbl.bbox
            )
            continue
        records = cells_to_records(cells)
        if records:
            blocks.append(ContentBlock(
                type="table",
                page=page_num,
                bbox=list(tbl.bbox),
                data=records,
                source="pdfplumber-lattice",
            ))
            valid_table_bboxes.append(tbl.bbox)

    # ── Passo 2: extrair texto FORA das bboxes de tabelas válidas
    if valid_table_bboxes:
        def outside_all_tables(obj):
            return not any(obj_inside_bbox(obj, bb) for bb in valid_table_bboxes)
        try:
            text = plumber_page.filter(outside_all_tables).extract_text() or ""
        except Exception as exc:
            logger.debug("filter pág %d falhou, usando extract_text: %s", page_num, exc)
            text = plumber_page.extract_text() or ""
    else:
        text = plumber_page.extract_text() or ""

    text = sanitize_text(text)
    if text:
        blocks.append(ContentBlock(
            type="text",
            page=page_num,
            content=text,
            source="pdfplumber",
        ))

    # ── Passo 3: ordenar por posição vertical
    def _sort_key(b: ContentBlock) -> float:
        return b.bbox[1] if b.bbox else 0.0

    blocks.sort(key=_sort_key)
    return blocks
