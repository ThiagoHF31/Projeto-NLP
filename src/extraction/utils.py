"""Funções auxiliares compartilhadas entre os módulos de extração."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import BBOX_TOLERANCE, MIN_TEXT_LENGTH, TABLE_EMPTY_CELL_THRESHOLD


def is_valid_text(text: str) -> bool:
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return False
    printable_ratio = sum(c.isprintable() for c in text) / len(text)
    return printable_ratio > 0.8


def sanitize_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def df_to_records(df: pd.DataFrame) -> list[dict]:
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    return df.to_dict(orient="records")


def table_is_real(cells: list[list]) -> bool:
    """
    Descarta detecções espúrias: tabelas onde a maioria das células é vazia
    são layouts de página, não dados tabulares.
    """
    if not cells:
        return False
    flat = [c for row in cells for c in row]
    if not flat:
        return False
    empty = sum(1 for c in flat if c is None or str(c).strip() == "")
    return (empty / len(flat)) <= TABLE_EMPTY_CELL_THRESHOLD


def cells_to_records(cells: list[list]) -> list[dict]:
    """Converte matriz de células (pdfplumber) em lista de dicts."""
    if not cells:
        return []
    header = [
        str(c).strip() if (c and str(c).strip()) else f"col_{i}"
        for i, c in enumerate(cells[0])
    ]
    records = []
    for row in cells[1:]:
        if not any(c for c in row):
            continue
        record = {header[i]: (row[i] if i < len(row) else None) for i in range(len(header))}
        records.append(record)
    return records


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def obj_inside_bbox(obj: dict, bbox: tuple, tol: float = BBOX_TOLERANCE) -> bool:
    """Verifica se um objeto pdfplumber está dentro de um bounding box."""
    x0, top, x1, bottom = bbox
    return (
        obj.get("x0", 0) >= x0 - tol
        and obj.get("x1", 0) <= x1 + tol
        and obj.get("top", 0) >= top - tol
        and obj.get("bottom", 0) <= bottom + tol
    )
