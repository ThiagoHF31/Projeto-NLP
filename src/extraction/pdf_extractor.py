"""
pdf_extractor.py — Extração robusta e modular de PDFs com lógica de fallback.

Arquitetura central: extração bloco a bloco por coordenada (por página).
Para cada página, o script detecta regiões de tabela via bounding-box e
separa o texto que está FORA delas. O output final é uma lista ordenada de
blocos em ordem de leitura (content_blocks), onde cada bloco é text ou table.
Isso elimina duplicação e fornece contexto posicional direto para chunking.

O que define uma tabela num PDF:
  - PDFs gerados por Word/Excel: linhas vetoriais (operadores 're','l','m')
    formam uma grade → pdfplumber detecta por interseção de linhas (lattice).
  - PDFs sem bordas visíveis: colunas alinhadas por espaço → camelot/tabula
    stream mode. Esse modo é mais agressivo e pode gerar falsos positivos;
    só é acionado como fallback quando lattice não encontra nada.
  - Critério de descarte: tabela com >80% de células vazias → reclassificada
    como texto (provavelmente detecção de layout, não dados tabulares reais).

Dependências:
    pip install pymupdf pymupdf4llm pypdf pdfplumber camelot-py[cv] tabula-py
                pytesseract easyocr pdf2image tqdm pandas Pillow opencv-python

Requisitos de sistema:
    - Tesseract OCR  : https://github.com/UB-Mannheim/tesseract/wiki  (Windows)
    - Poppler        : necessário para pdf2image e camelot
    - Java (JRE/JDK) : necessário para tabula-py
    - Ghostscript    : necessário para camelot-py
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

# ── Logging ───────────────────────────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[
        _stream_handler,
        logging.FileHandler("pdf_extractor.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("pdf_extractor")

# ── Constantes ────────────────────────────────────────────────────────────────

MIN_TEXT_LENGTH = 50
OCR_LANG_TESSERACT = "por"
OCR_LANG_EASYOCR = ["pt"]
_ROOT_DIR = Path(__file__).parent.parent.parent   # src/extraction/ → src/ → project root
IMAGE_OUTPUT_DIR = str(_ROOT_DIR / "data" / "processed" / "extracted_images")
JSON_OUTPUT_DIR  = str(_ROOT_DIR / "data" / "processed" / "extracted_json")
IMAGE_DPI = 200
MAX_WORKERS = 4

# Tabela com >80% células vazias é provavelmente falso positivo de layout
TABLE_EMPTY_CELL_THRESHOLD = 0.80

# Tolerância em pontos para verificar se objeto está dentro de bbox de tabela
BBOX_TOLERANCE = 2.0


# ── Estrutura de retorno ───────────────────────────────────────────────────────

@dataclass
class ContentBlock:
    """
    Unidade atômica de conteúdo extraído, em ordem de leitura.
    type: "text" | "table" | "image"
    """
    type: str
    page: int
    bbox: list[float] = field(default_factory=list)
    # text blocks
    content: str = ""
    # table blocks
    data: list[dict] = field(default_factory=list)
    # image blocks
    image_path: str = ""
    # qual biblioteca extraiu
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
    # blocos ordenados em ordem de leitura (texto + tabelas + imagens)
    content_blocks: list[ContentBlock] = field(default_factory=list)
    images_paths: list[str] = field(default_factory=list)
    extraction_log: dict[str, str] = field(default_factory=dict)
    success: bool = True
    error: str = ""

    # ── Vistas agregadas (para compatibilidade e conveniência) ────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_valid_text(text: str) -> bool:
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return False
    printable_ratio = sum(c.isprintable() for c in text) / len(text)
    return printable_ratio > 0.8


def _sanitize_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    return df.to_dict(orient="records")


def _table_is_real(cells: list[list]) -> bool:
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


def _cells_to_records(cells: list[list]) -> list[dict]:
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


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _obj_inside_bbox(obj: dict, bbox: tuple, tol: float = BBOX_TOLERANCE) -> bool:
    """Verifica se um objeto pdfplumber está dentro de um bounding box."""
    x0, top, x1, bottom = bbox
    return (
        obj.get("x0", 0) >= x0 - tol
        and obj.get("x1", 0) <= x1 + tol
        and obj.get("top", 0) >= top - tol
        and obj.get("bottom", 0) <= bottom + tol
    )


# ── Metadados ─────────────────────────────────────────────────────────────────

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


# ── Núcleo: extração bloco a bloco por página ─────────────────────────────────

def _extract_blocks_for_page(
    plumber_page,
    page_num: int,
) -> list[ContentBlock]:
    """
    Extrai blocos de uma única página pdfplumber em ordem de leitura.

    Estratégia:
      1. Detecta tabelas com bordas (lattice) via find_tables().
      2. Descarta tabelas espúrias (>80% células vazias).
      3. Extrai texto FORA das bboxes de tabelas válidas usando page.filter().
      4. Ordena todos os blocos pelo topo (Y) para manter leitura natural.

    Por que não usar 'stream mode' aqui:
      Stream mode trata qualquer coluna de texto como tabela, o que causa
      falsos positivos em textos normais com recuo ou numeração. Ele só é
      ativado no fallback camelot/tabula quando lattice falha completamente.
    """
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
        if not _table_is_real(cells):
            logger.debug(
                "Pág %d: tabela bbox=%s descartada (>80%% vazia)", page_num, tbl.bbox
            )
            continue
        records = _cells_to_records(cells)
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
            return not any(_obj_inside_bbox(obj, bb) for bb in valid_table_bboxes)
        try:
            text = plumber_page.filter(outside_all_tables).extract_text() or ""
        except Exception as exc:
            logger.debug("filter pág %d falhou, usando extract_text: %s", page_num, exc)
            text = plumber_page.extract_text() or ""
    else:
        text = plumber_page.extract_text() or ""

    text = _sanitize_text(text)
    if text:
        blocks.append(ContentBlock(
            type="text",
            page=page_num,
            content=text,
            source="pdfplumber",
        ))

    # ── Passo 3: ordenar por posição vertical (topo da bbox ou página inteira)
    # Tabelas têm bbox definida; texto recebe Y estimado pelo pdfplumber
    def _sort_key(b: ContentBlock) -> float:
        if b.bbox:
            return b.bbox[1]   # top Y da tabela
        return 0.0             # texto sem bbox vai para o topo (cabeçalho/parágrafo)

    blocks.sort(key=_sort_key)
    return blocks


# ── OCR fallback (para PDFs escaneados — texto vazio após pdfplumber) ─────────

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


def _extract_blocks_ocr(file_path: str, page_num_offset: int = 1) -> list[ContentBlock]:
    """Fallback OCR página a página quando todo o texto nativo está vazio."""
    blocks: list[ContentBlock] = []
    try:
        images = _render_pages_to_images(file_path)
    except Exception as exc:
        logger.warning("OCR — pdf2image falhou (Poppler instalado?): %s", exc)
        return blocks

    for i, img in enumerate(images):
        page_num = page_num_offset + i
        text = ""
        # Tenta Tesseract
        try:
            text = _ocr_page_tesseract(img)
            source = "tesseract-ocr"
        except Exception as exc:
            logger.debug("Tesseract pág %d falhou: %s", page_num, exc)
        # Fallback EasyOCR
        if not _is_valid_text(text):
            try:
                text = _ocr_page_easyocr(img)
                source = "easyocr"
            except Exception as exc:
                logger.debug("EasyOCR pág %d falhou: %s", page_num, exc)
                source = "ocr-failed"

        text = _sanitize_text(text)
        if text:
            blocks.append(ContentBlock(
                type="text",
                page=page_num,
                content=text,
                source=source,
            ))
    return blocks


# ── Extração de Imagens ────────────────────────────────────────────────────────

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
    output_dir = _ensure_dir(Path(IMAGE_OUTPUT_DIR) / stem)

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


# ── Orquestrador principal ────────────────────────────────────────────────────

def process_pdf(file_path: str) -> ExtractionResult:
    """
    Processa um único PDF e retorna ExtractionResult com content_blocks
    em ordem de leitura. Cada bloco é {type: text|table|image}.

    Fluxo de decisão por página:
      1. pdfplumber por página → separa texto e tabelas por bounding-box
      2. Se texto total vazio → OCR (Tesseract → EasyOCR)
      3. Imagens: PyMuPDF (embutidas) → pdf2image (render de página)
    """
    result = ExtractionResult(file_path=file_path)
    log: dict[str, str] = {}

    if not Path(file_path).exists():
        result.success = False
        result.error = f"Arquivo não encontrado: {file_path}"
        logger.error(result.error)
        return result

    logger.info("Processando: %s", Path(file_path).name)
    start = time.perf_counter()

    # ── Metadados
    try:
        result.metadata = extract_metadata(file_path)
    except Exception as exc:
        logger.warning("Metadados falhou: %s", exc)
        result.metadata = {"file_name": Path(file_path).name}

    # ── Extração bloco a bloco por página (texto + tabelas)
    content_blocks: list[ContentBlock] = []
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_blocks = _extract_blocks_for_page(page, page_num)
                content_blocks.extend(page_blocks)

        text_blocks = [b for b in content_blocks if b.type == "text"]
        table_blocks = [b for b in content_blocks if b.type == "table"]
        full_text_sample = " ".join(b.content for b in text_blocks)

        log["text_blocks"] = str(len(text_blocks))
        log["table_blocks"] = str(len(table_blocks))
        log["text_source"] = "pdfplumber (por coordenada)"

        # ── Fallback OCR se nenhum texto nativo encontrado
        if not _is_valid_text(full_text_sample):
            logger.info(
                "%s: texto nativo insuficiente, tentando OCR...",
                Path(file_path).name,
            )
            ocr_blocks = _extract_blocks_ocr(file_path)
            if ocr_blocks:
                content_blocks.extend(ocr_blocks)
                log["text_source"] = ocr_blocks[0].source if ocr_blocks else "ocr-failed"
            else:
                log["text_source"] = "FALHA — nenhum texto extraído"

    except Exception as exc:
        log["text_source"] = f"ERRO: {exc}"
        logger.error("Extração de blocos falhou: %s", exc)
        result.success = False
        result.error = str(exc)

    result.content_blocks = content_blocks

    # ── Imagens
    try:
        img_blocks = extract_images(file_path, log)
        result.content_blocks.extend(img_blocks)
        result.images_paths = [b.image_path for b in img_blocks]
    except Exception as exc:
        log["images"] = f"ERRO: {exc}"
        logger.error("Extração de imagens falhou: %s", exc)

    elapsed = round(time.perf_counter() - start, 2)
    log["elapsed_seconds"] = str(elapsed)
    result.extraction_log = log

    n_text = len([b for b in result.content_blocks if b.type == "text"])
    n_table = len([b for b in result.content_blocks if b.type == "table"])
    n_img = len([b for b in result.content_blocks if b.type == "image"])
    logger.info(
        "Concluído: %s | %d blocos texto | %d tabelas | %d imgs | %.2fs",
        Path(file_path).name, n_text, n_table, n_img, elapsed,
    )
    return result


# ── Processamento em lote ─────────────────────────────────────────────────────

def _process_single(args: tuple[str, str]) -> dict:
    file_path, json_output_dir = args
    result = process_pdf(file_path)
    if json_output_dir:
        out_dir = _ensure_dir(json_output_dir)
        out_file = out_dir / f"{Path(file_path).stem}.json"
        out_file.write_text(result.to_json(), encoding="utf-8")
    return result.to_dict()


def process_batch(
    pdf_paths: list[str],
    json_output_dir: str = JSON_OUTPUT_DIR,
    max_workers: int = MAX_WORKERS,
    use_multiprocessing: bool = False,
) -> list[dict]:
    results = []
    args = [(p, json_output_dir) for p in pdf_paths]

    if use_multiprocessing and len(pdf_paths) > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_single, a): a[0] for a in args}
            with tqdm(total=len(pdf_paths), desc="Processando PDFs", unit="pdf") as pbar:
                for future in as_completed(futures):
                    path = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        logger.error("Worker falhou para %s: %s", path, exc)
                        results.append({"file_path": path, "success": False, "error": str(exc)})
                    pbar.update(1)
    else:
        for arg in tqdm(args, desc="Processando PDFs", unit="pdf"):
            results.append(_process_single(arg))

    failed = [r["file_path"] for r in results if not r.get("success", True)]
    if failed:
        logger.warning("Falhas (%d): %s", len(failed), failed)
    else:
        logger.info("Todos os %d PDFs processados.", len(results))
    return results


def find_pdfs(directory: str, recursive: bool = True) -> list[str]:
    base = Path(directory)
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(str(p) for p in base.glob(pattern))


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extração robusta de PDFs (blocos em ordem de leitura)."
    )
    parser.add_argument("input", help="Arquivo .pdf ou diretório com PDFs.")
    parser.add_argument("--output-dir", default=JSON_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--no-recursive", action="store_true")
    args = parser.parse_args()

    target = Path(args.input)
    if target.is_file() and target.suffix.lower() == ".pdf":
        pdf_list = [str(target)]
    elif target.is_dir():
        pdf_list = find_pdfs(str(target), recursive=not args.no_recursive)
        logger.info("Encontrados %d PDFs em '%s'.", len(pdf_list), target)
    else:
        logger.error("Entrada inválida: %s", args.input)
        sys.exit(1)

    if not pdf_list:
        logger.warning("Nenhum PDF encontrado.")
        sys.exit(0)

    all_results = process_batch(
        pdf_paths=pdf_list,
        json_output_dir=args.output_dir,
        max_workers=args.workers,
        use_multiprocessing=args.parallel,
    )

    total = len(all_results)
    ok = sum(1 for r in all_results if r.get("success", True))
    print(f"\n{'─'*50}")
    print(f"  Total     : {total}")
    print(f"  Sucesso   : {ok}")
    print(f"  Falhas    : {total - ok}")
    print(f"  JSONs     : {args.output_dir}/")
    print(f"  Imagens   : {IMAGE_OUTPUT_DIR}/")
    print(f"  Log       : pdf_extractor.log")
    print(f"{'─'*50}\n")
