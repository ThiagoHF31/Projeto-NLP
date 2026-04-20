"""
pdf_extractor.py — Orquestrador principal + CLI.

Fluxo de decisão por página:
  1. pdfplumber (text_extractor) — separa texto e tabelas por bounding-box
  2. Se texto total vazio → OCR (Tesseract → EasyOCR)
  3. Imagens: PyMuPDF (embutidas) → pdf2image (render de página)

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

import logging
import sys
import time
from pathlib import Path

from .config import IMAGE_OUTPUT_DIR, JSON_OUTPUT_DIR, MAX_WORKERS
from .image_extractor import extract_images
from .metadata import extract_metadata
from .models import ContentBlock, ExtractionResult
from .ocr import extract_blocks_ocr
from .text_extractor import extract_blocks_for_page
from .utils import is_valid_text

# ── Logging ───────────────────────────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pdf_extractor.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ── Orquestrador ──────────────────────────────────────────────────────────────

def process_pdf(file_path: str) -> ExtractionResult:
    """Processa um único PDF e retorna ExtractionResult com content_blocks em ordem de leitura."""
    result = ExtractionResult(file_path=file_path)
    log: dict[str, str] = {}

    if not Path(file_path).exists():
        result.success = False
        result.error = f"Arquivo não encontrado: {file_path}"
        logger.error(result.error)
        return result

    logger.info("Processando: %s", Path(file_path).name)
    start = time.perf_counter()

    try:
        result.metadata = extract_metadata(file_path)
    except Exception as exc:
        logger.warning("Metadados falhou: %s", exc)
        result.metadata = {"file_name": Path(file_path).name}

    content_blocks: list[ContentBlock] = []
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                content_blocks.extend(extract_blocks_for_page(page, page_num))

        text_blocks = [b for b in content_blocks if b.type == "text"]
        table_blocks = [b for b in content_blocks if b.type == "table"]
        full_text_sample = " ".join(b.content for b in text_blocks)

        log["text_blocks"] = str(len(text_blocks))
        log["table_blocks"] = str(len(table_blocks))
        log["text_source"] = "pdfplumber (por coordenada)"

        if not is_valid_text(full_text_sample):
            logger.info("%s: texto nativo insuficiente, tentando OCR...", Path(file_path).name)
            ocr_blocks = extract_blocks_ocr(file_path)
            if ocr_blocks:
                content_blocks.extend(ocr_blocks)
                log["text_source"] = ocr_blocks[0].source
            else:
                log["text_source"] = "FALHA — nenhum texto extraído"

    except Exception as exc:
        log["text_source"] = f"ERRO: {exc}"
        logger.error("Extração de blocos falhou: %s", exc)
        result.success = False
        result.error = str(exc)

    result.content_blocks = content_blocks

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


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse
    from .batch import find_pdfs, process_batch

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


if __name__ == "__main__":
    _cli()
