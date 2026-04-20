"""Processamento em lote de múltiplos PDFs com suporte a paralelismo."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from .config import JSON_OUTPUT_DIR, MAX_WORKERS
from .utils import ensure_dir

logger = logging.getLogger(__name__)


def _process_single(args: tuple[str, str]) -> dict:
    from .pdf_extractor import process_pdf
    file_path, json_output_dir = args
    result = process_pdf(file_path)
    if json_output_dir:
        out_dir = ensure_dir(json_output_dir)
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
