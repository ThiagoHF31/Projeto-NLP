"""
run_extraction.py — Executa a extração completa de todos os PDFs do projeto.


py -3.13 run_extraction.py --parallel --workers 8


Uso:
    python run_extraction.py              # processa todos os PDFs novos
    python run_extraction.py --reprocess  # reprocessa mesmo os já extraídos
    python run_extraction.py --workers 8  # controla paralelismo
    python run_extraction.py --parallel   # usa multiprocessing (mais rápido em lotes grandes)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Caminhos do projeto ───────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).parent
PDF_DIR = ROOT_DIR / "data" / "pdfs"
JSON_DIR = ROOT_DIR / "data" / "processed" / "extracted_json"
LOG_FILE = ROOT_DIR / "logs" / "extraction.log"

# ── Configuração padrão ───────────────────────────────────────────────────────

DEFAULT_WORKERS = 4

# ── Logging ───────────────────────────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("run_extraction")


# ── Funções auxiliares ────────────────────────────────────────────────────────

def find_pending_pdfs(pdf_dir: Path, json_dir: Path, reprocess: bool) -> list[Path]:
    """Retorna PDFs que ainda não foram extraídos (ou todos se reprocess=True)."""
    all_pdfs = sorted(pdf_dir.glob("**/*.pdf"))
    if reprocess or not json_dir.exists():
        return all_pdfs

    already_done = {p.stem for p in json_dir.glob("*.json")}
    pending = [p for p in all_pdfs if p.stem not in already_done]
    return pending


def print_summary(total: int, ok: int, elapsed: float, json_dir: Path) -> None:
    failed = total - ok
    print(f"\n{'─' * 52}")
    print(f"  Total processado : {total}")
    print(f"  Sucesso          : {ok}")
    print(f"  Falhas           : {failed}")
    print(f"  Tempo total      : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  JSONs salvos em  : {json_dir}")
    print(f"  Log              : {LOG_FILE}")
    print(f"{'─' * 52}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Extração de PDFs ANEEL para JSON.")
    parser.add_argument("--reprocess", action="store_true",
                        help="Reprocessa PDFs que já têm JSON gerado.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Número de workers paralelos (padrão: {DEFAULT_WORKERS}).")
    parser.add_argument("--parallel", action="store_true",
                        help="Usa multiprocessing (recomendado para lotes grandes).")
    args = parser.parse_args()

    if not PDF_DIR.exists():
        logger.error("Pasta de PDFs não encontrada: %s", PDF_DIR)
        sys.exit(1)

    pending = find_pending_pdfs(PDF_DIR, JSON_DIR, args.reprocess)

    if not pending:
        logger.info("Nenhum PDF pendente. Use --reprocess para reprocessar tudo.")
        sys.exit(0)

    total_existing = sum(1 for _ in PDF_DIR.glob("**/*.pdf"))
    already_done = total_existing - len(pending)

    logger.info("PDFs encontrados : %d", total_existing)
    logger.info("Já processados   : %d", already_done)
    logger.info("Pendentes        : %d", len(pending))
    logger.info("Workers          : %d", args.workers)
    logger.info("Multiprocessing  : %s", args.parallel)
    logger.info("Saída JSON       : %s", JSON_DIR)
    print()

    from src.extraction.batch import process_batch

    start = time.perf_counter()
    results = process_batch(
        pdf_paths=[str(p) for p in pending],
        json_output_dir=str(JSON_DIR),
        max_workers=args.workers,
        use_multiprocessing=args.parallel,
    )
    elapsed = time.perf_counter() - start

    ok = sum(1 for r in results if r.get("success", True))
    print_summary(len(results), ok, elapsed, JSON_DIR)


if __name__ == "__main__":
    main()
