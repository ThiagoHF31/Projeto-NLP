"""Constantes de configuração do pipeline de extração."""

from pathlib import Path

_ROOT_DIR = Path(__file__).parent.parent.parent  # src/extraction/ → src/ → project root

MIN_TEXT_LENGTH = 50
OCR_LANG_TESSERACT = "por"
OCR_LANG_EASYOCR = ["pt"]

IMAGE_OUTPUT_DIR = str(_ROOT_DIR / "data" / "processed" / "extracted_images")
JSON_OUTPUT_DIR = str(_ROOT_DIR / "data" / "processed" / "extracted_json")
IMAGE_DPI = 200
MAX_WORKERS = 4

# Tabela com >80% células vazias é provavelmente falso positivo de layout
TABLE_EMPTY_CELL_THRESHOLD = 0.80

# Tolerância em pontos para verificar se objeto está dentro de bbox de tabela
BBOX_TOLERANCE = 2.0
