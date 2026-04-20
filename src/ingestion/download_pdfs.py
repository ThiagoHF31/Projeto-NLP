from __future__ import annotations

"""
Script para baixar todos os PDFs dos metadados ANEEL para uso em banco vetorial.

Estratégia: curl_cffi com impersonação Chrome120 para bypass de TLS fingerprinting.
O servidor ANEEL bloqueia requests Python/OpenSSL por fingerprint TLS diferente do browser.

Fontes: 3 arquivos JSON com legislação de 2016, 2021 e 2022
Destino: dados/pdfs/
Manifesto: dados/pdfs_manifesto.csv  (metadados + status)
Erros: dados/pdfs_erros.log
"""

import json
import csv
import logging
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from curl_cffi import requests as cf
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

ROOT_DIR   = Path(__file__).parent.parent.parent   # src/ingestion/ → src/ → project root
DATA_DIR   = ROOT_DIR / "data"
JSON_DIR   = DATA_DIR / "raw" / "metadata"
OUTPUT_DIR = DATA_DIR / "pdfs"
MANIFESTO_PATH = DATA_DIR / "pdfs_manifesto.csv"
LOG_PATH   = ROOT_DIR / "logs" / "download.log"

JSON_FILES = [
    JSON_DIR / "biblioteca_aneel_gov_br_legislacao_2016_metadados.json",
    JSON_DIR / "biblioteca_aneel_gov_br_legislacao_2021_metadados.json",
    JSON_DIR / "biblioteca_aneel_gov_br_legislacao_2022_metadados.json",
]

MAX_WORKERS    = 4      # paralelos (curl_cffi é mais leve, podemos usar 4)
TIMEOUT        = 30     # segundos por request
MAX_RETRIES    = 3      # tentativas por arquivo
RETRY_DELAY    = 8      # segundos entre tentativas
REQUEST_DELAY  = 1.0    # delay entre downloads no mesmo worker

EXTENSOES_VALIDAS = {".pdf", ".html", ".zip"}  # tipos aceitos no JSON

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_log_handler = logging.FileHandler(str(LOG_PATH), encoding="utf-8")
_log_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logging.getLogger().addHandler(_log_handler)
logging.getLogger().setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def limpar_url(url: str) -> str:
    """Corrige URLs malformadas e força HTTPS."""
    url = url.strip()
    # Remove prefixos duplicados: "http://  http://..." ou "http://  https://..."
    url = re.sub(r'^https?://\s+https?://', 'https://', url)
    url = re.sub(r'^https?://\s+http://',  'https://', url)
    # Força HTTPS
    url = re.sub(r'^http://', 'https://', url)
    return url


def criar_sessao() -> cf.Session:
    """Cria sessão curl_cffi impersonando Chrome120 (bypass TLS fingerprint)."""
    session = cf.Session(impersonate="chrome120")
    session.headers.update({
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8",
        "Referer":         "https://www2.aneel.gov.br/cedoc/",
    })
    return session


# Pool de sessões: uma por worker thread
_session_pool: dict[int, cf.Session] = {}
_pool_lock = Lock()

def _get_session() -> cf.Session:
    import threading
    tid = threading.get_ident()
    with _pool_lock:
        if tid not in _session_pool:
            _session_pool[tid] = criar_sessao()
    return _session_pool[tid]

# ---------------------------------------------------------------------------
# Extração dos metadados
# ---------------------------------------------------------------------------

def extrair_registros() -> list[dict]:
    registros = []
    for json_path in JSON_FILES:
        ano = json_path.stem.split("_")[-2]
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        for data_pub, bloco in data.items():
            for reg in bloco.get("registros", []):
                for pdf in reg.get("pdfs", []):
                    url     = pdf.get("url", "").strip()
                    arquivo = pdf.get("arquivo", "").strip()
                    if not url or not arquivo:
                        continue
                    registros.append({
                        "ano":            ano,
                        "data_publicacao": data_pub,
                        "titulo":         reg.get("titulo", ""),
                        "autor":          reg.get("autor", ""),
                        "material":       reg.get("material", ""),
                        "esfera":         reg.get("esfera", ""),
                        "situacao":       reg.get("situacao", ""),
                        "assinatura":     reg.get("assinatura", ""),
                        "assunto":        reg.get("assunto", ""),
                        "ementa":         reg.get("ementa", ""),
                        "tipo_pdf":       pdf.get("tipo", ""),
                        "url":            url,
                        "arquivo":        arquivo,
                    })
    return registros

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

_csv_lock = Lock()
_counters = {"ok": 0, "skip": 0, "erro": 0}
_counters_lock = Lock()

CAMPOS_CSV = [
    "ano", "data_publicacao", "titulo", "autor", "material", "esfera",
    "situacao", "assinatura", "assunto", "ementa", "tipo_pdf",
    "url", "arquivo", "status", "tamanho_bytes", "erro",
]


def _registrar(writer, reg: dict, status: str, tamanho: int, erro: str = ""):
    row = {**reg, "status": status, "tamanho_bytes": tamanho, "erro": erro}
    with _csv_lock:
        writer.writerow(row)


def download_pdf(reg: dict, writer, pbar) -> str:
    dest = OUTPUT_DIR / reg["arquivo"]

    # Arquivo já existe e não está vazio → pula
    if dest.exists() and dest.stat().st_size > 0:
        _registrar(writer, reg, "skip", dest.stat().st_size)
        with _counters_lock:
            _counters["skip"] += 1
        pbar.update(1)
        return "skip"

    url = limpar_url(reg["url"])
    session = _get_session()
    time.sleep(REQUEST_DELAY)

    for tentativa in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=TIMEOUT)
            resp.raise_for_status()

            conteudo = resp.content
            if len(conteudo) == 0:
                raise ValueError("Resposta vazia")

            # Valida que é PDF (ou outro tipo esperado) pelo magic bytes ou Content-Type
            ct = resp.headers.get("Content-Type", "").lower()
            ext = Path(reg["arquivo"]).suffix.lower()
            if ext == ".pdf" and not conteudo.startswith(b"%PDF") and "pdf" not in ct:
                raise ValueError(f"Conteudo nao e PDF (Content-Type: {ct})")

            dest.write_bytes(conteudo)

            _registrar(writer, reg, "ok", len(conteudo))
            with _counters_lock:
                _counters["ok"] += 1
            pbar.update(1)
            return "ok"

        except Exception as e:
            if tentativa < MAX_RETRIES:
                time.sleep(RETRY_DELAY * tentativa)
            else:
                logging.warning(f"ERRO [{reg['arquivo']}] {url} | {type(e).__name__}: {e}")
                if dest.exists():
                    dest.unlink()
                _registrar(writer, reg, "erro", 0, str(e))
                with _counters_lock:
                    _counters["erro"] += 1
                pbar.update(1)
                return "erro"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Lendo metadados dos JSONs...")
    registros = extrair_registros()
    print(f"Total de PDFs no metadado: {len(registros):,}")

    # Carrega manifesto existente para saber o que já foi baixado
    ja_ok: set[str] = set()
    modo_csv = "w"
    if MANIFESTO_PATH.exists():
        modo_csv = "a"
        with open(MANIFESTO_PATH, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") in ("ok", "skip"):
                    ja_ok.add(row["arquivo"])
        print(f"Manifesto existente: {len(ja_ok):,} arquivos já concluídos")

    pendentes = [r for r in registros if r["arquivo"] not in ja_ok]
    print(f"Pendentes (novos + erros anteriores): {len(pendentes):,}")

    if not pendentes:
        print("Nada a fazer — todos os PDFs já foram baixados.")
        return

    with open(MANIFESTO_PATH, modo_csv, newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CAMPOS_CSV)
        if modo_csv == "w":
            writer.writeheader()

        with tqdm(total=len(pendentes), unit="pdf", desc="Baixando PDFs") as pbar:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(download_pdf, reg, writer, pbar): reg
                    for reg in pendentes
                }
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        reg = futures[future]
                        logging.error(f"Exceção não tratada [{reg['arquivo']}]: {e}")

    ok    = _counters["ok"]
    skip  = _counters["skip"]
    erro  = _counters["erro"]
    total_ok = ok + skip
    print(f"\n{'='*52}")
    print(f"Concluído!")
    print(f"  Baixados agora         : {ok:,}")
    print(f"  Já existiam (pulados)  : {skip:,}")
    print(f"  Erros                  : {erro:,}")
    print(f"  Total disponível       : {total_ok:,}")
    if erro:
        print(f"  Log de erros em        : {LOG_PATH}")
    print(f"  Manifesto salvo em     : {MANIFESTO_PATH}")
    print(f"  PDFs em                : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
