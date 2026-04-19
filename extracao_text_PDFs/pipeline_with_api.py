#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_with_api.py — Pipeline RAG com Gemini 2.5 Flash API (async paralelo)

Arquitetura:
  - Enriquecimento via Gemini 2.5 Flash API — sem modelo local, sem consumo de VRAM.
  - Dois níveis de processamento:
      · PDFs <= SMALL_PDF_PAGES páginas (81% do acervo): zero chamadas API,
        contexto sintético a partir do nome e primeira página.
      · PDFs grandes: enriquecimento em batch (BATCH_SIZE chunks/chamada).
  - asyncio puro:
      · CPU-bound (extração, embeddings) → ThreadPoolExecutor
      · I/O-bound (Gemini API) → coroutines async com semáforo de concorrência
  - RateLimiter token bucket: respeita 15 RPM do plano gratuito.
  - Retry com backoff exponencial em erros 429/503.
  - Checkpointing atômico por PDF — retomada automática após qualquer falha.
"""

import asyncio
import gc
import hashlib
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ── UTF-8 no Windows ──────────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Env vars antes dos imports pesados ───────────────────────────────────────
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dotenv import load_dotenv
load_dotenv()

import fitz
import chromadb
import torch
from google import genai
from google.genai import types as genai_types
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Silencia loggers ruidosos
for _log in ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub",
             "huggingface_hub.file_download", "sentence_transformers",
             "google.api_core", "google.auth", "google.ai.generativelanguage",
             "google.generativeai", "google.genai"):
    logging.getLogger(_log).setLevel(logging.ERROR)

try:
    import pymupdf4llm
    HAS_PYMUPDF4LLM = True
except ImportError:
    HAS_PYMUPDF4LLM = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR        = Path(__file__).parent.parent  # raiz do projeto
PDF_DIR         = BASE_DIR / "dados" / "pdfs"
CHROMA_DIR      = BASE_DIR / "dados" / "chroma_db_api"
CHECKPOINT_FILE = BASE_DIR / "dados" / "checkpoint_api.json"
ERROR_LOG       = BASE_DIR / "dados" / "erros_pipeline_api.log"

# Gemini
# gemini-2.0-flash → 15 RPM grátis  |  gemini-2.5-flash → apenas 5 RPM grátis
GEMINI_MODEL          = "gemini-2.0-flash"
GEMINI_RPM            = 13                   # free tier = 15; usamos 13 com margem
GEMINI_MAX_CONCURRENT = 5                    # requisições simultâneas em voo
GEMINI_MAX_RETRIES    = 5
GEMINI_RETRY_BASE_S   = 2.0                  # dobra a cada tentativa

# Chunking
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

# Dois níveis de enriquecimento
SMALL_PDF_PAGES = 5    # PDFs com <= N páginas: zero chamadas API
BATCH_SIZE      = 8    # chunks por chamada Gemini
SUMMARY_PAGES   = 10   # páginas para sumário de PDFs grandes

# Embeddings (CPU, float16 para economizar RAM)
EMBED_MODEL_NAME = "BAAI/bge-m3"
EMBED_BATCH_SIZE = 32

# Paralelismo de PDFs
PDF_WORKERS = 4

CHROMA_COLLECTION = "aneel_legislacao_api"

# Limite de teste (None = todos)
TEST_LIMIT: Optional[int] = 5

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging() -> logging.Logger:
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)-8s] %(message)s"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(logging.StreamHandler(sys.stdout))
    root.addHandler(logging.FileHandler(ERROR_LOG, encoding="utf-8"))
    for h in root.handlers:
        h.setFormatter(logging.Formatter(fmt))
    for _log in ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub",
                 "huggingface_hub.file_download", "sentence_transformers",
                 "google.api_core", "google.auth", "google.ai.generativelanguage",
                 "google.generativeai", "google.genai"):
        logging.getLogger(_log).setLevel(logging.ERROR)
    return logging.getLogger(__name__)

logger = setup_logging()

# ══════════════════════════════════════════════════════════════════════════════
# RATE LIMITER — token bucket assíncrono
# ══════════════════════════════════════════════════════════════════════════════

class RateLimiter:
    """Distribui requisições uniformemente no tempo para respeitar N RPM."""

    def __init__(self, rpm: int) -> None:
        self._interval = 60.0 / rpm
        self._lock = asyncio.Lock()
        self._next_allowed: float = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self._next_allowed - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._next_allowed = asyncio.get_event_loop().time() + self._interval

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINTING
# ══════════════════════════════════════════════════════════════════════════════

def load_checkpoint() -> Set[str]:
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f).get("processed", []))
        except (json.JSONDecodeError, KeyError):
            logger.warning("Checkpoint corrompido — reiniciando do zero.")
    return set()


def save_checkpoint(processed: Set[str]) -> None:
    """Escrita atômica: evita checkpoint parcial em caso de crash."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"processed": sorted(processed)}, f, ensure_ascii=False)
    tmp.replace(CHECKPOINT_FILE)

# ══════════════════════════════════════════════════════════════════════════════
# EXTRAÇÃO DE PDF
# ══════════════════════════════════════════════════════════════════════════════

def pdf_num_pages(pdf_path: Path) -> int:
    doc = fitz.open(str(pdf_path))
    n = len(doc)
    doc.close()
    return n


def extract_pages_text(pdf_path: Path, start: int = 0, end: Optional[int] = None) -> str:
    doc = fitz.open(str(pdf_path))
    end = min(end if end is not None else len(doc), len(doc))
    parts = []
    for i in range(start, end):
        page = doc[i]
        text = page.get_text("text")
        if len(text.strip()) < 30 and page.get_images():
            parts.append(f"[Pagina {i+1}: imagem sem texto]")
        else:
            parts.append(text)
    doc.close()
    return "\n\n".join(parts)


def extract_pdf_as_markdown(pdf_path: Path) -> Tuple[str, int]:
    num_pages = pdf_num_pages(pdf_path)

    if HAS_PYMUPDF4LLM:
        try:
            md = pymupdf4llm.to_markdown(str(pdf_path))
            if md.strip():
                return md, num_pages
        except Exception as e:
            logger.debug(f"pymupdf4llm falhou em {pdf_path.name}: {e}")

    doc = fitz.open(str(pdf_path))
    parts = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            parts.append(f"## Pagina {i+1}\n\n{text}")
        elif page.get_images():
            parts.append(f"## Pagina {i+1}\n\n[imagem sem texto]")
    doc.close()
    return "\n\n".join(parts), num_pages

# ══════════════════════════════════════════════════════════════════════════════
# CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

_MD_HEADERS = [("#", "H1"), ("##", "H2"), ("###", "H3"), ("####", "H4")]


def chunk_markdown(md_text: str) -> List[Dict]:
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_MD_HEADERS,
        strip_headers=False,
    )
    rec_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    result = []
    for hc in md_splitter.split_text(md_text):
        if not hc.page_content.strip():
            continue
        for sc in rec_splitter.split_text(hc.page_content):
            if sc.strip():
                result.append({"content": sc, "metadata": dict(hc.metadata)})
    return result

# ══════════════════════════════════════════════════════════════════════════════
# GEMINI API — chamadas assíncronas com rate limiting e retry
# ══════════════════════════════════════════════════════════════════════════════

_BATCH_PROMPT = """\
Você é especialista em RAG para documentos regulatórios brasileiros da ANEEL.
Para cada CHUNK numerado abaixo, gere um contexto de 2-3 frases situando o trecho no documento.

REGRAS ESTRITAS:
- NÃO resuma o chunk. Explique de onde ele vem: tipo de norma, número, ano, entidade, tema.
- Exemplo bom: "Este trecho pertence à Resolução Normativa ANEEL nº 1000/2021 e detalha os critérios de reajuste tarifário das distribuidoras."
- Retorne SOMENTE JSON válido no formato: {{"contexts": ["ctx_1", ..., "ctx_{n}"]}}
- Exatamente {n} itens, na mesma ordem dos chunks. Sem markdown, sem tags XML.

SUMÁRIO DO DOCUMENTO:
{summary}

CHUNKS:
{chunks_block}

RESPOSTA JSON:"""

_SUMMARY_PROMPT = """\
Faça um resumo executivo de no máximo 3 linhas deste documento regulatório da ANEEL, \
identificando: Tipo de norma, Número/ano, Entidades envolvidas e Propósito.
Apenas o resumo, sem introduções.

DOCUMENTO:
{text}"""

_GENERATION_CONFIG = genai_types.GenerateContentConfig(
    temperature=0.1,
    max_output_tokens=1024,
    # Desabilita thinking do Gemini 2.5 Flash (mais rápido, resposta direta)
    thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
)


def _build_chunks_block(contents: List[str]) -> str:
    return "\n\n".join(f"[{i+1}] {c[:600]}" for i, c in enumerate(contents))


def _parse_json_contexts(raw: str, expected: int) -> List[Optional[str]]:
    """Extrai lista de contextos do JSON mesmo com texto antes/depois."""
    try:
        match = re.search(r'\{[^{}]*"contexts"\s*:\s*\[.*?\]\s*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            ctxs = data.get("contexts", [])
            if len(ctxs) == expected:
                return ctxs
    except Exception:
        pass
    return [None] * expected


async def _gemini_call(
    client: genai.Client,
    prompt: str,
    rate_limiter: RateLimiter,
    semaphore: asyncio.Semaphore,
) -> Optional[str]:
    """
    Chamada assíncrona ao Gemini com rate limiting e retry exponencial.
    Retorna o texto ou None após esgotar as tentativas.
    """
    async with semaphore:
        await rate_limiter.acquire()

        for attempt in range(1, GEMINI_MAX_RETRIES + 1):
            try:
                response = await client.aio.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=_GENERATION_CONFIG,
                )
                # Extrai texto ignorando partes de thinking (thought=True)
                for part in response.candidates[0].content.parts:
                    if part.text and not getattr(part, "thought", False):
                        return part.text.strip()
                return None

            except Exception as e:
                err = str(e).lower()
                is_daily_exhausted = "perday" in err or ("limit: 0" in err and "perday" in err) or "requests_per_day" in err or ("quota" in err and "daily" in err)
                is_rate   = ("429" in err or "resource_exhausted" in err) and not is_daily_exhausted
                is_server = "503" in err or "500" in err

                # Cota diária esgotada: não adianta tentar novamente hoje
                if is_daily_exhausted or ("limit: 0" in err and "perday" in str(e)):
                    logger.warning("  Gemini: cota DIARIA esgotada. Usando fallback sem enriquecimento.")
                    return None

                if (is_rate or is_server) and attempt < GEMINI_MAX_RETRIES:
                    # Extrai o retryDelay sugerido pela API se disponível
                    retry_match = re.search(r"retry in (\d+)", err)
                    suggested  = int(retry_match.group(1)) if retry_match else 0
                    wait = max(GEMINI_RETRY_BASE_S * (2 ** (attempt - 1)), suggested)
                    tag  = "rate-limit" if is_rate else "erro-servidor"
                    logger.warning(f"  Gemini {tag} — aguardando {wait:.0f}s (tentativa {attempt}/{GEMINI_MAX_RETRIES})")
                    await asyncio.sleep(wait)
                else:
                    logger.warning(f"  Gemini falhou ({attempt} tentativas): {type(e).__name__}")
                    return None

    return None


async def gemini_summarize(
    pdf_path: Path,
    num_pages: int,
    client: genai.Client,
    rate_limiter: RateLimiter,
    semaphore: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
) -> str:
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(
        executor, extract_pages_text, pdf_path, 0, min(SUMMARY_PAGES, num_pages)
    )
    if not text.strip():
        return "Documento sem texto nas primeiras paginas."
    prompt = _SUMMARY_PROMPT.format(text=text[:5000])
    return await _gemini_call(client, prompt, rate_limiter, semaphore) or "Sumario indisponivel."


async def gemini_enrich_batch(
    contents: List[str],
    global_summary: str,
    client: genai.Client,
    rate_limiter: RateLimiter,
    semaphore: asyncio.Semaphore,
) -> List[str]:
    """Enriquece um batch de chunks em UMA chamada Gemini."""
    n = len(contents)
    prompt = _BATCH_PROMPT.format(
        n=n,
        summary=global_summary[:600],
        chunks_block=_build_chunks_block(contents),
    )
    raw = await _gemini_call(client, prompt, rate_limiter, semaphore)
    if raw:
        contexts = _parse_json_contexts(raw, n)
        return [
            f"{ctx}\n\n{chunk}" if ctx else chunk
            for ctx, chunk in zip(contexts, contents)
        ]
    return contents  # fallback: chunk original sem enriquecimento


def enrich_small_pdf_fast(chunks: List[Dict], pdf_name: str, first_page: str) -> List[str]:
    """Contexto sintético para PDFs pequenos — zero chamadas API."""
    titulo = Path(pdf_name).stem.replace("_", " ").replace("-", " ")[:80]
    intro  = first_page.strip()[:200].replace("\n", " ")
    header = f"Documento ANEEL: {titulo}. {intro}"
    return [f"{header}\n\n{c['content']}" for c in chunks]

# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

def load_embedding_model() -> SentenceTransformer:
    logger.info(f"Carregando '{EMBED_MODEL_NAME}' (CPU, float16 — ~1.1 GB RAM)...")
    model = SentenceTransformer(
        EMBED_MODEL_NAME,
        device="cpu",
        model_kwargs={"torch_dtype": torch.float16},
    )
    logger.info("Modelo de embeddings pronto.")
    return model


def generate_embeddings(model: SentenceTransformer, texts: List[str]) -> List[List[float]]:
    return model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).tolist()

# ══════════════════════════════════════════════════════════════════════════════
# CHROMADB
# ══════════════════════════════════════════════════════════════════════════════

def get_chroma_collection() -> chromadb.Collection:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"ChromaDB pronto. Chunks ja indexados: {col.count()}")
    return col


def _doc_id(pdf_name: str, idx: int) -> str:
    return hashlib.md5(f"{pdf_name}::{idx:06d}".encode()).hexdigest()


def store_chunks(
    collection: chromadb.Collection,
    pdf_name: str,
    enriched: List[str],
    embeddings: List[List[float]],
    metas: List[Dict],
) -> None:
    ids = [_doc_id(pdf_name, i) for i in range(len(enriched))]
    metadatas = [
        {"source": pdf_name, "chunk_index": i, **{k: str(v) for k, v in m.items()}}
        for i, m in enumerate(metas)
    ]
    collection.add(ids=ids, documents=enriched, embeddings=embeddings, metadatas=metadatas)

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE: PROCESSAMENTO ASSÍNCRONO DE UM PDF
# ══════════════════════════════════════════════════════════════════════════════

async def process_pdf(
    pdf_path: Path,
    collection: chromadb.Collection,
    embed_model: SentenceTransformer,
    client: genai.Client,
    rate_limiter: RateLimiter,
    semaphore: asyncio.Semaphore,
    chroma_lock: asyncio.Lock,
    executor: ThreadPoolExecutor,
) -> Tuple[bool, str]:
    """Processa um PDF fim-a-fim. Retorna (sucesso, modo_usado)."""
    name = pdf_path.name
    loop = asyncio.get_event_loop()

    # Extração — thread pool (I/O bloqueante)
    try:
        md_text, num_pages = await loop.run_in_executor(
            executor, extract_pdf_as_markdown, pdf_path
        )
    except Exception as e:
        logger.error(f"[EXTRACAO] {name}: {e}")
        return False, "erro"

    if not md_text.strip():
        logger.warning(f"[SKIP] {name}: sem texto extraivel.")
        return True, "skip"

    # Chunking — thread pool
    try:
        chunks = await loop.run_in_executor(executor, chunk_markdown, md_text)
    except Exception as e:
        logger.error(f"[CHUNK] {name}: {e}")
        return False, "erro"

    if not chunks:
        logger.warning(f"[SKIP] {name}: nenhum chunk gerado.")
        return True, "skip"

    # Enriquecimento contextual
    if num_pages <= SMALL_PDF_PAGES:
        first_page = await loop.run_in_executor(
            executor, extract_pages_text, pdf_path, 0, 1
        )
        enriched_chunks = enrich_small_pdf_fast(chunks, name, first_page)
        mode = "rapido"
    else:
        global_summary = await gemini_summarize(
            pdf_path, num_pages, client, rate_limiter, semaphore, executor
        )
        contents = [c["content"] for c in chunks]

        # Dispara todos os batches em paralelo (limitados pelo semáforo)
        batch_tasks = [
            gemini_enrich_batch(
                contents[i: i + BATCH_SIZE],
                global_summary, client, rate_limiter, semaphore,
            )
            for i in range(0, len(contents), BATCH_SIZE)
        ]
        results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        enriched_chunks = []
        for bi, res in enumerate(results):
            start = bi * BATCH_SIZE
            if isinstance(res, Exception):
                logger.warning(f"  [ENRICH] batch {bi} de {name}: {res}")
                enriched_chunks.extend(contents[start: start + BATCH_SIZE])
            else:
                enriched_chunks.extend(res)

        mode = "gemini"

    chunk_metas = [c["metadata"] for c in chunks]

    # Embeddings — thread pool (CPU intensivo)
    try:
        embeddings = await loop.run_in_executor(
            executor, generate_embeddings, embed_model, enriched_chunks
        )
    except Exception as e:
        logger.error(f"[EMBED] {name}: {e}")
        return False, "erro"

    # ChromaDB — lock garante escrita segura entre workers paralelos
    try:
        async with chroma_lock:
            store_chunks(collection, name, enriched_chunks, embeddings, chunk_metas)
    except Exception as e:
        logger.error(f"[CHROMA] {name}: {e}")
        return False, "erro"

    logger.info(f"[OK/{mode}] {name} -- {num_pages}pag, {len(chunks)} chunks")

    del enriched_chunks, embeddings, chunks, chunk_metas
    gc.collect()

    return True, mode

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

async def main_async() -> None:
    logger.info("=" * 60)
    logger.info("Pipeline RAG -- Gemini 2.5 Flash API")
    logger.info("=" * 60)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY nao encontrada. Verifique o arquivo .env")
        sys.exit(1)

    if not PDF_DIR.exists():
        logger.error(f"Diretorio de PDFs nao encontrado: {PDF_DIR}")
        sys.exit(1)

    all_pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not all_pdfs:
        logger.error("Nenhum PDF encontrado.")
        sys.exit(1)

    processed = load_checkpoint()
    pending   = [p for p in all_pdfs if p.name not in processed]

    if TEST_LIMIT is not None:
        pending = pending[:TEST_LIMIT]
        logger.info(f"[MODO TESTE] Limitado a {TEST_LIMIT} PDFs.")

    logger.info(f"PDFs totais:    {len(all_pdfs)}")
    logger.info(f"Ja processados: {len(processed)}")
    logger.info(f"Pendentes:      {len(pending)}")
    logger.info(f"Modelo:         {GEMINI_MODEL}")
    logger.info(f"Rate limit:     {GEMINI_RPM} req/min | {GEMINI_MAX_CONCURRENT} simultaneos")
    logger.info(f"Estrategia:     sem API para <= {SMALL_PDF_PAGES} pags | batch={BATCH_SIZE}/chamada")

    if not pending:
        logger.info("Nenhum PDF pendente.")
        return

    # Inicializa recursos
    client       = genai.Client(api_key=api_key)
    rate_limiter = RateLimiter(rpm=GEMINI_RPM)
    semaphore    = asyncio.Semaphore(GEMINI_MAX_CONCURRENT)
    chroma_lock  = asyncio.Lock()
    ckpt_lock    = asyncio.Lock()

    collection  = get_chroma_collection()
    embed_model = load_embedding_model()

    n_ok = n_fail = n_rapido = n_gemini = 0
    cnt_lock = asyncio.Lock()

    pdf_semaphore = asyncio.Semaphore(PDF_WORKERS)

    async def process_with_guard(pdf_path: Path) -> None:
        nonlocal n_ok, n_fail, n_rapido, n_gemini

        async with pdf_semaphore:
            success, mode = await process_pdf(
                pdf_path, collection, embed_model,
                client, rate_limiter, semaphore,
                chroma_lock, executor,
            )

        async with cnt_lock:
            if success:
                processed.add(pdf_path.name)
                async with ckpt_lock:
                    save_checkpoint(processed)
                n_ok += 1
                if mode == "rapido":
                    n_rapido += 1
                elif mode == "gemini":
                    n_gemini += 1
            else:
                n_fail += 1
            pbar.set_postfix(ok=n_ok, falha=n_fail, rapido=n_rapido, gemini=n_gemini)
            pbar.update(1)

    with ThreadPoolExecutor(max_workers=PDF_WORKERS) as executor:
        with tqdm(total=len(pending), desc="PDFs", unit="pdf") as pbar:
            await asyncio.gather(*[process_with_guard(p) for p in pending])

    logger.info("=" * 60)
    logger.info(f"Finalizado -- OK: {n_ok} | Falha: {n_fail}")
    logger.info(f"  Sem API (rapido): {n_rapido} PDFs")
    logger.info(f"  Com Gemini:       {n_gemini} PDFs")
    logger.info(f"Total no ChromaDB:  {collection.count()} chunks")
    logger.info("=" * 60)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
