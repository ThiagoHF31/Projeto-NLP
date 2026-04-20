#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_rag_sota.py — Pipeline RAG SOTA para PDFs ANEEL (versão otimizada)

Otimizações vs versão anterior:
  - PDFs <= SMALL_PDF_PAGES paginas (81% do acervo): ZERO chamadas LLM.
    O contexto é gerado diretamente do nome do arquivo + sumario simples.
  - PDFs grandes: enriquecimento em BATCH (LLM_BATCH_SIZE chunks por chamada),
    reduzindo chamadas LLM em ~8x.
  - Embedding model em float16 na CPU: metade da RAM (~1.1 GB vs ~2.3 GB).
  - Logs HTTP do HuggingFace silenciados.
  - Encoding UTF-8 forçado para compatibilidade com Windows.
"""

import gc
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Força UTF-8 no stdout/stderr antes de qualquer print
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Silencia logs HTTP verbosos do HuggingFace e httpx ANTES dos imports
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import fitz  # PyMuPDF
import requests
import chromadb
import torch
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Silencia loggers HTTP apos os imports
for noisy in ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub",
              "huggingface_hub.file_download", "sentence_transformers"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

try:
    import pymupdf4llm
    HAS_PYMUPDF4LLM = True
except ImportError:
    HAS_PYMUPDF4LLM = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACAO
# ══════════════════════════════════════════════════════════════════════════════

ROOT_DIR        = Path(__file__).parent.parent.parent   # src/pipeline/ → src/ → project root
DATA_DIR        = ROOT_DIR / "data"
PDF_DIR         = DATA_DIR / "pdfs"
CHROMA_DIR      = DATA_DIR / "vector_store"
CHECKPOINT_FILE = DATA_DIR / "processed" / "checkpoint.json"
ERROR_LOG       = ROOT_DIR / "logs" / "pipeline.log"

# LM Studio
LLM_URL         = "http://127.0.0.1:1234/v1/chat/completions"
LLM_MODEL       = "qwen/qwen3-14b"
LLM_TIMEOUT_S   = 180
LLM_MAX_RETRIES = 3

# Chunking
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200

# --- Otimizacao de velocidade e memoria ---
# PDFs com ate N paginas NAO fazem chamada LLM (81% do acervo tem <= 5 pags)
SMALL_PDF_PAGES = 5
# Quantos chunks enviamos por chamada LLM (reduz chamadas em ~8x)
LLM_BATCH_SIZE  = 8
# Janela de contexto para PDFs grandes
SLIDE_WINDOW    = 2
# Paginas iniciais para sumario (PDFs grandes)
SUMMARY_PAGES   = 10

# Embeddings — float16 na CPU usa ~1.1 GB RAM (vs ~2.3 GB em float32)
EMBED_MODEL_NAME  = "BAAI/bge-m3"
EMBED_BATCH_SIZE  = 32

CHROMA_COLLECTION = "aneel_legislacao"

# Limite de teste (None = todos os PDFs)
TEST_LIMIT: Optional[int] = 2

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING — sem caracteres especiais para compatibilidade Windows CP1252
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging() -> logging.Logger:
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)-8s] %(message)s"
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(fmt))

    fh = logging.FileHandler(ERROR_LOG, encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt))

    root.handlers.clear()
    root.addHandler(sh)
    root.addHandler(fh)

    # Re-silencia loggers verbosos apos configurar o root
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub",
                  "huggingface_hub.file_download", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    return logging.getLogger(__name__)

logger = setup_logging()

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINTING
# ══════════════════════════════════════════════════════════════════════════════

def load_checkpoint() -> Set[str]:
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f).get("processed", []))
        except (json.JSONDecodeError, KeyError):
            logger.warning("Checkpoint corrompido, reiniciando do zero.")
    return set()


def save_checkpoint(processed: Set[str]) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"processed": sorted(processed)}, f, ensure_ascii=False)
    tmp.replace(CHECKPOINT_FILE)

# ══════════════════════════════════════════════════════════════════════════════
# EXTRACAO DE PDF
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
# LLM — CHAMADAS STATELESS AO LM STUDIO
# ══════════════════════════════════════════════════════════════════════════════

def _llm_call(prompt: str, max_tokens: int = 600) -> Optional[str]:
    """
    Chamada REST stateless. /no_think desabilita o thinking mode do Qwen3,
    tornando as respostas mais rapidas e sem tokens <think>...</think>.
    """
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": f"/no_think\n\n{prompt}"}],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "stream": False,
    }
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            r = requests.post(LLM_URL, json=payload, timeout=LLM_TIMEOUT_S)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.Timeout:
            logger.warning(f"  LLM timeout (tentativa {attempt}/{LLM_MAX_RETRIES})")
            time.sleep(5 * attempt)
        except requests.exceptions.ConnectionError:
            logger.error("  LM Studio nao esta respondendo em 127.0.0.1:1234")
            time.sleep(10)
        except Exception as e:
            logger.warning(f"  LLM erro (tentativa {attempt}/{LLM_MAX_RETRIES}): {e}")
            time.sleep(3)
    return None


_SUMMARY_PROMPT = """\
Faca um resumo executivo de no maximo 3 linhas deste documento regulatorio, \
identificando: Tema principal, Entidades envolvidas, Ano e Proposito.
Responda apenas com o resumo, sem introducoes.

DOCUMENTO:
{text}"""


def generate_global_summary(pdf_path: Path, num_pages: int) -> str:
    text = extract_pages_text(pdf_path, 0, min(SUMMARY_PAGES, num_pages))
    if not text.strip():
        return "Documento sem texto nas primeiras paginas."
    result = _llm_call(_SUMMARY_PROMPT.format(text=text[:5000]), max_tokens=200)
    return result or "Sumario indisponivel."


# Prompt de enriquecimento em BATCH — processa N chunks por chamada
_BATCH_ENRICH_PROMPT = """\
Voce e especialista em RAG (Retrieval-Augmented Generation).
Para cada CHUNK listado abaixo, gere um contexto de 2-3 frases que situe \
aquele trecho dentro do documento.
NAO resuma o chunk; explique de onde ele vem e a que entidade/topico pertence.

SUMARIO DO DOCUMENTO:
{summary}

CHUNKS PARA ENRIQUECER:
{chunks_block}

INSTRUCOES:
- Retorne SOMENTE um objeto JSON valido no formato: {{"contexts": ["ctx_1", "ctx_2", ...]}}
- A lista deve ter exatamente {n} itens, na mesma ordem dos chunks.
- Cada item: max 3 frases, sem tags XML, sem formatacao extra.
- Exemplo de item bom: "Este trecho pertence a Resolucao Normativa ANEEL 2022 \
sobre revisao tarifaria da distribuidora X e detalha os criterios de reajuste."

RESPOSTA JSON:"""


def _build_chunks_block(chunks_content: List[str]) -> str:
    lines = []
    for i, c in enumerate(chunks_content, 1):
        lines.append(f"[{i}] {c[:600]}")
    return "\n\n".join(lines)


def _parse_batch_response(raw: str, expected: int) -> List[Optional[str]]:
    """Extrai a lista de contextos do JSON retornado pelo LLM."""
    try:
        # Tenta encontrar o JSON mesmo que haja texto antes/depois
        match = re.search(r'\{.*"contexts"\s*:\s*\[.*?\]\s*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            ctxs = data.get("contexts", [])
            if len(ctxs) == expected:
                return ctxs
    except Exception:
        pass
    # Fallback: retorna None para todos (usara chunk original)
    return [None] * expected


def enrich_batch(
    chunks_content: List[str],
    global_summary: str,
) -> List[str]:
    """
    Enriquece uma lista de chunks em UMA UNICA chamada LLM.
    Retorna lista de chunks enriquecidos (ou originais em caso de falha).
    """
    n = len(chunks_content)
    prompt = _BATCH_ENRICH_PROMPT.format(
        summary=global_summary[:600],
        chunks_block=_build_chunks_block(chunks_content),
        n=n,
    )
    raw = _llm_call(prompt, max_tokens=100 * n + 50)
    if raw:
        contexts = _parse_batch_response(raw, n)
        return [
            f"{ctx}\n\n{chunk}" if ctx else chunk
            for ctx, chunk in zip(contexts, chunks_content)
        ]
    return chunks_content  # fallback: sem enriquecimento


def enrich_small_pdf(
    chunks: List[Dict],
    pdf_name: str,
    first_page_text: str,
) -> List[str]:
    """
    Para PDFs pequenos (<=5 paginas): zero chamadas LLM.
    Contexto sintetico gerado a partir do nome do arquivo e do texto da 1a pagina.
    """
    # Extrai um titulo simples do nome do arquivo (sem extensao e underscores)
    titulo = Path(pdf_name).stem.replace("_", " ").replace("-", " ")[:80]
    # Primeiras 200 chars da primeira pagina como identificador
    intro = first_page_text.strip()[:200].replace("\n", " ")
    header = f"Documento ANEEL: {titulo}. {intro}"
    return [f"{header}\n\n{c['content']}" for c in chunks]

# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDINGS — float16 para economizar RAM
# ══════════════════════════════════════════════════════════════════════════════

def load_embedding_model() -> SentenceTransformer:
    logger.info(f"Carregando modelo de embeddings '{EMBED_MODEL_NAME}' (CPU, float16)...")
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


def _doc_id(pdf_name: str, chunk_idx: int) -> str:
    return hashlib.md5(f"{pdf_name}::{chunk_idx:06d}".encode()).hexdigest()


def store_chunks(
    collection: chromadb.Collection,
    pdf_name: str,
    enriched_chunks: List[str],
    embeddings: List[List[float]],
    chunk_metas: List[Dict],
) -> None:
    ids = [_doc_id(pdf_name, i) for i in range(len(enriched_chunks))]
    metadatas = []
    for i, meta in enumerate(chunk_metas):
        m = {"source": pdf_name, "chunk_index": i}
        for k, v in meta.items():
            m[k] = str(v) if v is not None else ""
        metadatas.append(m)
    collection.add(
        ids=ids,
        documents=enriched_chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE: UM PDF
# ══════════════════════════════════════════════════════════════════════════════

def process_pdf(
    pdf_path: Path,
    collection: chromadb.Collection,
    embed_model: SentenceTransformer,
) -> bool:
    name = pdf_path.name

    # Extracao
    try:
        md_text, num_pages = extract_pdf_as_markdown(pdf_path)
    except Exception as e:
        logger.error(f"[EXTRACAO] {name}: {e}")
        return False

    if not md_text.strip():
        logger.warning(f"[SKIP] {name}: sem texto extraivel.")
        return True

    # Chunking
    try:
        chunks = chunk_markdown(md_text)
    except Exception as e:
        logger.error(f"[CHUNK] {name}: {e}")
        return False

    if not chunks:
        logger.warning(f"[SKIP] {name}: nenhum chunk gerado.")
        return True

    # ── Estrategia de enriquecimento ──────────────────────────────────────────
    if num_pages <= SMALL_PDF_PAGES:
        # PDF PEQUENO: sem LLM, contexto sintetico instantaneo
        first_page = extract_pages_text(pdf_path, 0, 1)
        enriched_chunks = enrich_small_pdf(chunks, name, first_page)
        mode = "rapido"
    else:
        # PDF GRANDE: sumario + enriquecimento em batch
        try:
            global_summary = generate_global_summary(pdf_path, num_pages)
        except Exception as e:
            logger.warning(f"[SUMARIO] {name}: {e}")
            global_summary = "Sumario indisponivel."

        enriched_chunks = []
        contents = [c["content"] for c in chunks]

        # Processa em batches de LLM_BATCH_SIZE
        for batch_start in range(0, len(contents), LLM_BATCH_SIZE):
            batch = contents[batch_start: batch_start + LLM_BATCH_SIZE]
            try:
                enriched_batch = enrich_batch(batch, global_summary)
            except Exception as e:
                logger.warning(f"  [ENRICH] batch {batch_start} de {name}: {e}")
                enriched_batch = batch
            enriched_chunks.extend(enriched_batch)
            gc.collect()

        mode = "llm-batch"

    chunk_metas = [c["metadata"] for c in chunks]

    # Embeddings
    try:
        embeddings = generate_embeddings(embed_model, enriched_chunks)
    except Exception as e:
        logger.error(f"[EMBED] {name}: {e}")
        return False

    # Salva no ChromaDB
    try:
        store_chunks(collection, name, enriched_chunks, embeddings, chunk_metas)
    except Exception as e:
        logger.error(f"[CHROMA] {name}: {e}")
        return False

    logger.info(f"[OK/{mode}] {name} -- {num_pages}pag, {len(chunks)} chunks")

    # Libera memoria da lista de chunks antes do proximo PDF
    del enriched_chunks, embeddings, chunks, chunk_metas
    gc.collect()

    return True

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    logger.info("=" * 55)
    logger.info("Pipeline RAG SOTA -- PDFs ANEEL (versao otimizada)")
    logger.info("=" * 55)

    if not HAS_PYMUPDF4LLM:
        logger.warning("pymupdf4llm ausente: pip install pymupdf4llm")

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
    logger.info(f"Estrategia:     sem LLM para PDFs <= {SMALL_PDF_PAGES} pags | batch={LLM_BATCH_SIZE} chunks/chamada")

    if not pending:
        logger.info("Nenhum PDF pendente.")
        return

    collection  = get_chroma_collection()
    embed_model = load_embedding_model()

    n_ok = n_fail = n_rapido = n_llm = 0

    with tqdm(total=len(pending), desc="PDFs", unit="pdf") as bar:
        for pdf_path in pending:
            # Conta antes para o log de modo
            pages = pdf_num_pages(pdf_path)
            is_small = pages <= SMALL_PDF_PAGES

            success = process_pdf(pdf_path, collection, embed_model)

            if success:
                processed.add(pdf_path.name)
                save_checkpoint(processed)
                n_ok += 1
                if is_small:
                    n_rapido += 1
                else:
                    n_llm += 1
            else:
                n_fail += 1

            bar.set_postfix(ok=n_ok, falha=n_fail, rapido=n_rapido, llm=n_llm)
            bar.update(1)

    logger.info("=" * 55)
    logger.info(f"Finalizado -- OK: {n_ok} | Falha: {n_fail}")
    logger.info(f"  Sem LLM (rapido): {n_rapido} PDFs")
    logger.info(f"  Com LLM (batch):  {n_llm} PDFs")
    logger.info(f"Total no ChromaDB: {collection.count()} chunks")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
