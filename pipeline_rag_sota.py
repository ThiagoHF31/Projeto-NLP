#!/usr/bin/env python3
"""
pipeline_rag_sota.py — Pipeline RAG SOTA para PDFs ANEEL

Fluxo por PDF:
  1. Extração → Markdown (PyMuPDF4LLM + fallback fitz)
  2. Sumário global via Qwen3 14B (LM Studio, chamadas STATELESS)
  3. Chunking estrutural: MarkdownHeaderSplitter + RecursiveCharacterSplitter
  4. Enriquecimento contextual com janela deslizante (Qwen3, stateless)
  5. Embeddings (BAAI/bge-m3, CPU) → ChromaDB persistente
  6. Checkpointing JSON para retomada automática após falhas

Requisitos:
    pip install pymupdf pymupdf4llm langchain-text-splitters chromadb
                sentence-transformers requests tqdm
"""

import gc
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import fitz  # PyMuPDF
import requests
import chromadb
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    import pymupdf4llm
    HAS_PYMUPDF4LLM = True
except ImportError:
    HAS_PYMUPDF4LLM = False
    print("[AVISO] pymupdf4llm não encontrado. Usando extração fitz como fallback.")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR          = Path(__file__).parent
PDF_DIR           = BASE_DIR / "dados" / "pdfs"
CHROMA_DIR        = BASE_DIR / "dados" / "chroma_db"
CHECKPOINT_FILE   = BASE_DIR / "dados" / "checkpoint.json"
ERROR_LOG         = BASE_DIR / "dados" / "erros_pipeline.log"

# LM Studio — API compatível com OpenAI em localhost
LLM_URL           = "http://127.0.0.1:1234/v1/chat/completions"
LLM_MODEL         = "qwen/qwen3-14b"
LLM_TIMEOUT_S     = 180          # timeout por chamada ao LLM (segundos)
LLM_MAX_RETRIES   = 3            # tentativas em caso de falha de rede

# Gestão de VRAM: a cada N chunks processados, força GC no Python.
# Como as chamadas já são stateless, isso apenas libera objetos Python acumulados.
VRAM_GC_EVERY     = 3

# Chunking
CHUNK_SIZE        = 1000
CHUNK_OVERLAP     = 200

# PDFs com mais páginas que este limiar usam estratégia de "sumário + janela"
LARGE_PDF_PAGES   = 50
SUMMARY_PAGES     = 10   # páginas iniciais para gerar o sumário global
SLIDE_WINDOW      = 2    # chunks vizinhos (antes e depois) para contexto

# Embeddings
EMBED_MODEL_NAME  = "BAAI/bge-m3"   # multilingual, ótimo para português
EMBED_BATCH_SIZE  = 32

CHROMA_COLLECTION = "aneel_legislacao"

# Limite de PDFs para teste (None = processa tudo)
TEST_LIMIT: Optional[int] = 100

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging() -> logging.Logger:
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)-8s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(ERROR_LOG, encoding="utf-8"),
        ],
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINTING
# ══════════════════════════════════════════════════════════════════════════════

def load_checkpoint() -> Set[str]:
    """Retorna o conjunto de nomes de arquivos já processados com sucesso."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f).get("processed", []))
        except (json.JSONDecodeError, KeyError):
            logger.warning("Checkpoint corrompido, iniciando do zero.")
    return set()


def save_checkpoint(processed: Set[str]) -> None:
    """Persiste o checkpoint de forma atômica (write-then-rename)."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"processed": sorted(processed)}, f, ensure_ascii=False)
    tmp.replace(CHECKPOINT_FILE)

# ══════════════════════════════════════════════════════════════════════════════
# EXTRAÇÃO DE PDF → MARKDOWN
# ══════════════════════════════════════════════════════════════════════════════

def pdf_num_pages(pdf_path: Path) -> int:
    doc = fitz.open(str(pdf_path))
    n = len(doc)
    doc.close()
    return n


def extract_pages_text(pdf_path: Path, start: int = 0, end: Optional[int] = None) -> str:
    """Extrai texto puro de um intervalo de páginas usando fitz."""
    doc = fitz.open(str(pdf_path))
    end = min(end if end is not None else len(doc), len(doc))
    parts = []
    for i in range(start, end):
        page = doc[i]
        text = page.get_text("text")
        # Detecta página quase em branco (provável imagem sem OCR)
        if len(text.strip()) < 30 and page.get_images():
            parts.append(f"[Página {i+1}: imagem sem texto extraível]")
        else:
            parts.append(text)
    doc.close()
    return "\n\n".join(parts)


def extract_pdf_as_markdown(pdf_path: Path) -> Tuple[str, int]:
    """
    Retorna (markdown_text, num_pages).

    Prioridade:
      1. pymupdf4llm  — converte estrutura (títulos, tabelas, listas) para MD
      2. fitz nativo  — extração página a página com separadores de seção
    """
    num_pages = pdf_num_pages(pdf_path)

    if HAS_PYMUPDF4LLM:
        try:
            md = pymupdf4llm.to_markdown(str(pdf_path))
            if md.strip():
                return md, num_pages
        except Exception as e:
            logger.debug(f"pymupdf4llm falhou em {pdf_path.name}: {e}. Usando fallback fitz.")

    # Fallback: extração fitz página a página
    doc = fitz.open(str(pdf_path))
    parts = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            parts.append(f"## Página {i+1}\n\n{text}")
        elif page.get_images():
            parts.append(f"## Página {i+1}\n\n[imagem sem texto extraível]")
    doc.close()
    return "\n\n".join(parts), num_pages

# ══════════════════════════════════════════════════════════════════════════════
# CHUNKING ESTRUTURAL
# ══════════════════════════════════════════════════════════════════════════════

_MARKDOWN_HEADERS = [
    ("#",    "H1"),
    ("##",   "H2"),
    ("###",  "H3"),
    ("####", "H4"),
]


def chunk_markdown(md_text: str) -> List[Dict]:
    """
    Dois estágios:
      1. MarkdownHeaderTextSplitter — respeita seções do documento
      2. RecursiveCharacterTextSplitter — quebra seções grandes

    Retorna lista de {"content": str, "metadata": dict}.
    """
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_MARKDOWN_HEADERS,
        strip_headers=False,
    )
    header_chunks = md_splitter.split_text(md_text)

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    result = []
    for hc in header_chunks:
        if not hc.page_content.strip():
            continue
        sub_chunks = recursive_splitter.split_text(hc.page_content)
        for sc in sub_chunks:
            if sc.strip():
                result.append({"content": sc, "metadata": dict(hc.metadata)})

    return result

# ══════════════════════════════════════════════════════════════════════════════
# LLM — CHAMADAS STATELESS AO LM STUDIO
# ══════════════════════════════════════════════════════════════════════════════

def _llm_call(prompt: str, temperature: float = 0.1, max_tokens: int = 512) -> Optional[str]:
    """
    Chamada REST STATELESS: envia apenas a mensagem atual, sem histórico.
    Isso garante que o LM Studio não acumule KV-cache entre requisições.
    O /no_think instrui o Qwen3 a desabilitar o modo de raciocínio encadeado
    (mais rápido e sem overhead de tokens de thinking).
    """
    # /no_think desabilita o "thinking mode" do Qwen3 (tokens <think>...</think>)
    full_prompt = f"/no_think\n\n{prompt}"

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            resp = requests.post(LLM_URL, json=payload, timeout=LLM_TIMEOUT_S)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.Timeout:
            logger.warning(f"  LLM timeout (tentativa {attempt}/{LLM_MAX_RETRIES})")
            time.sleep(5 * attempt)
        except requests.exceptions.ConnectionError:
            logger.error("  LM Studio não está respondendo em localhost:1234. Verifique se está rodando.")
            time.sleep(10)
        except Exception as e:
            logger.warning(f"  Erro LLM (tentativa {attempt}/{LLM_MAX_RETRIES}): {e}")
            time.sleep(3)

    return None


_SUMMARY_PROMPT = """\
Faça um resumo executivo de no máximo 4 linhas deste documento regulatório, \
identificando estritamente: Tema principal, Entidades/Empresas envolvidas, Ano e Propósito do texto.
Responda apenas com o resumo, sem introduções ou formatação extra.

DOCUMENTO:
{text}"""


def generate_global_summary(pdf_path: Path, num_pages: int) -> str:
    """Gera sumário executivo das primeiras páginas do PDF."""
    pages = min(SUMMARY_PAGES, num_pages)
    raw_text = extract_pages_text(pdf_path, 0, pages)

    if not raw_text.strip():
        return "Documento sem texto nas primeiras páginas."

    # Trunca para não estourar o contexto do LLM (≈ 5k chars ≈ 1.2k tokens)
    text = raw_text[:5000]
    result = _llm_call(_SUMMARY_PROMPT.format(text=text), temperature=0.1, max_tokens=256)
    return result or "Sumário indisponível."


_ENRICH_PROMPT = """\
Você é um especialista em estruturação de dados e extração de conhecimento para sistemas RAG.
Sua tarefa é analisar um trecho de texto (<chunk_alvo>) e gerar um contexto curto e explicativo \
que situe este trecho dentro do documento original.
O contexto deve responder implicitamente: "Sobre o que é este trecho, a que entidade ele se refere \
e qual é o tópico principal?".

<informacoes_do_documento>
<sumario_global>
{global_summary}
</sumario_global>
<contexto_anterior>
{previous_chunks}
</contexto_anterior>
<contexto_posterior>
{next_chunks}
</contexto_posterior>
</informacoes_do_documento>

<chunk_alvo>
{target_chunk}
</chunk_alvo>

INSTRUÇÕES RESTRITAS:
1. Analise o <chunk_alvo> em relação ao sumário global e aos contextos adjacentes.
2. Identifique nomes de empresas, pessoas, produtos, datas ou conceitos chave implícitos no \
<chunk_alvo>, mas explícitos no contexto.
3. Escreva um contexto conciso (máximo de 3 frases) que explique a situação do <chunk_alvo>.
4. NÃO resuma o chunk alvo; explique de onde ele vem. \
(Exemplo bom: "Este trecho pertence à Resolução Normativa da ANEEL de 2022 referente à \
revisão tarifária da distribuidora X e detalha os critérios de reajuste anual").
5. Retorne APENAS o texto do contexto gerado. Sem tags XML, sem formatação extra, sem introduções."""


def enrich_chunk(
    target_chunk: str,
    global_summary: str,
    prev_chunks: List[str],
    next_chunks: List[str],
) -> str:
    """
    Enriquece um chunk com contexto gerado pelo Qwen3.
    Retorna: "contexto_gerado\\n\\ntarget_chunk"
    Em caso de falha do LLM, retorna o chunk original sem enriquecimento.
    """
    prev_text = "\n---\n".join(prev_chunks) if prev_chunks else "(nenhum)"
    next_text = "\n---\n".join(next_chunks) if next_chunks else "(nenhum)"

    prompt = _ENRICH_PROMPT.format(
        global_summary=global_summary[:800],
        previous_chunks=prev_text[:2000],
        next_chunks=next_text[:2000],
        target_chunk=target_chunk[:3000],
    )

    context = _llm_call(prompt, temperature=0.1, max_tokens=300)
    if context:
        return f"{context}\n\n{target_chunk}"

    logger.debug(f"  Enriquecimento falhou; usando chunk original.")
    return target_chunk

# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

def load_embedding_model() -> SentenceTransformer:
    """
    Carrega bge-m3 na CPU para preservar VRAM para o Qwen3 14B.
    bge-m3 é multilingual e tem excelente desempenho em português.
    """
    logger.info(f"Carregando modelo de embeddings '{EMBED_MODEL_NAME}' na CPU...")
    model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    logger.info("Modelo de embeddings pronto.")
    return model


def generate_embeddings(model: SentenceTransformer, texts: List[str]) -> List[List[float]]:
    return model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,  # requerido pelo bge-m3 para cosine similarity
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
    logger.info(f"ChromaDB pronto. Documentos já indexados: {col.count()}")
    return col


def _doc_id(pdf_name: str, chunk_idx: int) -> str:
    """ID único e determinístico para cada chunk."""
    return hashlib.md5(f"{pdf_name}::{chunk_idx:06d}".encode()).hexdigest()


def store_chunks(
    collection: chromadb.Collection,
    pdf_name: str,
    enriched_chunks: List[str],
    embeddings: List[List[float]],
    chunk_metas: List[Dict],
) -> None:
    """Insere batch de chunks enriquecidos no ChromaDB."""
    ids = [_doc_id(pdf_name, i) for i in range(len(enriched_chunks))]

    metadatas = []
    for i, meta in enumerate(chunk_metas):
        m = {"source": pdf_name, "chunk_index": i}
        # Adiciona cabeçalhos Markdown como metadados de filtragem
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
# PIPELINE: PROCESSAMENTO DE UM PDF
# ══════════════════════════════════════════════════════════════════════════════

def process_pdf(
    pdf_path: Path,
    collection: chromadb.Collection,
    embed_model: SentenceTransformer,
) -> bool:
    """
    Pipeline completo para um único PDF.
    Retorna True em sucesso, False em erro irrecuperável.
    """
    name = pdf_path.name

    # ── Extração ──────────────────────────────────────────────────────────────
    try:
        md_text, num_pages = extract_pdf_as_markdown(pdf_path)
    except Exception as e:
        logger.error(f"[EXTRAÇÃO] {name}: {e}")
        return False

    if not md_text.strip():
        logger.warning(f"[SKIP] {name}: sem texto extraível.")
        return True  # não é erro; simplesmente vazio

    # ── Sumário global ────────────────────────────────────────────────────────
    try:
        global_summary = generate_global_summary(pdf_path, num_pages)
    except Exception as e:
        logger.warning(f"[SUMÁRIO] {name}: {e}. Usando fallback.")
        global_summary = "Sumário indisponível."

    # ── Chunking ──────────────────────────────────────────────────────────────
    try:
        chunks = chunk_markdown(md_text)
    except Exception as e:
        logger.error(f"[CHUNK] {name}: {e}")
        return False

    if not chunks:
        logger.warning(f"[SKIP] {name}: nenhum chunk gerado após split.")
        return True

    # ── Enriquecimento contextual + coleta ────────────────────────────────────
    enriched_chunks: List[str] = []
    chunk_metas: List[Dict]    = []

    with tqdm(
        total=len(chunks),
        desc=f"  chunks [{name[:35]}]",
        leave=False,
        unit="chunk",
    ) as pbar:
        for i, chunk in enumerate(chunks):
            target   = chunk["content"]
            prev_ctx = [c["content"] for c in chunks[max(0, i - SLIDE_WINDOW) : i]]
            next_ctx = [c["content"] for c in chunks[i + 1 : i + 1 + SLIDE_WINDOW]]

            try:
                enriched = enrich_chunk(target, global_summary, prev_ctx, next_ctx)
            except Exception as e:
                logger.warning(f"  [ENRICH] chunk {i} de {name}: {e}. Usando original.")
                enriched = target

            enriched_chunks.append(enriched)
            chunk_metas.append(chunk["metadata"])

            # Gestão de VRAM/memória Python: coleta periódica de lixo
            if (i + 1) % VRAM_GC_EVERY == 0:
                gc.collect()

            pbar.update(1)

    # ── Embeddings ────────────────────────────────────────────────────────────
    try:
        embeddings = generate_embeddings(embed_model, enriched_chunks)
    except Exception as e:
        logger.error(f"[EMBED] {name}: {e}")
        return False

    # ── Salva no ChromaDB ─────────────────────────────────────────────────────
    try:
        store_chunks(collection, name, enriched_chunks, embeddings, chunk_metas)
    except Exception as e:
        logger.error(f"[CHROMA] {name}: {e}")
        return False

    logger.info(f"[OK] {name} — {num_pages} págs, {len(chunks)} chunks indexados.")
    return True

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    logger.info("═" * 60)
    logger.info("Pipeline RAG SOTA — PDFs ANEEL")
    logger.info("═" * 60)

    if not PDF_DIR.exists():
        logger.error(f"Diretório de PDFs não encontrado: {PDF_DIR}")
        sys.exit(1)

    if not HAS_PYMUPDF4LLM:
        logger.warning("pymupdf4llm ausente. Instale com: pip install pymupdf4llm")

    all_pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not all_pdfs:
        logger.error(f"Nenhum PDF encontrado em {PDF_DIR}")
        sys.exit(1)

    processed = load_checkpoint()
    pending   = [p for p in all_pdfs if p.name not in processed]

    if TEST_LIMIT is not None:
        pending = pending[:TEST_LIMIT]
        logger.info(f"[MODO TESTE] Limitado a {TEST_LIMIT} PDFs.")

    logger.info(f"PDFs totais:     {len(all_pdfs)}")
    logger.info(f"Já processados:  {len(processed)}")
    logger.info(f"Pendentes:       {len(pending)}")

    if not pending:
        logger.info("Nenhum PDF pendente. Pipeline finalizado.")
        return

    # Inicializa recursos
    collection  = get_chroma_collection()
    embed_model = load_embedding_model()

    n_ok   = 0
    n_fail = 0

    with tqdm(total=len(pending), desc="PDFs", unit="pdf") as pdf_bar:
        for pdf_path in pending:
            success = process_pdf(pdf_path, collection, embed_model)

            if success:
                processed.add(pdf_path.name)
                save_checkpoint(processed)
                n_ok += 1
            else:
                n_fail += 1

            pdf_bar.set_postfix(ok=n_ok, falha=n_fail)
            pdf_bar.update(1)

    logger.info("═" * 60)
    logger.info(f"Pipeline finalizado — OK: {n_ok} | Falha: {n_fail}")
    logger.info(f"Total indexado no ChromaDB: {collection.count()} chunks")
    logger.info(f"Log de erros: {ERROR_LOG}")
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
