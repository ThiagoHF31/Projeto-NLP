"""
aplicacao_ocr.py — Pipeline de indexação de documentos ANEEL no ChromaDB.

Lê os JSONs extraídos dos PDFs, chunka o full_text de cada documento e
armazena os embeddings no banco vetorial para uso pelo RAG.
"""
import sys
import time
import hashlib
import traceback
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from tqdm import tqdm

# ── Configurações ─────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent.parent
CAMINHO_JSONS = ROOT / "data" / "processed" / "extracted_json"
VECTOR_STORE  = str(ROOT / "data" / "vector_store")
LOG_DIR       = ROOT / "logs"

MODELO        = "intfloat/multilingual-e5-small"
NOME_COLECAO  = "embeddings_salvar"

# e5-small suporta 512 tokens.
# 450 + "passage: " (3 tokens) + [CLS][SEP] (2 tokens) = 455 — margem segura.
CHUNK_SIZE    = 450
CHUNK_OVERLAP = 45

# Documentos com menos de MIN_TEXTO chars são cabeçalhos/imagens sem conteúdo
MIN_TEXTO     = 150

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


# ── Inicialização ──────────────────────────────────────────────────────────────
LOG_DIR.mkdir(exist_ok=True)
LOG_ERROS  = LOG_DIR / "indexacao_erros.txt"
LOG_PULADO = LOG_DIR / "indexacao_pulados.txt"

# Limpa logs anteriores
LOG_ERROS.write_text("", encoding="utf-8")
LOG_PULADO.write_text("", encoding="utf-8")

print(f"[init] Device     : {DEVICE}")
print(f"[init] Modelo     : {MODELO}")
print(f"[init] JSONs      : {CAMINHO_JSONS}")
print(f"[init] VectorStore: {VECTOR_STORE}")
print()

tokenizer = AutoTokenizer.from_pretrained(MODELO)

modelo_embedding = HuggingFaceEmbeddings(
    model_name=MODELO,
    model_kwargs={"device": DEVICE},
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 64,
    },
)

client = chromadb.PersistentClient(path=VECTOR_STORE)

# Recria a collection com distância cosine.
# Cosine é obrigatório para embeddings normalizados: score = cosine_similarity.
# L2 (padrão do chromadb) causa divergência no cálculo de score do LangChain.
try:
    client.delete_collection(NOME_COLECAO)
    print(f"[init] Collection '{NOME_COLECAO}' anterior removida.")
except Exception:
    pass

collection = client.create_collection(
    name=NOME_COLECAO,
    metadata={"hnsw:space": "cosine"},
)
print(f"[init] Collection '{NOME_COLECAO}' criada com distância cosine.\n")

splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


# ── Funções auxiliares ─────────────────────────────────────────────────────────
def _md5(text: str) -> str:
    """Hash do conteúdo para deduplicação de documentos idênticos."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _sanitize_metadata(meta: dict) -> dict:
    """ChromaDB só aceita str, int, float ou bool como valores de metadado."""
    return {
        k: v if isinstance(v, (str, int, float, bool)) else str(v)
        for k, v in meta.items()
    }


# ── Loop principal ─────────────────────────────────────────────────────────────
arquivos  = sorted(p for p in CAMINHO_JSONS.iterdir() if p.suffix == ".json")
vistos    = set()   # hashes MD5 de full_text já indexados
proc_ok   = 0
pulados   = 0
erros     = 0

tempo_inicio = time.time()

for caminho in tqdm(arquivos, desc="Indexando", unit="doc"):
    try:
        with open(caminho, encoding="utf-8") as f:
            doc = json.load(f)

        full_text = doc.get("full_text", "").strip()

        # ── Filtragem ──────────────────────────────────────────────────────────
        if len(full_text) < MIN_TEXTO:
            LOG_PULADO.open("a", encoding="utf-8").write(
                f"CURTO ({len(full_text)}c): {caminho.name}\n"
            )
            pulados += 1
            continue

        h = _md5(full_text)
        if h in vistos:
            LOG_PULADO.open("a", encoding="utf-8").write(
                f"DUPLICADO: {caminho.name}\n"
            )
            pulados += 1
            continue
        vistos.add(h)

        # ── Chunking ───────────────────────────────────────────────────────────
        chunks = splitter.split_text(full_text)
        if not chunks:
            pulados += 1
            continue

        # ── Metadados base do documento ────────────────────────────────────────
        meta = doc.get("metadata", {})
        base_meta = {
            "file_name" : meta.get("file_name", caminho.name),
            "num_pages" : meta.get("num_pages", 0),
            "author"    : meta.get("author", ""),
            "source_json": caminho.name,
        }

        # ── Embeddings ─────────────────────────────────────────────────────────
        # O modelo e5 exige prefixo "passage: " para documentos indexados.
        # A query usará "query: " (ver fazer_melhorar_perguntas.py → E5Embeddings).
        textos_embed = [f"passage: {c}" for c in chunks]
        vetores      = modelo_embedding.embed_documents(textos_embed)

        # ── Persistência ───────────────────────────────────────────────────────
        stem    = caminho.stem
        ids     = [f"{stem}_chunk_{i}" for i in range(len(chunks))]
        metas   = [
            _sanitize_metadata({
                **base_meta,
                "chunk_index" : i,
                "total_chunks": len(chunks),
            })
            for i in range(len(chunks))
        ]

        collection.upsert(
            ids=ids,
            documents=chunks,     # texto limpo SEM prefixo
            embeddings=vetores,   # vetores gerados COM prefixo "passage: "
            metadatas=metas,
        )

        proc_ok += 1

    except Exception as e:
        erros += 1
        with LOG_ERROS.open("a", encoding="utf-8") as lf:
            lf.write(f"{caminho.name}\t{e}\n{traceback.format_exc()}\n\n")

# ── Resumo ─────────────────────────────────────────────────────────────────────
tempo = time.time() - tempo_inicio
print(f"\n{'─'*50}")
print(f"✅ Indexados  : {proc_ok}")
print(f"⏭️  Pulados    : {pulados}")
print(f"❌ Erros      : {erros}")
print(f"📦 Total na coleção: {collection.count()}")
print(f"⏱️  Tempo      : {int(tempo // 60)}m {tempo % 60:.1f}s")
print(f"{'─'*50}")
if erros:
    print(f"   Detalhes dos erros: {LOG_ERROS}")
if pulados:
    print(f"   Detalhes dos pulados: {LOG_PULADO}")
