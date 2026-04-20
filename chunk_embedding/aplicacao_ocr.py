import time
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
import traceback
import json
from langchain_core.documents import Document


# ─────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────
CAMINHO_PDFS  = Path("C:\\Users\\paogr\\Desktop\\Projeto-NLP\\data\\processed\\extracted_json")
MODELO_TOKEN  = "intfloat/multilingual-e5-small"
MODELO_EMBED  = "intfloat/multilingual-e5-small"
NOME_COLECAO  = "embeddings_salvar"
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 103
MAX_ARQUIVOS  = 1


# ─────────────────────────────────────────
# INICIALIZAÇÃO
# ─────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODELO_TOKEN)

modelo_embedding = HuggingFaceEmbeddings(
    model_name=MODELO_EMBED,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

client     = chromadb.PersistentClient(path="C:\\Users\\paogr\\Desktop\\NLP\\chroma_db")
collection = client.get_or_create_collection(name=NOME_COLECAO)

splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


# ─────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ─────────────────────────────────────────
def _table_to_text(data: list[dict]) -> str:
    """Converte lista de dicts (linhas da tabela) em texto legível por pipe."""
    if not data:
        return ""

    def clean(s) -> str:
        return str(s).replace("\n", " ").strip()

    headers = list(data[0].keys())
    linhas  = [" | ".join(clean(h) for h in headers), "-" * 60]

    for row in data:
        linhas.append(" | ".join(clean(row.get(h, "")) for h in headers))

    return "\n".join(linhas)


def _sanitize_metadata(metadata: dict) -> dict:
    """Garante que todos os valores sejam scalars aceitos pelo ChromaDB."""
    return {
        k: v if isinstance(v, (str, int, float, bool)) else str(v)
        for k, v in metadata.items()
    }


def json_document(nome_arquivo: str) -> list[Document]:
    """Lê um arquivo JSON extraído de PDF e retorna lista de Documents."""
    documentos_total = []

    with open(nome_arquivo, encoding="utf-8") as arquivo:
        documento = json.load(arquivo)

    for block in documento["content_blocks"]:
        metadata = {
            **documento["metadata"],
            "file_path":    documento["file_path"],
            "page":         block["page"],
            "block_type":   block["type"],
            "block_source": block["source"],
        }

        if block["type"] == "text":
            content = block.get("content", "").replace("\\n", "\n")

        elif block["type"] == "table":
            metadata["bbox"] = str(block.get("bbox", []))
            content = _table_to_text(block.get("data", []))

        else:
            content = block.get("content", str(block))

        if content.strip():
            documentos_total.append(
                Document(page_content=content, metadata=metadata)
            )

    return documentos_total


def _sanitize_metadata(metadata: dict) -> dict:
    """Garante que todos os valores sejam scalars aceitos pelo ChromaDB."""
    return {
        k: v if isinstance(v, (str, int, float, bool)) else str(v)
        for k, v in metadata.items()
    }


# ─────────────────────────────────────────
# PROCESSAMENTO PRINCIPAL
# ─────────────────────────────────────────
tempo_inicio  = time.time()
arquivos_proc = 0

try:
    for pdf in CAMINHO_PDFS.iterdir():
        print(f"\nProcessando: {pdf.name} …")

        pages  = json_document(str(pdf))
        chunks = splitter.split_documents(pages)

        # ✅ Prefixo "passage: " obrigatório para indexação no E5
        textos_raw     = [chunk.page_content for chunk in chunks]
        textos_prefixo = [f"passage: {t}" for t in textos_raw]

        vetores   = modelo_embedding.embed_documents(textos_prefixo)
        ids       = [f"{pdf.stem}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [_sanitize_metadata(chunk.metadata) for chunk in chunks]

        collection.upsert(
            ids=ids,
            documents=textos_raw,       # ✅ armazena o texto limpo (sem prefixo)
            embeddings=vetores,          # ✅ vetores gerados COM prefixo
            metadatas=metadatas,
        )

        print(f"  ✅ {len(textos_raw)} chunks adicionados à coleção '{NOME_COLECAO}'.")

        arquivos_proc += 1
        if MAX_ARQUIVOS is not None and arquivos_proc >= MAX_ARQUIVOS:
            break

    tempo_total = time.time() - tempo_inicio
    minutos     = int(tempo_total // 60)
    segundos    = tempo_total % 60

    print(f"\n📦 Total de itens na coleção: {collection.count()}")
    print(f"⏱️  Tempo total: {minutos}m {segundos:.2f}s")

except Exception as e:
    with open("log.txt", mode="a", encoding="utf-8") as arquivo:
        arquivo.write("=== ERRO ===\n")
        arquivo.write(traceback.format_exc())
        arquivo.write("\n")
    print(f"❌ Erro registrado em log.txt: {e}")