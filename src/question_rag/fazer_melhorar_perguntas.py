"""
fazer_melhorar_perguntas.py — Módulo central do pipeline RAG.

Expõe as funções de carregamento do banco, criação do retriever e
chains de resposta. Importado por chat_rag.py.
"""
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# ── Configurações ─────────────────────────────────────────────────────────────
ROOT              = Path(__file__).resolve().parent.parent.parent
VECTOR_STORE_PATH = str(ROOT / "data" / "vector_store")

MODELO_EMBED    = "intfloat/multilingual-e5-small"
COLLECTION_NAME = "embeddings_salvar"
MODELO_LLM      = "llama-3.1-8b-instant"

# Docs relevantes para perguntas sobre ANEEL ficam acima de 0.85.
# 0.65 elimina ruído (footnotes, cabeçalhos) sem descartar docs legítimos.
SCORE_THRESHOLD = 0.65
K_DOCS          = 5


# ── Embedding com prefixos corretos para e5 ───────────────────────────────────
class E5Embeddings(HuggingFaceEmbeddings):
    """
    O modelo intfloat/multilingual-e5 foi treinado com instruction tuning.
    Ele exige prefixos diferentes para documentos e queries:
      - Documentos indexados: 'passage: {texto}'  → feito em aplicacao_ocr.py
      - Queries de busca:     'query: {texto}'    → feito aqui em embed_query

    Sem esse prefixo, query e documentos ficam em regiões diferentes do espaço
    vetorial e a busca por similaridade retorna resultados incorretos.
    """
    def embed_query(self, text: str) -> list[float]:
        return super().embed_documents([f"query: {text}"])[0]


# ── Funções públicas ──────────────────────────────────────────────────────────
def carregar_banco() -> Chroma:
    modelo_embedding = E5Embeddings(
        model_name=MODELO_EMBED,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=modelo_embedding,
    )


def criar_llm() -> ChatGroq:
    return ChatGroq(
        model=MODELO_LLM,
        temperature=0.2,
        max_tokens=1024,
    )


def criar_retriever(banco: Chroma):
    """
    Busca direta por similaridade cosine com prefixo 'query:' via E5Embeddings.

    O RePhraseQueryRetriever (LLM converte pergunta em palavras-chave) foi removido
    porque degradava a qualidade: keywords genéricas batem com footnotes e rodapés
    em vez do conteúdo real. O modelo e5 com instrução 'query:' já é otimizado
    para perguntas em linguagem natural — não precisa de pré-processamento.
    """
    return banco.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": K_DOCS, "score_threshold": SCORE_THRESHOLD},
    )


def criar_chain_resposta(llm):
    prompt = PromptTemplate(
        input_variables=["contexto", "pergunta"],
        template="""Você é um assistente especializado em regulações da ANEEL e do mercado de energia elétrica brasileiro.
Responda com base EXCLUSIVAMENTE no contexto abaixo.
Se a informação não estiver no contexto, responda: "Não encontrei essa informação nos documentos disponíveis."

Contexto:
{contexto}

Pergunta: {pergunta}

Resposta:""",
    )
    return prompt | llm | StrOutputParser()


def recuperar_documentos(retriever, pergunta: str) -> list:
    return retriever.invoke(pergunta)


def formatar_contexto(documentos: list) -> str:
    partes = []
    for i, doc in enumerate(documentos):
        fonte = doc.metadata.get("file_name", "Desconhecido")
        partes.append(f"[Documento {i+1} — {fonte}]\n{doc.page_content}")
    return "\n\n---\n\n".join(partes)


# ── Teste rápido ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    banco     = carregar_banco()
    print(f"Total de documentos no banco: {banco._collection.count()}")

    llm       = criar_llm()
    retriever = criar_retriever(banco)
    chain     = criar_chain_resposta(llm)

    pergunta  = "O que é necessário para se cadastrar como agente na CCEE?"
    docs      = recuperar_documentos(retriever, pergunta)
    print(f"Documentos recuperados: {len(docs)}")

    if not docs:
        print(f"Nenhum documento passou o filtro de relevância (score < {SCORE_THRESHOLD}).")
    else:
        contexto = formatar_contexto(docs)
        print("\n=== CONTEXTO (primeiros 1500 chars) ===")
        print(contexto[:1500])
        resposta = chain.invoke({"contexto": contexto, "pergunta": pergunta})
        print("\n=== RESPOSTA ===")
        print(resposta)
