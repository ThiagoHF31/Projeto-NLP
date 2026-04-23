"""
chat_rag.py — Interface de chat interativo com RAG sobre documentos ANEEL.

Mantém histórico de conversa e mostra as fontes usadas em cada resposta.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from fazer_melhorar_perguntas import (
    MODELO_LLM,
    carregar_banco,
    criar_llm,
    criar_retriever,
    formatar_contexto,
    recuperar_documentos,
)

MAX_HISTORICO = 5   # turnos anteriores mantidos no contexto


def _formatar_historico(historico: list) -> str:
    linhas = []
    for turno in historico:
        linhas.append(f"Usuário: {turno['pergunta']}")
        linhas.append(f"Assistente: {turno['resposta']}")
    return "\n".join(linhas)


def criar_chain_chat(llm):
    prompt = PromptTemplate(
        input_variables=["historico_bloco", "contexto", "pergunta"],
        template="""Você é um assistente especializado em regulações da ANEEL e do mercado de energia elétrica brasileiro.
Responda com base EXCLUSIVAMENTE no contexto de documentos abaixo.
Se a informação não estiver no contexto, responda: "Não encontrei essa informação nos documentos disponíveis."

{historico_bloco}Contexto dos documentos:
{contexto}

Pergunta atual: {pergunta}

Resposta:""",
    )
    return prompt | llm | StrOutputParser()


def _fontes(documentos: list) -> str:
    vistas = []
    for doc in documentos:
        nome = doc.metadata.get("file_name", "Desconhecido")
        if nome not in vistas:
            vistas.append(nome)
    return ", ".join(vistas)


def main():
    print("Carregando banco vetorial...")
    banco = carregar_banco()
    print(f"Banco carregado — {banco._collection.count()} chunks indexados.")
    print(f"Modelo LLM : {MODELO_LLM} (Groq)\n")

    llm        = criar_llm()
    retriever  = criar_retriever(banco)
    chain_chat = criar_chain_chat(llm)

    historico: list = []
    print("Chat RAG ANEEL iniciado. Digite 'sair' para encerrar.\n")
    print("─" * 60)

    while True:
        try:
            pergunta = input("\nVocê: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando chat.")
            break

        if not pergunta:
            continue
        if pergunta.lower() in ("sair", "exit", "quit"):
            print("Encerrando chat.")
            break

        documentos = recuperar_documentos(retriever, pergunta)

        if not documentos:
            msg = "Não encontrei documentos relevantes para essa pergunta. Tente reformular com termos mais técnicos."
            print(f"\nAssistente: {msg}\n")
            historico.append({"pergunta": pergunta, "resposta": msg})
            continue

        contexto = formatar_contexto(documentos)

        historico_bloco = ""
        if historico:
            hist_str = _formatar_historico(historico[-MAX_HISTORICO:])
            historico_bloco = f"Histórico da conversa:\n{hist_str}\n\n"

        resposta = chain_chat.invoke({
            "historico_bloco": historico_bloco,
            "contexto"       : contexto,
            "pergunta"       : pergunta,
        })

        print(f"\nAssistente: {resposta}")
        print(f"\n  Fontes ({len(documentos)} chunk(s)): {_fontes(documentos)}")
        print("─" * 60)

        historico.append({"pergunta": pergunta, "resposta": resposta})


if __name__ == "__main__":
    main()
