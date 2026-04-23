from langchain_community.llms import Ollama
from langchain_classic.retrievers import RePhraseQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

MODELO_EMBED = "intfloat/multilingual-e5-small"

modelo_embedding = HuggingFaceEmbeddings(
    model_name=MODELO_EMBED,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

banco = Chroma(
    collection_name="embeddings_salvar",
    persist_directory=r"C:\Users\paogr\Desktop\Projeto-NLP\data\vector_store",
    embedding_function=modelo_embedding
)

local_llm = Ollama(base_url="http://localhost:11434", model="llama3")

prompt_reescrita = PromptTemplate(
    input_variables=["question"],
    template="""Você é um especialista em busca de informações.
Sua tarefa é converter a pergunta natural do usuário em uma consulta de busca otimizada.
Remova saudações, palavras de preenchimento e foque apenas nas palavras-chave essenciais.

Pergunta original do usuário: {question}

Consulta otimizada:"""
)

banco_retriever = banco.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 4, "score_threshold": 0.8}  # Comece com 0.5 para testar
)

retriever_avancado = RePhraseQueryRetriever.from_llm(
    retriever=banco_retriever,
    llm=local_llm,
    prompt=prompt_reescrita
)

prompt_resposta = PromptTemplate(
    input_variables=["contexto", "pergunta"],  # ← CORRIGIDO para "pergunta"
    template="""Você é um assistente que responde perguntas com base SOMENTE no contexto abaixo.
Se a resposta não estiver no contexto, diga: "Não encontrei essa informação nos documentos."

Contexto:
{contexto}

Pergunta: {pergunta}

Resposta:"""
)

cadeia_resposta = prompt_resposta | local_llm | StrOutputParser()

pergunta = "Me explique como se dá o cadastro de agentes?"

print("Total de documentos no banco:", banco._collection.count())

documentos_recuperados = retriever_avancado.invoke(pergunta)
print(f"\nDocumentos recuperados: {len(documentos_recuperados)}")

if not documentos_recuperados:
    print("Nenhum documento passou pelo filtro de score.")
else:
    contexto = "\n\n---\n\n".join(
        [f"Documento {i+1}:\n{doc.page_content}" for i, doc in enumerate(documentos_recuperados)]
    )

    # DEBUG: mostra o que vai ser enviado para a LLM
    print("\n=== CONTEXTO ENVIADO PARA A LLM ===")
    print(contexto[:1000])

    resposta_final = cadeia_resposta.invoke({
        "contexto": contexto,
        "pergunta": pergunta  # ← CORRIGIDO: era "question", agora é "pergunta"
    })

    print("\n=== RESPOSTA FINAL ===")
    print(resposta_final)