from langchain_community.llms import Ollama
from langchain_classic.retrievers import RePhraseQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


MODELO_EMBED  = "intfloat/multilingual-e5-small"
modelo_embedding = HuggingFaceEmbeddings(
    model_name=MODELO_EMBED,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


banco = Chroma(
    collection_name="embeddings_salvar",  # <-- Adicione esta linha!
    persist_directory=r"C:\Users\paogr\Desktop\Projeto-NLP\data\vector_store\vector_store",
    embedding_function=modelo_embedding
)

local_llm = Ollama(base_url="http://localhost:11434", model="llama3")
prompt_reescrita = PromptTemplate(
    input_variables=["question"],
    template="""Você é um especialista em busca de informações.
Sua tarefa é converter a pergunta natural do usuário em uma consulta de busca otimizada para um banco de dados vetorial.
Remova saudações, palavras de preenchimento e foque apenas nas palavras-chave essenciais e conceitos principais.

Pergunta original do usuário: {question}

Consulta otimizada:"""
)


banco_retriever = banco.as_retriever(search_kwargs = {"k" : 4})

retriever_avancado = RePhraseQueryRetriever.from_llm(
    retriever=banco_retriever,
    llm=local_llm,
    prompt=prompt_reescrita
)


pergunta = "Me explique como se dá o cadastro de agentes?"

# Verifica quantos documentos existem na collection
print("Total de documentos no banco:", banco._collection.count())

# Testa a busca diretamente (sem o RePhraseQueryRetriever)
resultados_diretos = banco.similarity_search("anos dados", k=4)
print("Busca direta:", len(resultados_diretos), "documentos")
for doc in resultados_diretos:
    print(doc.page_content[:200])

documentos_recuperados = retriever_avancado.invoke(pergunta)

