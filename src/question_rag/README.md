# RAG — Recuperação e Geração de Respostas

## Arquivos

| Arquivo | Função |
|---|---|
| `fazer_melhorar_perguntas.py` | Módulo central: embeddings, retriever, chains — importado por outros scripts |
| `chat_rag.py` | Interface de chat interativo no terminal com histórico de conversa |

---

## `fazer_melhorar_perguntas.py`

### Fluxo interno

```
Pergunta do usuário
       │
       ▼
RePhraseQueryRetriever
  └─ Groq llama-3.1-8b reescreve em palavras-chave técnicas
       │
       ▼
E5Embeddings.embed_query()
  └─ Adiciona "query: " antes da pergunta
  └─ Gera vetor de 384 dimensões
       │
       ▼
ChromaDB similarity search
  └─ Distância cosine — top-5 chunks com score ≥ 0.4
       │
       ▼
formatar_contexto()
  └─ Monta bloco com fonte [Documento N — arquivo.pdf]
       │
       ▼
PromptTemplate | ChatGroq | StrOutputParser
  └─ Resposta baseada SOMENTE no contexto recuperado
```

### Componente crítico: `E5Embeddings`

```python
class E5Embeddings(HuggingFaceEmbeddings):
    def embed_query(self, text: str) -> list[float]:
        return super().embed_documents([f"query: {text}"])[0]
```

O modelo `intfloat/multilingual-e5-small` foi treinado com **instruction tuning** — ele aprende a distinguir documentos de queries pelo prefixo:

| Operação | Prefixo | Quem aplica |
|---|---|---|
| Indexação de documentos | `"passage: "` | `aplicacao_ocr.py` |
| Busca por query | `"query: "` | `E5Embeddings.embed_query()` |

Sem essa distinção, o retrieval retorna resultados incorretos — o modelo vê query e documentos como objetos da mesma natureza, quando na verdade são assimétricos por design.

### `RePhraseQueryRetriever`

Antes de buscar no banco vetorial, a pergunta passa pelo LLM que a converte em palavras-chave técnicas. Exemplo:

```
Entrada : "O que é necessário para se cadastrar como agente na CCEE?"
Saída LLM: "cadastro agentes CCEE requisitos documentação processo"
```

A query otimizada tem maior overlap semântico com os chunks indexados (que são trechos técnicos de regulações), melhorando significativamente o recall.

---

## `chat_rag.py`

### Fluxo de uma conversa

```
Usuário digita pergunta
       │
       ▼
[mesmo fluxo de fazer_melhorar_perguntas.py]
       │
       ▼
Monta prompt com:
  ├─ Histórico dos últimos 5 turnos (contexto conversacional)
  ├─ Contexto dos documentos recuperados
  └─ Pergunta atual
       │
       ▼
Groq gera resposta
       │
       ▼
Exibe resposta + fontes (nomes dos PDFs usados)
```

### Histórico de conversa

O chat mantém os últimos `MAX_HISTORICO=5` turnos no prompt. Isso permite perguntas de acompanhamento naturais:

```
Você: O que é necessário para cadastro de agentes?
Assistente: [resposta sobre requisitos]

Você: E qual o prazo para aprovação?   ← refere-se ao cadastro sem repetir
Assistente: [resposta com contexto anterior]
```

---

## Como rodar

```bash
cd "C:\Users\User\Desktop\Projeto NLP\src\question_rag"
py -3.13 chat_rag.py
```

## Dependências

Listadas em `requirements.txt`:
- `langchain`, `langchain-classic`, `langchain-community`, `langchain-core`
- `langchain-chroma`, `langchain-huggingface`, `langchain-groq`
- `sentence-transformers`, `transformers`, `torch`
- `chromadb`, `python-dotenv`

## Variáveis de ambiente (`.env` na raiz do projeto)

```
GROQ_API_KEY=sua_chave_aqui
```
