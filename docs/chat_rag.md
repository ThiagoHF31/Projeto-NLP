# Sistema de Chat RAG — Documentação Técnica

## Sumário

1. [Visão geral](#1-visão-geral)
2. [Arquitetura do pipeline](#2-arquitetura-do-pipeline)
3. [E5Embeddings — o componente crítico](#3-e5embeddings--o-componente-crítico)
4. [Retrieval por similaridade cosine](#4-retrieval-por-similaridade-cosine)
5. [Geração de resposta com Groq](#5-geração-de-resposta-com-groq)
6. [Histórico de conversa](#6-histórico-de-conversa)
7. [Arquivos do módulo](#7-arquivos-do-módulo)
8. [Como executar](#8-como-executar)
9. [Uso como biblioteca](#9-uso-como-biblioteca)
10. [Constantes configuráveis](#10-constantes-configuráveis)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Visão geral

O módulo `src/question_rag/` implementa a interface de usuário final do sistema RAG. Dado um banco vetorial já indexado (`data/vector_store/`), permite fazer perguntas em linguagem natural sobre o acervo de documentos da ANEEL e receber respostas geradas pelo modelo `llama-3.1-8b-instant` via API Groq.

**Entrada:** pergunta do usuário em linguagem natural  
**Saída:** resposta baseada exclusivamente nos documentos da ANEEL + fontes citadas

---

## 2. Arquitetura do pipeline

```
Pergunta do usuário
       │
       ▼
E5Embeddings.embed_query()
  └─ Adiciona prefixo "query: " antes da pergunta
  └─ Gera vetor de 384 dimensões
       │
       ▼
ChromaDB similarity search
  └─ Distância cosine
  └─ Retorna top-5 chunks com score ≥ 0.65
  └─ score = cosine_similarity(query_vector, chunk_vector)
       │
       ▼
formatar_contexto()
  └─ Monta bloco de texto:
     "[Documento 1 — arquivo.pdf]\nconteúdo do chunk..."
       │
       ▼
PromptTemplate
  └─ Histórico da conversa (últimos 5 turnos)
  └─ Contexto dos documentos
  └─ Pergunta atual
       │
       ▼
ChatGroq (llama-3.1-8b-instant)
  └─ temperature=0.2 (respostas mais determinísticas)
  └─ max_tokens=1024
       │
       ▼
StrOutputParser → resposta como string
       │
       ▼
Exibição + fontes (nomes dos PDFs usados)
```

---

## 3. E5Embeddings — o componente crítico

```python
class E5Embeddings(HuggingFaceEmbeddings):
    def embed_query(self, text: str) -> list[float]:
        return super().embed_documents([f"query: {text}"])[0]
```

O modelo `intfloat/multilingual-e5-small` foi treinado com **instruction tuning assimétrico**: documentos e queries vivem em regiões diferentes do espaço vetorial, separadas pelo prefixo de instrução.

### Por que isso importa

```
Documento indexado (aplicacao_ocr.py):
  embed("passage: 3.1. O agente deve manter cadastro atualizado na CCEE...")
  → vetor aponta para região "documentos" do espaço

Query SEM prefixo (errado):
  embed("O que é necessário para cadastrar como agente?")
  → vetor aponta para região "linguagem geral" — longe dos documentos
  → similarity é baixa mesmo com documento relevante

Query COM prefixo (correto):
  embed("query: O que é necessário para cadastrar como agente?")
  → vetor aponta para região "queries" — complementar à região "documentos"
  → similarity alta com chunks relevantes (0.89+)
```

### Verificação experimental

| Prefixo usado na query | Top resultado | Score |
|---|---|---|
| Sem prefixo | `"Agente responsável"` (cabeçalho de tabela) | 0.804 |
| Com `"query: "` | `"3.4. O agente ou candidato a agente..."` | 0.896 |

A diferença de 0.09 no score representa a diferença entre uma resposta incorreta e uma correta.

---

## 4. Retrieval por similaridade cosine

```python
retriever_base = banco.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": K_DOCS, "score_threshold": SCORE_THRESHOLD},
)
# K_DOCS = 5, SCORE_THRESHOLD = 0.65
```

### Como o LangChain calcula o score

A collection ChromaDB usa `hnsw:space=cosine`. O ChromaDB retorna a **distância cosine** (0 = idêntico, 2 = oposto). O LangChain converte para **similarity score** com:

```
score = 1 - cosine_distance = cosine_similarity
```

Portanto `score_threshold=0.65` significa: *retornar apenas chunks com cosine similarity ≥ 0.65*.

### Por que 0.65 e não outro valor

Análise dos scores observados neste projeto:

| Tipo de chunk | Score típico |
|---|---|
| Altamente relevante (mesma regulação) | 0.85–0.90 |
| Parcialmente relevante (tema próximo) | 0.70–0.85 |
| Tangencialmente relacionado | 0.50–0.70 |
| Não relacionado (footnotes, rodapés) | < 0.50 |

O threshold de 0.65 captura os dois primeiros grupos e exclui ruído.

### Comportamento quando nenhum chunk passa o threshold

```python
if not documentos:
    msg = "Não encontrei documentos relevantes para essa pergunta. Tente reformular."
    print(f"\nAssistente: {msg}\n")
```

O sistema falha com grace — informa o usuário em vez de alucinar uma resposta sem contexto.

---

## 5. Geração de resposta com Groq

```python
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    max_tokens=1024,
)
```

### Prompt de sistema

```
Você é um assistente especializado em regulações da ANEEL e do mercado de
energia elétrica brasileiro.
Responda com base EXCLUSIVAMENTE no contexto abaixo.
Se a informação não estiver no contexto, responda:
"Não encontrei essa informação nos documentos disponíveis."
```

A instrução `EXCLUSIVAMENTE` é fundamental: sem ela, o LLM usaria seu conhecimento de treinamento para complementar respostas, podendo citar informações incorretas ou desatualizadas sobre regulações da ANEEL.

### Por que `temperature=0.2`

Respostas sobre regulações devem ser determinísticas e precisas. `temperature=0.2` reduz variabilidade mantendo fluência — o modelo quase sempre produz a mesma resposta para a mesma pergunta com o mesmo contexto.

---

## 6. Histórico de conversa

O chat mantém os últimos `MAX_HISTORICO=5` turnos no prompt:

```python
historico_bloco = f"Histórico da conversa:\n{hist_str}\n\n"
```

Isso permite perguntas de acompanhamento naturais sem repetir contexto:

```
Você: Quais são os requisitos para cadastro de agente na CCEE?
Assistente: [lista de requisitos]

Você: E qual o prazo?            ← refere ao cadastro implicitamente
Assistente: [resposta com contexto herdado do turno anterior]

Você: Como é feito o cancelamento?   ← ainda sobre cadastro
Assistente: [resposta coerente]
```

**Limitação:** o histórico é mantido apenas em memória. Ao encerrar o chat (`sair`), o histórico é perdido. Para persistência, seria necessário salvar em arquivo ou banco de dados.

---

## 7. Arquivos do módulo

### `fazer_melhorar_perguntas.py`

Módulo central — contém toda a lógica de RAG. Pode ser importado por outros scripts.

| Função/Classe | Descrição |
|---|---|
| `E5Embeddings` | Subclasse de HuggingFaceEmbeddings com `embed_query` corrigido |
| `carregar_banco()` | Conecta ao ChromaDB com E5Embeddings |
| `criar_llm()` | Instancia ChatGroq |
| `criar_retriever(banco)` | Cria retriever com threshold cosine |
| `criar_chain_resposta(llm)` | Chain: PromptTemplate → ChatGroq → StrOutputParser |
| `recuperar_documentos(retriever, pergunta)` | Executa a busca vetorial |
| `formatar_contexto(documentos)` | Formata os chunks com nomes dos arquivos |

### `chat_rag.py`

Interface de chat interativo. Importa tudo de `fazer_melhorar_perguntas.py`.

| Função | Descrição |
|---|---|
| `criar_chain_chat(llm)` | Chain de chat com histórico no prompt |
| `_formatar_historico(historico)` | Formata turnos anteriores como texto |
| `_fontes(documentos)` | Extrai nomes únicos dos PDFs usados |
| `main()` | Loop interativo principal |

---

## 8. Como executar

```bash
cd "C:\Users\User\Desktop\Projeto NLP\src\question_rag"
py -3.13 chat_rag.py
```

**Ou a partir da raiz do projeto:**
```bash
py -3.13 src/question_rag/chat_rag.py
```

**Pré-requisito:** o banco vetorial deve estar indexado (`data/vector_store/` com a collection `embeddings_salvar`). Veja [Passo 3 do README](../README.md#passo-3--indexação-vetorial).

---

## 9. Uso como biblioteca

```python
import sys
sys.path.insert(0, "src/question_rag")

from fazer_melhorar_perguntas import (
    carregar_banco, criar_llm, criar_retriever,
    criar_chain_resposta, recuperar_documentos, formatar_contexto
)

# Inicialização (uma vez)
banco     = carregar_banco()
llm       = criar_llm()
retriever = criar_retriever(banco)
chain     = criar_chain_resposta(llm)

# Pergunta
pergunta = "Quais são os requisitos para cadastro de agente na CCEE?"
docs     = recuperar_documentos(retriever, pergunta)

if not docs:
    print("Sem resultados relevantes.")
else:
    contexto = formatar_contexto(docs)
    resposta = chain.invoke({"contexto": contexto, "pergunta": pergunta})
    print(resposta)
    
    # Fontes
    for doc in docs:
        print(doc.metadata.get("file_name"))
```

---

## 10. Constantes configuráveis

```python
# src/question_rag/fazer_melhorar_perguntas.py

MODELO_EMBED    = "intfloat/multilingual-e5-small"
# Deve ser o mesmo modelo usado na indexação (aplicacao_ocr.py)
# Trocar aqui E em aplicacao_ocr.py juntos — modelos diferentes = busca incorreta

MODELO_LLM      = "llama-3.1-8b-instant"
# Modelos Groq disponíveis: llama-3.1-8b-instant, llama-3.1-70b-versatile
# 70b tem maior qualidade mas é mais lento e consome mais tokens

SCORE_THRESHOLD = 0.65
# Cosine similarity mínima para incluir um chunk no contexto
# Aumentar → mais seletivo, pode retornar 0 resultados para perguntas ambíguas
# Diminuir → menos seletivo, pode incluir chunks pouco relevantes

K_DOCS          = 5
# Número máximo de chunks retornados por busca
# Aumentar → mais contexto para o LLM, mas prompt maior e mais caro
```

```python
# src/question_rag/chat_rag.py

MAX_HISTORICO = 5
# Número de turnos anteriores incluídos no prompt
# Aumentar → mais contexto conversacional, prompt maior
# Diminuir → menos contexto, respostas menos coerentes em conversas longas
```

---

## 11. Troubleshooting

### "Não encontrei documentos relevantes"

Causas possíveis:
1. **Banco não indexado:** rode `py -3.13 src/chunk_embedding/aplicacao_ocr.py` primeiro
2. **Pergunta muito genérica:** reformule com termos técnicos do setor elétrico
3. **Threshold muito alto:** reduza `SCORE_THRESHOLD` de 0.65 para 0.5

### Erro de API Groq

```
AuthenticationError: Invalid API key
```

Verifique se o arquivo `.env` existe na raiz do projeto e contém `GROQ_API_KEY=sua_chave`.

### Embedding model carregando lentamente

O modelo `multilingual-e5-small` (~120 MB) é baixado automaticamente na primeira execução via HuggingFace Hub. As execuções seguintes usam o cache local.

### Aviso `UNEXPECTED: embeddings.position_ids`

```
BertModel LOAD REPORT: embeddings.position_ids | UNEXPECTED
```

Este aviso é inofensivo — indica apenas que o modelo foi carregado de uma arquitetura diferente (e5 usa BERT como base com pequenas modificações). Não afeta a qualidade dos embeddings.
