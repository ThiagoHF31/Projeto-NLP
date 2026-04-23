# Pipeline de Indexação Vetorial — `aplicacao_ocr.py`

## Sumário

1. [O que este módulo faz](#1-o-que-este-módulo-faz)
2. [Por que full_text e não content_blocks](#2-por-que-full_text-e-não-content_blocks)
3. [O modelo de embedding e os prefixos de instrução](#3-o-modelo-de-embedding-e-os-prefixos-de-instrução)
4. [Distância cosine no ChromaDB](#4-distância-cosine-no-chromadb)
5. [Deduplicação por MD5](#5-deduplicação-por-md5)
6. [Fluxo de execução completo](#6-fluxo-de-execução-completo)
7. [Estrutura do ChromaDB](#7-estrutura-do-chromadb)
8. [Como executar](#8-como-executar)
9. [Constantes configuráveis](#9-constantes-configuráveis)
10. [Logs gerados](#10-logs-gerados)

---

## 1. O que este módulo faz

O `aplicacao_ocr.py` é a ponte entre os JSONs estruturados gerados pela extração e o banco vetorial usado pelo chat RAG.

**Entrada:** `data/processed/extracted_json/*.json` (~27.121 arquivos)  
**Saída:** ChromaDB em `data/vector_store/` com 126k+ chunks indexados

Para cada JSON:
1. Lê o campo `full_text` (texto limpo do documento inteiro)
2. Divide em chunks de 450 tokens com 45 tokens de sobreposição
3. Gera embeddings com `intfloat/multilingual-e5-small` (prefixo `"passage: "`)
4. Armazena texto + vetores + metadados no ChromaDB

---

## 2. Por que `full_text` e não `content_blocks`

A versão anterior deste script iterava sobre `content_blocks` individualmente. Isso criava um problema grave de qualidade no retrieval:

### O problema dos chunks curtos

```
content_blocks de um documento típico:
├── texto (pág 1):  "Submódulo 1.2 – Cadastro de agentes..."  → 124 chars
├── tabela (pág 2): "Agente responsável"                       → 20 chars  ← PROBLEMA
├── texto (pág 3):  "3.1. O agente ou candidato a agente..."  → 2.222 chars
```

O chunk `"Agente responsável"` (2 palavras) tem alta concentração semântica em torno do conceito "agente". Na busca por similaridade cosine, ele compete com os blocos de texto ricos e, por ser tão específico, muitas vezes **vence** — retornando como "mais similar" apesar de não ter informação útil.

**Resultado observado:** query "O que é necessário para cadastrar como agente?" retornava cabeçalhos de tabela `"Agente responsável"` em vez de parágrafos com as regras reais.

### A solução: `full_text`

O campo `full_text` do JSON já é a concatenação limpa de todos os blocos de texto em ordem de leitura — sem ruído de tabelas, sem cabeçalhos repetidos de página, com contexto contínuo entre parágrafos.

```
full_text do mesmo documento: 53.875 chars de texto contínuo sobre
"Submódulo 1.2 – Cadastro de agentes, Módulo 1 – Agentes, INTRODUÇÃO,
OBJETIVO, PREMISSAS, 3.1 O agente deve manter atualizado seu cadastro..."
```

Chunks gerados a partir do `full_text` têm contexto semântico rico e produzem retrieval de qualidade muito superior.

---

## 3. O modelo de embedding e os prefixos de instrução

O modelo `intfloat/multilingual-e5-small` pertence à família E5 (**E**mbeddings from **E**nglish **E**xample). É um modelo de instrução: ele foi treinado para distinguir documentos de queries através de prefixos de texto.

### Assimetria documentos × queries

| Papel | Prefixo | Quem aplica |
|---|---|---|
| Documento indexado | `"passage: {texto}"` | `aplicacao_ocr.py` na hora de indexar |
| Query de busca | `"query: {texto}"` | `E5Embeddings.embed_query()` na hora de buscar |

```python
# Na indexação (aplicacao_ocr.py):
textos_embed = [f"passage: {chunk}" for chunk in chunks]
vetores = modelo_embedding.embed_documents(textos_embed)

# O texto armazenado no ChromaDB NÃO tem o prefixo:
collection.upsert(documents=chunks, embeddings=vetores, ...)
```

### O que acontece sem os prefixos

Sem `"passage: "` e `"query: "`, o modelo trata documentos e queries como objetos idênticos no mesmo espaço vetorial. A assimetria que foi treinada desaparece — a busca por similaridade retorna resultados aleatórios (matches incorretos com alta confiança aparente).

**Verificação experimental realizada neste projeto:**

```
Com prefixo "query: ":
  score=0.896 | 01-ANEXO I - Cadastro de agentes_v9.0.pdf | "3.4. O agente ou candidato..."
  score=0.895 | 01-ANEXO I - Cadastro de agentes_v9.0.pdf | "3.18. A CCEE, em hipótese..."

Sem prefixo:
  score=0.196 | 10-ANEXO X - Contratos ACL_v5.0.pdf       | "Agente responsável"  ← ERRADO
  score=0.196 | 10-ANEXO X - Contratos ACL_v5.0.pdf       | "Agente responsável"  ← ERRADO
```

---

## 4. Distância cosine no ChromaDB

O ChromaDB cria collections com distância **L2 (euclidiana)** por padrão. Para embeddings normalizados, L2 e cosine são matematicamente equivalentes, mas o LangChain calcula o `relevance_score` de forma diferente para cada métrica:

| Métrica | Fórmula LangChain | `score_threshold=0.65` significa |
|---|---|---|
| L2 (padrão) | `score = 1 - distance / 2` | `distance ≤ 0.70` — interpretação ambígua |
| **Cosine** | `score = 1 - distance = cosine_similarity` | `cosine_similarity ≥ 0.65` ✅ direto |

A collection é criada explicitamente com `metadata={"hnsw:space": "cosine"}` para garantir que `score_threshold` no LangChain seja interpretado diretamente como cosine similarity.

---

## 5. Deduplicação por MD5

Os JSONs extraídos podem conter arquivos com conteúdo idêntico. Isso acontece porque:
- O mesmo PDF pode ter sido extraído duas vezes (gerando `doc.json` e `doc_1.json`)
- PDFs diferentes podem ter sido copiados do mesmo documento original

O script calcula o hash MD5 do `full_text` de cada documento e mantém um conjunto de hashes já vistos. Se o hash já existe, o arquivo é pulado e registrado em `logs/indexacao_pulados.txt`.

```
Verificação: 27.121 JSONs → ~24.296 únicos (~2.825 duplicatas)
```

---

## 6. Fluxo de execução completo

```
data/processed/extracted_json/*.json
         │
         ▼
  Para cada JSON (ordenado por nome):
         │
         ├─ Lê full_text
         ├─ len(full_text) < 150?  → PULADO (cabeçalho/imagem)
         ├─ MD5 já visto?          → PULADO (duplicata)
         │
         ▼
  RecursiveCharacterTextSplitter
  chunk_size=450 tokens | overlap=45 tokens
         │
         ▼
  Para cada chunk:
  texto_para_embed = f"passage: {chunk}"
         │
         ▼
  HuggingFaceEmbeddings.embed_documents(textos_para_embed)
  modelo: intfloat/multilingual-e5-small
  device: CUDA (fallback CPU)
  batch_size: 64
         │
         ▼
  ChromaDB.upsert(
    ids       = ["{stem}_chunk_{i}", ...],
    documents = [chunk_sem_prefixo, ...],   ← texto limpo para exibição
    embeddings= [vetor_com_prefixo, ...],   ← vetores para busca
    metadatas = [{file_name, num_pages, ...}, ...]
  )
         │
         ▼
  data/vector_store/  (collection "embeddings_salvar", hnsw:space=cosine)
```

---

## 7. Estrutura do ChromaDB

### Collection: `embeddings_salvar`

| Campo | Tipo | Exemplo |
|---|---|---|
| `id` | string | `"01-ANEXO I - Cadastro_chunk_3"` |
| `document` | string | `"3.1. O agente ou candidato a agente deve manter atualizado o seu cadastro na CCEE..."` |
| `embedding` | float[384] | Vetor gerado de `"passage: 3.1. O agente..."` |
| `metadata.file_name` | string | `"01-ANEXO I - 1.2 - Cadastro de agentes_v9.0.pdf"` |
| `metadata.num_pages` | int | `28` |
| `metadata.author` | string | `"psampaio"` |
| `metadata.chunk_index` | int | `3` |
| `metadata.total_chunks` | int | `45` |
| `metadata.source_json` | string | `"01-ANEXO I - 1.2 - Cadastro de agentes_v9.0.json"` |

### Estatísticas após indexação completa

| Métrica | Valor |
|---|---|
| JSONs processados | ~24.296 únicos |
| Total de chunks | 126.993 |
| Dimensão dos vetores | 384 |
| Distância | cosine |
| Espaço em disco | ~2 GB |

---

## 8. Como executar

```bash
# A partir da raiz do projeto
py -3.13 src/chunk_embedding/aplicacao_ocr.py
```

O script **recria a collection** do zero a cada execução (`delete_collection` + `create_collection`). Não é incremental — sempre reindexação completa.

---

## 9. Constantes configuráveis

```python
# src/chunk_embedding/aplicacao_ocr.py

MODELO     = "intfloat/multilingual-e5-small"
# Trocar por "intfloat/multilingual-e5-large" para maior qualidade (mais lento/pesado)
# Trocar por "BAAI/bge-m3" para qualidade máxima (muito mais pesado)

NOME_COLECAO  = "embeddings_salvar"
# Nome da collection no ChromaDB

CHUNK_SIZE    = 450
# Tokens por chunk. e5-small suporta até 512 — reservamos margem para
# prefixo "passage: " (3 tokens) + tokens especiais [CLS][SEP] (2 tokens)

CHUNK_OVERLAP = 45
# Sobreposição entre chunks consecutivos (~10% de CHUNK_SIZE)
# Preserva contexto nas fronteiras entre chunks

MIN_TEXTO     = 150
# Descarta JSONs com full_text menor que isso (caracteres)
# PDFs de imagem pura ou documentos com apenas cabeçalho

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
# Auto-detectado. Forçar CPU: DEVICE = "cpu"
```

---

## 10. Logs gerados

| Arquivo | Conteúdo |
|---|---|
| `logs/indexacao_erros.txt` | JSONs que falharam com traceback completo |
| `logs/indexacao_pulados.txt` | JSONs pulados com motivo: `CURTO` ou `DUPLICADO` |
