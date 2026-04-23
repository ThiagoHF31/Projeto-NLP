# Pipeline de Indexação — `aplicacao_ocr.py`

## O que este script faz

Lê os JSONs extraídos dos PDFs da ANEEL, divide o texto em chunks e armazena os embeddings num banco vetorial ChromaDB — preparando os dados para busca semântica no RAG.

---

## Fluxo passo a passo

```
data/processed/extracted_json/*.json
         │
         ▼
  1. Lê full_text do JSON
         │
         ├─ len < 150 chars? → PULADO (cabeçalho/imagem)
         ├─ MD5 já visto?    → PULADO (duplicata)
         │
         ▼
  2. RecursiveCharacterTextSplitter
     chunk_size=500 tokens | overlap=50 tokens
         │
         ▼
  3. Prefixo "passage: " + HuggingFaceEmbeddings
     modelo: intfloat/multilingual-e5-small
     device: CUDA (fallback CPU)
         │
         ▼
  4. ChromaDB.upsert()
     texto: sem prefixo
     vetor: com prefixo "passage: "
     meta:  file_name, num_pages, author, chunk_index
         │
         ▼
  data/vector_store/   (collection: embeddings_salvar, distância: cosine)
```

---

## Decisões técnicas

### Por que `full_text` e não `content_blocks`

A versão anterior iterava sobre `content_blocks` individualmente. Isso criava chunks problemáticos que dominavam a busca semântica:

| Tipo de chunk | Conteúdo | Problema |
|---|---|---|
| Cabeçalho de tabela | "Agente responsável" | 3 palavras, altíssima concentração semântica |
| Célula de tabela | "col_0 / Candidato a a" | Ruído puro |
| Bloco de texto completo | 2.500 chars de regulação | Correto, mas perdia para os dois acima |

`full_text` é a concatenação limpa de todos os blocos de texto em ordem de leitura — sem ruído de tabelas, sem cabeçalhos de página repetidos, com contexto contínuo entre parágrafos.

### Por que o prefixo `"passage: "`

O modelo `multilingual-e5-small` usa **instruction tuning**: ele aprendeu que documentos e queries são entidades diferentes no espaço vetorial.

- Documentos indexados → `"passage: {texto}"`
- Queries de busca → `"query: {texto}"` (ver `fazer_melhorar_perguntas.py`)

Omitir os prefixos faz query e documentos navegarem em regiões incompatíveis do espaço vetorial — o retrieval falha silenciosamente retornando documentos não relacionados.

### Por que distância cosine no ChromaDB

O ChromaDB cria collections com distância **L2 (euclidiana)** por padrão. Para embeddings normalizados (que é o nosso caso — `normalize_embeddings=True`), L2 e cosine são matematicamente equivalentes. Mas o LangChain calcula o `relevance_score` de forma diferente para cada métrica:

- **L2**: `score = 1 - distance / 2` (errado para threshold de similaridade)
- **Cosine**: `score = 1 - distance = cosine_similarity` ✅

Usando cosine explicitamente, `score_threshold=0.4` significa exatamente "cosine similarity ≥ 0.4".

### Deduplicação por MD5

Os JSONs extraídos podem conter arquivos duplicados (ex: `documento_v9.0.json` e `documento_v9.0_1.json` com conteúdo idêntico). O hash MD5 do `full_text` garante que cada conteúdo único seja indexado apenas uma vez.

---

## Como rodar

```bash
cd "C:\Users\User\Desktop\Projeto NLP\src\chunk_embedding"
py -3.13 aplicacao_ocr.py
```

**Tempo estimado:** 15–45 min dependendo da quantidade de documentos e GPU disponível.

## Logs gerados

| Arquivo | Conteúdo |
|---|---|
| `logs/indexacao_erros.txt` | JSONs que falharam com traceback completo |
| `logs/indexacao_pulados.txt` | JSONs pulados (curtos ou duplicados) |
