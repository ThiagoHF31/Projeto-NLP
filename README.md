# ANEEL RAG — Pipeline de NLP sobre Legislação Regulatória

Sistema de Retrieval-Augmented Generation (RAG) sobre o acervo legislativo da ANEEL (~27 mil documentos PDF), desenvolvido pelo grupo de estudos da UFG.

---

## Índice

1. [Estrutura do Projeto](#estrutura-do-projeto)
2. [Pré-requisitos](#pré-requisitos)
3. [Instalação](#instalação)
4. [Guia de Uso — Passo a Passo Completo](#guia-de-uso--passo-a-passo-completo)
5. [Módulos](#módulos)
6. [Uso como biblioteca Python](#uso-como-biblioteca-python)
7. [Contexto do Projeto](#contexto-do-projeto)

---

## Estrutura do Projeto

```
Projeto NLP/
├── src/
│   ├── ingestion/          # Passo 1 — Download dos PDFs da ANEEL
│   ├── extraction/         # Passo 2 — Extração estruturada (texto, tabelas, imagens)
│   ├── chunk_embedding/    # Passo 3 — Indexação vetorial no ChromaDB
│   └── question_rag/       # Passo 4 — Interface de chat RAG
│
├── docs/                   # Documentação técnica detalhada por módulo
├── notebooks/              # Análises exploratórias e experimentos
│
├── data/                   # [gitignored]
│   ├── raw/metadata/       # JSONs de metadados ANEEL (2016, 2021, 2022)
│   ├── pdfs/               # ~27.039 PDFs baixados
│   ├── processed/
│   │   ├── extracted_json/     # JSONs gerados pela extração
│   │   └── extracted_images/   # Imagens extraídas dos PDFs
│   └── vector_store/       # Banco vetorial ChromaDB (126k chunks)
│
├── logs/                   # [gitignored] — logs de execução
├── run_extraction.py       # Entry point para extração em lote
├── .env                    # [gitignored] — chaves de API
├── .env.example            # Template de variáveis de ambiente
└── requirements.txt
```

---

## Pré-requisitos

### Python
```
Python 3.13 (recomendado)
```

### GPU (recomendado para indexação)
A etapa de indexação usa o modelo de embedding `intfloat/multilingual-e5-small`. Com GPU CUDA o processo é ~10x mais rápido. Sem GPU, roda na CPU (mais lento, mas funcional).

### Dependências de sistema (para PDFs escaneados)

> **Nota:** Para os PDFs da ANEEL (gerados por Word), essas ferramentas NÃO são necessárias. Só entram em ação se o PDF for escaneado (imagem digitalizada sem camada de texto).

```bash
# Windows
# Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
# Poppler:   https://github.com/oschwartz10612/poppler-windows/releases

# Linux
sudo apt install tesseract-ocr tesseract-ocr-por poppler-utils
```

---

## Instalação

```bash
# 1. Clone o repositório
git clone <url-do-repositorio>
cd "Projeto NLP"

# 2. Crie e ative o ambiente virtual (recomendado)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure as variáveis de ambiente
copy .env.example .env         # Windows
# cp .env.example .env         # Linux/Mac
# Edite o .env e insira sua GROQ_API_KEY
```

Para obter uma chave Groq gratuitamente: https://console.groq.com

---

## Guia de Uso — Passo a Passo Completo

O pipeline tem 4 etapas sequenciais. Execute na ordem abaixo.

---

### Passo 1 — Download dos PDFs da ANEEL

> Pule este passo se já tiver os PDFs em `data/pdfs/`.

```bash
py -3.13 src/ingestion/download_pdfs.py
```

**O que acontece:**
- Lê os 3 arquivos de metadados em `data/raw/metadata/` (2016, 2021, 2022)
- Baixa ~27.039 PDFs para `data/pdfs/` usando 4 workers paralelos
- Usa bypass de TLS fingerprinting (`curl_cffi`) para contornar bloqueio do servidor ANEEL
- É retomável — se interromper, na próxima execução pula os já baixados

**Saída:** `data/pdfs/` com os PDFs + `data/pdfs_manifesto.csv` com status de cada arquivo

**Tempo estimado:** 2–6 horas dependendo da conexão

---

### Passo 2 — Extração dos PDFs

```bash
py -3.13 run_extraction.py --parallel --workers 4
```

**O que acontece:**
- Processa cada PDF em `data/pdfs/` com pdfplumber
- Extrai texto (fora das tabelas), tabelas (lattice/bordas visíveis) e imagens
- Para PDFs escaneados, aciona OCR automaticamente (Tesseract → EasyOCR)
- Salva um JSON estruturado por documento com os campos `full_text`, `content_blocks`, `tables`

**Flags disponíveis:**
```bash
py -3.13 run_extraction.py                    # sequencial, 4 workers
py -3.13 run_extraction.py --parallel         # multiprocessing (mais rápido)
py -3.13 run_extraction.py --workers 8        # controla paralelismo
py -3.13 run_extraction.py --reprocess        # reprocessa mesmo os já extraídos
```

**Saída:** `data/processed/extracted_json/*.json` (~27.121 arquivos)

**Tempo estimado:** 30–120 minutos dependendo do hardware

---

### Passo 3 — Indexação Vetorial

```bash
py -3.13 src/chunk_embedding/aplicacao_ocr.py
```

**O que acontece:**
- Lê o campo `full_text` de cada JSON extraído
- Divide em chunks de 450 tokens com 45 tokens de sobreposição
- Gera embeddings com `intfloat/multilingual-e5-small` (CUDA se disponível)
- Armazena os 126k+ chunks no ChromaDB em `data/vector_store/`
- Deduplica automaticamente documentos com conteúdo idêntico (por hash MD5)

**Saída:** `data/vector_store/` com a collection `embeddings_salvar` (distância cosine)

**Tempo estimado:** 20–45 minutos com GPU / 2–4 horas com CPU

---

### Passo 4 — Chat RAG

```bash
py -3.13 src/question_rag/chat_rag.py
```

**O que acontece:**
- Carrega o banco vetorial e o modelo de embedding
- Entra em loop interativo de perguntas e respostas
- Para cada pergunta: busca os 5 chunks mais relevantes (cosine similarity ≥ 0.65), monta o contexto e envia para o modelo `llama-3.1-8b-instant` via Groq API
- Mantém histórico dos últimos 5 turnos para perguntas de acompanhamento
- Mostra as fontes (nomes dos PDFs) usadas em cada resposta

**Comandos no chat:**
```
Você: O que é necessário para cadastrar como agente na CCEE?
Você: Quais documentos são obrigatórios para modelagem de ativo?
Você: sair   (encerra o chat)
```

---

## Módulos

### `src/ingestion/`

| Arquivo | Descrição |
|---------|-----------|
| `download_pdfs.py` | Baixa os PDFs da ANEEL com bypass de TLS fingerprinting. Suporta retomada automática. |

Documentação completa: [`docs/ingestion.md`](docs/ingestion.md)

---

### `src/extraction/`

| Arquivo | Descrição |
|---------|-----------|
| `pdf_extractor.py` | Orquestrador principal. Extrai texto (fora de tabelas), tabelas (lattice) e imagens por coordenada, em ordem de leitura. |
| `batch.py` | Processamento em lote com ProcessPoolExecutor. |
| `config.py` | Constantes centralizadas (caminhos, limites, idiomas OCR). |
| `models.py` | Dataclasses: `ContentBlock` e `ExtractionResult`. |
| `text_extractor.py` | Extração página a página via pdfplumber. |
| `image_extractor.py` | Extração de imagens (PyMuPDF → pdf2image). |
| `ocr.py` | OCR fallback para PDFs escaneados (Tesseract → EasyOCR). |
| `metadata.py` | Metadados do PDF via PyMuPDF. |
| `utils.py` | Funções auxiliares compartilhadas. |

Documentação completa: [`docs/pdf_extractor.md`](docs/pdf_extractor.md)

---

### `src/chunk_embedding/`

| Arquivo | Descrição |
|---------|-----------|
| `aplicacao_ocr.py` | Indexa os JSONs extraídos no ChromaDB. Usa `full_text`, chunks de 450 tokens, embeddings e5-small com prefixo `"passage: "`, distância cosine. |

Documentação completa: [`docs/indexacao.md`](docs/indexacao.md)

---

### `src/question_rag/`

| Arquivo | Descrição |
|---------|-----------|
| `fazer_melhorar_perguntas.py` | Módulo central RAG: `E5Embeddings` com prefixo `"query: "`, retriever com threshold cosine, chain de resposta com Groq. |
| `chat_rag.py` | Interface de chat interativo com histórico de conversa e exibição de fontes. |

Documentação completa: [`docs/chat_rag.md`](docs/chat_rag.md)

---

## Uso como biblioteca Python

```python
# Extrair um PDF individualmente
from src.extraction import process_pdf

result = process_pdf("data/pdfs/documento.pdf")

print(result.full_text[:500])       # texto completo
for block in result.content_blocks: # blocos em ordem de leitura
    print(block.type, block.page)
```

```python
# Fazer perguntas diretamente (sem chat interativo)
import sys
sys.path.insert(0, "src/question_rag")

from fazer_melhorar_perguntas import carregar_banco, criar_llm, criar_retriever
from fazer_melhorar_perguntas import criar_chain_resposta, recuperar_documentos, formatar_contexto

banco     = carregar_banco()
llm       = criar_llm()
retriever = criar_retriever(banco)
chain     = criar_chain_resposta(llm)

docs      = recuperar_documentos(retriever, "Como se cadastrar como agente?")
contexto  = formatar_contexto(docs)
resposta  = chain.invoke({"contexto": contexto, "pergunta": "Como se cadastrar como agente?"})
print(resposta)
```

---

## Contexto do Projeto

| Item | Detalhe |
|---|---|
| **Instituição** | Universidade Federal de Goiás (UFG) |
| **Fonte dos dados** | [Biblioteca ANEEL](https://www2.aneel.gov.br/cedoc/) — Despachos, Resoluções, Portarias (2016–2022) |
| **Volume** | ~27.039 PDFs, ~3 GB |
| **Embedding** | `intfloat/multilingual-e5-small` (384 dim, 512 tokens max) |
| **LLM** | `llama-3.1-8b-instant` via [Groq](https://console.groq.com) |
| **Banco vetorial** | ChromaDB com distância cosine — 126.993 chunks indexados |
| **Objetivo** | RAG sobre legislação do setor elétrico brasileiro para suporte à pesquisa regulatória |
