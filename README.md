# ANEEL RAG — Pipeline de NLP sobre Legislação Regulatória

Sistema de Retrieval-Augmented Generation (RAG) sobre o acervo legislativo da ANEEL (~27 mil documentos PDF), desenvolvido pelo grupo de estudos da UFG.

---

## Estrutura do Projeto

```
Projeto NLP/
├── src/
│   ├── ingestion/          # Coleta de dados: download dos PDFs da ANEEL
│   ├── extraction/         # Extração estruturada de PDFs (texto, tabelas, imagens)
│   └── pipeline/           # Pipeline RAG: chunking, embedding e indexação vetorial
│
├── notebooks/              # Análises exploratórias e experimentos
├── docs/                   # Documentação técnica detalhada de cada módulo
│
├── data/
│   ├── raw/
│   │   ├── metadata/       # JSONs de metadados ANEEL (2016, 2021, 2022)
│   │   └── other_formats/  # Arquivos XLSX auxiliares
│   ├── pdfs/               # 27.039 PDFs baixados [gitignored]
│   ├── processed/          # Saídas do pdf_extractor (JSON + imagens) [gitignored]
│   └── vector_store/       # Banco vetorial ChromaDB [gitignored]
│
├── logs/                   # Logs de execução [gitignored]
├── .env                    # Chaves de API [gitignored — use .env.example como base]
├── .env.example            # Template de variáveis de ambiente
└── requirements.txt        # Dependências Python
```

---

## Módulos

### `src/ingestion/` — Coleta de dados

| Arquivo | Descrição |
|---------|-----------|
| `download_pdfs.py` | Baixa os PDFs da ANEEL usando `curl_cffi` para bypass de TLS fingerprinting. Suporta retomada automática via manifesto CSV. |
| `retirada_pastas.py` | Utilitário de extração e organização dos arquivos zip originais. |

Documentação completa: [`docs/ingestion.md`](docs/ingestion.md)

### `src/extraction/` — Extração de PDFs

| Arquivo | Descrição |
|---------|-----------|
| `pdf_extractor.py` | Extração robusta e modular de PDFs. Separa texto, tabelas e imagens por coordenada, produzindo blocos em ordem de leitura. Suporte a OCR para PDFs escaneados. |

Documentação completa: [`docs/pdf_extractor.md`](docs/pdf_extractor.md)

### `src/pipeline/` — Pipeline RAG

| Arquivo | Descrição |
|---------|-----------|
| `pipeline_rag_sota.py` | Pipeline completo: extração via pymupdf4llm → resumo com Qwen3-14B (LM Studio) → embedding com bge-m3 → indexação no ChromaDB. Checkpointing para retomada. |

---

## Setup

### 1. Pré-requisitos do sistema

```bash
# Tesseract OCR (para PDFs escaneados)
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
# Linux:
sudo apt install tesseract-ocr tesseract-ocr-por

# Poppler (para pdf2image)
# Windows: https://github.com/oschwartz10612/poppler-windows/releases
# Linux:
sudo apt install poppler-utils
```

### 2. Dependências Python

```bash
pip install -r requirements.txt
```

### 3. Variáveis de ambiente

```bash
cp .env.example .env
# edite .env com suas chaves
```

### 4. LM Studio (pipeline RAG)

O `pipeline_rag_sota.py` requer o LM Studio rodando localmente com o modelo `qwen/qwen3-14b` na porta `1234`.

---

## Uso rápido

```bash
# Baixar PDFs da ANEEL
python src/ingestion/download_pdfs.py

# Extrair um PDF (texto + tabelas + imagens)
python src/extraction/pdf_extractor.py data/pdfs/documento.pdf

# Extrair um diretório completo
python src/extraction/pdf_extractor.py data/pdfs/ --output-dir data/processed/extracted_json

# Rodar o pipeline RAG completo
python src/pipeline/pipeline_rag_sota.py
```

---

## Usar `pdf_extractor` como módulo Python

```python
from src.extraction.pdf_extractor import process_pdf, process_batch, find_pdfs

# Arquivo único
result = process_pdf("data/pdfs/documento.pdf")

# Iterar blocos em ordem de leitura
for block in result.content_blocks:
    if block.type == "text":
        print(f"Pág {block.page}: {block.content[:100]}")
    elif block.type == "table":
        print(f"Pág {block.page}: tabela com {len(block.data)} linhas")

# Lote com barra de progresso
pdfs = find_pdfs("data/pdfs/")
results = process_batch(pdfs, json_output_dir="data/processed/extracted_json")
```

---

## Contexto do Projeto

- **Instituição**: Universidade Federal de Goiás (UFG)
- **Fonte dos dados**: [Biblioteca ANEEL](https://www2.aneel.gov.br/cedoc/) — Despachos, Resoluções, Portarias (2016–2022)
- **Volume**: ~27.039 PDFs, ~3 GB
- **Objetivo**: RAG sobre legislação do setor elétrico brasileiro para suporte à pesquisa regulatória
