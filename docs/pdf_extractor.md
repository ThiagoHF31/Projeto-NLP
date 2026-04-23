# pdf_extractor.py — Documentação Técnica Completa

## Sumário

1. [O problema que este script resolve](#1-o-problema-que-este-script-resolve)
2. [O que define uma tabela num PDF](#2-o-que-define-uma-tabela-num-pdf)
3. [Por que as abordagens ingênuas falham](#3-por-que-as-abordagens-ingênuas-falham)
4. [Arquitetura central: blocos por coordenada](#4-arquitetura-central-blocos-por-coordenada)
5. [Fluxo de execução completo](#5-fluxo-de-execução-completo)
6. [Estruturas de dados](#6-estruturas-de-dados)
7. [Cada função explicada](#7-cada-função-explicada)
8. [Lógica de fallback por camada](#8-lógica-de-fallback-por-camada)
9. [O JSON de saída](#9-o-json-de-saída)
10. [Como usar o output no pipeline de chunking](#10-como-usar-o-output-no-pipeline-de-chunking)
11. [Instalação e dependências](#11-instalação-e-dependências)
12. [CLI — linha de comando](#12-cli--linha-de-comando)
13. [Constantes configuráveis](#13-constantes-configuráveis)

---

## 1. O problema que este script resolve

### O cenário real

PDFs de legislação e documentos técnicos — como os da ANEEL — são arquivos heterogêneos. Dentro de um mesmo documento de 28 páginas você pode ter:

- Parágrafos de texto corrido (premissas, descrição de atividades)
- Tabelas com bordas visíveis (histórico de revisões, fluxos de processo)
- Imagens embutidas (logos, diagramas)
- Páginas inteiramente tabulares (dados financeiros por transmissora)
- Páginas com texto + tabela intercalados na mesma página

O desafio não é só "extrair texto". É extrair **cada tipo de conteúdo corretamente, sem mistura e sem perda**, de forma que um pipeline de chunking downstream saiba exatamente o que é texto narrativo e o que é dado estruturado.

### O que estava errado antes

A abordagem ingênua — usada na maioria dos scripts de extração — faz o seguinte:

```python
# Abordagem ERRADA
full_text = ""
for page in pdf.pages:
    full_text += page.extract_text()   # dump de TUDO, inclusive células de tabelas

tables = []
for page in pdf.pages:
    tables += page.extract_tables()    # as mesmas células, de novo
```

**Resultado**: o conteúdo das células de tabela aparece duas vezes — uma em `full_text` e outra em `tables`. Além disso, a ordem de leitura é perdida: você tem uma string gigante de texto e uma lista separada de tabelas, sem saber onde no documento cada tabela aparece.

Para um documento como o "Cadastro de Agentes" (28 páginas), a pág. 24 tem um título, um parágrafo de introdução, e depois 7 tabelas de fluxo de processo. Com a abordagem ingênua:

- O título e o parágrafo aparecem em `full_text` ✓
- As 7 tabelas aparecem em `tables` ✓  
- **Mas o texto das células das tabelas também aparece em `full_text`** ✗
- **E você não sabe que as tabelas estão entre o parágrafo e o próximo título** ✗

Isso é fatal para RAG: o chunker vai incluir texto de tabela no meio de parágrafos, gerando chunks sem sentido semântico.

---

## 2. O que define uma tabela num PDF

Entender isso é fundamental para saber qual biblioteca chamar e quando.

### Nível físico: o que está no arquivo

Um PDF é um arquivo de instruções gráficas. Não existe o conceito semântico de "tabela" na especificação base do PDF — o que existe são:

- **Operador `re`** (rectangle): desenha um retângulo preenchido ou com borda
- **Operadores `m`/`l`/`S`** (move/line/stroke): desenham linhas
- **Objetos de texto** (`BT`/`ET`): posicionam strings em coordenadas X,Y

Uma tabela é, no fundo, um conjunto de retângulos e linhas que formam uma grade, com objetos de texto dentro das células. O PDF não diz "isso é uma tabela" — apenas posiciona os elementos gráficos.

### Dois tipos de tabela no mundo real

#### Tipo 1 — Lattice (grade explícita)
Tabelas com bordas visíveis. O PDF contém linhas horizontais e verticais que se cruzam formando células.

```
┌─────────────┬──────────────┬────────────┐
│ Revisão     │ Motivo       │ Data       │
├─────────────┼──────────────┼────────────┤
│ 1.0         │ Consulta...  │ 16.10.2012 │
└─────────────┴──────────────┴────────────┘
```

Como detectar: o pdfplumber analisa os objetos de linha/retângulo da página e encontra interseções. Quando linhas horizontais e verticais se cruzam em múltiplos pontos formando uma grade, a região é classificada como tabela.

**Este é o tipo predominante nos PDFs Word/Excel da ANEEL** — e é o modo padrão do script.

#### Tipo 2 — Stream (alinhamento por espaço)
Tabelas sem bordas visíveis. O alinhamento é dado por espaços ou tabulações.

```
Transmissora        Contrato    RAP (R$)
AETE                008/2004    43.051.289,05
AFLUENTE T          001/2010    86.442.450,32
```

Como detectar: análise estatística das posições X dos textos na página. Clusters de X consistentes indicam colunas. É uma heurística — muito mais propensa a falsos positivos.

**Por que não usar stream mode como padrão**: qualquer lista com recuo, numeração, ou texto com colunas decorativas vira "tabela". O resultado são centenas de micro-tabelas de 1-2 linhas que são, na realidade, parágrafos normais. O script usa stream mode **apenas como fallback de última instância** via camelot/tabula.

### O critério de descarte: 80% de células vazias

Mesmo o lattice mode às vezes detecta falsos positivos — bordas decorativas de layout, caixas de rodapé, separadores. O script descarta automaticamente qualquer tabela onde mais de 80% das células estão vazias:

```python
TABLE_EMPTY_CELL_THRESHOLD = 0.80

def _table_is_real(cells):
    flat = [c for row in cells for c in row]
    empty = sum(1 for c in flat if c is None or str(c).strip() == "")
    return (empty / len(flat)) <= TABLE_EMPTY_CELL_THRESHOLD
```

Uma tabela real de dados tem conteúdo na maioria das células. Uma borda decorativa detectada como tabela terá quase todas as células vazias.

---

## 3. Por que as abordagens ingênuas falham

### Problema 1: `extract_text()` não sabe o que é tabela

O método `page.extract_text()` do pdfplumber (e equivalentes em outras bibliotecas) faz um dump linear de todos os objetos de texto da página, ordenados por posição Y (vertical) e X (horizontal). Ele não distingue se um texto está dentro de uma célula de tabela ou em um parágrafo.

```python
# O que extract_text() retorna na pág 24 do Cadastro de Agentes:
# "6. DESCRIÇÃO DE ATIVIDADES\nInclusão de Cadastro...\nATIVIDADE\nRESPONSÁVEL\n
#  DETALHAMENTO\nEnviar solicitação...\nO candidato a agente deve..."
#
# Problema: "ATIVIDADE", "RESPONSÁVEL", "DETALHAMENTO" são headers de tabela,
# mas aparecem como texto corrido
```

### Problema 2: dupla extração gera duplicação

```python
# Fluxo problemático:
full_text = page.extract_text()    # inclui "Enviar solicitação..."
tables = page.extract_tables()     # também inclui "Enviar solicitação..."

# A mesma informação está em dois lugares diferentes
# O chunker vai criar chunks com conteúdo duplicado
```

### Problema 3: sem posição, sem contexto

Com a abordagem de dois baldes (`full_text` + `tables`), você perde a informação posicional:

- Você não sabe que a Tabela 3 aparece entre o parágrafo 2 e o parágrafo 3
- Você não pode chunkar respeitando a fronteira entre texto e tabela
- Um chunk pode começar no meio de uma tabela e terminar no meio de um parágrafo

---

## 4. Arquitetura central: blocos por coordenada

### A solução

Em vez de dois baldes separados, o script produz uma **lista única de blocos em ordem de leitura**, onde cada bloco sabe seu tipo, sua página e sua posição:

```
content_blocks = [
  {type: "text",  page: 24, content: "6. DESCRIÇÃO DE ATIVIDADES..."},
  {type: "table", page: 24, bbox: [42.6, 170.5, 742.8, 200.0], data: [...]},
  {type: "table", page: 24, bbox: [42.6, 229.1, 742.8, 248.6], data: [...]},
  {type: "text",  page: 25, content: "Inclusão de Cadastro dos Demais..."},
  {type: "table", page: 25, bbox: [...], data: [...]},
  ...
]
```

### O mecanismo: `page.filter()` com coordenadas

O pdfplumber expõe o método `page.filter(predicate)` que retorna uma sub-página contendo apenas os objetos que satisfazem o predicado. O predicado é uma função que recebe cada objeto da página (caractere, linha, retângulo) e retorna `True` se deve ser incluído.

O script usa isso para extrair **apenas o texto que está fora das regiões de tabela**:

```python
def outside_all_tables(obj):
    return not any(_obj_inside_bbox(obj, bb) for bb in valid_table_bboxes)

text = plumber_page.filter(outside_all_tables).extract_text()
```

A função `_obj_inside_bbox` verifica se as coordenadas do objeto (x0, x1, top, bottom) estão dentro do bounding box de alguma tabela válida, com uma tolerância de 2 pontos para bordas:

```python
def _obj_inside_bbox(obj, bbox, tol=2.0):
    x0, top, x1, bottom = bbox
    return (
        obj["x0"] >= x0 - tol and obj["x1"] <= x1 + tol and
        obj["top"] >= top - tol and obj["bottom"] <= bottom + tol
    )
```

### Ordenação por eixo Y

Depois de coletar os blocos de tabela (com suas bboxes conhecidas) e o bloco de texto (sem bbox individual), os blocos são ordenados pelo topo da bbox — coordenada Y no espaço da página:

```python
def _sort_key(b):
    if b.bbox:
        return b.bbox[1]   # Y do topo da tabela
    return 0.0             # texto sem bbox: vai para o início (cabeçalho)

blocks.sort(key=_sort_key)
```

Resultado: para uma página com texto no topo e tabela no meio, a ordem de leitura é preservada — `[text, table, table]`.

---

## 5. Fluxo de execução completo

```
process_pdf(file_path)
│
├── 1. extract_metadata()
│       └── PyMuPDF: num_pages, título, autor, tamanho do arquivo
│
├── 2. Loop página a página (pdfplumber)
│   │
│   └── _extract_blocks_for_page(page, page_num)
│       │
│       ├── Passo 1: find_tables()  ← detecta tabelas por linhas vetoriais (lattice)
│       │   ├── Para cada tabela detectada:
│       │   │   ├── _table_is_real()  ← descarta se >80% células vazias
│       │   │   └── _cells_to_records()  ← converte matriz → lista de dicts
│       │   └── Guarda bboxes das tabelas válidas
│       │
│       ├── Passo 2: page.filter(outside_all_tables).extract_text()
│       │   └── Extrai APENAS texto fora das regiões de tabela
│       │
│       └── Passo 3: sort por Y  ← ordena blocos em ordem de leitura
│
├── 3. Verifica se texto total é válido
│   └── Se vazio/ilegível → fallback OCR
│       ├── pdf2image: renderiza páginas como imagens (requer Poppler)
│       ├── Tesseract: OCR em português
│       └── EasyOCR: fallback se Tesseract falhar
│
└── 4. extract_images()
    ├── PyMuPDF: extrai objetos de imagem embutidos (JPEG, PNG, etc.)
    └── pdf2image: fallback — renderiza páginas inteiras como PNG
```

### Por que o OCR só é acionado no final

O OCR é lento e impreciso comparado à extração nativa. O script só o aciona se, **depois de processar todas as páginas com pdfplumber**, o texto total acumulado for menor que `MIN_TEXT_LENGTH` (50 chars) ou tiver menos de 80% de caracteres imprimíveis. Isso indica um PDF escaneado (imagem digitalizada sem camada de texto).

```python
full_text_sample = " ".join(b.content for b in text_blocks)

if not _is_valid_text(full_text_sample):
    # PDF escaneado — acionar OCR
    ocr_blocks = _extract_blocks_ocr(file_path)
```

---

## 6. Estruturas de dados

### `ContentBlock`

Unidade atômica de conteúdo. Cada bloco representa um elemento indivisível da página.

```python
@dataclass
class ContentBlock:
    type: str           # "text" | "table" | "image"
    page: int           # número da página (começa em 1)
    bbox: list[float]   # [x0, top, x1, bottom] em pontos PDF — vazio para texto puro
    content: str        # conteúdo se type=="text"
    data: list[dict]    # linhas se type=="table", ex: [{"Revisão": "1.0", "Data": "2012"}]
    image_path: str     # caminho do arquivo salvo se type=="image"
    source: str         # qual biblioteca extraiu: "pdfplumber", "tesseract-ocr", "pymupdf"...
```

**Por que bbox pode ser vazio para texto**: o pdfplumber extrai o texto fora das tabelas como uma string consolidada da sub-página filtrada. A bbox agregada de todos os fragmentos de texto seria complexa de calcular e pouco útil — o número da página já localiza o bloco com precisão suficiente para chunking.

### `ExtractionResult`

Resultado completo de um PDF. Contém os blocos e oferece duas vistas agregadas via `@property`:

```python
@dataclass
class ExtractionResult:
    file_path: str
    metadata: dict           # num_pages, autor, título, tamanho_kb...
    content_blocks: list[ContentBlock]   # A fonte da verdade
    images_paths: list[str]  # atalho para paths de imagens
    extraction_log: dict     # quais bibliotecas foram usadas, tempo, contagens
    success: bool
    error: str

    @property
    def full_text(self) -> str:
        # Concatena blocos de texto em ordem → sem conteúdo de tabelas
        return "\n\n".join(b.content for b in self.content_blocks if b.type == "text")

    @property
    def tables(self) -> list[dict]:
        # Lista de {page, bbox, source, data} para acesso rápido às tabelas
        return [{...} for b in self.content_blocks if b.type == "table"]
```

**Importante**: `full_text` e `tables` são vistas derivadas de `content_blocks` — não armazenam dados duplicados. Alterar `content_blocks` atualiza automaticamente as duas propriedades.

---

## 7. Cada função explicada

### `extract_metadata(file_path)`

Usa PyMuPDF para ler o dicionário de metadados embutido no PDF (XMP/Info dictionary). Inclui nome do arquivo, tamanho em KB, número de páginas, título, autor, data de criação e produtor (ex: "Microsoft Word para Microsoft 365").

```python
meta = extract_metadata("documento.pdf")
# → {"file_name": "documento.pdf", "file_size_kb": 245.3,
#    "num_pages": 28, "author": "psampaio", "creator": "Microsoft® Word..."}
```

### `_extract_blocks_for_page(plumber_page, page_num)`

**Núcleo do script.** Processa uma página e retorna seus blocos em ordem de leitura.

1. Chama `find_tables()` — analisa as linhas vetoriais da página e retorna objetos `Table` com bbox e células
2. Para cada tabela, verifica autenticidade com `_table_is_real()` e converte com `_cells_to_records()`
3. Chama `page.filter(outside_all_tables).extract_text()` para pegar apenas texto fora das tabelas
4. Ordena pela coordenada Y do topo

### `_table_is_real(cells)`

Recebe a matriz de células `[[linha0_col0, linha0_col1...], [linha1_col0...]]` e retorna `False` se mais de 80% das células estiverem vazias. Isso elimina bordas decorativas e caixas de layout que o pdfplumber detecta erroneamente como tabelas.

### `_cells_to_records(cells)`

Converte a matriz bruta do pdfplumber em lista de dicionários, usando a primeira linha como header. Células da primeira linha que estão vazias recebem o nome `col_0`, `col_1`... Linhas completamente vazias são descartadas.

```python
# Entrada (matriz bruta):
[["Revisão", "Motivo", "Data"], ["1.0", "Consulta...", "16.10.2012"]]

# Saída (lista de dicts):
[{"Revisão": "1.0", "Motivo": "Consulta...", "Data": "16.10.2012"}]
```

### `_obj_inside_bbox(obj, bbox, tol=2.0)`

Verifica se um objeto pdfplumber (caractere, palavra, linha) está dentro de um bounding box. A tolerância de 2 pontos absorve imprecisões de arredondamento nas coordenadas do PDF — sem ela, caracteres na borda exata da tabela poderiam vazar para o texto externo.

### `_is_valid_text(text)`

Duas condições para texto ser considerado válido:
1. Pelo menos 50 caracteres (`MIN_TEXT_LENGTH`)
2. Pelo menos 80% dos caracteres são imprimíveis

A segunda condição detecta PDFs escaneados: quando o texto nativo retorna uma série de caracteres inválidos ou nulos, o ratio de imprimíveis cai abaixo de 0.8 e o script aciona OCR.

### `_sanitize_text(text)`

Remove caracteres nulos (`\x00`) que alguns PDFs embutem nos strings de texto, e colapsa sequências de 3+ quebras de linha em apenas 2. Isso normaliza o espaçamento sem perder separação entre parágrafos.

### `extract_images(file_path, log)`

Orquestra a extração de imagens com fallback:

1. **PyMuPDF** (`doc.get_images()` + `doc.extract_image(xref)`): extrai os objetos de imagem binários embutidos no PDF (JPEG, PNG, JBIG2, etc.) diretamente, sem reprocessamento. É o método mais fiel — preserva a imagem original sem recompressão.

2. **pdf2image** (fallback): renderiza cada página inteira como um PNG raster usando a engine Poppler. Usado quando o PDF não tem imagens embutidas como objetos (ex: PDFs de texto puro com vetores). Requer Poppler instalado no sistema.

### `process_pdf(file_path)`

Orquestrador principal. Coordena as etapas na ordem correta, captura exceções em cada etapa de forma independente (falha em tabelas não aborta extração de texto), e popula o `extraction_log` com diagnóstico de qual biblioteca foi usada em cada camada.

### `process_batch(pdf_paths, json_output_dir, max_workers, use_multiprocessing)`

Processa uma lista de PDFs com barra de progresso tqdm. Suporta dois modos:

- **Sequencial** (padrão): roda um PDF de cada vez no processo principal. Mais seguro com bibliotecas que têm estado global (EasyOCR, pdfplumber).
- **Paralelo** (`use_multiprocessing=True`): usa `ProcessPoolExecutor` — cada PDF roda em um processo separado. Mais rápido para lotes grandes, mas requer que todas as bibliotecas sejam seguras para multiprocessing. Cuidado com EasyOCR que carrega modelos de rede neural por processo.

---

## 8. Lógica de fallback por camada

O script tem três camadas de fallback independentes. Uma falha em qualquer camada não afeta as outras.

### Camada 1: Texto

```
pdfplumber (lattice, por coordenada)
    ↓ se texto total < 50 chars ou < 80% imprimível
Tesseract OCR (página a página via pdf2image)
    ↓ se Tesseract falhar ou retornar texto inválido
EasyOCR (página a página, GPU opcional)
    ↓ se EasyOCR falhar
→ log["text_source"] = "FALHA — nenhum texto extraído"
```

**Quando cada um é acionado:**

| Situação | Biblioteca usada |
|----------|-----------------|
| PDF gerado por Word/LibreOffice | pdfplumber |
| PDF gerado por Excel ou sistema | pdfplumber |
| PDF escaneado (foto digitalizada) | Tesseract |
| PDF escaneado com fonte não-latina | EasyOCR |
| PDF corrompido ou protegido | Falha registrada no log |

### Camada 2: Tabelas (dentro do pdfplumber)

```
find_tables() com lattice (linhas vetoriais)
    ↓ se retornar 0 tabelas com lattice
→ tabelas são nulas nessa página (stream mode não é aplicado por página)
```

O stream mode via camelot/tabula é reservado para casos onde o documento inteiro não tem nenhuma tabela com bordas. Na prática, os PDFs da ANEEL são todos Word-gerados e têm bordas explícitas, então lattice resolve 100% dos casos.

### Camada 3: Imagens

```
PyMuPDF extract_image() — extrai objetos binários embutidos
    ↓ se não encontrar nenhuma imagem embutida
pdf2image — renderiza páginas inteiras como PNG (requer Poppler)
    ↓ se Poppler não estiver instalado
→ log["images"] = "nenhuma imagem extraída"
```

---

## 9. O JSON de saída

Cada PDF gera um arquivo JSON com a seguinte estrutura:

```json
{
  "file_path": "dados/pdfs/01-ANEXO I - 1.2 - Cadastro de agentes_v9.0.pdf",

  "metadata": {
    "file_name": "01-ANEXO I - 1.2 - Cadastro de agentes_v9.0.pdf",
    "file_size_kb": 1842.5,
    "num_pages": 28,
    "author": "psampaio",
    "creator": "Microsoft® Word para Microsoft 365",
    "creationDate": "D:20220209151424-03'00'"
  },

  "content_blocks": [
    {
      "type": "text",
      "page": 1,
      "source": "pdfplumber",
      "content": "Submódulo 1.2 – Cadastro de agentes\nMódulo 1 – Agentes\n..."
    },
    {
      "type": "text",
      "page": 2,
      "source": "pdfplumber",
      "content": "ÍNDICE\n1. INTRODUÇÃO\n2. OBJETIVO\n..."
    },
    {
      "type": "table",
      "page": 2,
      "source": "pdfplumber-lattice",
      "bbox": [57.1, 367.6, 552.3, 684.8],
      "data": [
        {"Revisão": "1.0", "Motivo da Revisão": "Consulta Pública n° 05/2012",
         "Instrumento de aprovação pela ANEEL": "Despacho nº 3.215/2012",
         "Data de Vigência": "16.10.2012"},
        {"Revisão": "2.0", "Motivo da Revisão": "Adequação à REN n° 583/2013...",
         ...}
      ]
    },
    {
      "type": "text",
      "page": 3,
      "source": "pdfplumber",
      "content": "1. INTRODUÇÃO\nEste submódulo estabelece as atividades..."
    }
  ],

  "full_text": "Submódulo 1.2 ...\n\nÍNDICE\n...\n\n1. INTRODUÇÃO\n...",
  
  "tables": [
    {
      "page": 2,
      "bbox": [57.1, 367.6, 552.3, 684.8],
      "source": "pdfplumber-lattice",
      "data": [...]
    }
  ],

  "images_paths": [
    "extracted_images/01-ANEXO I.../p1_img0.png",
    "extracted_images/01-ANEXO I.../p1_img1.jpeg"
  ],

  "extraction_log": {
    "text_blocks": "28",
    "table_blocks": "15",
    "text_source": "pdfplumber (por coordenada)",
    "images": "PyMuPDF (84 imagens embutidas)",
    "elapsed_seconds": "2.76"
  },

  "success": true,
  "error": ""
}
```

### Campos explicados

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `content_blocks` | lista | **A fonte da verdade.** Blocos em ordem de leitura. Iterate aqui para processar o documento. |
| `full_text` | string | Concatenação dos blocos de texto apenas. Conveniente para busca por palavra-chave, mas não use para chunking — perde o contexto das tabelas. |
| `tables` | lista | Vista agregada de todas as tabelas. Conveniente para acessar dados estruturados diretamente. |
| `bbox` | lista de 4 floats | `[x0, top, x1, bottom]` em pontos PDF (1 ponto = 1/72 de polegada). Permite saber exatamente onde na página a tabela está. |
| `source` | string | Qual biblioteca gerou o bloco. Útil para diagnóstico de qualidade. |
| `extraction_log` | dict | Diagnóstico técnico: contagens, bibliotecas usadas, tempo total. |

---

## 10. Como usar o output no pipeline de chunking

### Estratégia recomendada: iterate por `content_blocks`

```python
import json

with open("saida_json/documento.json", encoding="utf-8") as f:
    doc = json.load(f)

chunks = []
current_text_buffer = []

for block in doc["content_blocks"]:
    
    if block["type"] == "image":
        # Imagens: guardar referência com página para multimodal RAG
        continue

    elif block["type"] == "text":
        # Acumular texto — o chunker de texto vai fatiar depois
        current_text_buffer.append(block["content"])

    elif block["type"] == "table":
        # Antes de processar a tabela: flush do buffer de texto
        if current_text_buffer:
            chunks.append({
                "type": "text",
                "page": block["page"],
                "content": "\n\n".join(current_text_buffer)
            })
            current_text_buffer = []

        # Serializar a tabela como markdown para embedding
        rows = block["data"]
        if rows:
            header = "| " + " | ".join(rows[0].keys()) + " |"
            sep    = "| " + " | ".join("---" for _ in rows[0]) + " |"
            body   = "\n".join(
                "| " + " | ".join(str(v or "") for v in row.values()) + " |"
                for row in rows
            )
            md_table = f"{header}\n{sep}\n{body}"
            chunks.append({
                "type": "table",
                "page": block["page"],
                "content": md_table
            })

# Flush final
if current_text_buffer:
    chunks.append({"type": "text", "content": "\n\n".join(current_text_buffer)})
```

### Por que serializar tabelas como Markdown

Modelos de embedding como `bge-m3` processam texto. Tabelas em formato dict não têm representação textual natural. O formato Markdown preserva a estrutura relacional (coluna-valor) de forma que o modelo consegue entender:

```markdown
| Revisão | Motivo da Revisão          | Data       |
| ---     | ---                        | ---        |
| 1.0     | Consulta Pública n° 05/2012 | 16.10.2012 |
| 2.0     | Adequação à REN n° 583/2013 | 22.12.2014 |
```

Isso é semanticamente muito melhor para embedding do que `{"Revisão": "1.0", "Motivo": "Consulta..."}`.

### Não misture `full_text` com `tables` para chunking

```python
# ERRADO — perde ordem de leitura e contexto posicional
text_chunks = chunk_text(doc["full_text"])
for table in doc["tables"]:
    text_chunks.append(serialize_table(table))  # tabelas jogadas no final

# CORRETO — preserva contexto e ordem
chunks = process_content_blocks(doc["content_blocks"])  # como mostrado acima
```

---

## 11. Instalação e dependências

### Dependências Python

```bash
pip install pymupdf pymupdf4llm pypdf pdfplumber "camelot-py[cv]" tabula-py \
            pytesseract easyocr pdf2image tqdm pandas Pillow opencv-python
```

| Biblioteca | Versão mínima | Papel |
|------------|--------------|-------|
| `pymupdf` | >= 1.24.0 | Metadados, extração de imagens embutidas |
| `pdfplumber` | >= 0.11.0 | Núcleo da extração: texto + tabelas lattice |
| `pypdf` | >= 4.0.0 | Fallback de texto nativo |
| `pytesseract` | >= 0.3.10 | OCR via Tesseract (para PDFs escaneados) |
| `easyocr` | >= 1.7.0 | OCR fallback do Tesseract |
| `pdf2image` | >= 1.17.0 | Renderização de páginas para OCR e imagens |
| `camelot-py[cv]` | >= 0.11.0 | Fallback de tabelas stream mode |
| `tabula-py` | >= 2.9.0 | Fallback de tabelas stream mode |
| `pandas` | >= 2.0.0 | Manipulação de DataFrames para tabelas |
| `tqdm` | >= 4.66.0 | Barra de progresso no processamento em lote |
| `Pillow` | >= 10.0.0 | Manipulação de imagens |
| `opencv-python` | >= 4.9.0 | Dependência do camelot |

### Dependências de sistema

| Ferramenta | Necessária para | Instalação Windows | Instalação Linux |
|------------|----------------|-------------------|-----------------|
| **Tesseract OCR** | `pytesseract` (OCR) | [UB-Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki) | `sudo apt install tesseract-ocr tesseract-ocr-por` |
| **Poppler** | `pdf2image`, `camelot` | [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases) | `sudo apt install poppler-utils` |
| **Java JRE** | `tabula-py` | [java.com](https://www.java.com) | `sudo apt install default-jre` |
| **Ghostscript** | `camelot-py` | [ghostscript.com](https://www.ghostscript.com) | `sudo apt install ghostscript` |

> **Nota**: Para os PDFs da ANEEL (gerados por Word), apenas PyMuPDF e pdfplumber são necessários. Tesseract e Poppler só entram em ação para PDFs escaneados.

---

## 12. CLI — linha de comando

### Processar um arquivo único

```bash
python pdf_extractor.py "dados/pdfs/documento.pdf"
```

### Processar um diretório inteiro

```bash
python pdf_extractor.py "dados/pdfs/" --output-dir "saida_json"
```

### Processar em paralelo (lotes grandes)

```bash
python pdf_extractor.py "dados/pdfs/" --parallel --workers 8 --output-dir "saida_json"
```

### Processar sem subdiretórios

```bash
python pdf_extractor.py "dados/pdfs/" --no-recursive
```

### Argumentos disponíveis

| Argumento | Padrão | Descrição |
|-----------|--------|-----------|
| `input` | (obrigatório) | Caminho para `.pdf` ou diretório |
| `--output-dir` | `extracted_json` | Diretório para salvar os JSONs |
| `--workers` | `4` | Número de processos paralelos |
| `--parallel` | `False` | Habilita ProcessPoolExecutor |
| `--no-recursive` | `False` | Não busca em subdiretórios |

### Usar como módulo Python

```python
from src.extraction import process_pdf, process_batch, find_pdfs

# Arquivo único
result = process_pdf("dados/pdfs/documento.pdf")
print(result.full_text[:500])
for block in result.content_blocks:
    if block.type == "table":
        print(f"Tabela na pág {block.page}: {len(block.data)} linhas")

# Lote
pdfs = find_pdfs("dados/pdfs/", recursive=True)
results = process_batch(pdfs, json_output_dir="saida_json")

# Lote paralelo
results = process_batch(
    pdfs,
    json_output_dir="saida_json",
    max_workers=8,
    use_multiprocessing=True
)
```

---

## 13. Constantes configuráveis

Todas as constantes estão no topo do arquivo e podem ser ajustadas sem modificar a lógica:

```python
MIN_TEXT_LENGTH = 50
# Mínimo de caracteres para considerar texto válido.
# Aumentar se seus PDFs têm páginas de capa com poucos chars que
# disparam OCR desnecessariamente.

TABLE_EMPTY_CELL_THRESHOLD = 0.80
# Tabelas com mais de 80% de células vazias são descartadas.
# Diminuir (ex: 0.60) se tabelas legítimas com muitos campos opcionais
# estiverem sendo descartadas.

BBOX_TOLERANCE = 2.0
# Tolerância em pontos para considerar um objeto dentro de uma tabela.
# Aumentar se texto de borda de tabela estiver vazando para o bloco de texto.

IMAGE_DPI = 200
# DPI para renderização pdf2image e OCR.
# Aumentar para 300 se a qualidade do OCR for insatisfatória.
# Diminuir para 150 para processar mais rápido com menor qualidade.

MAX_WORKERS = 4
# Workers padrão para processamento paralelo.

OCR_LANG_TESSERACT = "por"
# Idioma Tesseract. "por" = português, "eng" = inglês, "por+eng" = ambos.

OCR_LANG_EASYOCR = ["pt"]
# Idioma EasyOCR.

IMAGE_OUTPUT_DIR = "extracted_images"
JSON_OUTPUT_DIR  = "extracted_json"
# Diretórios de saída padrão.
```
