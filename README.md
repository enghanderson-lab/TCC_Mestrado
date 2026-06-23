# TCC Mestrado — Busca Semântica em Vídeo via VLM

Projeto de mestrado para desenvolvimento de um pipeline de busca semântica em
gravações de vídeo de longa duração (ex.: 12h), usando modelos de licença
permissiva (Apache 2.0 / MIT, sem royalties comerciais), permitindo que o
usuário pesquise por linguagem natural (ex.: "homem de camisa branca e boné
vermelho") e receba os frames correspondentes com grau de confiança associado.
O resultado será disponibilizado como microsserviço.

## Estrutura do repositório

```
00_proposta/              Projeto de pesquisa / anteprojeto
01_revisao_bibliografica/ Mapeamento de literatura, candidatos a VLM
02_dissertacao/           Documento da dissertação (LaTeX/abnTeX2)
03_codigo/                Implementação do pipeline
  API.md                  Guia de uso da API HTTP (endpoints, exemplos)
  src/video_search/
    cli.py                Ponto de entrada CLI: comandos index e search
    api.py                Microsserviço HTTP (FastAPI): endpoints index/search
    pipeline.py           run_index/run_search — lógica compartilhada por cli.py e api.py
    siglip_embedder.py    SigLIPEmbedder — encoder multilingue (Apache 2.0)
    vlm_describer.py      Qwen2VLDescriber — legenda condicionada a query
    embedding_store.py    Índice em memória (numpy), com metadata.json
    frame_extractor.py    Extração de frames por timestamp (OpenCV)
04_dados/
  raw/                    Vídeos de teste e frames de validação
  index/                  Índices SigLIP gerados (embeddings.npy + records.json)
05_experimentos/          Resultados, métricas e logs
```

## Pipeline implementado

O pipeline opera em dois estágios separados:

### Estágio 1 — Indexação (executado uma vez, offline)

```
Vídeo .mp4
  └─ Extração de frames a cada N segundos (OpenCV)
       └─ SigLIP encode_image em batch (GPU, ~ms/frame)
            └─ embeddings.npy + records.json + metadata.json
```

Custo: ~3–4 minutos para 6 horas de vídeo (intervalo de 1s, GPU RTX 4060).

### Estágio 2 — Busca (sob demanda)

```
Query em português
  └─ SigLIP encode_text (suporte nativo a PT, sem tradução)
       └─ similaridade contra embeddings.npy (numpy, <1s)
            └─ Top-K frames
                 └─ Qwen2.5-VL describe_for_query (1 frase focada na query)
                      └─ sentence-transformer: confiança = cos_sim(query, legenda)
                           └─ Resultado rerankeado por confiança
```

O VLM (Qwen2.5-VL) roda apenas sobre os top-K resultados do SigLIP — nunca em
todos os frames.

## Modelos utilizados

| Modelo | Papel | Licença |
|--------|-------|---------|
| SigLIP multilingual base-patch16-256 | Indexação e retrieval imagem/texto (PT nativo) | Apache 2.0 |
| Qwen2.5-VL-3B-Instruct (4-bit) | Legenda condicionada à query (reranking) | Apache 2.0 |
| paraphrase-multilingual-mpnet-base-v2 | Score de confiança (sentence-transformer) | Apache 2.0 |

## Uso rápido

```powershell
cd 03_codigo
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"

# Indexar um vídeo (SigLIP)
python -m video_search.cli index "video.mp4" --output "..\04_dados\index\meu_video" --interval 1

# Buscar
python -m video_search.cli search "homem com caderno de anotações" --index "..\04_dados\index\meu_video"

# Buscar sem VLM (apenas SigLIP, mais rápido)
python -m video_search.cli search "homem com caderno de anotações" --index "..\04_dados\index\meu_video" --no-caption
```

### Via microsserviço HTTP

```powershell
cd 03_codigo
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
uvicorn video_search.api:app --host 0.0.0.0 --port 8000
```

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/health` | Sanity check |
| GET | `/indexes` | Lista índices existentes (nome, nº de frames, modelo) |
| POST | `/index` | `{video_path, name, interval, batch_size, limit}` → dispara indexação em background, retorna `{job_id, status}` |
| GET | `/index/jobs/{job_id}` | Status do job (`queued`/`running`/`done`/`error`) |
| POST | `/search` | `{query, index, top_k, with_caption}` → lista de resultados (frame, score, confiança, legenda) |

`video_path` é um caminho em disco acessível ao processo do serviço (sem
upload via multipart — vídeos de horas não cabem num upload HTTP razoável).
`name`/`index` são resolvidos dentro de `04_dados/index` (configurável via
`VIDEO_SEARCH_INDEX_ROOT`).

Guia completo com exemplos de requisição/resposta para cada endpoint:
[03_codigo/API.md](03_codigo/API.md).

## Achados de validação

### Evolução do modelo de retrieval: CLIP → tradução de query → M-CLIP → SigLIP

O retrieval passou por três abordagens antes de convergir para SigLIP:

**1. CLIP (OpenCLIP ViT-B/32, laion2b) com query em português.** Baseline
inicial. Como o modelo foi treinado majoritariamente em inglês, retrieval com
query em PT era fraco.

**2. CLIP + tradução automática da query (PT→EN via Qwen2-VL).** Tentativa de
mitigar a limitação acima traduzindo a query antes do `encode_text`. Resultado:
ganho **modesto** — `clip_score` do top-1 subiu de 0,185 para 0,193 no vídeo
de teste. Custo extra: uma chamada ao VLM só para tradução, antes mesmo do
retrieval, e dependência de uma tradução automática nem sempre fiel.

**3. M-CLIP (`clip-ViT-B-32-multilingual-v1` + OpenAI ViT-B/32) — suporte
nativo a PT, sem tradução.** Comparação direta com CLIP+tradução na mesma
query ("homem com caderno de anotações"):

| Modelo | Posição do frame correto | retrieval_score |
|--------|---------------------------|------------------|
| CLIP (+tradução) | 3º lugar | 0,1339 |
| M-CLIP | **1º lugar** | **0,2258** (+68%) |

M-CLIP superou claramente o CLIP traduzido, eliminando a necessidade do passo
de tradução e do viés introduzido por ele.

**4. SigLIP multilingual (`google/siglip-base-patch16-256-multilingual`) —
escolha final.** Mesma vantagem de suporte nativo a PT do M-CLIP, mas treinado
com *sigmoid loss* em vez de softmax, o que produz scores de similaridade mais
discriminativos (cada par imagem-texto é pontuado independentemente, em vez de
competir dentro do batch). Embedding dim=768 (vs 512 do CLIP/M-CLIP).

**Decisão**: diante dessa progressão, **CLIP, M-CLIP e a tradução de query via
Qwen2-VL foram descontinuados** — o pipeline usa exclusivamente **SigLIP**
como modelo de retrieval. O código correspondente (`embedder.py`,
`mclip_embedder.py`, `translate_to_english()`, opções `--model clip/mclip` da
CLI) foi removido.

**Limitação residual conhecida do SigLIP**: confusão entre objetos
visualmente parecidos (ex.: escada vs. rack de teto em vídeo de câmera de
segurança elevada) — o VLM de reranking também tende a confirmar o objeto
errado nesses casos (zero-shot embeddings têm dificuldade em distinções
visuais finas). Considerada aceitável; ver discussão na dissertação.

### Reranking por confiança corrige erros do retrieval

Teste com `teste05.mp4`, query "homem com caderno de anotações" (na época,
indexado com CLIP):

| Rank retrieval | Rank VLM | t | retrieval_score | confiança |
|-----------------|----------|---|----------------|-----------|
| 1 | 3 | 00:32 | 0,1878 | 63,5% |
| 2 | 4 | 00:00 | 0,1614 | 64,8% |
| **3** | **1** | **00:31** | **0,1373** | **76,3%** |
| 4 | 2 | 00:01 | 0,1217 | 65,6% |
| 5 | 5 | 00:21 | 0,1176 | 15,4% ← caminhão, rejeitado |

**O frame correto estava em 3º lugar no retrieval e foi elevado para 1º pelo
VLM.** Confirmado manualmente: o homem com caderno aparece exatamente no
frame 31 (t=00:31). O reranking por confiança é essencial para recuperar o
resultado correto.

### Queries específicas (placas) produzem scores mais altos — mas com risco de alucinação

Teste com `teste04.mp4`, query "veículo com a placa IYS5B37":

- retrieval_score: ~0,29 (bem acima do ~0,19 de queries semânticas genéricas)
- confiança VLM: 82–86% nos frames corretos

**Limitação identificada no modelo 2B**: ao buscar por uma placa ligeiramente
diferente ("IVS5B37" em vez de "IYS5B37"), o Qwen2-VL-2B alucinava a placa —
gerou legendas dizendo "IVS5B37" mesmo o vídeo contendo "IYS5B37", resultando
em confiança artificialmente alta (até 93%).

Isso é **viés de confirmação em OCR**: o modelo de 2B parâmetros tende a
repetir o texto da query ao invés de ler os pixels com precisão. O pipeline foi
atualizado para **Qwen2.5-VL-3B-Instruct** (Apache 2.0, com quantização 4-bit
para caber em GPUs de 6–8GB), com legenda condicionada à query
(`describe_for_query`) em vez da pergunta direta "Esta imagem mostra X?", que
induzia o mesmo tipo de confirmação.

### Qualidade de confiança por tipo de query

| Query | Frames corretos acima de 80% | Observação |
|-------|------------------------------|------------|
| "carro estacionado na rua" | 4/5 | Semântica visual simples |
| "ponte sobre o rio" | 3/3 | Alta separação match/mismatch |
| "estrada com linha amarela" | 3/3 | Atributo visual específico |
| "montanhas cobertas de árvores" | 1/3 (top-1 = 80,2%) | Cena ampla, boa discriminação |
| "mulher de jaqueta preta na calçada" | 0/3 (28–47%) | Conteúdo ausente no vídeo — sinalizado corretamente |

## Próximos passos

### 1. Indexação ao vivo via stream RTSP / RTMP

A arquitetura atual exige que o vídeo seja gravado antes de indexar. Uma
evolução natural é indexar em tempo real enquanto o vídeo é gravado:

```
Câmera IP (RTSP/RTMP)
  └─ OpenCV VideoCapture(rtsp://...)
       └─ Frame a cada N segundos → SigLIP encode_image (tempo real)
            └─ Índice crescente em disco
                 └─ Busca disponível imediatamente durante a gravação
```

O SigLIP continua sendo usado apenas sob demanda na busca (top-K).
Vantagem principal: elimina a etapa de indexação pós-gravação — o índice
está pronto no momento em que a busca é necessária.

Protótipo planejado: novo comando `stream-index` no CLI com suporte a URLs
RTSP e RTMP via OpenCV.

### 2. Endurecer o microsserviço FastAPI para produção

O microsserviço (`api.py`) já expõe `index`/`search` via HTTP (ver seção
"Uso rápido"), mas fora de escopo na v1: autenticação/autorização (assumido
em rede interna por ora) e persistência dos jobs de indexação em disco/banco
(hoje em memória, perdidos ao reiniciar o processo).
