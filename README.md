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
  src/video_search/
    cli.py                Ponto de entrada: comandos index e search
    embedder.py           ClipEmbedder (OpenCLIP ViT-B/32, MIT)
    mclip_embedder.py     MCLIPEmbedder — encoder multilingue (Apache 2.0)
    vlm_describer.py      Qwen2VLDescriber — legenda + tradução (Apache 2.0)
    embedding_store.py    Índice em memória (numpy), com metadata.json
    frame_extractor.py    Extração de frames por timestamp (OpenCV)
04_dados/
  raw/                    Vídeos de teste e frames de validação
  index/                  Índices CLIP/M-CLIP gerados (embeddings.npy + records.json)
05_experimentos/          Resultados, métricas e logs
```

## Pipeline implementado

O pipeline opera em dois estágios separados:

### Estágio 1 — Indexação (executado uma vez, offline)

```
Vídeo .mp4
  └─ Extração de frames a cada N segundos (OpenCV)
       └─ CLIP encode_image em batch (GPU, ~ms/frame)
            └─ embeddings.npy + records.json + metadata.json
```

Custo: ~3–4 minutos para 6 horas de vídeo (intervalo de 1s, GPU RTX 4060).

### Estágio 2 — Busca (sob demanda, ~17–18 segundos por consulta)

```
Query em português
  └─ Qwen2-VL traduz para inglês (encode_text do CLIP é treinado em EN)
       └─ CLIP encode_text → similaridade contra embeddings.npy (numpy, <1s)
            └─ Top-K frames
                 └─ Qwen2-VL describe_for_query (1 frase focada na query)
                      └─ sentence-transformer: confiança = cos_sim(query, legenda)
                           └─ Resultado rerankeado por confiança
```

O VLM (Qwen2-VL) roda apenas sobre os top-K resultados do CLIP — nunca em
todos os frames. O tempo de busca é **fixo em ~17s** independente da duração
do vídeo.

## Modelos utilizados

| Modelo | Papel | Licença |
|--------|-------|---------|
| OpenCLIP ViT-B/32 (laion2b) | Indexação e retrieval de imagens | MIT |
| clip-ViT-B-32-multilingual-v1 | Encoder de texto multilingue (M-CLIP) | Apache 2.0 |
| Qwen2-VL-2B-Instruct | Legenda condicional + tradução PT→EN | Apache 2.0 |
| paraphrase-multilingual-mpnet-base-v2 | Score de confiança (sentence-transformer) | Apache 2.0 |

## Uso rápido

```powershell
cd 03_codigo
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"

# Indexar um vídeo (CLIP, padrão)
python -m video_search.cli index "video.mp4" --output "..\04_dados\index\meu_video" --interval 1

# Indexar com M-CLIP (multilingue, sem precisar de tradução)
python -m video_search.cli index "video.mp4" --output "..\04_dados\index\meu_video_mclip" --model mclip --interval 1

# Buscar
python -m video_search.cli search "homem com caderno de anotações" --index "..\04_dados\index\meu_video"

# Buscar sem VLM (apenas CLIP, mais rápido)
python -m video_search.cli search "homem com caderno de anotações" --index "..\04_dados\index\meu_video" --no-caption
```

## Achados de validação

### Reranking por confiança corrige erros do CLIP

Teste com `teste05.mp4`, query "homem com caderno de anotações":

| Rank CLIP | Rank VLM | t | retrieval_score | confiança |
|-----------|----------|---|----------------|-----------|
| 1 | 3 | 00:32 | 0,1878 | 63,5% |
| 2 | 4 | 00:00 | 0,1614 | 64,8% |
| **3** | **1** | **00:31** | **0,1373** | **76,3%** |
| 4 | 2 | 00:01 | 0,1217 | 65,6% |
| 5 | 5 | 00:21 | 0,1176 | 15,4% ← caminhão, rejeitado |

**O frame correto estava em 3º lugar no CLIP e foi elevado para 1º pelo VLM.**
Confirmado manualmente: o homem com caderno aparece exatamente no frame 31
(t=00:31). O reranking por confiança é essencial para recuperar o resultado
correto.

### Queries específicas (placas) produzem scores mais altos — mas com risco de alucinação

Teste com `teste04.mp4`, query "veículo com a placa IYS5B37":

- retrieval_score: ~0,29 (bem acima do ~0,19 de queries semânticas genéricas)
- confiança VLM: 82–86% nos frames corretos

**Limitação crítica identificada**: ao buscar por uma placa ligeiramente
diferente ("IVS5B37" em vez de "IYS5B37"), o Qwen2-VL-2B alucinous a placa
correta — gerou legendas dizendo "IVS5B37" mesmo o vídeo contendo "IYS5B37",
resultando em confiança artificialmente alta (até 93%).

Isso é **viés de confirmação em OCR**: o prompt `describe_for_query` condiciona
o VLM a descrever elementos relacionados à query, e o modelo de 2B parâmetros
tende a repetir o texto da query ao invés de ler os pixels da placa com
precisão. O pipeline é confiável para queries semânticas/visuais (pessoas,
roupas, objetos, ações), mas **não deve ser usado como sistema de OCR de
placas** sem um módulo dedicado de leitura de texto (ex.: EasyOCR, PaddleOCR).

### Qualidade de confiança por tipo de query

| Query | Frames corretos acima de 80% | Observação |
|-------|------------------------------|------------|
| "carro estacionado na rua" | 4/5 | Semântica visual simples |
| "ponte sobre o rio" | 3/3 | Alta separação match/mismatch |
| "estrada com linha amarela" | 3/3 | Atributo visual específico |
| "montanhas cobertas de árvores" | 1/3 (top-1 = 80,2%) | Cena ampla, boa discriminação |
| "mulher de jaqueta preta na calçada" | 0/3 (28–47%) | Conteúdo ausente no vídeo — sinalizado corretamente |

## Próximos passos

### 1. Avaliar M-CLIP vs CLIP para queries em português

O M-CLIP (`clip-ViT-B-32-multilingual-v1` + OpenAI ViT-B/32) já está
implementado (`--model mclip`). Falta comparar retrieval_scores com o CLIP
(laion2b) + tradução automática para determinar qual entrega melhores resultados
em PT sem depender do Qwen2-VL para tradução.

### 2. Indexação ao vivo via stream RTSP / RTMP

A arquitetura atual exige que o vídeo seja gravado antes de indexar. Uma
evolução natural é indexar em tempo real enquanto o vídeo é gravado:

```
Câmera IP (RTSP/RTMP)
  └─ OpenCV VideoCapture(rtsp://...)
       └─ Frame a cada N segundos → CLIP encode_image (tempo real, ~5–10ms)
            └─ Índice crescente em disco
                 └─ Busca disponível imediatamente durante a gravação
```

O CLIP é rápido o suficiente para acompanhar 1 frame/s em tempo real.
O Qwen2-VL continua sendo usado apenas sob demanda na busca (top-K).
Vantagem principal: elimina a etapa de indexação pós-gravação — o índice
está pronto no momento em que a busca é necessária.

Protótipo planejado: novo comando `stream-index` no CLI com suporte a URLs
RTSP e RTMP via OpenCV.

### 3. Microsserviço FastAPI

Expor `index` e `search` como endpoints HTTP para integração com o VSM em Qt.
