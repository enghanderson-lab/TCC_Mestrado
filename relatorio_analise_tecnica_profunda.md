# Relatório de Análise Técnica Profunda — TCC de Mestrado
**Busca Semântica em Gravações de Vídeo de Longa Duração por Linguagem Natural Utilizando Modelos de Visão-Linguagem de Código Aberto**

*Análise gerada em 04/07/2026, com base em leitura integral do código-fonte, documentação e dados presentes no repositório `TCC_Mestrado`.*

> Nota: o repositório já contém dois relatórios de análise anteriores
> (`relatorio_analise_critica.md`, 27/06/2026, e `estrategia_otimizacao.md`,
> 28/06/2026). Desde então o código passou por uma refatoração grande
> (reorganização em subpacotes `models/`, `media/`, `indexing/`, `utils/`,
> adição de FAISS, MMR reranker, pipeline assíncrono por filas, cache de
> embeddings SQLite). Este relatório reflete o estado **atual** do
> repositório e sinaliza explicitamente onde diverge das análises
> anteriores.

---

## 0. Estrutura geral do repositório

```
TCC_Mestrado/
├── README.md                        Documentação principal do projeto (científica + técnica)
├── relatorio_analise_critica.md     Análise crítica anterior (27/06/2026) — parcialmente desatualizada
├── estrategia_otimizacao.md         Plano de otimização (28/06/2026) — parcialmente já implementado
├── 00_proposta/
│   └── projeto_pesquisa.md          Anteprojeto/rascunho de pesquisa (não formatado em ABNT)
├── 03_codigo/                       Único subprojeto de código — implementação Python do pipeline
│   ├── README.md
│   ├── requirements.txt             Sem versões pinadas (exceto torch/torchvision via constraints)
│   ├── constraints-cuda.txt         Trava torch/torchvision na build CUDA cu121
│   ├── pytest.ini
│   ├── configs/multi_index.yaml     Config do pipeline multi-câmera
│   ├── .venv/                       Ambiente virtual Windows (Python 3.12, não versionado)
│   ├── src/video_search/            Pacote principal (ver seção 3)
│   ├── scripts/                     10 scripts utilitários — TODOS quebrados (ver seção "Inconsistências")
│   └── tests/                       7 arquivos de teste (357 linhas), atualizados e consistentes
└── 04_dados/
    ├── README.md                    Diretriz de LGPD/anonimização
    ├── raw/                         Só imagens de preview/debug (9 .jpg) — nenhum vídeo versionado
    ├── index/meu_video/             1 índice residual (formato legado numpy), referencia vídeo externo
    └── models/qwen25vl3b-nf4/       Cache do Qwen2.5-VL-3B já quantizado em 4-bit (gerado em disco)
```

**Não há múltiplos subprojetos.** Existe um único projeto de código
(`03_codigo/`), de um único autor, evoluindo de forma linear (confirmado
pelo `git log`, 24 commits, todos na branch `master`). As pastas
`01_revisao_bibliografica/` e `02_dissertacao/`, referenciadas tanto no
`README.md` principal quanto em `00_proposta/projeto_pesquisa.md`,
**não existem fisicamente no repositório**. A pasta `05_experimentos/`
mencionada no relatório crítico anterior também não existe mais (foi
removida no commit `1bdca25 Remove videos de teste indexados e pasta
05_experimentos nao utilizada`).

Estado do git no momento da análise: há um `git status` com uma mistura de
renomeações já staged (a refatoração de `src/video_search/*.py` para
subpastas `indexing/`, `media/`, `models/`, `utils/`) e diversos arquivos
novos ainda não commitados (`async_pipeline.py`, `embedding_cache.py`,
`faiss_store.py`, `reranker.py`, a pasta `profiling/` inteira, e o próprio
`estrategia_otimizacao.md`). Ou seja, **o repositório está em estado de
refatoração incompleta e não commitada** — um `git commit` faltando para
consolidar a reorganização em subpacotes.

---

## 1. Arquitetura geral do sistema (fluxo ponta a ponta)

```
┌─────────────────────────── ESTÁGIO 1 — INDEXAÇÃO (offline) ───────────────────────────┐
│                                                                                          │
│  Vídeo(s) .mp4 (1 ou N câmeras)                                                          │
│    └─ VideoReader (thread por câmera, OpenCV)                                            │
│         └─ [opcional] MotionFilter — descarta frame se diff de pixels < motion_ratio      │
│              └─ BatchCollector — agrupa frames de TODAS as câmeras em lotes              │
│                   (batch_size ou batch_timeout_ms, o que vier primeiro)                   │
│                    └─ SigLIPWorker (GPU, serializado por _gpu_lock)                       │
│                         └─ [opcional] EmbeddingCache (SQLite) — pula reencode se já visto │
│                              └─ [opcional] EmbeddingSimilarityFilter — descarta embedding  │
│                                 quase idêntico ao último ACEITO daquela câmera            │
│                                   └─ StoreWriter → FaissStore (ou EmbeddingStore legado)  │
│                                        └─ grava faiss.index + records.json + metadata.json│
└──────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────── ESTÁGIO 2 — BUSCA (sob demanda) ───────────────────────────┐
│                                                                                          │
│  Query em português (usuário)                                                            │
│    └─ SigLIP encode_text (suporte nativo PT, sem tradução)                                │
│         └─ modo FAST:                                                                    │
│              └─ FaissStore.search(top_k) → resultado imediato (sem VLM)                  │
│         └─ modo DETAILED (default):                                                       │
│              └─ FaissStore.search_with_embeddings(retrieval_k=100)                        │
│                   └─ MmrReranker (MMR: λ·relevância − (1−λ)·redundância) → top_k          │
│                        └─ extract_frames_at() — abre vídeo 1x, extrai os top_k frames     │
│                             └─ Qwen2.5-VL-3B describe_for_query_batch() — 1 chamada       │
│                                generate() para todo o lote                                │
│                                  └─ sentence-transformer encode(query + legendas)          │
│                                       └─ confiança = cos_sim(query_emb, legenda_emb)       │
│                                            └─ resultados reordenados por confiança         │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

Ponto arquitetural central: o pipeline é **um processo CLI que nasce e
morre a cada invocação**. Não há servidor HTTP residente em produção (a
API FastAPI foi prototipada e removida — commit `fbecca6`). O
`ModelManager` resolve a questão de "não recarregar 3× dentro do mesmo
processo", mas **entre invocações separadas do CLI, os modelos são
recarregados do zero** (~8-10s de overhead fixo por busca).

---

## 2. Documentação existente e objetivos declarados

### 2.1 `00_proposta/projeto_pesquisa.md` (rascunho de anteprojeto)

Título provisório: *"Busca Semântica em Gravações de Vídeo de Longa Duração
por Linguagem Natural Utilizando Modelos de Visão-Linguagem de Código
Aberto"*.

Problema de pesquisa declarado: viabilizar, **num sistema de
videomonitoramento (VMS) desenvolvido em Qt com arquitetura de
microsserviços**, busca por linguagem natural via chat em gravações de
~12h, retornando frames com grau de confiança, usando **exclusivamente
VLMs de licença permissiva** (Apache 2.0/MIT, sem royalties comerciais).

Seis objetivos específicos declarados:
1. Levantar/comparar VLMs open-source com licença permissiva. **Feito**
   (CLIP → M-CLIP → SigLIP; Qwen2.5-VL vs Moondream2 vs Florence-2).
2. Definir estratégia de amostragem de frames para vídeo longo. **Feito**
   (extração por intervalo + motion detection + filtro de similaridade).
3. Projetar pipeline de indexação e busca por similaridade semântica.
   **Feito**.
4. Implementar microsserviço (API) que receba consulta via chat.
   **Não cumprido** — API FastAPI foi prototipada e removida (ver seção 9).
5. Integrar o microsserviço a um protótipo de interface Qt do VSM.
   **Não cumprido** — não há nenhum código Qt/frontend no repositório.
6. Avaliar precisão, recall e latência num conjunto de vídeos de teste com
   ground truth. **Não cumprido formalmente** — existem observações
   pontuais manuais (5-10 queries, poucos frames), mas nenhum dataset
   anotado, nenhuma métrica de Precision@K/Recall@K calculada
   sistematicamente.

O documento é explicitamente marcado como "rascunho de trabalho... após
consolidado, deve ser formatado em ABNT" — isto é, **não é a dissertação
final**, é uma nota de alinhamento com o orientador, com cronograma "a
definir".

### 2.2 `README.md` (raiz) — a documentação de fato mais completa

Funciona como um diário de bordo técnico-científico muito bem escrito:
descreve a arquitetura, a evolução de modelos (com números de experimentos
reais), a otimização de latência (tabela antes/depois com números
concretos) e limitações conhecidas documentadas com honestidade
(hubness do SigLIP, viés de confirmação do VLM em OCR). É, de fato, como
apontado no relatório crítico anterior, um esboço muito forte para os
capítulos de Implementação e Experimentos da dissertação — mas **não é a
dissertação em si**, nem está em formato acadêmico (ABNT/citações formais).

### 2.3 `03_codigo/README.md`

Documentação operacional (como instalar, rodar, testar). Menciona
"Ideias de stack (a confirmar): FAISS/Qdrant" como algo futuro — **isso já
está desatualizado**: FAISS já foi implementado (`faiss_store.py`) e é o
store default (`PipelineConfig.use_faiss: bool = True`).

### 2.4 `04_dados/README.md`

Curto, só com diretriz de cuidado com LGPD (dados de vigilância podem
conter pessoas identificáveis) e estrutura sugerida (`raw/`, `anotacoes/`,
`amostras/`). As pastas `anotacoes/` e `amostras/` sugeridas **não
existem** — reforça a ausência de dataset de avaliação formal.

### 2.5 `relatorio_analise_critica.md` e `estrategia_otimizacao.md`

Dois relatórios de auto-análise já produzidos anteriormente (aparentemente
por uma sessão de IA anterior). Contêm diagnósticos corretos na época, mas
**parcialmente superados pelo código atual**:
- M1/M3 (dissertação e revisão bibliográfica ausentes): **ainda válido**,
  nenhuma das duas pastas existe.
- M2 (dataset com ground truth): **ainda válido**, nenhum dataset anotado
  encontrado.
- A "Estratégia A" (microsserviço com modelos persistentes via FastAPI
  lifespan) do `estrategia_otimizacao.md` **foi parcialmente implementada
  de outra forma**: em vez de um servidor HTTP, o `ModelManager` cumpre o
  mesmo papel dentro de um único processo CLI de vida longa — mas o
  problema original (CLI nasce e morre a cada busca do usuário final) só
  é resolvido se o processo for mantido rodando entre buscas, o que a CLI
  atual não faz por padrão.
- "Estratégia C" (QB-Norm para mitigar hubness): **não encontrada
  implementada** em nenhum arquivo do `src/`. Permanece como proposta.
- "Estratégia B" (YOLO + crop SigLIP para sujeito secundário): **não
  implementada**. Não há nenhuma referência a `ultralytics`/YOLO no
  `requirements.txt` nem no código.
- M9 (pinar versões no requirements.txt): **ainda não feito** —
  `requirements.txt` continua sem versões (exceto via
  `constraints-cuda.txt` para torch/torchvision).

---

## 3. Código-fonte: linguagem, frameworks e módulos

**Linguagem**: Python 3.12 exclusivamente. Não há JavaScript/TypeScript,
não há Qt/C++, não há frontend de nenhum tipo no repositório — apesar de o
anteprojeto declarar "VSM desenvolvido em Qt" como sistema-alvo de
integração, esse sistema Qt **não está neste repositório** (deve ser um
projeto externo/produto comercial da empresa do autor, fora de escopo do
código do TCC).

**Frameworks/bibliotecas principais** (de `requirements.txt` e imports):
- `torch` 2.5.1+cu121, `torchvision` 0.20.1+cu121 — base de deep learning
- `transformers` 5.12.0 — carregamento de SigLIP e Qwen2.5-VL
- `bitsandbytes` 0.49.2 — quantização 4-bit (NF4) do Qwen
- `accelerate` — device_map do Qwen
- `qwen-vl-utils` 0.0.14 — pré-processamento de imagens/vídeo para Qwen2.5-VL
- `sentence-transformers` 5.5.1 — modelo de confiança textual
- `faiss-cpu` 1.14.3 — índice vetorial (`faiss-gpu` sugerido como opção)
- `opencv-python` — extração de frames e motion detection
- `Pillow` — manipulação de imagens
- `pydantic` — validação do schema YAML (`MultiIndexConfig`)
- `PyYAML` — parsing do config multi-câmera
- `pytest` — testes
- `psutil`, opcional `nvidia-ml-py` — profiling de hardware
- `matplotlib` — gráfico de profiling
- **Resíduo encontrado no venv**: `open_clip_torch` 3.3.0 está instalado
  mas **não é mais importado por nenhum arquivo em `src/`** — vestígio da
  época em que o pipeline usava CLIP/OpenCLIP, já removido do código
  (commit `1204b64 Remove CLIP, M-CLIP e traducao de query`). Não consta
  em `requirements.txt`, então não seria reinstalado num ambiente limpo —
  é sujeira apenas do `.venv` local, não do requirements declarado.

### 3.1 Estrutura de `src/video_search/` (3.594 linhas de código)

```
video_search/
├── cli.py                    Entry point: comandos `index`, `index-multi`, `search`
├── indexing/
│   ├── pipeline.py           run_index() / run_search() — orquestração de alto nível
│   ├── async_pipeline.py     AsyncIndexPipeline — pipeline produtor-consumidor multi-thread
│   ├── multi_index.py        run_multi_index() — indexação concorrente de N câmeras
│   ├── batch_dispatcher.py   BatchDispatcher — CÓDIGO MORTO (não usado fora dos testes, ver seção 8)
│   ├── embedding_store.py    EmbeddingStore (numpy legado) + load_store() (dispatch automático)
│   ├── faiss_store.py        FaissStore — índice vetorial FAISS (IndexFlatIP)
│   ├── embedding_filter.py   EmbeddingSimilarityFilter — descarta embeddings redundantes
│   ├── embedding_cache.py    EmbeddingCache — cache SQLite de embeddings por hash SHA-256
│   └── reranker.py           MmrReranker — reranking por Maximal Marginal Relevance
├── media/
│   ├── frame_extractor.py    extract_frames() / extract_frames_at() (OpenCV)
│   └── motion_filter.py      MotionFilter — descarte por diff de pixels
├── models/
│   ├── vlm_abc.py            VisionLanguageModel (ABC): load/warmup/infer/unload
│   ├── model_config.py       ModelConfig (dataclass): device, precisão, warmup, compile
│   ├── model_manager.py      ModelManager — carrega e mantém residentes os 3 modelos
│   ├── siglip_embedder.py    SigLIPEmbedder — encoder multilingue de imagem/texto
│   └── vlm_describer.py      Qwen2VLDescriber — legendagem condicionada à query
├── utils/
│   ├── config.py             MultiIndexConfig (pydantic) — schema do YAML multi-câmera
│   └── hf_utils.py           load_offline_first() — evita round-trip ao HF Hub
└── profiling/
    ├── context.py            ProfilingContext — coleta de métricas thread-safe
    ├── hardware.py           HardwareMonitor — amostragem de CPU/RAM/VRAM/GPU%
    ├── report.py             generate_report() — relatório texto + gráfico
    └── chart.py              gráfico de barras (matplotlib) do tempo por etapa
```

Nenhum backend web, nenhuma API HTTP ativa, nenhum banco de dados
relacional/NoSQL tradicional — o "banco de dados" do sistema é
integralmente um índice vetorial em arquivo (FAISS ou numpy) mais um
cache SQLite opcional de embeddings.

---

## 4. Modelos de IA utilizados (nomes exatos, arquivo e trecho)

| Papel | Modelo exato | Licença | Onde no código |
|---|---|---|---|
| Retrieval imagem/texto | `google/siglip-base-patch16-256-multilingual` | Apache 2.0 | `src/video_search/models/siglip_embedder.py:50` — `MODEL_NAME = "google/siglip-base-patch16-256-multilingual"` |
| Legendagem/VLM de reranking | `Qwen/Qwen2.5-VL-3B-Instruct` (4-bit NF4 via bitsandbytes) | Apache 2.0 | `src/video_search/models/vlm_describer.py:68` — `model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"` |
| Score de confiança (similaridade texto-texto) | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | Apache 2.0 | `src/video_search/indexing/pipeline.py:28` — `CONFIDENCE_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"` |
| Índice vetorial | FAISS `IndexFlatIP` | MIT | `src/video_search/indexing/faiss_store.py:82` |

**Modelos avaliados e descartados** (documentados no README com evidência
experimental, código removido):
- **OpenCLIP ViT-B/32 (laion2b)** — baseline inicial, fraco em PT.
- **CLIP + tradução PT→EN via Qwen2-VL** — ganho modesto (0.185→0.193),
  descontinuado.
- **M-CLIP** (`clip-ViT-B-32-multilingual-v1` + OpenAI ViT-B/32) —
  superou CLIP+tradução (+68%), mas superado por SigLIP.
- **Qwen2-VL-2B** — alucinava OCR de placas veiculares (viés de
  confirmação), substituído por Qwen2.5-VL-3B.
- **Moondream2** (`vikhyatk/moondream2`) — rejeitado por dependência
  frágil de `libvips` no Windows e falta de suporte a batch/PT.
- **Florence-2-large** (`microsoft/Florence-2-large`) — rejeitado por 5
  incompatibilidades com `transformers==5.12.0` no código
  `trust_remote_code`, a última não corrigível sem reescrever o loop de
  geração (`EncoderDecoderCache` não subscriptável).

Essas duas últimas avaliações têm scripts dedicados: `scripts/
benchmark_vlm_florence2.py` (309 linhas) e evidências no README — mas o
próprio script Florence-2, como os demais, **tem imports quebrados** (ver
seção "Inconsistências").

---

## 5. Pipeline de processamento — vídeo/imagem até resultado consultável

Detalhamento passo a passo (modo `index-multi`, o caminho mais completo):

1. **Leitura**: uma `threading.Thread` por câmera chama
   `extract_frames(video_path, interval_sec)` (`media/frame_extractor.py`),
   que abre o vídeo via `cv2.VideoCapture`, calcula `frame_step = round(fps
   * interval_sec)` e itera lendo frame a frame, só convertendo/emitindo
   (`Frame(index, timestamp_sec, image)`, BGR→RGB via PIL) quando `frame_no
   % frame_step == 0`. Ou seja, a decodificação lê **todos** os frames do
   vídeo sequencialmente (sem seek), descartando os que não caem no
   intervalo — não há skip real de decodificação.
2. **Filtro de movimento** (opcional, default `enabled: true`): cada frame
   é redimensionado para 320×180, convertido para grayscale, comparado por
   `cv2.absdiff` contra o último frame **aceito** daquela câmera; se a
   fração de pixels com diferença ≥ `pixel_threshold=25` for menor que
   `motion_ratio=0.02`, o frame é descartado antes de qualquer custo de GPU.
3. **Batching cross-câmera**: um `BatchCollector` (dentro de
   `async_pipeline.py`, função `_batch_collector_worker`) drena uma fila
   compartilhada por todas as câmeras e monta lotes de até `batch_size=16`
   itens ou até `batch_timeout_ms=200ms` (o que vier primeiro) — frames de
   câmeras diferentes podem coexistir no mesmo lote GPU.
4. **Cache opcional de embeddings** (`embedding_cache.py`): se fornecido
   (`embedding_cache=` em `run_index`, não exposto por nenhuma flag da
   CLI atualmente — ver seção Inconsistências), calcula SHA-256 do buffer
   bruto da imagem e consulta um SQLite (`WAL` mode) antes de rodar
   inferência; só as imagens com cache miss vão à GPU.
5. **Embedding SigLIP** (`models/siglip_embedder.py`): o processor do
   HuggingFace faz o pré-processamento (resize 256×256, normalização),
   depois passa por `model.vision_model(pixel_values).pooler_output` — a
   nota de implementação explica que `get_image_features()` não é usado
   diretamente porque, em `transformers` 5.x, esse método retorna
   dataclasses em vez de tensors quando o modelo é carregado via
   `AutoModel`. O vetor resultante (768-dim) é L2-normalizado.
6. **Filtro de similaridade de embeddings** (opcional, default
   `enabled: true`, `similarity_threshold=0.97`): produto interno entre o
   novo embedding e o último **aceito** daquela câmera; se ≥ 0.97, descarta
   (frame redundante) sem gerar legenda depois.
7. **Persistência**: `FaissStore.add()` insere no índice `IndexFlatIP`
   (lazy-inicializado na dimensão do primeiro vetor) e acumula um
   `FrameRecord` (index, timestamp, nome/caminho do vídeo, caption vazio).
   `save()` grava `faiss.index` + `records.json` + `metadata.json`.
8. **Busca**: a query em português é encodada por
   `SigLIPEmbedder.encode_text()` (mesmo espaço vetorial). Em modo
   `detailed`, o FAISS retorna `retrieval_k=100` candidatos com embeddings
   reconstruídos (`IndexFlatIP.reconstruct`), o `MmrReranker` seleciona os
   `top_k` mais relevantes-e-diversos (fórmula MMR clássica, Carbonell &
   Goldstein 1998), os frames correspondentes são extraídos sob demanda
   (`extract_frames_at`, abre o vídeo 1 vez e faz seek para cada
   timestamp), o Qwen2.5-VL gera 1 legenda por frame **numa única chamada
   de `generate()` para todo o lote** (`describe_for_query_batch`), e a
   confiança final é a similaridade de cosseno entre o embedding da query
   e o embedding da legenda, ambos via `sentence-transformers`.

---

## 6. Objetivos e funcionalidades: implementado vs. planejado

| Item | Status | Evidência |
|---|---|---|
| Extração de frames por intervalo | ✅ Implementado | `media/frame_extractor.py` |
| Motion detection (descarte de frames estáticos) | ✅ Implementado | `media/motion_filter.py` |
| Batching cross-câmera na GPU | ✅ Implementado | `async_pipeline.py::_batch_collector_worker` |
| Filtro de similaridade de embeddings | ✅ Implementado | `indexing/embedding_filter.py` |
| Cache de embeddings por hash (SQLite) | ✅ Implementado, mas **não conectado à CLI** | `embedding_cache.py` existe; `cli.py` não expõe flag para ativá-lo |
| Índice vetorial FAISS | ✅ Implementado (substituiu numpy legado como default) | `faiss_store.py`, `PipelineConfig.use_faiss=True` |
| Reranking por diversidade (MMR) | ✅ Implementado | `indexing/reranker.py` |
| Legendagem condicionada à query (VLM) | ✅ Implementado | `models/vlm_describer.py::describe_for_query_batch` |
| Score de confiança via similaridade texto-texto | ✅ Implementado (não calibrado, ver seção 7) | `pipeline.py::_search_detailed` |
| Indexação multi-câmera concorrente | ✅ Implementado | `indexing/multi_index.py` |
| Profiling de performance (CPU/RAM/VRAM/GPU) | ✅ Implementado | pacote `profiling/` completo |
| Microsserviço HTTP (API REST) | ❌ Removido (prototipado e descartado) | Objetivo 4 da proposta; `git log` commit `fbecca6` |
| Integração com VSM Qt | ❌ Não iniciado | Nenhum código Qt no repo; Objetivo 5 da proposta |
| Avaliação formal (Precision@K/Recall@K, dataset anotado) | ❌ Não implementado | Nenhum dataset em `04_dados/anotacoes/`; nenhum script de métricas de IR |
| Calibração do score de confiança | ⚠️ Script existe, não integrado ao pipeline | `scripts/calibrate_confidence.py` calcula floor/ceil mas `pipeline.py` não os usa |
| Indexação ao vivo via RTSP/RTMP | ❌ Não implementado (planejado) | README seção "Próximos passos", comando `stream-index` não existe no `cli.py` |
| Detecção de pessoas + re-embedding por recorte (YOLO) | ❌ Não implementado (planejado em `estrategia_otimizacao.md`) | Nenhuma menção a `ultralytics`/YOLO no código |
| QB-Norm para mitigar hubness | ❌ Não implementado (planejado) | Nenhuma função `qb_norm` encontrada em `src/` |
| Revisão bibliográfica formal | ❌ Não iniciado | Pasta `01_revisao_bibliografica/` inexistente |
| Dissertação (documento acadêmico) | ❌ Não iniciado | Pasta `02_dissertacao/` inexistente |

---

## 7. Casos de uso suportados

1. **Indexar um único vídeo** (`cli.py index`): extrai frames a cada N
   segundos, gera embeddings SigLIP, salva índice. Sem motion
   detection/filtro de similaridade neste caminho (`MotionDetectionConfig(enabled=False)`
   e `EmbeddingFilterConfig(enabled=False)` são forçados explicitamente em
   `run_index`, ver `pipeline.py:115-116`).
2. **Indexar múltiplas câmeras simultaneamente** (`cli.py index-multi`):
   com motion detection, batching cross-câmera e filtro de similaridade
   ativos via YAML.
3. **Busca rápida sem VLM** (`--mode fast` ou `--no-caption`): retorna
   apenas ranking SigLIP bruto, sem legenda nem confiança calibrada.
4. **Busca detalhada com reranking e legendas** (`--mode detailed`,
   default): FAISS → MMR → Qwen, com confiança textual.
5. **Geração de relatório de profiling** (`--profiling-dir`): mede tempo
   por etapa, uso de hardware, throughput, gera `.txt` e `.png`.

Não há: interface de chat, interface web, autenticação, multiusuário,
streaming ao vivo, ou qualquer forma de consumo que não seja a linha de
comando local.

---

## 8. Telas / frontend

**Não existe frontend.** Nenhum arquivo `.html`, `.jsx`, `.tsx`, `.vue`,
nenhum framework de UI, nenhuma pasta `frontend/` ou similar. A menção a
"integração com protótipo de interface Qt" no anteprojeto (`00_proposta/
projeto_pesquisa.md`) refere-se a um sistema externo (o VMS comercial da
empresa do autor), que **não está incluído neste repositório** — é
apontado como objetivo futuro/fora do escopo atual.

A única "interface" existente é a saída de texto formatada do `cli.py`
(`cmd_search`), que imprime timestamp, índice de frame, score de
retrieval, confiança (%), legenda e nome do vídeo por linha.

---

## 9. APIs (REST/GraphQL)

**Não há API ativa no código atual.** Um microsserviço FastAPI
(`api.py`, com endpoints implícitos `/index` e `/search`, conforme
commits `91c474c Adiciona microsservico FastAPI para index/search` e
`cbc6aaf Adiciona guia de uso da API HTTP (API.md)`) foi criado e
**depois inteiramente removido** no commit `fbecca6 Remove API e CLIP;
otimiza latencia de busca com legenda (73s -> 37s)`. Não há vestígio de
`api.py` nem de `API.md` no estado atual do repositório (confirmado por
busca — nenhum arquivo `api.py` ou `API.md` existe em `03_codigo/`).

O arquivo `.claude/settings.local.json` contém um resquício de uma
sessão anterior de desenvolvimento com uma permissão de comando
`curl -X POST http://localhost:8002/index ...`, evidenciando que a API
chegou a ser testada manualmente antes de ser removida.

O `estrategia_otimizacao.md` propõe recriar essa API com FastAPI
`lifespan` para manter modelos residentes — isso continua como proposta
não implementada.

---

## 10. Banco de dados

**Não há banco de dados relacional ou NoSQL tradicional.** Persistência é
feita inteiramente em arquivos:

- **Índice vetorial** (equivalente a "vector store"): FAISS `IndexFlatIP`
  (produto interno = similaridade de cosseno para vetores L2-normalizados),
  serializado em `<dir>/faiss.index` via `faiss.write_index`. Alternativa
  legada: `EmbeddingStore` com `embeddings.npy` (array numpy empilhado) +
  busca por força bruta (`matrix @ query_vector`).
- **Metadados por frame**: `<dir>/records.json` — lista de `FrameRecord`
  (index, timestamp_sec, video, video_path, caption).
- **Metadados do índice**: `<dir>/metadata.json` — `{model_name,
  store_type, dim}`.
- **Cache de embeddings** (opcional, não conectado à CLI): SQLite
  (`embedding_cache.py`), schema:
  ```sql
  CREATE TABLE embeddings (
      image_hash TEXT PRIMARY KEY,   -- SHA-256 de modo+tamanho+pixels
      embedding  BLOB NOT NULL,       -- np.float32 serializado
      dim        INTEGER NOT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
  )
  ```
  Usa `PRAGMA journal_mode=WAL` e uma conexão SQLite por thread
  (`threading.local`), com lock explícito só nas escritas.
- **Cache de pesos de modelo quantizado**: não é "dado", mas vale notar —
  `04_dados/models/qwen25vl3b-nf4/` guarda o checkpoint Qwen2.5-VL-3B já
  quantizado em NF4 (safetensors + configs), gerado automaticamente na
  primeira execução (`vlm_describer.py::load`, linha ~126) para evitar
  requantizar (~25s) a cada `from_pretrained`.

`load_store()` (`embedding_store.py:86`) decide automaticamente entre
FAISS e numpy legado checando se `faiss.index` existe na pasta —
compatibilidade retroativa com índices antigos (como o `04_dados/index/
meu_video/`, que é formato numpy legado, sem `faiss.index`).

---

## 11. "Backend" (estrutura e serviços)

Não há backend no sentido de serviço de rede. O mais próximo é o
`ModelManager` (`models/model_manager.py`), que atua como um container de
inicialização/injeção de dependência dos 3 modelos residentes, usado como
context manager (`with ModelManager(config) as mm:`) pelo `cli.py`. Fora
disso, a "camada de serviço" é a função `run_index`/`run_search` em
`indexing/pipeline.py`, chamada diretamente pela CLI — não há separação
entre camada de aplicação e camada de transporte porque não há transporte
de rede.

Concorrência: usa `threading` puro (não asyncio), com filas `queue.Queue`
bounded entre estágios (produtor-consumidor clássico) e um
`_gpu_lock` global (`threading.Lock()` em `pipeline.py:30`) que serializa
todo acesso à GPU entre SigLIP e Qwen — nenhuma chamada de GPU roda
concorrentemente dentro do mesmo processo.

---

## 12. Frontend

Inexistente (ver seção 8).

---

## 13. Processamento de vídeo (extração de frames, sampling)

- **Formatos suportados**: qualquer container/codec que o OpenCV/FFmpeg
  consiga abrir via `cv2.VideoCapture` (mp4 testado nos exemplos).
- **Sampling**: por tempo, não por número de frames — `interval_sec`
  (default 2.0s no `cli.py`, mas 2.0s também no README como exemplo de "1
  frame por segundo", **inconsistência de default vs. texto do README**,
  ver seção "Inconsistências").
- **Método de sampling**: decodificação sequencial completa
  (`cap.read()` em loop), sem seek/skip real; o frame é convertido para
  PIL Image apenas quando cai no passo calculado
  (`frame_no % frame_step == 0`). Ou seja, para vídeos longos (12h), o
  custo de decodificação de **todos** os frames intermediários ainda é
  pago, mesmo que só uma fração seja processada pelo SigLIP — isso é uma
  limitação de custo de I/O/CPU não discutida no README (o benchmark de
  performance foca no custo de GPU, não no custo de decodificação bruta).
- **Acesso aleatório** (para reextrair frames específicos do top-K na
  busca): `extract_frames_at()` abre o vídeo **uma única vez** e usa
  `cap.set(cv2.CAP_PROP_POS_FRAMES, ...)` + `cap.read()` para cada
  timestamp, processando em ordem crescente de timestamp para minimizar
  custo de seek, mas retornando na ordem de entrada original.
- **Motion detection**: resize para 320×180, grayscale, `cv2.absdiff`
  contra o último frame aceito, comparado a `pixel_threshold=25` por
  pixel e `motion_ratio=0.02` de fração de pixels alterados.
- **Redimensionamento para VLM**: frames maiores que 768px no lado maior
  são redimensionados antes do Qwen2.5-VL (`_resize_for_vlm`,
  `vlm_describer.py:38`) — motivado por um bug real documentado: vídeos 4K
  sem esse limite geram até 16384 tokens visuais, estourando VRAM da RTX
  4060 e travando o driver NVIDIA no Windows inteiro (não só o processo
  Python).

---

## 14. Processamento de imagens

Após extração, cada frame é um `PIL.Image.Image` em RGB. Dois caminhos de
pré-processamento distintos, cada um pelo seu próprio `AutoProcessor` do
HuggingFace:
- **SigLIP**: resize para 256×256 (implícito no processor do modelo,
  `google/siglip-base-patch16-256-multilingual`), normalização padrão do
  encoder.
- **Qwen2.5-VL**: resize condicional (`_resize_for_vlm`, máx. 768px),
  depois `qwen_vl_utils.process_vision_info()` monta os tensores de
  imagem no formato esperado pelo modelo (patches dinâmicos, específico
  da arquitetura Qwen2.5-VL).

Não há pipeline de pré-processamento clássico de visão computacional
(equalização, filtros, aumento de contraste) — o processamento é
inteiramente delegado aos processors dos modelos.

---

## 15. Geração de descrições (captioning)

Modelo: `Qwen/Qwen2.5-VL-3B-Instruct`, quantizado em NF4 4-bit via
`bitsandbytes` (`BitsAndBytesConfig(load_in_4bit=True,
bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)`).

**Dois modos de prompt existem no código**, mas só um é usado no
pipeline atual:

1. `describe()` — legenda genérica: *"Descreva esta imagem em uma frase,
   focando em pessoas, roupas e objetos visíveis."* (`vlm_describer.py:207`).
   Usada apenas no `warmup()` e em scripts avulsos.
2. `describe_for_query()` / `describe_for_query_batch()` — legenda
   **condicionada à query** (usada em produção pelo `_search_detailed`):
   ```
   Estou procurando por: '{query}'. Sem assumir que isso esta na imagem,
   observe com atencao e descreva em 1 frase curta o que realmente
   aparece, nomeando o tipo exato dos objetos (ex.: moto ou bicicleta,
   van ou caminhonete, escada ou rack de teto). Se o que aparece for
   apenas parecido mas diferente do que foi pedido, diga o que e de
   fato, em vez de confirmar.
   ```
   (`vlm_describer.py:220-227`). Esse prompt foi desenhado especificamente
   para **evitar viés de confirmação** — documentado como resposta a um
   bug real observado (Qwen2-VL-2B alucinando placas veiculares).

Há também um terceiro método, `match_confidence()` (linha 294), que
pergunta sim/não diretamente e extrai a probabilidade do primeiro token
via `log_softmax` sobre todo o vocabulário — mas este método **não é mais
usado no pipeline de busca atual** (foi a abordagem original, abandonada
por viés de confirmação; permanece no código como método morto/histórico,
usado apenas em `scripts/debug_match_confidence.py`).

Geração em lote: `describe_for_query_batch` monta todos os prompts,
usa `padding_side="left"` no tokenizer (necessário para batch de geração
causal) e faz **uma única chamada** `model.generate()` para todas as
imagens do lote — otimização central que cortou ~26s da latência de busca
segundo o README.

---

## 16. Busca semântica

Implementada via embeddings SigLIP compartilhados entre imagem e texto no
mesmo espaço vetorial (768-dim), com métrica de produto interno
(equivalente a cosseno, pois os vetores são L2-normalizados em
`encode_text`/`encode_images`). Dois back-ends de índice intercambiáveis:
`FaissStore` (`IndexFlatIP`, default) e `EmbeddingStore` (numpy brute-force,
legado, mantido só para compatibilidade com índices antigos).

Reranking por diversidade (MMR) acontece **antes** do VLM, para evitar
gastar geração de texto em frames redundantes dentre os top-100
candidatos do FAISS — puramente algébrico (produto interno entre
embeddings), sem custo de modelo adicional.

---

## 17. RAG (Retrieval-Augmented Generation)

Não há RAG no sentido clássico (não há geração de texto livre condicionada
a documentos recuperados para responder uma pergunta aberta). O que existe
é **retrieval multimodal + geração condicionada curta** (legenda de 1
frase por frame, não uma resposta sintetizada a partir de múltiplos
frames recuperados). Não há um LLM que sintetize/resuma os resultados —
a saída final para o usuário é uma lista estruturada de frames com
timestamp, score e legenda individual, não uma resposta em linguagem
natural agregada.

Vector store: FAISS (`IndexFlatIP`), local, sem persistência em nuvem,
sem particionamento, sem suporte a atualização/exclusão granular pós-save
(o índice é escrito de uma vez ao final da indexação; adição incremental é
suportada em memória via `.add()`, mas não há API pública de update de um
índice já salvo em disco).

---

## 18. Embeddings

| Aspecto | Detalhe |
|---|---|
| Modelo gerador | `google/siglip-base-patch16-256-multilingual` |
| Dimensão | 768 (vs. 512 do CLIP/M-CLIP anteriores) |
| Normalização | L2, tanto para imagem quanto para texto (`features / features.norm(dim=-1, keepdim=True)`) |
| Precisão de armazenamento | `float32` (convertido de bf16 antes de salvar: `.cpu().float().numpy()`) |
| Onde são gerados | `models/siglip_embedder.py::encode_images` / `encode_text` |
| Onde são armazenados | `<index_dir>/faiss.index` (FAISS) ou `<index_dir>/embeddings.npy` (legado); cache intermediário opcional em SQLite (`embedding_cache.py`) |
| Precisão do modelo em memória | `bf16` (default), configurável para `fp16`/`fp32` via `ModelConfig.siglip_precision` |

---

## 19. VLM (Vision-Language Model)

Dois VLMs desempenham papéis distintos e **não devem ser confundidos**:

- **SigLIP** — tecnicamente um modelo de embedding contrastivo
  imagem-texto (não gerativo), usado para retrieval. Chamado de "VLM" de
  forma um pouco imprecisa em alguns trechos da documentação, mas
  funcionalmente é um dual-encoder, não um modelo gerativo.
- **Qwen2.5-VL-3B-Instruct** — VLM gerativo real (decoder-only,
  instruction-following), usado exclusivamente para legendagem
  condicionada dos top-K resultados do SigLIP, nunca para retrieval em
  massa.

O README é preciso nessa distinção ao descrever o pipeline, mas o título
do projeto ("busca semântica... via VLM") mistura os dois papéis sob o
mesmo termo guarda-chuva.

---

## 20. LLM

Não há um LLM de propósito geral (tipo GPT/Llama para texto puro) no
pipeline atual. O componente textual mais próximo de um "LLM" é o próprio
Qwen2.5-VL-3B (que tem um backbone de linguagem Qwen2.5), usado
estritamente para gerar 1 frase de legenda — não para diálogo, não para
sumarização, não para raciocínio em texto livre. Não há execução via API
externa (OpenAI, Anthropic, etc.) — todos os modelos rodam localmente
(inferência on-premise, GPU do próprio autor), alinhado ao requisito de
licença permissiva/sem custo de royalties do anteprojeto.

O `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` também não
é um LLM gerativo — é um encoder de sentenças (BERT-like) usado só para
calcular similaridade textual (score de confiança).

---

## 21. Fluxo completo do ponto de vista do usuário final

1. Usuário (pesquisador/desenvolvedor, via terminal) roda:
   ```powershell
   python -m video_search.cli index-multi cam1.mp4 cam2.mp4 --output ..\04_dados\index --config configs\multi_index.yaml
   ```
   Aguarda a indexação terminar (segundos a minutos, dependendo do vídeo).
2. Roda uma busca:
   ```powershell
   python -m video_search.cli search "homem de camisa branca e bone vermelho" --index ..\04_dados\index\cam1
   ```
3. Aguarda ~37s (segundo o benchmark documentado) enquanto: 3 modelos são
   carregados do zero (SigLIP, Qwen2.5-VL 4-bit, sentence-transformer),
   FAISS busca top-100, MMR seleciona top-12, frames são extraídos do
   vídeo, Qwen gera 12 legendas em lote, confiança é calculada.
4. Recebe no terminal uma lista ordenada por confiança:
   ```
   t=00:31.00  frame=    31  retrieval_score=0.1373  confianca=76.3%  legenda='...' video=teste05.mp4
   ```
5. Não há interface gráfica, não há chat, não há visualização de imagem
   inline — o usuário precisa abrir o vídeo manualmente no timestamp
   indicado para conferir visualmente.

Este fluxo é adequado para validação técnica/experimental do autor, mas
**não corresponde ao caso de uso final declarado no anteprojeto**
("busca por linguagem natural via chat" integrada a um VSM Qt) — esse
salto (CLI local → chat integrado a um produto) ainda não foi dado.

---

## 22. INCONSISTÊNCIAS, BUGS, CÓDIGO INCOMPLETO E DÉBITOS TÉCNICOS

Esta seção é intencionalmente extensa e crítica, por pedido explícito.

### 22.1 CRÍTICO — Todos os 10 scripts em `scripts/` estão com imports quebrados

A refatoração que moveu `src/video_search/*.py` para subpacotes
(`indexing/`, `media/`, `models/`, `utils/`) — commit staged mas não
finalizado no momento desta análise — **não atualizou nenhum script em
`03_codigo/scripts/`**. Todos os 10 arquivos ainda importam dos caminhos
antigos (pré-refatoração):

```
scripts/validate_confidence_full.py:13   from video_search.siglip_embedder import SigLIPEmbedder
scripts/validate_confidence_full.py:14   from video_search.embedding_store import EmbeddingStore
scripts/validate_confidence_full.py:15   from video_search.frame_extractor import extract_frame_at
scripts/validate_confidence_full.py:16   from video_search.vlm_describer import Qwen2VLDescriber
scripts/check_query_bias.py:19,20        from video_search.frame_extractor / vlm_describer
scripts/smoke_test_qwen2vl.py:11         from video_search.vlm_describer import Qwen2VLDescriber
scripts/smoke_test_pipeline_v2.py:15,16  from video_search.siglip_embedder / vlm_describer
scripts/benchmark_optimizations.py:25-29 from video_search.config / frame_extractor / multi_index / pipeline / vlm_describer
scripts/debug_match_confidence.py:11     from video_search.vlm_describer import Qwen2VLDescriber
scripts/timing_caption.py:10,11          from video_search.frame_extractor / vlm_describer
scripts/calibrate_confidence.py:13       from video_search.siglip_embedder import SigLIPEmbedder
scripts/benchmark_vlm_florence2.py:29-31 from video_search.frame_extractor / hf_utils / vlm_describer
```

Todos falhariam com `ModuleNotFoundError` ao executar, pois os módulos
corretos hoje são `video_search.models.siglip_embedder`,
`video_search.media.frame_extractor`, `video_search.models.vlm_describer`,
`video_search.indexing.embedding_store`, `video_search.utils.config`,
`video_search.indexing.multi_index`, `video_search.indexing.pipeline`,
`video_search.utils.hf_utils`. Isso inclui exatamente os scripts que
produziram os números reportados no README (`benchmark_optimizations.py`,
`calibrate_confidence.py`, `check_query_bias.py`) — ou seja, **os
resultados documentados no README não são mais reproduzíveis com um
`git pull` + execução direta do script**, até que os imports sejam
corrigidos. É um risco real para a banca pedir para reproduzir um número
do README.

### 22.2 CRÍTICO — Índice de dados residual aponta para caminho fora do repositório

`04_dados/index/meu_video/records.json` referencia
`"video_path": "C:\\Users\\Lucas\\Downloads\\Vídeos_teste\\teste02.mp4"`
— um caminho absoluto na máquina do autor, fora do repositório e fora de
`04_dados/raw/`. Combinado com o fato de `04_dados/raw/` só conter 9
imagens `.jpg` de preview/debug (nenhum `.mp4`), **nenhum vídeo de teste
está de fato versionado ou reproduzível a partir do repositório
sozinho** — mesmo ignorando LGPD, isso quebra reprodutibilidade científica
básica (a banca não consegue rodar `search` sobre um índice existente sem
ter acesso aos vídeos originais do autor).

### 22.3 ALTO — `run_multi_index` gera legendas genéricas nunca usadas (M4 do relatório anterior, ainda não resolvido)

Verificado no código atual: `run_multi_index` (`multi_index.py`) monta o
`AsyncIndexPipeline` sem nenhuma etapa de legendagem — `FrameRecord.caption`
fica como `""` (default) em todo índice gerado por `index-multi` (visto
diretamente no `records.json` real de `04_dados/index/meu_video/`, todos
os campos `"caption": ""`). O apontamento do relatório anterior (M4) sobre
"legenda genérica gerada na indexação e nunca usada na busca" **parece ter
sido resolvido pela refatoração** — hoje `index`/`index-multi` não geram
legenda alguma, e a legenda é sempre gerada sob demanda em
`_search_detailed`. Isso é consistente entre os dois caminhos agora,
mas **não está documentado explicitamente** que essa era a resolução
escolhida (Opção A do relatório anterior) — vale registrar a decisão na
dissertação.

### 22.4 ALTO — Score de "confiança" não é uma probabilidade calibrada

Confirmado no código atual: `results[i].confidence = float(np.dot(query_emb,
caption_emb))` (`pipeline.py:301`) é diretamente a similaridade de
cosseno entre dois embeddings de sentence-transformer, multiplicada por
100 e exibida como "%" na CLI (`cli.py:119`,
`f"confianca={r.confidence * 100:.1f}%"`). Não há sigmoid, não há
normalização floor/ceil, não há calibração isotônica. O script
`calibrate_confidence.py` calcula floor (max mismatch) e ceil (média de
paráfrases), mas essa calibração **não é lida nem aplicada em nenhum
lugar de `indexing/pipeline.py`** — é um artefato de análise isolado, não
integrado. Esse ponto já estava listado como M5 no relatório crítico
anterior e **continua sem solução no código atual**.

### 22.5 ALTO — Objetivos 4 e 5 da proposta continuam não cumpridos

Nenhuma mudança desde o relatório anterior: a API HTTP foi removida (não
recriada) e não há nenhum código de integração com Qt/VSM. Ver seção 6 e
9 acima. A decisão de descontinuar precisa constar explicitamente na
dissertação como mudança de escopo justificada, não como pendência
silenciosa.

### 22.6 MÉDIO — `embedding_cache` (SQLite) implementado mas não exposto pela CLI

`run_index()` aceita um parâmetro `embedding_cache=None`
(`indexing/pipeline.py:80`) e o `_siglip_worker` do `async_pipeline.py`
sabe usá-lo (linhas 473-509 de `async_pipeline.py`), mas **nenhum comando
do `cli.py` cria ou passa um `EmbeddingCache`** — não há flag
`--cache-dir` ou equivalente em `p_index`/`p_multi`. Ou seja, existe uma
feature inteira (cache SQLite de embeddings) implementada e testada
implicitamente pelo design, mas **inacessível ao usuário final da CLI**.
Parece um recurso "quase pronto", esquecido antes da última etapa de
integração.

### 22.7 MÉDIO — `batch_dispatcher.py` é código morto mantido só para os testes

`indexing/batch_dispatcher.py` (`BatchDispatcher`, 110 linhas) não é
importado por nenhum módulo de produção (`pipeline.py`, `multi_index.py`,
`async_pipeline.py`) — o batching real hoje é feito pelo
`_batch_collector_worker` interno ao `async_pipeline.py`, que reimplementa
uma lógica equivalente (drena fila, batch_size ou timeout) de forma
específica ao pipeline assíncrono. `BatchDispatcher` só é usado por
`tests/test_batch_dispatcher.py` (98 linhas) — isto é, há uma classe
inteira coberta por testes que não faz mais parte do sistema em produção.
Não é um bug funcional, mas é dívida técnica de limpeza (a classe deveria
ser removida, ou os testes deveriam ser removidos, ou a documentação
deveria deixar claro que é uma implementação alternativa mantida por outro
motivo — nada disso está explicado no código).

### 22.8 MÉDIO — Inconsistência de default de intervalo de amostragem

O `README.md` (raiz) descreve, na seção "Uso rápido": *"Indexar um vídeo
(SigLIP)"* com `--interval 1` no exemplo de comando, e depois, na tabela de
otimização, refere-se a "6 horas de vídeo (intervalo de 1s)". Já
`03_codigo/README.md` usa `--interval 2` nos exemplos e o **default real
do argparse é `2.0`** (`cli.py:150`, `p_index.add_argument("--interval",
type=float, default=2.0, ...)`). A documentação não é internamente
consistente sobre qual é o intervalo "padrão" do sistema — pequeno, mas
gera confusão sobre qual configuração foi de fato usada nos benchmarks
reportados.

### 22.9 MÉDIO — `match_confidence()` é código morto de produção (mas não de scripts)

O método `Qwen2VLDescriber.match_confidence()` (`vlm_describer.py:294-323`,
30 linhas, incluindo lógica delicada de log-softmax sobre o vocabulário
inteiro para extrair P(Sim)/P(Não)) não é chamado por nenhum caminho de
produção (`pipeline.py`, `multi_index.py`) — foi a abordagem original,
abandonada por viés de confirmação e substituída por
`describe_for_query`. Permanece no arquivo principal do describer (não
isolado em scripts), o que mistura código de produção com código
histórico/experimental na mesma classe. Só é referenciado por
`scripts/debug_match_confidence.py` (que, como todo o resto de `scripts/`,
tem import quebrado — seção 22.1).

### 22.10 MÉDIO — `requirements.txt` sem versões pinadas (M9, ainda não resolvido)

Confirmado: `numpy`, `opencv-python`, `Pillow`, `transformers`,
`sentence-transformers`, `accelerate`, `qwen-vl-utils`, `bitsandbytes`,
etc. — todos sem `==versão`. Apenas torch/torchvision são fixados,
via `constraints-cuda.txt` (`torch==2.5.1+cu121`). Dado que o próprio
código depende de comportamento específico de `transformers==5.12.0`
(citado explicitamente na docstring de `siglip_embedder.py` como razão
para uma escolha de implementação, e nas tabelas de rejeição de
Florence-2/Moondream2), a ausência de pin é uma armadilha real de
reprodutibilidade — um `pip install -r requirements.txt` daqui a alguns
meses pode trazer uma versão de `transformers` que quebra o SigLIP da
mesma forma que quebrou Florence-2 e Moondream2.

### 22.11 BAIXO — Ausência de dataset de avaliação formal (M2, ainda não resolvido)

Nenhuma pasta `04_dados/anotacoes/` ou equivalente; nenhum arquivo CSV/
JSON de ground truth encontrado; nenhum script que calcule Precision@K,
Recall@K ou F1. Todas as "validações" no README são inspeção manual de 3
a 20 queries por experimento, sem repetição estatística nem
protocolo formal de anotação. Esse é provavelmente o maior risco
metodológico para a defesa (conforme já sinalizado com prioridade CRÍTICA
no relatório anterior).

### 22.12 BAIXO — Estado do git com refatoração não commitada

No momento da análise, `git status` mostra 14 arquivos renomeados
("staged" mas não commitados) mais ~14 arquivos "modified" (não staged)
mais ~7 arquivos novos completamente não rastreados (`async_pipeline.py`,
`embedding_cache.py`, `faiss_store.py`, `reranker.py`, todo o pacote
`profiling/`, `estrategia_otimizacao.md`, `04_dados/index/`). Ou seja, boa
parte da arquitetura mais sofisticada do sistema (FAISS, MMR, pipeline
assíncrono, profiling) **ainda não tem um commit correspondente** — um
`git log` até este ponto não reflete o estado real mais avançado do
código. Risco prático: perda de trabalho se o diretório de trabalho for
corrompido/perdido antes do commit.

### 22.13 BAIXO — Resíduo de dependência não declarada no ambiente

`open_clip_torch==3.3.0` está instalado no `.venv` mas não consta em
`requirements.txt` nem é importado por nada em `src/` — resíduo órfão de
quando o pipeline usava CLIP. Não é um bug funcional (um ambiente limpo
via `pip install -r requirements.txt` não o instalaria), mas pode confundir
quem inspeciona o ambiente ativo do autor tentando entender quais
dependências são realmente necessárias.

### 22.14 BAIXO — `03_codigo/README.md` desatualizado em relação ao FAISS

A seção "Ideias de stack (a confirmar)" do `03_codigo/README.md` ainda
lista "Indexação vetorial: FAISS / Qdrant (avaliar para escala)" como algo
futuro/a decidir — mas o `PipelineConfig.use_faiss: bool = True`
(`async_pipeline.py:60`) já usa FAISS como default há pelo menos um commit
anterior à data desta análise. A seção "Limitações conhecidas" do mesmo
arquivo também ainda diz "Busca por força bruta em numpy não escala
indefinidamente... avaliar FAISS/Qdrant", quando isso já foi endereçado.

### 22.15 BAIXO — Bloco de código malformado no README raiz (M12, ainda não corrigido)

Confirmado na leitura atual: no bloco de "Uso rápido", há um code fence
que fecha (```) após o exemplo de `index-multi` (linha ~174) e depois uma
linha solta de exemplo de busca (`.venv\Scripts\python.exe -m
video_search.cli search "procure por uma van escolar" --index
..\04_dados\index\teste9_siglip`) seguida de um ``` de fechamento órfão,
sem abertura correspondente — o markdown renderiza essa linha fora do
bloco de código, quebrando a formatação visual.

---

## 23. Síntese avaliativa

| Dimensão | Estado em 04/07/2026 | Mudança desde o relatório de 27/06 |
|---|---|---|
| Engenharia de software / arquitetura | Excelente e mais madura (FAISS, MMR, pipeline assíncrono, profiling, cache SQLite) | Evoluiu significativamente |
| Consistência entre módulos e scripts auxiliares | Quebrada — todos os scripts de validação têm imports obsoletos | Regressão introduzida pela refatoração |
| Rigor metodológico / avaliação quantitativa | Ainda ausente (sem dataset anotado, sem Precision@K/Recall@K) | Sem mudança |
| Cobertura de literatura (`01_revisao_bibliografica/`) | Ainda inexistente | Sem mudança |
| Documento formal (`02_dissertacao/`) | Ainda inexistente | Sem mudança |
| Completude dos objetivos da proposta | 4/6 cumpridos (API e integração Qt permanecem pendentes) | Sem mudança |
| Reprodutibilidade | Piorou pontualmente (scripts quebrados, requirements sem pin, vídeos de teste fora do repo, refatoração não commitada) | Regressão |
| Score de confiança calibrado | Ainda não integrado ao pipeline | Sem mudança |

O núcleo técnico do pipeline é robusto e vem sendo aprimorado com boas
práticas de engenharia (pipeline assíncrono, FAISS, MMR, profiling
detalhado). Entretanto, a velocidade da refatoração recente introduziu uma
dívida técnica imediata e concreta — os scripts que geram os números
citados no README estão quebrados — que precisa ser corrigida antes de
qualquer nova rodada de experimentos, sob risco de os resultados relatados
deixarem de ser reproduzíveis a partir do próprio repositório. As lacunas
estruturais maiores para a defesa do TCC (revisão bibliográfica, dataset
de avaliação com métricas formais, e o documento da dissertação em si)
continuam integralmente por fazer.
