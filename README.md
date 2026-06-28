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
  configs/
    multi_index.yaml      Config do pipeline multi-câmera (motion/batch/similaridade)
  src/video_search/
    cli.py                Ponto de entrada CLI: comandos index, index-multi e search
    pipeline.py           run_index/run_search — lógica usada por cli.py
    siglip_embedder.py    SigLIPEmbedder — encoder multilingue (Apache 2.0)
    vlm_describer.py      Qwen2VLDescriber — legenda condicionada a query
    embedding_store.py    Índice em memória (numpy), com metadata.json
    frame_extractor.py    Extração de frames por timestamp (OpenCV)
    config.py             Schema (pydantic) do YAML de configuração multi-câmera
    motion_filter.py      MotionFilter — descarta frames sem mudança significativa
    embedding_filter.py   EmbeddingSimilarityFilter — descarta embeddings redundantes
    batch_dispatcher.py   BatchDispatcher — agrupa frames de várias câmeras num lote de GPU
    multi_index.py        run_multi_index — orquestra N câmeras com as 3 otimizações
    hf_utils.py           load_offline_first — pula round-trip ao HF Hub p/ modelo já em cache
04_dados/
  raw/                    Vídeos de teste e frames de validação
  index/                  Índices SigLIP gerados (embeddings.npy + records.json)
  models/                 Cache local de pesos já quantizados (Qwen2.5-VL 4-bit), gerado na 1ª busca
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

### Otimização da latência de busca

Uma busca com legenda (top_k=12) chegou a levar **~73s**, e quase tudo era
*carregamento de modelo*, não geração de texto:

| Etapa | Antes | Depois | O que mudou |
|---|---|---|---|
| Carregar SigLIP | ~9s | ~1,4s | `local_files_only=True` evita round-trip ao Hugging Face Hub quando o modelo já está em cache (ver `hf_utils.load_offline_first`) |
| Extrair os 12 frames | ~5,4s | ~3,7s | `extract_frames_at` abre o vídeo uma única vez e faz seek de todos os timestamps, em vez de reabrir o arquivo por resultado |
| **Carregar Qwen2.5-VL 4-bit** | **~33s** | **~3,8s** | A quantização NF4 (bitsandbytes) é recalculada na CPU a cada `from_pretrained` — agora o modelo já quantizado é cacheado em `04_dados/models/qwen25vl3b-nf4/` na 1ª execução e as próximas carregam os pesos prontos (mesmo checkpoint, mesma config de quantização — sem trocar de modelo) |
| Gerar as 12 legendas | ~11s/legenda (estimado, sequencial) | ~10s **para as 12 juntas** | `describe_for_query_batch`: uma única chamada a `generate()` para todo o lote, em vez de uma por imagem |
| Carregar sentence-transformer | ~6s | ~2-5s | mesmo truque de `local_files_only=True` |
| **Total medido (CLI, ponta a ponta)** | **~73s** | **~37s** | |

Não chegou aos <15s pedidos: o piso restante é, na prática, carregar 3
modelos + correr generate() em GPU dentro de um processo que nasce e morre a
cada busca. Testado e descartado nesse caminho:

- **Checkpoint 4-bit pré-quantizado de terceiros** (`unsloth/...`): carregaria
  em ~4s, mas quebra durante a geração (erro interno do bitsandbytes,
  incompatível com as versões instaladas) — e trocaria o checkpoint usado
  nos experimentos de validação já documentados nesta seção, então foi
  descartado.
- **Reduzir `top_k` legendado**: testado com 5/8/12 imagens no mesmo lote —
  efeito quase nulo (a GPU absorve o lote sem saturar), não é uma alavanca
  útil aqui.
- **Reduzir `max_new_tokens`** (40→16): corta ~35% do tempo de geração
  (9,8s→6,4s), mas encurta a legenda — troca qualidade por velocidade, não
  aplicado por padrão.

Para cravar <15s de forma confiável restaria manter os modelos carregados
entre buscas (processo residente) — exatamente o papel que o microsserviço
HTTP cumpriria; ver "Microsserviço HTTP (deferido)" abaixo.

### Estágio 1b — Indexação otimizada para múltiplas câmeras (`index-multi`)

Para viabilizar ~8 câmeras simultâneas sem explodir o custo de GPU, a
indexação ganhou um caminho alternativo (`run_multi_index`, comando
`index-multi`) com três otimizações em cima do Estágio 1, todas configuráveis
num único YAML (`configs/multi_index.yaml`):

```
Vídeo (uma thread leitora por câmera)
  └─ Detecção de movimento (OpenCV, resize+grayscale+absdiff vs. último frame
     ACEITO daquela câmera) → descarta frames quase idênticos
       └─ Fila compartilhada entre câmeras
            └─ Batch Inference (SigLIP) — um único encode_images() por lote,
               podendo misturar frames de câmeras diferentes
                 └─ Filtro de similaridade de embeddings (cosseno vs. último
                    embedding ACEITO daquela câmera) → descarta frames
                    redundantes, sem legendar
                      └─ Legenda (Qwen2.5-VL) só para os frames que sobraram
                           └─ embeddings.npy + records.json (campo `caption`
                              populado) — mesmo formato do Estágio 1
```

Cada câmera mantém estado independente (último frame aceito, último
embedding aceito) — nunca comparado entre câmeras diferentes. `run_index`/
`run_search`/CLI/API existentes não foram alterados; é um ponto de entrada
aditivo que reaproveita `SigLIPEmbedder`/`Qwen2VLDescriber`.

**Benchmark (1 vídeo, 15 frames, RTX 4060):**

| Etapa | Sem otimizações | Com otimizações |
|---|---|---|
| Indexação (SigLIP) | 16,12s (15 frames) | ~4,2s (motion 0,26s + espera de lote 3,94s) |
| VLM (legenda) | 114,99s — todos os 15 frames (7,67s/frame) | 41,28s — só 6/15 frames (9 redundantes descartados) |
| **Total** | **131,11s** | **87,66s** (-33%) |

Reprodutível com `03_codigo/scripts/benchmark_optimizations.py video.mp4`.

Em validação com 3 câmeras simultâneas (183 frames lidos no total), o filtro
de similaridade reduziu as chamadas ao VLM em 76% (183 → 44 frames
legendados) e os logs confirmaram lotes de SigLIP misturando frames de
câmeras diferentes (ex.: `lote executado tamanho=3` no primeiro flush, com
uma câmera de cada fonte) — evidência direta do batching cross-câmera.

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
python -m video_search.cli index "video.mp4" --output "..\04_dados\index\teste9" --interval 1

# Buscar

.venv\Scripts\python.exe -m video_search.cli search "procure por um carro branco com uma escada sobre o teto" --index ..\04_dados\index\teste7_siglip
.venv\Scripts\python.exe -m video_search.cli search "procure por uma van" --index ..\04_dados\index\teste6_siglip


# Buscar sem VLM (apenas SigLIP, mais rápido)
python -m video_search.cli search "homem com caderno de anotações" --index "..\04_dados\index\meu_video" --no-caption

# Indexar várias câmeras simultaneamente (motion detection + batch cross-câmera + filtro de similaridade)
python -m video_search.cli index-multi cam1.mp4 cam2.mp4 cam3.mp4 --output ..\04_dados\index --config configs\multi_index.yaml
```

.venv\Scripts\python.exe -m video_search.cli search "procure por uma van escolar" --index ..\04_dados\index\teste9_siglip
```

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

### Limitação de retrieval: SigLIP perde recall em sujeito secundário numa cena com "hub"

Teste com `teste02.mp4`, query "uma mulher com cabelos escuros, tênis brancos
e usando uma mochila nas costas" (índice com 56 frames, intervalo ~2s,
`top_k=12`). Inspeção manual do vídeo confirmou que a mulher descrita
aparece com clareza em **t=95,32s** (frame 48), caminhando em direção à
porta da loja — visível, não oclusa, em escala comparável à de outras
pessoas no quadro.

**O SigLIP rankeou esse frame em 40º lugar de 56** (score -0,1029) — bem
fora do `top_k=12`, então o reranking por VLM nunca chegou a avaliá-lo.
Confirmado manualmente: ao rodar o Qwen2.5-VL **diretamente** sobre esse
frame (fora do pipeline normal), ele identifica a mulher corretamente, com
67,1% de confiança — ou seja, **o reranking funcionaria se tivesse recebido
o frame**; o gargalo é só o retrieval.

**Hipótese investigada 1 (descartada por evidência visual)**: o sujeito
estaria pequeno/distante demais para o SigLIP discriminar. Refutada por
inspeção visual direta do frame — a mulher está visível e em escala normal,
comparável à de outras pessoas na cena.

**Hipótese investigada 2 (confirmada parcialmente, mas mitigação testada
falhou)**: o frame que ficou em 1º lugar no retrieval original (t=89,36s)
é um **"hub"** — um embedding que pontua anomalamente alto contra **qualquer**
texto, fenômeno documentado em espaços de embedding de alta dimensão
(hubness, Radovanović et al.). Evidência: rankeando esse mesmo frame contra
20 queries genéricas e **não relacionadas** à cena (ex.: "uma montanha
nevada", "um carro azul estacionado"), ele ficou entre os 10 primeiros (de
56) em **10 das 20** queries:

| Query (não relacionada à cena) | Rank do frame t=89,36s |
|---|---|
| "uma montanha nevada" | 2º/56 |
| "um carro azul estacionado" | 2º/56 |
| "um gato preto em cima do sofá" | 2º/56 |
| "uma xícara de café fumegante" | 2º/56 |
| "uma ponte sobre um rio" | 3º/56 |

Isso é real, mas **a tentativa de mitigação (subtrair de cada frame sua
similaridade média contra um conjunto de referência de 20 queries
genéricas, técnica relacionada a CSLS) não corrigiu o ranking** — testado
com a média sobre as top-3, top-5, top-8 e todas as 20 referências; em
nenhum dos quatro casos o frame correto (t=95,32s) subiu para perto do
top-12 (ficou entre 38º e 43º em todos). O motivo: o "bias" médio do frame
correto contra o conjunto de referência é **parecido ou até maior** (em
módulo) do que o do frame-hub, então subtrair o bias não fecha a diferença
no score da query real.

**Conclusão (em aberto)**: o efeito hub é real e mensurável, mas não explica
sozinho a perda de recall, e a correção simples testada não funciona. A
explicação mais provável, ainda não comprovada por mitigação eficaz, é a
limitação conhecida de embeddings globais (um vetor por imagem) tipo
SigLIP/CLIP: o frame que venceu no retrieval mostra a mulher **dominando o
quadro** (close-up, câmera próxima), enquanto no frame correto ela é só uma
entre várias pessoas num quadro amplo e visualmente cheio — o sinal
semântico dela se dilui no embedding global da cena. Mitigações reais
exigiriam uma mudança de arquitetura (ex.: detecção de pessoa + reembedding
por recorte, em vez de um único vetor para o frame inteiro) — fora do
escopo atual, registrado como trabalho futuro.

**Tentativa de mitigação prática (parcialmente descartada)**: a ideia óbvia
seria aumentar `--top-k` para o reranking por VLM alcançar o frame. Mas o
frame correto está em **40º lugar de 56** — aumentar `--top-k` para algo
como 20-25 não seria suficiente; seria preciso `top_k≈40`, ou seja, quase o
índice inteiro, o que deixa de ser uma filtragem útil e aproxima o custo de
`generate()` do "legendar tudo" que a otimização de latência evitou. Para
este caso específico, aumentar `top_k` não é uma mitigação prática — reforça
que a causa é estrutural (embedding global) e não um limiar mal calibrado.

### VLM de legendagem: Qwen2.5-VL-3B vs. Moondream2

Com a busca otimizada (ver "Otimização da latência de busca" em
[03_codigo/README.md](03_codigo/README.md#otimização-da-latência-de-busca)),
avaliou-se a troca do Qwen2.5-VL-3B pelo **Moondream2** (`vikhyatk/moondream2`,
Apache 2.0, ~2B parâmetros) como modelo de legendagem, por ser um VLM menor e
anunciado como mais rápido. A avaliação foi prática (tentativas reais de
carregar e rodar o modelo no ambiente do projeto), não apenas leitura de
documentação, e resultou na **rejeição da troca**:

| Critério | Qwen2.5-VL-3B (atual) | Moondream2 |
|---|---|---|
| Licença | Apache 2.0 | Apache 2.0 |
| Parâmetros | 3B (cache 4-bit, ~2,4GB VRAM) | ~2B (sem 4-bit simples via `transformers`) |
| Geração em lote (batch) | Sim — testado e em produção (`describe_for_query_batch`, ver seção de latência) | Não documentado; benchmarks externos confirmam que "Moondream doesn't handle batching" |
| Suporte a português | Nativo, validado nos experimentos acima | Sem documentação de multilíngue; tokenizer ("superword") focado em inglês |
| Instalação no Windows | Só `pip` (`transformers`, `bitsandbytes`) | Exige `trust_remote_code=True` + `einops` + `pyvips`, que por sua vez exige a DLL nativa `libvips-42.dll` — **falhou** no ambiente do projeto: `OSError: cannot load library 'libvips-42.dll'` |
| Compatibilidade de versão | Estável com `transformers==5.12.0` (instalado) | Revisões antigas (sem `pyvips`) quebram no `transformers` atual: `AttributeError: 'PhiConfig' object has no attribute 'pad_token_id'` |

**Decisão**: manter o Qwen2.5-VL-3B. Os dois pontos que mais importam para
este pipeline — geração em lote (que já cortou ~26s da busca) e suporte
nativo a português (requisito de design desde a escolha do SigLIP sobre
CLIP, ver acima) — não são garantidos no Moondream2, e a instalação no
Windows depende de uma biblioteca nativa frágil (`libvips`). O ganho de
tamanho de modelo (2B vs. 3B) também já havia sido absorvido pelo cache do
checkpoint 4-bit (ver seção de latência).

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

### 2. Microsserviço HTTP (deferido)

Um microsserviço FastAPI (`index`/`search` via HTTP) foi prototipado e
depois removido para focar a otimização de desempenho do pipeline via CLI
(ver seção "Otimização da latência de busca"). Retomar como evolução futura,
já considerando autenticação/autorização e persistência dos jobs de
indexação em disco/banco (em vez de em memória).



### 3. Implementado: motion detection, batch cross-câmera e filtro de similaridade

As três ideias acima (extrator de frame por mudança de pixel, evitar
reprocessar vetores redundantes, processar vários vídeos ao mesmo tempo)
foram implementadas no comando `index-multi` — ver seção "Estágio 1b" acima
para a arquitetura e o benchmark.

