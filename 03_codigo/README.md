# Código

Implementação do pipeline de busca semântica em vídeo.

```
.venv/      Ambiente virtual Python (não versionar)
configs/    Config YAML do pipeline multi-câmera (motion/batch/similaridade)
src/        Pacote video_search: extração de frames, embeddings, índice, CLI
scripts/    Scripts utilitários (smoke tests, benchmark, etc.)
tests/      Testes automatizados (pytest)
```

## Setup

Ambiente: Python 3.12, GPU NVIDIA RTX 4060 (8GB) com CUDA 12.1.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip

# 1) Instalar PyTorch com CUDA primeiro
.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 2) Demais dependências, travadas no torch CUDA do passo anterior via
#    constraints-cuda.txt (ver nota abaixo):
.venv\Scripts\python.exe -m pip install -r requirements.txt -c constraints-cuda.txt
```

> ⚠️ **Pegadinha**: instalar dependências sem o `-c constraints-cuda.txt` pode
> fazer o pip resolver uma versão mais nova do `torch`/`torchvision` **sem
> CUDA** (build `+cpu`) ao satisfazer outro pacote (ex.: `transformers`).
> Sempre confirme depois com:
> ```powershell
> .venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
> ```
> Se voltar `False`, reinstale `torch`/`torchvision` com `--index-url
> https://download.pytorch.org/whl/cu121`.

## Pipeline atual

Pipeline usando exclusivamente **SigLIP multilingual**
(`google/siglip-base-patch16-256-multilingual`, licença Apache 2.0) para
embeddings de imagem e texto no mesmo espaço vetorial, com suporte nativo a
português (sem necessidade de tradução de query). Busca por similaridade de
cosseno (força bruta em numpy — adequado até centenas de milhares de
frames). Ver achados de validação e a evolução CLIP → M-CLIP → SigLIP no
[README do projeto](../README.md#achados-de-validação).

```powershell
# Indexar um vídeo (extrai 1 frame por segundo por padrão)
.venv\Scripts\python.exe -m video_search.cli index ..\04_dados\raw\camera1.mp4 --output index\camera1 --interval 2

# Buscar por descrição em linguagem natural
.venv\Scripts\python.exe -m video_search.cli search "homem de camisa branca e bone vermelho" --index index\camera1
```

### Indexação multi-câmera (motion detection + batch + filtro de similaridade)

Para indexar várias câmeras/vídeos ao mesmo tempo com menos chamadas de GPU
(ver detalhes e benchmark no [README do projeto](../README.md#estágio-1b--indexação-otimizada-para-múltiplas-câmeras-index-multi)):

```powershell
.venv\Scripts\python.exe -m video_search.cli index-multi cam1.mp4 cam2.mp4 cam3.mp4 --output ..\04_dados\index --config configs\multi_index.yaml
```

Benchmark antes/depois das otimizações, num único vídeo:

```powershell
.venv\Scripts\python.exe scripts\benchmark_optimizations.py caminho\do\video.mp4
```

Rodar os testes:

```powershell
.venv\Scripts\python.exe -m pytest -v
```

Smoke test do Qwen2.5-VL (baixa os pesos do modelo na primeira execução):

```powershell
.venv\Scripts\python.exe scripts\smoke_test_qwen2vl.py
```

## Limitações conhecidas / próximos passos

- SigLIP dá um score de similaridade, não uma "confiança" calibrada — por
  isso o reranking via Qwen2.5-VL (`describe_for_query` + sentence-transformer)
  é a fonte de confiança exibida na busca (ver `00_proposta/projeto_pesquisa.md`
  e achados no [README do projeto](../README.md#achados-de-validação)).
- Busca por força bruta em numpy não escala indefinidamente; para vídeos de
  12h com amostragem densa, avaliar FAISS/Qdrant.
- Limitação residual: confusão entre objetos visualmente parecidos (ex.:
  escada vs. rack de teto) — tanto no retrieval do SigLIP quanto no
  reranking do VLM.
- Microsserviço HTTP (`api.py`) removido por ora — uso via CLI apenas;
  retomar a API como evolução futura.

## Ideias de stack (a confirmar)

- Indexação vetorial: FAISS / Qdrant (avaliar para escala)
- Processamento de vídeo: OpenCV / ffmpeg
