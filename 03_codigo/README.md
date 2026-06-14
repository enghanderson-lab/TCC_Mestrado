# Código

Implementação do pipeline de busca semântica em vídeo e do microsserviço.

```
.venv/      Ambiente virtual Python (não versionar)
src/        Pacote video_search: extração de frames, embeddings, índice, CLI
scripts/    Scripts utilitários (smoke tests, etc.)
notebooks/  Experimentos exploratórios (seleção de VLM, protótipos)
tests/      Testes automatizados (pytest)
```

## Setup

Ambiente: Python 3.12, GPU NVIDIA RTX 4060 (8GB) com CUDA 12.1.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip

# 1) Instalar PyTorch com CUDA primeiro
.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 2) Demais dependências (open_clip_torch SEM --no-deps reinstala o torch em
#    versão CPU mais nova — ver nota abaixo). Use --no-deps e instale o
#    restante separadamente:
.venv\Scripts\python.exe -m pip install numpy opencv-python Pillow pytest
.venv\Scripts\python.exe -m pip install open_clip_torch --no-deps
```

> ⚠️ **Pegadinha**: instalar `open_clip_torch` normalmente (sem `--no-deps`)
> faz o pip resolver uma versão mais nova do `torch`/`torchvision` **sem
> CUDA** (build `+cpu`), porque o índice padrão do PyPI não tem a variante
> `cu121` dessas versões. Sempre confirme depois com:
> ```powershell
> .venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
> ```
> Se voltar `False`, reinstale `torch`/`torchvision` com `--index-url
> https://download.pytorch.org/whl/cu121` e reinstale `open_clip_torch`
> com `--no-deps`.

## Pipeline atual (protótipo v1)

Baseline usando **OpenCLIP** (`laion/CLIP-ViT-B-32-laion2B-s34B-b79K`,
licença MIT) para embeddings de imagem e texto no mesmo espaço vetorial.
Busca por similaridade de cosseno (força bruta em numpy — adequado até
centenas de milhares de frames).

```powershell
# Indexar um vídeo (extrai 1 frame por segundo por padrão)
.venv\Scripts\python.exe -m video_search.cli index ..\04_dados\raw\camera1.mp4 --output index\camera1 --interval 2

# Buscar por descrição em linguagem natural
.venv\Scripts\python.exe -m video_search.cli search "homem de camisa branca e bone vermelho" --index index\camera1
```

Rodar os testes:

```powershell
.venv\Scripts\python.exe -m pytest -v
```

Smoke test do CLIP (baixa os pesos do modelo na primeira execução, ~600MB):

```powershell
.venv\Scripts\python.exe scripts\smoke_test_clip.py
```

## Limitações conhecidas / próximos passos

- CLIP dá um score de similaridade, não uma "confiança" calibrada — discutir
  calibração na metodologia (ver `00_proposta/projeto_pesquisa.md`).
- Busca por força bruta em numpy não escala indefinidamente; para vídeos de
  12h com amostragem densa, avaliar FAISS/Qdrant.
- Próximo passo de modelo: avaliar um VLM (Qwen2-VL, MiniCPM-V, etc.) para
  descrição/verificação de atributos compostos (ex.: "camisa branca" + "boné
  vermelho"), possivelmente como segunda etapa (re-ranking) sobre os
  candidatos retornados pelo CLIP.
- Empacotar como microsserviço (FastAPI) para integração com o VSM (Qt).

## Ideias de stack (a confirmar)

- API do microsserviço: FastAPI
- Indexação vetorial: FAISS / Qdrant (avaliar para escala)
- Processamento de vídeo: OpenCV / ffmpeg
