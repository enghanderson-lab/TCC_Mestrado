# Estratégia de Otimização — TCC Busca Semântica em Vídeo
*Gerado em 28/06/2026 com base na análise do repositório e pesquisa bibliográfica*

---

## Diagnóstico: Os Dois Achados Críticos

A análise do README e do `relatorio_analise_critica.md` identificou dois problemas técnicos abertos que afetam diretamente os resultados da dissertação:

### Achado 1 — Latência de busca travada em ~37s (meta: <15s)

O pipeline atual nasce e morre a cada busca (processo CLI). Cada execução recarrega três modelos do disco:

| Componente | Custo de carga atual |
|---|---|
| SigLIP | ~1,4s |
| Qwen2.5-VL-3B (4-bit do cache) | ~3,8s |
| sentence-transformer | ~2–5s |
| Geração das 12 legendas (batch) | ~10s |
| Extração de 12 frames | ~3,7s |
| **Total** | **~37s** |

O piso não é a geração — é o custo de carregar modelos do disco a cada chamada. Nenhuma otimização dentro do processo CLI resolve isso de forma limpa.

### Achado 2 — SigLIP perde recall quando o sujeito é secundário na cena

Frame correto ficou em **40º lugar de 56** (score −0,1029) com a query "uma mulher com cabelos escuros, tênis brancos e usando uma mochila nas costas". O VLM reconheceria a mulher se recebesse o frame (67,1% de confiança confirmado em teste direto), mas o retrieval nunca entrega o frame.

**Causa raiz confirmada:** embedding global do SigLIP produz um único vetor por frame inteiro. Quando a mulher é um dos vários elementos visuais da cena (câmera aberta, várias pessoas no quadro), o sinal semântico dela se dilui no embedding.

**Tentativas que não funcionaram:**
- Correção de hubness via subtração de similaridade média (CSLS-like): não corrigiu o ranking. O bias do frame correto contra o conjunto de referência era parecido com o do frame-hub, então subtrair não fechou a diferença.
- Aumentar `top_k`: frame correto em 40º/56 — precisaria de `top_k≈40`, deixando de ser filtragem útil.

---

## Estratégia de Otimização

As três estratégias abaixo são ordenadas por impacto e viabilidade no contexto do hardware disponível (RTX 4060 8GB VRAM).

---

### Estratégia A — Microsserviço com modelos persistentes (resolve Achado 1)

**Princípio:** Em vez de um processo CLI que nasce e morre, manter os três modelos carregados em VRAM/RAM entre buscas. O overhead de carga (~8–10s) vira zero nas buscas seguintes.

**Implementação sugerida:**

```
03_codigo/src/video_search/
  api.py          ← FastAPI com lifespan (startup carrega os 3 modelos)
  api_runner.py   ← script de start: uvicorn api:app --host 0.0.0.0 --port 8000
```

**Estrutura do `api.py` com lifespan:**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carrega uma vez na inicialização
    from video_search.siglip_embedder import SigLIPEmbedder
    from video_search.vlm_describer import Qwen2VLDescriber
    from sentence_transformers import SentenceTransformer
    _models["siglip"]   = SigLIPEmbedder()
    _models["qwen"]     = Qwen2VLDescriber()
    _models["sentence"] = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    yield
    _models.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/search")
async def search(query: str, index_path: str, top_k: int = 12):
    from video_search.pipeline import run_search
    return run_search(query, index_path, top_k,
                      siglip=_models["siglip"],
                      describer=_models["qwen"],
                      sentence_model=_models["sentence"])
```

**Adaptação necessária em `pipeline.py`:**
- Adicionar parâmetros opcionais `siglip=None`, `describer=None`, `sentence_model=None` ao `run_search()`.
- Quando recebidos, reutilizar os objetos passados em vez de instanciar novos.
- A CLI atual (`cli.py`) continua funcionando: sem argumentos, instancia os modelos normalmente.

**Ganho esperado:**
- 1ª busca: ~37s (igual, carrega os modelos)
- 2ª busca em diante: ~10–15s (só geração + extração de frames, sem carga de modelo)
- Meta de <15s alcançável a partir da segunda busca

**Referência:** documentação oficial vLLM para Qwen2.5-VL confirma esse padrão de servidor persistente com FastAPI. Para o modelo de 3B com 4-bit, o servidor uvicorn direto é suficiente — vLLM seria necessário apenas para múltiplos usuários simultâneos.

---

### Estratégia B — Detecção de pessoas + re-embed por recorte (resolve Achado 2)

**Princípio:** Em vez de um embedding global para o frame inteiro, detectar objetos/pessoas no frame, recortar cada um individualmente e gerar um embedding SigLIP por recorte. A busca passa a comparar a query contra embeddings de regiões específicas, não do frame completo.

**Arquitetura proposta:**

```
Frame (960x540, câmera de vigilância)
  └─ YOLO-World v2 (open-vocabulary, detecção de "person")
       └─ Bounding boxes de cada pessoa detectada
            └─ Crop de cada região → SigLIP encode_image
                 └─ Embedding da região (dim=768), salvo com metadado:
                    {frame_ts, camera_id, bbox: [x1,y1,x2,y2], type: "crop"}
```

**Impacto no índice:**
- Um frame com 3 pessoas detectadas gera 4 embeddings (1 global + 3 crops)
- A busca por "mulher com mochila" compara contra os embeddings de crop, não o embedding do frame cheio
- O VLM de reranking continua recebendo o frame completo, só o retrieval muda

**Implementação sugerida:**

```python
# 03_codigo/src/video_search/region_embedder.py

from ultralytics import YOLO  # pip install ultralytics
import torch

class RegionEmbedder:
    """Detecção de regiões + embedding por crop via SigLIP."""

    def __init__(self, siglip_embedder, yolo_model="yolov8n.pt",
                 classes=("person",), min_area_ratio=0.02):
        self.detector = YOLO(yolo_model)
        self.embedder = siglip_embedder
        self.classes = classes
        self.min_area_ratio = min_area_ratio  # descarta boxes muito pequenos

    def embed_frame_with_regions(self, image_pil):
        """
        Retorna lista de (embedding, metadata).
        Sempre inclui o embedding global + embeddings por crop.
        """
        results = []
        h, w = image_pil.height, image_pil.width
        frame_area = h * w

        # Embedding global (compatível com índice existente)
        global_emb = self.embedder.encode_image(image_pil)
        results.append((global_emb, {"type": "global"}))

        # Detecção de regiões
        det = self.detector(image_pil, verbose=False)[0]
        for box in det.boxes:
            cls_name = self.detector.names[int(box.cls)]
            if cls_name not in self.classes:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop_area = (x2 - x1) * (y2 - y1)
            if crop_area / frame_area < self.min_area_ratio:
                continue  # descarta pessoas muito pequenas
            crop = image_pil.crop((x1, y1, x2, y2))
            crop_emb = self.embedder.encode_image(crop)
            results.append((crop_emb, {
                "type": "crop", "class": cls_name,
                "bbox": [x1, y1, x2, y2]
            }))

        return results
```

**Adaptação no `EmbeddingStore`:**
- Adicionar campo `region_type` e `bbox` no `FrameRecord`
- Durante a busca, retornar o `frame_ts` do crop mais similar (o frame completo é extraído para o VLM)
- Índice separado para crops: `embeddings_crops.npy` + `records_crops.json`

**Ganho esperado:**
- Resolve o caso em que o sujeito está presente mas é secundário na cena
- Custo extra na indexação: YOLO-World (~5ms/frame em GPU) + N crops adicionais pelo SigLIP
- Custo na busca: zero (mesma busca por similaridade, índice maior)

**Referência:** Patchify usando SigLIP atinge mAP 79.54% em benchmarks two-stage (2024). YOLO-World v2.1 (fev/2025) integrado com open-vocabulary permite detectar além de "person".

---

### Estratégia C — QB-Norm para mitigação de hubness (complementa Estratégia B)

**Princípio:** A tentativa de CSLS feita no projeto (subtrair similaridade média contra 20 queries de referência) foi aplicada corretamente na intuição, mas a técnica correta é **Querybank Normalisation (QB-Norm)**, apresentada no CVPR 2022 e validada em múltiplos trabalhos de 2024.

**Diferença entre CSLS e QB-Norm:**

| | CSLS (testado) | QB-Norm (a testar) |
|---|---|---|
| O que normaliza | Similaridade média do *candidato* contra K vizinhos de cada query | Similaridade máxima da *query* contra um banco de queries diversas |
| Opera sobre | Lado dos candidatos (frames) | Lado das queries |
| Problema | O bias do frame correto se cancela com o do hub ao subtrair | Re-escala cada score pela "facilidade" da query, sem tocar os candidatos |

**Fórmula QB-Norm:**

```
score_normalizado(q, c) = score_bruto(q, c) / max_{q' in Q_bank} score_bruto(q', c)
```

Onde `Q_bank` é um banco de queries diversas (as mesmas 20 usadas no teste de hubness já são suficientes).

**Implementação sugerida em `pipeline.py`:**

```python
import numpy as np

def qb_norm(query_emb, candidate_embs, query_bank_embs):
    """
    Query Bank Normalisation (Bogolin et al., CVPR 2022).
    
    Args:
        query_emb: (768,) — embedding da query atual
        candidate_embs: (N, 768) — todos os embeddings do índice
        query_bank_embs: (M, 768) — banco de queries de referência diversas
    
    Returns:
        scores normalizados: (N,)
    """
    raw_scores = candidate_embs @ query_emb          # (N,)
    bank_scores = candidate_embs @ query_bank_embs.T  # (N, M)
    normalization = bank_scores.max(axis=1)            # (N,) — hubness score por candidato
    normalization = np.maximum(normalization, 1e-8)    # evita divisão por zero
    return raw_scores / normalization

# Em run_search(), substituir:
#   scores = embeddings @ query_emb
# por:
#   scores = qb_norm(query_emb, embeddings, QUERY_BANK_EMBS)
```

**Geração do `query_bank_embs`:**
- Encodificar as 20 queries genéricas usadas no teste de hubness uma única vez durante o startup
- Salvar como `query_bank.npy` no diretório do índice (ou embutir no código como constante)

**Ganho esperado:**
- Sem re-treinamento, sem alteração no índice, sem GPU extra
- Custo adicional na busca: uma multiplicação de matriz (N, 768) × (M=20, 768)ᵀ — negligível
- Probabilidade de corrigir o ranking do frame em 40º lugar: moderada. QB-Norm é mais robusto que CSLS quando o bias do frame correto é comparável ao do hub (exatamente o caso documentado no README)

---

## Plano de Testes e Validação

### Fase 1 — Baseline (pré-mudança)

Antes de qualquer alteração, documentar o estado atual como linha de base:

```bash
# 1. Medir latência atual (5 execuções, média e desvio)
for i in {1..5}; do
  time python -m video_search.cli search "mulher com mochila nas costas" \
    --index ../04_dados/index/teste02_siglip
done

# 2. Registrar o ranking atual para a query problemática
python -m video_search.cli search "uma mulher com cabelos escuros..." \
  --index ../04_dados/index/teste02_siglip --top-k 56 --no-caption \
  > baseline_ranking_teste02.txt
```

Métricas a registrar:
- Latência total (s), por componente
- Rank do frame correto (t=95,32s) — esperado: 40º/56
- Score SigLIP do frame correto — esperado: −0,1029

---

### Fase 2 — Validar Estratégia A (microsserviço)

```bash
# Iniciar o servidor
uvicorn video_search.api:app --host 127.0.0.1 --port 8000

# Medir latência com servidor quente
for i in {1..5}; do
  time curl -X POST "http://localhost:8000/search" \
    -d '{"query":"mulher com mochila", "index_path":"../04_dados/index/teste02_siglip"}'
done
```

**Critério de sucesso:** média de latência < 15s nas buscas 2ª em diante (excluindo cold start).

**Critério de falha:** latência > 20s mesmo com modelos carregados (indicaria gargalo na geração, não no carregamento).

---

### Fase 3 — Validar Estratégia C (QB-Norm) — sem dependência de hardware extra

QB-Norm é a mais barata de testar (nenhuma mudança no índice). Testar primeiro:

```python
# 03_codigo/scripts/test_qbnorm.py

import numpy as np
from video_search.embedding_store import EmbeddingStore
from video_search.siglip_embedder import SigLIPEmbedder

QUERY_BANK = [
    "uma montanha nevada",
    "um carro azul estacionado",
    "um gato preto em cima do sofá",
    "uma xícara de café fumegante",
    "uma ponte sobre um rio",
    "uma criança brincando no parque",
    "um avião no céu",
    "uma praia com palmeiras",
    "um escritório com computadores",
    "uma sala de aula vazia",
    # ... completar até 20
]

store = EmbeddingStore.load("../04_dados/index/teste02_siglip")
embedder = SigLIPEmbedder()

# Encodificar banco de queries
bank_embs = np.array([embedder.encode_text(q) for q in QUERY_BANK])  # (20, 768)

# Query problemática
query_emb = embedder.encode_text("uma mulher com cabelos escuros, tênis brancos e usando uma mochila nas costas")

# Score bruto (atual)
raw_scores = store.embeddings @ query_emb
rank_bruto = np.argsort(-raw_scores)
frame_target_ts = 95.32
frame_target_idx = next(i for i, r in enumerate(store.records) if abs(r.timestamp - frame_target_ts) < 1)
rank_antes = np.where(rank_bruto == frame_target_idx)[0][0] + 1

# Score QB-Norm
bank_scores = store.embeddings @ bank_embs.T        # (N, 20)
normalization = bank_scores.max(axis=1)              # (N,)
qb_scores = raw_scores / np.maximum(normalization, 1e-8)
rank_qb = np.argsort(-qb_scores)
rank_depois = np.where(rank_qb == frame_target_idx)[0][0] + 1

print(f"Rank ANTES (SigLIP puro):  {rank_antes}/56  (score={raw_scores[frame_target_idx]:.4f})")
print(f"Rank DEPOIS (QB-Norm):     {rank_depois}/56  (score_norm={qb_scores[frame_target_idx]:.4f})")
print(f"Score frame-hub t=89.36s:  raw={raw_scores[hub_idx]:.4f}  qb={qb_scores[hub_idx]:.4f}")
```

**Critério de sucesso:** frame correto (t≈95,32s) entra no top-15 com QB-Norm.

**Critério de falha:** frame continua fora do top-20 → confirma que o problema é estrutural (embedding global) e migrar para Estratégia B é obrigatório.

---

### Fase 4 — Validar Estratégia B (detecção + crop) — se QB-Norm falhar

```bash
pip install ultralytics  # YOLOv8 incluso

# Indexar teste02.mp4 com modo de crops
python -m video_search.cli index-region "../../04_dados/raw/teste02.mp4" \
  --output "../../04_dados/index/teste02_region" \
  --interval 2

# Buscar com índice de regions
python -m video_search.cli search "mulher com cabelos escuros, tênis brancos e mochila" \
  --index "../../04_dados/index/teste02_region" --top-k 12
```

**Critério de sucesso:** frame t≈95,32s aparece no top-5 da busca por crops.

**Comparação a documentar na dissertação:**

| Modo | Rank do frame correto | Score | Latência de indexação |
|---|---|---|---|
| SigLIP global (atual) | 40º/56 | −0,1029 | baseline |
| SigLIP global + QB-Norm | ? | ? | +~0ms |
| SigLIP por crops (YOLO-World) | ? | ? | +~N×5ms/frame |

---

## Resumo Executivo

| Estratégia | Problema que resolve | Esforço | Impacto esperado |
|---|---|---|---|
| **A — FastAPI lifespan** | Latência ~37s → <15s na 2ª busca | Médio (3–4h) | Alto — resolve Achado 1 para o microsserviço |
| **B — YOLO + crop SigLIP** | Sujeito secundário perdido no retrieval | Alto (1–2 dias) | Alto — resolve Achado 2 estruturalmente |
| **C — QB-Norm** | Hubness + sujeito secundário | Baixo (2–3h) | Moderado — testa antes do B, sem mudança no índice |

**Ordem recomendada de execução:** C → A → B (menor esforço primeiro, validação rápida, evolução incremental).

---

*Referências utilizadas:*
- Bogolin et al. "Cross Modal Retrieval with Querybank Normalisation." CVPR 2022.
- Radovanović et al. Hubness problem in high-dimensional embeddings (referenciado no README do projeto).
- YOLO-World v2.1 (fev/2025): https://github.com/AILab-CVC/YOLO-World
- Documentação vLLM para Qwen2.5-VL: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen2.5-VL.html
- "Hubness Reduction with Dual Bank Sinkhorn Normalization for Cross-Modal Retrieval" (2025): https://arxiv.org/html/2508.02538
- "Patch-wise Retrieval: A Bag of Practical Techniques" (2024): https://arxiv.org/pdf/2512.12610
