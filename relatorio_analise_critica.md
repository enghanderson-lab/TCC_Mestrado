# Relatório de Análise Crítica — TCC de Mestrado
**Busca Semântica em Gravações de Vídeo de Longa Duração por Linguagem Natural Utilizando Modelos de Visão-Linguagem de Código Aberto**

*Análise gerada em 27 de junho de 2026*

---

## 1. Resumo do Trabalho

O projeto desenvolve um pipeline de busca semântica em vídeos longos (até 12h) via linguagem natural, usando exclusivamente modelos com licença permissiva (Apache 2.0). O usuário digita uma query como "homem de camisa branca e boné vermelho" e o sistema retorna os frames correspondentes com timestamps e um grau de confiança associado.

**Arquitetura do pipeline (dois estágios):**

- **Estágio 1 — Indexação (offline):** Extração de frames via OpenCV → SigLIP encode_image em batch (GPU) → salva `embeddings.npy + records.json`.
- **Estágio 2 — Busca (sob demanda):** Query em PT → SigLIP encode_text → busca por similaridade de cosseno (brute-force numpy) → top-K frames → Qwen2.5-VL-3B (4-bit) gera legenda condicionada à query → sentence-transformer calcula confiança (similaridade texto-texto) → resultados rerankeados.
- **Estágio 1b — Indexação multi-câmera (`index-multi`):** Três otimizações sobre o Estágio 1: (a) filtro de movimento por câmera (pixel diff), (b) batch cross-câmera do SigLIP (`BatchDispatcher`), e (c) filtro de similaridade de embeddings para não legendar frames redundantes.

**Modelos utilizados:** SigLIP multilingual base-patch16-256 (retrieval), Qwen2.5-VL-3B-Instruct 4-bit (legenda/reranking), paraphrase-multilingual-mpnet-base-v2 (score de confiança).

**Estado atual do repositório:**
- `00_proposta/projeto_pesquisa.md` — proposta (rascunho, não formatada em ABNT)
- `03_codigo/` — implementação Python completa com testes
- `04_dados/README.md` — estrutura definida, mas **sem dados**
- `05_experimentos/README.md` — estrutura definida, mas **sem experimentos**
- **`01_revisao_bibliografica/`** — **pasta não existe**
- **`02_dissertacao/`** — **pasta não existe**

---

## 2. Pontos Fortes

**Engenharia de software de alta qualidade.** O código é excepcionalmente bem documentado para um TCC. Cada módulo tem docstrings que explicam *por que* a decisão foi tomada (não apenas *o que* ela faz), como a escolha de `sigmoid loss` vs. `softmax` no SigLIP, o trade-off de quantização 4-bit vs. 7B em bfloat16, e a razão para `device_map` explícito no Qwen. Isso é raro e vai diretamente para a dissertação.

**Evolução de modelos documentada com dados.** A progressão CLIP → CLIP+tradução → M-CLIP → SigLIP é justificada com benchmarks reais (ex.: M-CLIP superou CLIP+tradução em +68% no retrieval_score para "homem com caderno de anotações", com o frame correto saltando do 3º para o 1º lugar). Essa comparação é um experimento de ablation válido para a dissertação.

**Identificação e mitigação do viés de confirmação no VLM.** A transição de `match_confidence()` (pergunta sim/não que induzia confirmação) para `describe_for_query()` (legenda condicionada, sem confirmar) foi motivada por evidência empírica documentada no README — o Qwen2-VL-2B alucinava a placa "IVS5B37" em vez de ler "IYS5B37". A solução (prompt de descrição + upgrade para 3B) foi adequada.

**Benchmark mensurável das otimizações.** O README apresenta uma tabela antes/após com redução de 33% no tempo total (131s → 87s) e 76% nas chamadas ao VLM (183 → 44 frames legendados em teste de 3 câmeras). O script `benchmark_optimizations.py` é reprodutível.

**Cobertura de testes adequada.** Há testes unitários para `BatchDispatcher`, `EmbeddingSimilarityFilter`, `MotionFilter`, `MultiIndexConfig` e `run_multi_index`. Os testes usam fakes determinísticos (sem GPU/rede), o que os torna rápidos e confiáveis.

**Atenção a restrições de hardware reais.** A escolha de 4-bit NF4, o `_gpu_lock` para serializar acesso à GPU, o `_resize_for_vlm()` que evita OOM com vídeos 4K, e a limpeza de VRAM com `torch.cuda.empty_cache()` demonstram que o código foi validado em hardware real (RTX 4060 8GB).

---

## 3. Melhorias Sugeridas (Ordem de Prioridade)

### 🔴 CRÍTICO — Sem isso, o TCC não pode ser defendido

---

#### M1 — Criar a dissertação (`02_dissertacao/`)

**Problema:** A pasta `02_dissertacao/` não existe. O documento acadêmico formal não foi iniciado. A proposta em `00_proposta/projeto_pesquisa.md` é explicitamente um "rascunho para alinhamento com o orientador" com cronograma "a definir".

**Impacto:** Sem a dissertação, tudo o que existe é um ótimo repositório de código — mas não um TCC de mestrado.

**Como implementar:**
1. Criar `02_dissertacao/` com template abnTeX2 (classe `abntex2.cls`).
2. A estrutura mínima de capítulos para este trabalho é: Introdução → Revisão Bibliográfica → Proposta de Solução → Implementação → Experimentos e Resultados → Conclusão → Referências.
3. Muito do conteúdo já existe no README e nos docstrings — precisam ser reformatados em LaTeX com citações formais.
4. **O README atual é um esboço excelente do Capítulo de Implementação e do Capítulo de Experimentos.** Reaproveitá-lo é o caminho mais rápido.

---

#### M2 — Construir dataset com ground truth e calcular precisão/recall

**Problema:** O "Capítulo de Experimentos" descrito em `05_experimentos/README.md` está vazio. Os resultados no README são anecdotais: 5 queries, 3–5 frames por query, avaliação manual. Não há dataset anotado, nem precisão, nem recall, nem F1.

**Trecho problemático (README):**
> "Qualidade de confiança por tipo de query — 'mulher de jaqueta preta na calçada': 0/3 (28–47%) — Conteúdo ausente no vídeo — sinalizado corretamente"

Esse resultado é observado manualmente em 3 frames. Não é uma métrica.

**Como implementar:**
1. **Anotar um dataset de avaliação** com pelo menos 5–10 vídeos e 10–20 queries por vídeo. Para cada query, listar os timestamps corretos (ground truth positivos) e os negativos. Não precisa ser enorme — 100 queries anotadas com 5 frames de referência cada já é suficiente para um TCC.
2. **Calcular Precision@K e Recall@K** para K = 1, 3, 5 e 10. O frame correto no rank 1 é mais valioso que no rank 5.
3. **Medir latência de busca** como função do tamanho do índice (número de frames), para validar a claim de que numpy brute-force é viável até "centenas de milhares de frames".
4. **Comparar com e sem reranking** (SigLIP apenas vs. SigLIP + Qwen2.5-VL) para quantificar o ganho do reranking em termos de Precision@1.
5. O `05_experimentos/README.md` propõe a estrutura correta (selecao_vlm / pipeline / integracao_vsm) — basta populá-la.

---

#### M3 — Criar a revisão bibliográfica (`01_revisao_bibliografica/`)

**Problema:** A pasta não existe. O arquivo `mapeamento_vlms.md` referenciado na proposta (`Candidatos a VLM: ver 01_revisao_bibliografica/mapeamento_vlms.md`) não existe. Sem revisão bibliográfica formal, o trabalho carece de fundamentação teórica.

**Como implementar:**
1. Criar `01_revisao_bibliografica/mapeamento_vlms.md` com tabela comparativa dos VLMs avaliados (CLIP, M-CLIP, SigLIP, Qwen2-VL 2B vs 3B, e outros candidatos descartados).
2. Referenciar os artigos originais: SigLIP (Zhai et al., 2023), Qwen2.5-VL (Bai et al., 2023), paraphrase-multilingual-mpnet (Reimers & Gurevych, 2020), e trabalhos relacionados de busca semântica em vídeo.
3. Cobrir as principais abordagens da literatura para busca em vídeo: sistemas como CLIP4Clip, VideoCLIP, InternVideo — e explicar por que a abordagem de extração de frames + retrieval por frame é justificada para vídeos longos (vs. abordagens que tratam o vídeo como sequência temporal).
4. Incluir uma seção sobre VLMs em cenários de vigilância (privacy-preserving, on-premise), que é o contexto aplicado do trabalho.

---

### 🟠 ALTO — Afeta a integridade científica do trabalho

---

#### M4 — Corrigir inconsistência arquitetural em `multi_index.py` linha 92

**Problema:** Na função `_process_camera()` em `multi_index.py`, linha 92:

```python
caption = describer.describe(frame.image)  # ← legenda GENÉRICA
```

A indexação multi-câmera gera legendas genéricas (sem saber a query). Porém, ao buscar via `run_search()` (`pipeline.py`), o sistema **ignora essas legendas armazenadas** e chama `describe_for_query_batch()` novamente on-demand para os top-K resultados. As legendas armazenadas no índice multi-câmera nunca são usadas na busca — são overhead puro.

Há dois cenários possíveis para corrigir, dependendo da intenção de design:

**Opção A (mais simples) — não legendar durante a indexação:** Remover a chamada ao Qwen durante `run_multi_index`, deixando `caption=""` no `FrameRecord`. Isso é consistente com `run_index()` (Estágio 1), que também não gera legendas na indexação. Legends são geradas on-demand na busca para todos os índices.

**Opção B (alternativa) — índice com legendas genéricas para busca offline:** Se a intenção for ter legendas pré-geradas para consulta sem query condicionada (ex: busca por palavras-chave simples), a Opção A não serve. Nesse caso, deixar claro na dissertação que `index-multi` gera um índice com legendas genéricas, enquanto `index` gera um índice sem legendas.

A inconsistência atual precisa ser resolvida e documentada — não pode ficar ambígua na dissertação.

---

#### M5 — Calibrar e expor o score de confiança corretamente

**Problema:** O pipeline reporta similaridade de cosseno bruta como "confiança" (ex: `confianca=76.3%`). O script `calibrate_confidence.py` existe e calcula floor/ceil para normalizar os scores, mas **não é integrado ao pipeline**. Uma confiança de 64% em similaridade de cosseno não tem a mesma semântica que 64% de probabilidade.

**Trecho relevante (projeto_pesquisa.md):**
> "Definir critério objetivo de 'grau de certeza' (score de similaridade do embedding? confiança do VLM? calibração?) — PONTO EM ABERTO"

Esse ponto continua em aberto no código atual.

**Como implementar:**
1. Rodar `calibrate_confidence.py` sobre o dataset de avaliação (M2) para estimar floor (similaridade máxima entre pares não-relacionados) e ceil (similaridade média entre paráfrases).
2. Integrar a normalização em `run_search()`:
   ```python
   confidence_normalized = (raw_confidence - FLOOR) / (CEIL - FLOOR)
   confidence_normalized = max(0.0, min(1.0, confidence_normalized))
   ```
3. Reportar na dissertação a separação floor/ceil medida empiricamente — isso é um resultado de avaliação válido.

---

#### M6 — Documentar e justificar os objetivos específicos não alcançados

**Problema:** A proposta tem 6 objetivos específicos. Dois não foram cumpridos:
- Obj. 4: "Implementar um microsserviço (API) que receba a consulta via chat" — **removido** (`api.py` descontinuado).
- Obj. 5: "Integrar o microsserviço a um protótipo da interface do VSM (Qt)" — **não feito**.

Isso não é necessariamente um problema — é normal um TCC ajustar o escopo durante a pesquisa. Mas a dissertação precisa explicar explicitamente por que esses objetivos foram descontinuados e o que foi produzido em seu lugar.

**Como implementar:**
1. Na proposta/dissertação, reformular os objetivos específicos para refletir o escopo atual:
   - Substituir Obj. 4 por algo como "Implementar um pipeline CLI executável localmente, validando a viabilidade operacional sem infraestrutura de microsserviço".
   - Substituir Obj. 5 por "Validar o pipeline end-to-end em vídeos de vigilância reais com queries em português".
2. Ou retomar os objetivos originais: implementar ao menos uma API FastAPI mínima com endpoints `/index` e `/search` — o README menciona que isso já foi prototipado e depois removido, então o esforço seria baixo.

---

### 🟡 MÉDIO — Afeta rigor e reprodutibilidade

---

#### M7 — Validar escalabilidade em vídeos de 12h reais

**Problema:** O benchmark usa 15 frames de um único vídeo curto. Para 12h com amostragem a 1fps = 43.200 frames, o `EmbeddingStore` carrega tudo em memória como lista de arrays numpy — o `load()` materializa um `np.stack` com dimensão `[43200, 768]`, o que consome ~130 MB de RAM, que é aceitável. Mas a latência de busca com `matrix @ query_vector` e o `np.argsort(-scores)` devem ser medidos nessa escala.

**Como implementar:**
1. Indexar um vídeo de teste longo (ou um vídeo de 1h como proxy) e medir:
   - Tempo de busca bruta como função do número de frames (índice com 1k, 10k, 43k frames).
   - Memória RAM utilizada.
2. Reportar na dissertação com um gráfico de latência vs. tamanho do índice.
3. Incluir comparação com FAISS (flat index, brute-force equivalente) para mostrar que numpy é competitivo até determinada escala.

---

#### M8 — Adicionar análise de sensibilidade dos parâmetros de filtragem

**Problema:** Os valores `motion_ratio=0.02` e `similarity_threshold=0.97` são defaults sem justificativa empírica. O README reporta que o filtro reduziu chamadas VLM em 76% — mas não analisa como esse número muda com diferentes thresholds, nem o impacto na qualidade (será que frames relevantes estão sendo descartados?).

**Como implementar:**
1. Para `similarity_threshold`, testar variação em {0.90, 0.93, 0.95, 0.97, 0.99} e medir:
   - % de frames descartados.
   - Se frames do ground truth (M2) foram descartados incorretamente (falsos negativos do filtro).
2. Para `motion_ratio`, similar: testar {0.01, 0.02, 0.05, 0.10}.
3. Produzir uma tabela de sensibilidade. Isso justifica os valores default escolhidos e é um resultado de experimento valioso para o Capítulo 5.

---

#### M9 — Pinar versões em `requirements.txt`

**Problema:** O arquivo `requirements.txt` lista apenas nomes de pacotes sem versões (`numpy`, `transformers`, `sentence-transformers`, etc.). Em pesquisa, reprodutibilidade exige versões fixas — a versão de `transformers` em particular pode quebrar a compatibilidade com Qwen2.5-VL.

**Como implementar:**
1. Gerar um `requirements-frozen.txt` com versões pinadas:
   ```
   numpy==1.26.4
   opencv-python==4.10.0.84
   transformers==4.49.0
   ...
   ```
2. Manter o `requirements.txt` atual como "requisitos mínimos" e adicionar o frozen como "ambiente reprodutível do experimento".
3. Documentar a versão do Python (3.12) e CUDA (12.1) como parte do ambiente experimental na dissertação.

---

### 🟢 MENOR — Qualidade e clareza

---

#### M10 — Adicionar type hints e docstrings nos scripts em `scripts/`

Os scripts de validação (`validate_confidence_full.py`, `calibrate_confidence.py`, `check_query_bias.py`) não têm type hints nem docstrings nos módulos. Como serão referenciados na dissertação, convém documentá-los com o mesmo padrão dos módulos em `src/`.

---

#### M11 — Documentar limitação do benchmark (amostra pequena)

O benchmark no README usa apenas 15 frames de um vídeo de teste para demonstrar a redução de 33%. Isso é suficiente como demonstração de conceito, mas na dissertação precisa ser explicitado que é um benchmark de microbenchmark em escala pequena, não um resultado de avaliação em escala real.

---

#### M12 — Corrigir o código de uso rápido no README

O `README.md` tem um bloco de código malformado (linha 141 fora do bloco de código, `backtick` de fechamento em excesso):

```markdown
.venv\Scripts\python.exe -m video_search.cli search "procure por uma van escolar" ...
```

Essa linha fica fora do bloco de código acidentalmente (o bloco fecha na linha 139 e a linha 141 não pertence a nenhum bloco).

---

## 4. Sugestões de Próximos Passos (Roteiro Prático)

O trabalho tem um núcleo técnico sólido. O que falta é transformá-lo em um documento acadêmico completo. Sugiro esta ordem de prioridade:

**Fase 1 — Consolidação (imediata):**
1. Criar `02_dissertacao/` com template abnTeX2 e esboço de capítulos.
2. Migrar o conteúdo do README para o Capítulo de Implementação (reescrevendo em linguagem acadêmica).
3. Criar `01_revisao_bibliografica/mapeamento_vlms.md` com os candidatos já avaliados.

**Fase 2 — Avaliação formal (mais urgente após a Fase 1):**
4. Anotar um dataset de ground truth com pelo menos 10 queries por vídeo.
5. Implementar Precision@K e Recall@K, comparando SigLIP-only vs. SigLIP+Qwen (M2).
6. Medir latência de busca em função do tamanho do índice (M7).
7. Calibrar o score de confiança (M5) e integrá-lo ao pipeline.

**Fase 3 — Refinamentos:**
8. Resolver a inconsistência em `multi_index.py` linha 92 (M4) e documentar a decisão.
9. Realizar análise de sensibilidade dos thresholds (M8).
10. Considerar reimplementar o microsserviço FastAPI mínimo (M6) — o esforço é baixo e completa o Obj. 4.

---

## Síntese

| Dimensão | Avaliação |
|---|---|
| Implementação / código | ⭐⭐⭐⭐⭐ Excelente — bem acima da média para um TCC |
| Rigor metodológico | ⭐⭐ Insuficiente — sem dataset anotado nem métricas formais |
| Cobertura de literatura | ⭐ Não iniciada — pasta ausente |
| Documento formal (dissertação) | ⭐ Não iniciada — pasta ausente |
| Completude dos objetivos | ⭐⭐⭐ Parcial — 4/6 objetivos cumpridos |
| Reprodutibilidade | ⭐⭐⭐ Parcial — código reprodutível, dados ausentes |

O código é o ponto de partida mais sólido possível para um TCC de mestrado em sistemas. A lacuna principal é a ausência do documento acadêmico e da avaliação quantitativa formal. Ambos são construíveis em cima do que já existe — o trabalho de engenharia que gera os resultados já está feito.

---

*Relatório produzido por análise automatizada da estrutura de arquivos, código-fonte, testes, scripts e documentação do repositório TCC_Mestrado.*
