# Conteúdo mestre — Apresentação TCC vs Intelbras IAX + Proposta de Plataforma

(Documento de trabalho interno para montagem do PPTX. Baseado em: análise do PDF Intelbras IAX, análise técnica profunda do código do TCC, e pesquisa de mercado 2026.)

---

## ETAPA 3 — BENCHMARKING TCC vs INTELBRAS IAX

### Funcionalidades em comum
- Busca de conteúdo visual por linguagem natural (texto → imagem/frame)
- Uso de embeddings/similaridade semântica como mecanismo central de retrieval
- Suporte a português (o IAX não traduz, processa nativo; o TCC usa SigLIP multilingue nativo PT)
- Score de confiança/similaridade exibido ao usuário
- Processamento local/on-premise (nenhum dos dois depende de API cloud para o core)
- Foco em vídeo de segurança/vigilância como caso de uso primário

### Funcionalidades exclusivas do Intelbras IAX
- Alarme/evento em tempo real disparado por prompt textual (streaming contínuo sobre novos snapshots)
- Prompt de exclusão (negativo) nativo na UI para reduzir falsos positivos
- Integração com ecossistema completo de câmera/NVR: LPR, reconhecimento facial (200k faces), EPI, linha virtual, mapa de calor, contagem de pessoas, detecção comportamental (queda, violência, aglomeração)
- Produto comercial certificado, com datasheet de performance (5 alvos/s), suporte, garantia, canal de vendas B2B
- App mobile (iSIC Lite) para acesso remoto multi-câmera
- Interface gráfica completa e testada em campo por operadores não técnicos

### Funcionalidades exclusivas do projeto de TCC
- Reranking por diversidade (MMR — Maximal Marginal Relevance) para evitar resultados redundantes
- Legendagem gerativa condicionada à query via VLM (Qwen2.5-VL-3B) — gera explicação textual de por que o frame corresponde à busca, mitigando viés de confirmação por design de prompt
- Pipeline assíncrono produtor-consumidor com batching cross-câmera configurável
- Cache de embeddings por hash SHA-256 (evita reprocessamento)
- Indexação multi-câmera concorrente configurável via YAML
- Profiling de hardware integrado (CPU/RAM/VRAM/GPU) com relatório e gráfico
- Transparência experimental total: comparação documentada de 6 modelos VLM/embedding diferentes com números reais (CLIP, M-CLIP, SigLIP, Qwen2-VL-2B, Qwen2.5-VL-3B, Moondream2, Florence-2)
- Código aberto, sem custo de licença/royalties (modelos Apache 2.0/MIT)

### Pontos onde o projeto de TCC supera o Intelbras IAX
1. **Explicabilidade**: o TCC gera uma legenda textual justificando cada resultado (por que aquele frame corresponde à busca); o IAX apenas retorna um score numérico de similaridade, sem explicação.
2. **Mitigação documentada de viés de confirmação**: o TCC identificou e corrigiu experimentalmente um bug real de alucinação (Qwen2-VL-2B confirmando placas incorretas) — nível de rigor metodológico que o material comercial do IAX não demonstra publicamente.
3. **Transparência de modelo**: o TCC documenta exatamente qual modelo/arquitetura é usado em cada etapa; o IAX nunca revela nomes de modelo (caixa-preta comercial).
4. **Custo de licenciamento**: modelos 100% open-source/Apache-MIT vs. hardware proprietário Intelbras vendido só via projeto registrado.
5. **Flexibilidade de deployment**: o pipeline do TCC roda em qualquer GPU CUDA (mesmo consumer-grade, ex. RTX 4060), enquanto o IAX exige linha específica de NVR "IAX FT" com módulo de IA dedicado.

### Pontos onde o Intelbras ainda tem vantagem
1. **Maturidade de produto**: interface gráfica completa, testada em campo, com anos de UX iterado para operadores de central de monitoramento.
2. **Alarme em tempo real (streaming)**: o TCC opera em modo batch/consulta sob demanda (CLI que nasce e morre); o IAX dispara alertas continuamente.
3. **Ecossistema integrado**: LPR, facial, EPI, comportamento — o IAX é parte de uma suíte completa; o TCC resolve apenas um problema (busca), sem as demais analíticas tradicionais.
4. **Confiabilidade de produção**: suporte técnico, SLA, hardware certificado, garantia — o TCC é um protótipo de pesquisa, sem dataset de avaliação formal (nenhuma métrica de Precision@K/Recall@K calculada).
5. **Interface/acessibilidade para usuário final não técnico**: o IAX tem UI gráfica e app mobile; o TCC hoje só roda via linha de comando.
6. **Latência e prontidão**: o IAX responde em segundos dentro do fluxo do NVR; o pipeline do TCC recarrega os 3 modelos do zero a cada execução (~8-10s de overhead fixo, chegando a ~37s por busca).

### Recursos que poderiam ser incorporados (de um para o outro)
- Do IAX para o TCC: alarme contínuo por prompt, prompt de exclusão, interface gráfica web, modo streaming/RTSP.
- Do TCC para o IAX (hipoteticamente, se a Intelbras adotasse): legendagem explicativa gerativa por resultado, benchmarking público de modelos, opção open-weight auditável.

### Recursos inovadores que ainda não existem em nenhum dos dois
- Chat conversacional multi-turno sobre o conteúdo do vídeo com memória de contexto (nenhum dos dois oferece isso — é o estado da arte 2026 segundo Forbes/SourceSecurity: "AI agents" que respondem perguntas complexas e agem автономamente)
- Relatórios executivos gerados automaticamente por LLM a partir de múltiplos eventos agregados (RAG real, não apenas retrieval)
- Explicabilidade visual (heatmap de atenção mostrando qual região da imagem motivou o resultado)
- Busca federada multi-site em linguagem natural com um único prompt (tendência 2026 confirmada por Genetec/BriefCam)
- Detecção de anomalia zero-shot (sem necessidade de regra pré-definida)
- Agentes de IA que tomam ação orquestrada (não apenas alertam, mas sugerem/executam resposta)

### Potencial de mercado
Alto. O segmento de busca de vídeo por linguagem natural em CFTV é comprovadamente quente: Hikvision (AcuSeek, 2025), Genetec, Milestone/BriefCam e startups como a **Conntour** (captou US$ 7M da General Catalyst/YC em 2026 para exatamente este problema, rodando em GPU de consumo) validam a demanda. O TCC ataca o mesmo problema com abordagem tecnicamente competitiva e custo de licenciamento zero.

### Potencial acadêmico
Alto, condicionado à conclusão dos itens pendentes (revisão bibliográfica, dataset de avaliação com métricas formais, dissertação). O pipeline técnico e a documentação de experimentos (comparação de 6 modelos, decisões de engenharia bem justificadas) já formam uma base sólida para os capítulos de Implementação e Experimentos.

### Potencial industrial
Médio-alto no curto prazo (protótipo/demo), alto no médio prazo se evoluir para produto com API, interface web, avaliação formal e integração com um VMS real — exatamente o gap identificado entre o estado atual (CLI local) e o objetivo declarado no anteprojeto (chat integrado a um VMS Qt).

### Grau de inovação
Médio-alto: a combinação SigLIP (retrieval multilingue) + MMR (diversidade) + VLM gerativo condicionado à query (explicabilidade) + engenharia cuidadosa anti-alucinação é uma composição técnica sofisticada e bem executada, ainda que cada peça individual (SigLIP, MMR, VLM) já exista na literatura. A inovação está na integração criteriosa e na validação experimental, não na invenção de um método novo.

---

## Notas de 0 a 10 (projeto de TCC, estado atual)

| Critério | Nota | Justificativa |
|---|---|---|
| Inovação | 7 | Composição técnica bem pensada (SigLIP+MMR+VLM condicionado), mas usa componentes existentes; falta um diferencial algorítmico realmente novo (ex. RAG completo, agentes). |
| Arquitetura | 7 | Pipeline assíncrono, cache, FAISS, MMR, profiling — engenharia de software madura para um TCC. Perde pontos por: ausência de serviço persistente (recarrega modelos a cada execução), refatoração recente não commitada, scripts de validação quebrados. |
| Escalabilidade | 5 | FAISS `IndexFlatIP` não escala para milhões de vetores (força bruta); sem sharding; sem servidor persistente; decodificação de vídeo é sequencial completa (sem seek real), custosa para vídeos muito longos. |
| Experiência do usuário | 2 | Não existe interface gráfica nem web — só CLI que imprime texto no terminal. Não atende ao usuário final (operador de central) declarado no próprio objetivo do projeto. |
| Facilidade de uso | 3 | Requer conhecimento de linha de comando, configuração YAML, ambiente Python/CUDA — inacessível a um gestor ou operador não técnico. |
| Inteligência empregada | 8 | Uso correto e bem justificado de VLM dual-encoder (SigLIP) + VLM gerativo (Qwen2.5-VL) + reranking por diversidade + encoder de sentenças para confiança — combinação tecnicamente sólida e documentada com rigor experimental incomum para um TCC. |
| Potencial comercial | 6 | Resolve um problema real e validado por mercado (busca por linguagem natural em vídeo), mas ainda como protótipo de pesquisa — falta produto (API, UI, SLA, dataset validado) para ser vendável. |
| Potencial científico | 8 | Comparação experimental de 6 modelos com números reais, identificação e correção de um bug real de alucinação, documentação técnica rigorosa — pronto para virar capítulos fortes de dissertação, uma vez preenchidas as lacunas de revisão bibliográfica e avaliação formal (Precision@K/Recall@K). |

**Média geral: 5,75/10** — tecnicamente forte no núcleo de IA, mas incompleto como produto/entrega (UX, escalabilidade, avaliação formal são os maiores gaps).

---

## ETAPA 4 — TENDÊNCIAS DE MERCADO 2026 (síntese)

- **Natural Language Search** é o novo padrão competitivo: Hikvision (AcuSeek), Genetec (Security Center SaaS), Milestone/BriefCam (AI Search, GA fim de 2026), Verkada, Eagle Eye Networks (Eeva) e a startup Conntour (YC/General Catalyst, 2026) — todos correndo atrás do mesmo recurso que o TCC já implementa.
- **AI Agents / Agentic AI**: a fronteira 2026 é sair do "retrieval" puro e ir para agentes que investigam, correlacionam e até sugerem resposta automaticamente (Forbes Business Council, maio/2026; SourceSecurity).
- **Edge vs Cloud**: fabricantes de hardware (Intelbras, Hikvision, Verkada) processam no edge por privacidade; players de cloud (AWS Rekognition, Azure Video Indexer, Google Vertex AI, Twelve Labs) priorizam profundidade semântica e escala, mas dependem de dados saindo da rede do cliente.
- **VLMs especializados de domínio**: Milestone treina VLMs em datasets específicos de segurança (não modelos genéricos adaptados) — o TCC já segue essa lógica ao testar/rejeitar modelos genéricos (Florence-2, Moondream2) em favor de SigLIP+Qwen2.5-VL ajustados ao caso de uso.
- **Mercado**: analytics de vídeo com IA projetado para US$ 17 bi até 2031 (+22% a.a.); mercado geral de videovigilância deve superar US$ 80 bi até 2030.
- **RAG real para vídeo** ainda é incipiente mesmo nos líderes — nenhum concorrente comercial entrega hoje uma resposta narrativa agregada (LLM sintetizando múltiplos eventos); é um espaço em aberto.
- **Explainable AI / Dashboards executivos**: tendência de exigir que a IA "mostre seu raciocínio" — relatórios automáticos explicáveis via agentes orquestrados (Google Gemini + RAG contextual) é citado como fronteira emergente.

### Funcionalidades modernas propostas para tornar o TCC competitivo
1. Alarme contínuo por prompt (modo streaming), com prompt de exclusão — paridade com IAX/AcuSeek.
2. Chat conversacional multi-turno com memória (verdadeiro RAG: LLM sintetiza resposta a partir de múltiplos frames recuperados, não apenas retorna lista).
3. API HTTP persistente com modelos residentes (recriar o que foi removido, mas com lifespan management).
4. Dashboard web executivo com timeline, mapa de calor e KPIs.
5. Dataset de avaliação com Precision@K/Recall@K — necessário tanto para credibilidade científica quanto comercial.
6. Explicabilidade visual (destacar região da imagem que gerou o match).
7. Suporte a busca federada multi-câmera/multi-site num único prompt.

---

## ETAPA 5-6 — PROPOSTA DE PLATAFORMA E ARQUITETURA

### Identidade visual (inspirada na WEG, sem copiar layout)
- Azul profundo corporativo (#003DA5 / #0B2D5C) — confiança e engenharia
- Verde tecnológico (#00A651) — indicadores positivos, sustentabilidade/eficiência
- Grafite escuro (#1A1F2B) para navegação, cinza claro (#F4F6F8) para fundo, branco para cards
- Tipografia sans-serif (Inter), muito espaço em branco, cantos arredondados 8-12px, sombras suaves
- Nome do produto demo: **Videre AI**

### Arquitetura proposta (visão de produto, evoluindo do protótipo de pesquisa)

```
┌───────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (SPA)                                │
│  React + TypeScript + Tailwind + shadcn/ui                                  │
│  Páginas: Dashboard, Monitoramento, Busca NL, Busca Semântica, Timeline,    │
│  Chat IA, Relatórios, Estatísticas, Admin, Modelos IA, Histórico, Auditoria │
│  Comunicação: REST/JSON + WebSocket (eventos em tempo real)                 │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                 │ HTTPS / WSS
┌───────────────────────────────▼─────────────────────────────────────────────┐
│                         API GATEWAY / BACKEND (FastAPI)                     │
│  Auth (JWT/OAuth2, RBAC: admin/operador/visualizador)                       │
│  Endpoints: /search, /alerts, /cameras, /models, /reports, /audit           │
│  Orquestra: ModelManager (mantém SigLIP, Qwen2.5-VL, encoder residentes)    │
└───────┬───────────────┬───────────────┬───────────────┬─────────────────────┘
        │               │               │               │
        ▼               ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌─────────────┐ ┌────────────────────┐
│ Fila de       │ │ Vetor DB     │ │ Banco        │ │ Armazenamento de    │
│ processamento │ │ (Qdrant /    │ │ relacional   │ │ objetos (vídeos,    │
│ (Redis/Celery │ │ FAISS+shard) │ │ (PostgreSQL) │ │ frames, relatórios) │
│ ou RabbitMQ)  │ │ embeddings   │ │ usuários,    │ │ (S3/MinIO)          │
│ indexação     │ │ 768-dim      │ │ câmeras,     │ │                     │
│ assíncrona    │ │              │ │ eventos, logs│ │                     │
└──────┬────────┘ └──────────────┘ └─────────────┘ └────────────────────┘
       │
       ▼
┌───────────────────────────────────────────────────────────────────────────┐
│              WORKERS DE INFERÊNCIA (GPU, escaláveis horizontalmente)        │
│  SigLIP (embeddings) · Qwen2.5-VL (legendagem/RAG) · sentence-transformers  │
│  Motion filter · Frame extraction (OpenCV/FFmpeg) · MMR reranking          │
└───────────────────────────────────────────────────────────────────────────┘
       ▲
       │ RTSP/RTMP (streaming ao vivo) ou upload de arquivo
┌──────┴────────────────────────────────────────────────────────────────────┐
│                          CÂMERAS / NVRs / ARQUIVOS DE VÍDEO                 │
└─────────────────────────────────────────────────────────────────────────────┘

Observabilidade transversal: Logs estruturados (ELK/Loki) + métricas (Prometheus/Grafana)
+ tracing (OpenTelemetry) + profiling de GPU (já existe no TCC, reaproveitar).
```

### Justificativas técnicas principais
- **FastAPI com lifespan**: resolve o maior débito técnico atual do TCC (API removida, modelos recarregados a cada chamada) — já era a "Estratégia A" do próprio `estrategia_otimizacao.md` do projeto, nunca implementada.
- **Fila de processamento (Celery/Redis ou RabbitMQ)**: desacopla indexação (pesada, assíncrona) de busca (interativa, latência baixa); permite escalar workers de GPU horizontalmente.
- **Vetor DB dedicado (Qdrant) em vez de FAISS puro**: FAISS `IndexFlatIP` é força bruta em memória — não escala para volume de produção (múltiplas câmeras, meses de retenção). Qdrant oferece filtragem por metadados, sharding e persistência sem reescrever tudo do zero — é a evolução natural do que o TCC já validou conceitualmente.
- **PostgreSQL para dados relacionais**: usuários, câmeras, papéis, eventos estruturados, auditoria — hoje inexistente no TCC (tudo em JSON/arquivo).
- **Armazenamento de objetos (S3/MinIO)**: vídeos e frames extraídos hoje ficam em caminhos locais fora do repositório (um dos problemas de reprodutibilidade identificados); um object store resolve isso de forma profissional.
- **WebSocket para eventos em tempo real**: necessário para a página de Monitoramento em Tempo Real e para paridade com o "Alarme por Texto" do IAX.
- **RBAC (Admin/Operador/Visualizador)**: requisito básico de qualquer produto corporativo B2B, ausente no protótipo atual.
- **Observabilidade (logs, métricas, tracing)**: o TCC já tem profiling de hardware — a evolução natural é integrá-lo a Prometheus/Grafana para monitoramento contínuo em produção, não apenas relatórios pontuais.
- **Escalabilidade**: workers de inferência em containers (Docker/Kubernetes) permitem escalar por GPU disponível; a fila absorve picos de indexação sem degradar a busca interativa.

---

## ETAPA 7 — FRONTEND: WIREFRAMES TEXTUAIS DAS 16 PÁGINAS

Protótipo real construído no Lovable (React + TypeScript + Tailwind + shadcn/ui) — projeto "Videre AI". Especificação de cada página:

| # | Página | Componentes principais | Comportamento esperado | APIs | UX |
|---|---|---|---|---|---|
| 1 | Login | Logo, campos e-mail/senha, botão entrar, "esqueci a senha", imagem split-screen industrial | Autentica e redireciona ao Dashboard; valida campos; mostra erro de credencial | `POST /auth/login` | Split-screen corporativo, transmite confiabilidade antes mesmo do login |
| 2 | Dashboard | Cards de KPI (câmeras online, alertas hoje, buscas realizadas, latência média), gráfico de eventos/hora, lista de alertas críticos, status dos modelos | Atualiza KPIs periodicamente; clique no alerta abre detalhe | `GET /dashboard/summary`, `GET /alerts?limit=5` | Visão executiva "de relance", sem necessidade de navegar |
| 3 | Monitoramento em Tempo Real | Grid de câmeras com thumbnail/status, painel de eventos ao vivo | Novo evento aparece via WebSocket com destaque visual | `WS /events/stream`, `GET /cameras` | Sensação de "sala de controle" viva |
| 4 | Busca por Linguagem Natural | Campo de busca estilo chat, sugestões, grade de resultados com score | Envia prompt, mostra loading, renderiza resultados ordenados por confiança | `POST /search` | Baixa fricção — qualquer gestor entende sem treinamento |
| 5 | Busca Semântica | Slider de similaridade, seleção de câmeras/grupo, prompt de exclusão | Refina resultados em tempo real ao mover slider | `POST /search/advanced` | Público mais técnico (operador avançado) |
| 6 | Consulta por Eventos | Tabela filtrável (tipo, câmera, data, severidade), exportação | Filtros combináveis, paginação, export CSV/PDF | `GET /events?filters=...` | Familiar a quem usa VMS tradicional |
| 7 | Timeline dos Vídeos | Linha do tempo horizontal por câmera, marcadores de evento, preview on-hover | Scroll/zoom na timeline, clique abre player no timestamp | `GET /timeline?camera_id=` | Navegação rápida em long-form video |
| 8 | Visualização das Câmeras | Grid/lista de câmeras, status, localização, config | Clique abre detalhes/edição de câmera | `GET /cameras`, `PATCH /cameras/:id` | Inventário claro de ativos |
| 9 | Chat com IA | Interface conversacional, respostas citando frame+timestamp | Pergunta em linguagem natural, resposta com evidência visual anexada | `POST /chat` (RAG) | Reduz a busca a uma conversa, o diferencial mais "wow" da demo |
| 10 | Relatórios | Cards de relatórios recentes, botão gerar novo, preview/exportação | Geração assíncrona com notificação ao concluir | `POST /reports`, `GET /reports` | Entrega valor direto para gestores não técnicos |
| 11 | Estatísticas | Heatmap de ocupação, linha do tempo de eventos, distribuição por câmera/tipo | Filtros de período, exportação de gráfico | `GET /stats?range=` | Storytelling de dados para reuniões executivas |
| 12 | Configurações | Preferências de conta, notificações, idioma, tema | Salva preferências localmente/servidor | `PATCH /users/me/settings` | Personalização padrão de SaaS |
| 13 | Administração | Tabela de usuários, convite, papéis (admin/operador/visualizador) | CRUD de usuários com confirmação | `GET/POST/PATCH /admin/users` | Controle de acesso corporativo |
| 14 | Gerenciamento de Modelos de IA | Lista de modelos (SigLIP, Qwen2.5-VL, sentence-transformers), status, versão, métricas de uso | Ativar/desativar/trocar modelo, ver latência/GPU | `GET /models`, `PATCH /models/:id` | Transparência técnica para stakeholders técnicos |
| 15 | Histórico | Log cronológico de buscas/ações do usuário | Busca/filtro por usuário, data, ação | `GET /history` | Rastreabilidade de uso |
| 16 | Auditoria | Log de segurança/compliance (LGPD), acessos, exportações | Filtros por usuário/ação, exportação para compliance | `GET /audit-log` | Confiança para compliance/jurídico |

Layout comum: sidebar fixa esquerda (colapsável) com as 16 rotas, header com busca global/notificações/avatar. Login é a única página sem esse shell.

---

## ETAPA 8 — ROADMAP EM 5 FASES

### Fase 1 — MVP (fundação técnica)
- Funcionalidades: corrigir débitos técnicos atuais (imports quebrados, requirements pinados, commit da refatoração), expor `EmbeddingCache` na CLI, recriar API HTTP mínima (FastAPI, `/index`, `/search`) com modelos residentes.
- Prioridade: crítica
- Complexidade: baixa-média
- Tempo estimado: 2-3 semanas
- Riscos: nenhum estrutural — é dívida técnica já mapeada
- Dependências: nenhuma

### Fase 2 — Protótipo executivo
- Funcionalidades: dashboard web mínimo (React), autenticação simples, páginas de Busca por Linguagem Natural e Monitoramento, WebSocket para eventos simulados/reais.
- Prioridade: alta
- Complexidade: média
- Tempo estimado: 4-6 semanas
- Riscos: integração frontend-backend, definição de contrato de API estável
- Dependências: Fase 1 concluída

### Fase 3 — Versão para demonstração (a que este protótipo Lovable antecipa)
- Funcionalidades: as 16 páginas completas (dashboard, busca semântica, timeline, chat IA, relatórios, estatísticas, administração, gerenciamento de modelos, histórico, auditoria), dados reais ou semi-reais, identidade visual corporativa completa.
- Prioridade: alta (é a entrega de maior impacto para banca/gestores)
- Complexidade: média-alta
- Tempo estimado: 6-8 semanas
- Riscos: escopo grande demais para um time pequeno; priorizar páginas de maior impacto de demonstração primeiro (Busca NL, Chat IA, Dashboard)
- Dependências: Fase 2

### Fase 4 — Versão piloto
- Funcionalidades: dataset de avaliação com métricas formais (Precision@K/Recall@K), alarme contínuo em tempo real (paridade com IAX), vetor DB escalável (Qdrant), RBAC completo, testes com vídeo real de um ambiente controlado (ex. um andar do prédio da universidade ou empresa parceira).
- Prioridade: alta
- Complexidade: alta
- Tempo estimado: 2-3 meses
- Riscos: obtenção de dados reais anotados (LGPD), tempo de validação com usuários reais
- Dependências: Fase 3, aprovação de comitê de ética/LGPD se houver dados de pessoas identificáveis

### Fase 5 — Versão industrial
- Funcionalidades: multi-tenant, alta disponibilidade, observabilidade completa (Prometheus/Grafana/ELK), integração com VMS real (retomando objetivo original do anteprojeto), certificação de segurança, SLA.
- Prioridade: média (depende de tração comercial)
- Complexidade: muito alta
- Tempo estimado: 4-6 meses
- Riscos: custo de infraestrutura GPU em escala, necessidade de equipe além do autor único
- Dependências: Fase 4 validada com usuários reais, decisão de investimento

---

## ETAPA 9 — PLANO DE IMPLEMENTAÇÃO

### Melhorias imediatas (dias)
- Corrigir imports quebrados nos 10 scripts de `scripts/`
- Commitar a refatoração pendente (subpacotes, FAISS, MMR, async pipeline)
- Pinar versões no `requirements.txt`
- Corrigir bloco de código malformado no README
- Remover vídeo de índice residual que aponta para caminho fora do repositório (ou documentar como não-reproduzível)

### Melhorias de médio prazo (semanas)
- Expor `EmbeddingCache` via flag da CLI
- Integrar calibração de confiança (`calibrate_confidence.py`) ao pipeline de busca
- Recriar API HTTP com FastAPI lifespan (modelos residentes)
- Construir dataset mínimo de avaliação com ground truth e calcular Precision@K/Recall@K
- Remover ou documentar código morto (`BatchDispatcher`, `match_confidence()`)

### Melhorias de longo prazo (meses)
- Migrar de FAISS `IndexFlatIP` para Qdrant (ou FAISS com índice aproximado tipo HNSW/IVF) para escalar
- Implementar alarme contínuo por prompt (modo streaming RTSP)
- Implementar chat com memória multi-turno (RAG real: LLM sintetizando resposta a partir de múltiplos eventos recuperados)
- Frontend web completo (as 16 páginas) com autenticação e RBAC
- Revisão bibliográfica formal e redação da dissertação

### Backlog priorizado (ordem sugerida)
1. Corrigir scripts quebrados + commit da refatoração (crítico, bloqueia reprodutibilidade)
2. Dataset de avaliação formal + métricas de IR (crítico para defesa acadêmica)
3. API HTTP persistente
4. Frontend mínimo (Dashboard + Busca NL)
5. Calibração de confiança integrada
6. Vetor DB escalável
7. Alarme em tempo real
8. Chat com RAG real
9. Frontend completo (16 páginas)
10. Integração com VMS real / piloto em ambiente controlado

### Arquitetura final proposta
Ver diagrama da Etapa 5-6 acima (FastAPI + Qdrant + PostgreSQL + Redis/Celery + object storage + workers GPU + frontend React).

### Tecnologias recomendadas e justificativa
- **FastAPI**: já validado experimentalmente pelo próprio autor no protótipo anterior (removido, mas funcional); baixo atrito para retomar.
- **Qdrant**: open-source, já cogitado no README do projeto ("FAISS/Qdrant a confirmar"), suporta filtros por metadados essenciais para multi-câmera/multi-tenant.
- **PostgreSQL**: padrão de mercado para dados relacionais, suporte maduro a RBAC e auditoria.
- **Redis + Celery**: solução comprovada para filas de processamento assíncrono em Python, linguagem já usada em 100% do projeto.
- **React + TypeScript + Tailwind + shadcn/ui**: acelera prototipação (via Lovable) e evolui para produto real sem reescrita.
- **Docker/Kubernetes**: necessário para escalar workers de GPU horizontalmente na Fase 5.

### Riscos técnicos
- Dependência de versão específica de `transformers` (já causou rejeição de Florence-2/Moondream2) — precisa de suíte de testes de regressão ao atualizar dependências.
- Custo de GPU em escala (cada busca detalhada usa Qwen2.5-VL, computacionalmente caro) — considerar cache agressivo e modos "fast" sem VLM para a maioria das consultas.
- LGPD: dados de vídeo com pessoas identificáveis exigem anonimização e base legal antes de qualquer piloto com dados reais.

### Oportunidades de pesquisa
- Calibração de score de confiança (isotônica ou Platt scaling) para VLMs de retrieval — lacuna documentada no próprio projeto.
- Mitigação de "hubness" em espaços de embedding multilingues (QB-Norm, proposto mas não implementado).
- Avaliação de MMR vs. outras estratégias de diversidade em busca de vídeo de vigilância — pouco explorado na literatura em português.

### Oportunidades de publicação científica
- Artigo sobre o prompt anti-viés-de-confirmação desenhado para Qwen2.5-VL em legendagem condicionada (achado experimental original e replicável).
- Artigo de benchmarking comparativo de VLMs open-source para retrieval multilingue em vídeo de vigilância (CLIP vs M-CLIP vs SigLIP vs Qwen2-VL vs Moondream2 vs Florence-2) — dado já coletado, falta só formalizar com dataset de avaliação.

### Oportunidades de patente
- Baixa probabilidade de patente sobre o pipeline em si (composição de técnicas conhecidas), mas o método específico de prompt anti-alucinação condicionado à query, se validado formalmente com métricas, pode ter potencial de proteção como método/processo, a avaliar com o núcleo de inovação da universidade/empresa.

### Diferenciais competitivos
- Explicabilidade (legenda gerativa por resultado) — hoje ausente nos concorrentes analisados
- Custo zero de licenciamento (modelos open-source)
- Flexibilidade de hardware (GPU de consumo, não exige NVR proprietário)
- Rigor experimental documentado (transparência de modelo, ao contrário da caixa-preta comercial)
