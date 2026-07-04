# Video Search API

API HTTP (FastAPI) para indexação e busca semântica multi-câmera, expondo o
pipeline SigLIP + Qwen2.5-VL usado pela CLI. Pensada para consumo por um
frontend (ex.: Lovable) ou outro serviço.

Status: **Beta 1** — uso interno/local, sem deploy em produção ainda.

## Subir a API

```powershell
.venv\Scripts\python.exe -m uvicorn video_search.api.main:app --host 0.0.0.0 --port 8000
```

Docs interativas (Swagger) em `http://localhost:8000/docs`.

### Variáveis de ambiente

| Variável           | Default       | Descrição                                              |
|---------------------|---------------|---------------------------------------------------------|
| `VS_API_KEY`         | `dev-key`     | Chave exigida no header `X-Api-Key`                      |
| `VS_DB_PATH`         | `cameras.db`  | Caminho do SQLite que guarda o registro de câmeras       |
| `VS_CORS_ORIGINS`    | *(vazio)*     | Origens CORS permitidas, separadas por vírgula — **obrigatório configurar antes de qualquer deploy real** |

> ⚠️ Sem `VS_CORS_ORIGINS`, nenhuma origem cross-origin é liberada
> (fail-safe) e um warning é logado no startup. `allow_credentials=True` é
> fixo no middleware, e a spec de CORS proíbe combinar credentials com
> wildcard (`*`) — então configure sempre a origem exata do frontend, ex.:
> `VS_CORS_ORIGINS=https://seu-projeto.lovable.app`.

No lifespan da aplicação os modelos (SigLIP, Qwen2.5-VL, sentence-transformer)
são carregados uma única vez via `ModelManager` — a primeira requisição já
encontra tudo pronto, mas o boot do processo demora até o load terminar.

Todos os endpoints (exceto `/health`) exigem o header `X-Api-Key: <VS_API_KEY>`.
Prefixo comum: **`/api/v1`**.

## Fluxo típico

1. `POST /api/v1/index/start` com o vídeo de uma câmera → recebe `job_id`.
2. Acompanhar progresso via `GET /api/v1/index/{job_id}/status` (polling) ou
   `WS /api/v1/index/ws/{job_id}` (streaming).
3. Quando `status == "done"`, a câmera aparece em `GET /api/v1/cameras`.
4. `POST /api/v1/search` com uma query em linguagem natural, opcionalmente
   filtrando `camera_ids`.
5. Cada resultado traz `frame_url` — buscar o JPEG em
   `GET /api/v1/frames/{camera_id}/{timestamp_ms}`.

## Endpoints

### `GET /health`

Sem autenticação. Sem prefixo `/api/v1`.

```json
{
  "status": "ok",
  "models_loaded": true,
  "vram_used_mb": 3421.0,
  "vram_total_mb": 6144.0,
  "active_jobs": 1
}
```

### `POST /api/v1/index/start`

Inicia a indexação de um vídeo em background (thread pool, 2 workers).

Request:
```json
{
  "camera_id": "cam1",
  "video_path": "C:\\videos\\cam1.mp4",
  "interval": 2.0,
  "batch_size": 16
}
```

- `interval`: segundos entre frames amostrados (0.1–60, default 2.0).
- `batch_size`: tamanho do batch de inferência (1–64, default 16).

Response (`202`-like, mas retorna `200` com o job recém-criado):
```json
{
  "job_id": "3f1b...",
  "camera_id": "cam1",
  "status": "running",
  "frames_indexed": 0,
  "frames_estimated": 450,
  "elapsed_sec": 0.0,
  "fps": 0.0,
  "error": null
}
```

Erros: `400` se `video_path` não existir no filesystem do servidor.

### `GET /api/v1/index/{job_id}/status`

Retorna o mesmo shape acima com os contadores atualizados. `404` se o
`job_id` não existir.

### `DELETE /api/v1/index/{job_id}`

Marca o job como `cancelled` (não interrompe o worker em execução, apenas
sinaliza o status). `400` se o job já não estiver `running`. `404` se não
existir.

### `WS /api/v1/index/ws/{job_id}?api_key=<key>`

WebSocket de progresso (autenticação via query param, já que browsers não
enviam headers customizados em handshakes de WS). Envia o mesmo JSON de
status a cada 0.5s até o job terminar (`done`, `error` ou `cancelled`), então
fecha a conexão. Se a chave for inválida, envia `{"error": "unauthorized"}`
e fecha com code `1008`.

### `GET /api/v1/cameras`

Lista todas as câmeras já indexadas.

```json
[
  {
    "camera_id": "cam1",
    "video_path": "C:\\videos\\cam1.mp4",
    "output_dir": "index\\cam1",
    "frame_count": 450,
    "indexed_at": "2026-07-03T18:22:10.123456+00:00",
    "status": "indexed"
  }
]
```

### `DELETE /api/v1/cameras/{camera_id}`

Remove o registro da câmera e apaga o diretório de índice (`output_dir`) do
disco. `404` se a câmera não existir.

### `POST /api/v1/search`

Busca semântica textual sobre uma ou mais câmeras já indexadas.

Request:
```json
{
  "query": "homem de camisa branca e bone vermelho",
  "mode": "detailed",
  "camera_ids": ["cam1", "cam2"],
  "top_k": 5,
  "mmr_lambda": 0.7
}
```

- `mode`:
  - `fast` — apenas busca FAISS por similaridade SigLIP, sem legendas.
  - `detailed` — aplica MMR cross-câmera nos candidatos e gera legenda +
    confidence score via Qwen2.5-VL para os `top_k` finais (mais lento).
- `camera_ids`: vazio = busca em todas as câmeras indexadas.
- `mmr_lambda`: trade-off relevância vs. diversidade no MMR (0=diversidade
  máxima, 1=relevância pura). Só tem efeito em `mode=detailed`.

Response:
```json
{
  "query_time_ms": 812,
  "mode": "detailed",
  "results": [
    {
      "camera_id": "cam1",
      "timestamp_sec": 134.0,
      "score": 0.312,
      "frame_url": "/api/v1/frames/cam1/134000",
      "description": "Homem parado próximo à entrada, camisa branca...",
      "confidence": 0.71
    }
  ]
}
```

`description`/`confidence` só vêm preenchidos em `mode=detailed`. Erros:
`404` se nenhuma câmera estiver indexada, ou se `camera_ids` não bater com
nenhuma câmera existente.

### `GET /api/v1/frames/{camera_id}/{timestamp_ms}`

Extrai on-demand (não fica em cache) o frame mais próximo do timestamp
informado (em milissegundos) e retorna como `image/jpeg`. `404` se a câmera
ou o arquivo de vídeo não existirem, ou se o frame não puder ser extraído.

## Autenticação

Todos os endpoints com `dependencies=[Depends(require_api_key)]` exigem:

```
X-Api-Key: <VS_API_KEY>
```

Chave inválida ou ausente → `401 Invalid or missing API key`.

## Exemplos (curl)

```bash
# Indexar
curl -X POST http://localhost:8000/api/v1/index/start \
  -H "X-Api-Key: dev-key" -H "Content-Type: application/json" \
  -d '{"camera_id":"cam1","video_path":"C:/videos/cam1.mp4","interval":2.0}'

# Status
curl http://localhost:8000/api/v1/index/<job_id>/status -H "X-Api-Key: dev-key"

# Buscar
curl -X POST http://localhost:8000/api/v1/search \
  -H "X-Api-Key: dev-key" -H "Content-Type: application/json" \
  -d '{"query":"pessoa correndo","mode":"fast","top_k":5}'
```

## Limitações conhecidas

- `require_api_key` compara a chave em texto puro (`==`), sem constant-time
  compare — aceitável para uso local/dev, revisar antes de expor
  publicamente.
- Cancelamento de job (`DELETE /index/{job_id}`) é cooperativo: o leitor de
  frames para na próxima iteração do loop, mas os frames já em trânsito nas
  filas internas (algumas dezenas, no pior caso) ainda são processados antes
  do job encerrar.
- Sem paginação em `GET /cameras` nem no `SearchResponse`.
- CORS fail-safe (nenhuma origem por padrão) — configure `VS_CORS_ORIGINS`
  com a origem exata do frontend antes de qualquer deploy exposto à internet.
