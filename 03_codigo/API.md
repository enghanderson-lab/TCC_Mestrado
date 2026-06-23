# API do microsserviço (`video_search.api`)

Guia de uso da API HTTP que expõe o pipeline de indexação/busca (ver
[README.md](README.md) e o [README do projeto](../README.md) para a visão
geral do pipeline e os achados de validação).

## Subindo o servidor

```powershell
cd 03_codigo
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
uvicorn video_search.api:app --host 0.0.0.0 --port 8000
```

Documentação interativa (Swagger UI), gerada automaticamente pelo FastAPI:
`http://localhost:8000/docs`.

Por padrão os índices ficam em `04_dados/index` (na raiz do projeto).
Para usar outra pasta, defina a variável de ambiente antes de subir o
servidor:

```powershell
$env:VIDEO_SEARCH_INDEX_ROOT = "D:\indices_video_search"
```

Não há autenticação — o serviço é pensado para rodar numa rede
interna/confiável junto com o VMS (Qt) que o consome.

## Conceitos

- **`video_path`**: caminho de um arquivo de vídeo **no disco do servidor**
  (não é upload via multipart — vídeos de horas de duração não cabem num
  upload HTTP razoável).
- **`name`** / **`index`**: identificador do índice. É o nome da subpasta
  criada dentro de `INDEX_ROOT`. Aceita apenas letras, números, `_` e `-`
  (validação contra path traversal).

## Endpoints

### `GET /health`

Sanity check.

```powershell
curl http://localhost:8000/health
```
```json
{"status": "ok"}
```

### `GET /indexes`

Lista os índices já gerados.

```powershell
curl http://localhost:8000/indexes
```
```json
[
  {"name": "camera1", "frame_count": 1800, "model_name": "siglip"}
]
```

### `POST /index`

Inicia a indexação de um vídeo **em background** (a chamada retorna
imediatamente; o processamento continua no servidor). Use o `job_id`
retornado para acompanhar o progresso em `GET /index/jobs/{job_id}`.

| Campo | Tipo | Default | Descrição |
|-------|------|---------|-----------|
| `video_path` | string | — (obrigatório) | Caminho do vídeo no disco do servidor |
| `name` | string | — (obrigatório) | Nome do índice a criar |
| `interval` | float | `1.0` | Intervalo entre frames amostrados (segundos) |
| `batch_size` | int | `16` | Tamanho do lote de inferência |
| `limit` | int | `0` | Limita o nº de frames processados (`0` = sem limite) |

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/index -ContentType "application/json" -Body (@{
    video_path = "D:\videos\camera1.mp4"
    name       = "camera1"
    interval   = 2
} | ConvertTo-Json)
```

> Use `Invoke-RestMethod` (PowerShell nativo) para requisições com corpo
> JSON, em vez de `curl -d '...'`: o `curl` do PowerShell é só um alias para
> `Invoke-WebRequest` (não aceita `-X`/`-H`/`-d`), e mesmo chamando o
> `curl.exe` real, o parsing de aspas no shell para executáveis nativos do
> Windows é inconsistente e quebra JSON com aspas internas. `curl` simples
> (sem corpo) funciona bem para os endpoints `GET` abaixo.

Resposta (`202`-like, mas retornada como `200` com status `queued`):

```json
{
  "job_id": "3f9c1e2b8a4d4e7f9b6c1a2d3e4f5a6b",
  "status": "queued",
  "frame_count": 0,
  "output_dir": "...\\04_dados\\index\\camera1",
  "error": null
}
```

Erros: `404` se `video_path` não existir; `400` se `name` tiver caracteres
inválidos.

### `GET /index/jobs/{job_id}`

Consulta o status de um job de indexação. Faça polling até `status` virar
`"done"` (ou `"error"`).

```powershell
curl http://localhost:8000/index/jobs/3f9c1e2b8a4d4e7f9b6c1a2d3e4f5a6b
```

```json
{
  "job_id": "3f9c1e2b8a4d4e7f9b6c1a2d3e4f5a6b",
  "status": "running",
  "frame_count": 320,
  "output_dir": "...\\04_dados\\index\\camera1",
  "error": null
}
```

`status` é um de: `queued`, `running`, `done`, `error` (nesse caso, `error`
traz a mensagem da exceção). Erro: `404` se `job_id` não existir.

> Jobs ficam em memória — são perdidos se o processo do servidor reiniciar.

### `POST /search`

Busca os frames mais relevantes para uma descrição em português.

| Campo | Tipo | Default | Descrição |
|-------|------|---------|-----------|
| `query` | string | — (obrigatório) | Descrição em linguagem natural (PT) |
| `index` | string | — (obrigatório) | Nome do índice (criado via `/index`) |
| `top_k` | int | `12` | Número de resultados |
| `with_caption` | bool | `true` | Gera legenda + confiança via Qwen2.5-VL (reranking). `false` retorna só o score de retrieval do SigLIP, bem mais rápido |

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/search -ContentType "application/json" -Body (@{
    query = "homem com caderno de anotações"
    index = "camera1"
    top_k = 5
} | ConvertTo-Json)
```

Resposta — lista ordenada por relevância:

```json
[
  {
    "frame_index": 31,
    "timestamp_sec": 31.0,
    "video": "camera1.mp4",
    "retrieval_score": 0.1373,
    "confidence": 0.763,
    "caption": "pessoa segurando caderno em asfalto marcado"
  }
]
```

`confidence`/`caption` vêm `null` quando `with_caption: false`. Erro: `404`
se `index` não existir.

## Fluxo completo (indexar + buscar)

```powershell
# 1) Disparar indexação
$job = Invoke-RestMethod -Method Post http://localhost:8000/index -ContentType "application/json" -Body (@{
    video_path = "D:\videos\camera1.mp4"
    name       = "camera1"
} | ConvertTo-Json)

# 2) Aguardar conclusão (polling simples)
do {
    Start-Sleep -Seconds 5
    $status = Invoke-RestMethod "http://localhost:8000/index/jobs/$($job.job_id)"
    Write-Host "status=$($status.status) frames=$($status.frame_count)"
} while ($status.status -in @("queued", "running"))

# 3) Buscar
Invoke-RestMethod -Method Post http://localhost:8000/search -ContentType "application/json" -Body (@{
    query = "carro estacionado na rua"
    index = "camera1"
} | ConvertTo-Json)
```

## Fora de escopo (v1)

- Autenticação/autorização.
- Persistência de jobs entre restarts do processo.
- Upload de vídeo via HTTP (somente caminho em disco).
