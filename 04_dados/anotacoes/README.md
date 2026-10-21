# Anotações para avaliação da confiança

Ground truth usado por `python -m video_search evaluate` (ver
`03_codigo/src/video_search/application/evaluation.py`).

> `consultas.exemplo.jsonl` mostra só o formato: os trechos ali **não** são
> anotações reais. Crie `consultas.jsonl` com as suas.

## Formato

Uma consulta por linha (JSONL):

```json
{"id": "p01", "query": "uma van branca estacionada", "camera": "Teste13", "spans": [[12.0, 18.5]]}
{"id": "n01", "query": "um carro com uma escada no teto", "camera": "Teste13", "spans": []}
```

| Campo    | Conteúdo |
|----------|----------|
| `id`     | identificador único (sugestão: `p..` positivas, `n..` negativas) |
| `query`  | o texto exatamente como um operador digitaria |
| `camera` | nome da pasta do índice em `03_codigo/index/` (ex.: `Teste13`) |
| `spans`  | trechos `[início, fim]` em segundos em que o alvo **aparece** no vídeo; `[]` = consulta negativa |

## Como anotar

- **Positivas (40–50):** assista ao vídeo e anote *todos* os trechos em que o
  alvo aparece, do primeiro ao último segundo visível. Um resultado conta como
  acerto se cair dentro do trecho, com tolerância de ±2 s (`--tolerance`).
- **Negativas (10–15):** objetos que **não** estão no vídeo, mas parecem com
  algo que está (negativos adversariais, como no POPE): "escada no teto" quando
  só há rack de teto, "moto" quando só há bicicleta. Negativos óbvios ("um
  elefante") quase não medem nada.
- Varie o tipo de alvo: pessoas e roupas, veículos, objetos, ações; e varie a
  dificuldade (alvo grande e central vs. pequeno ou no fundo da cena).
- Anote sem olhar o que o sistema retorna, para não enviesar o ground truth.

## Rodar

No diretório `03_codigo`, com os índices já gerados:

```powershell
.venv\Scripts\python.exe -m video_search evaluate ..\04_dados\anotacoes\consultas.jsonl --index-root index --output avaliacao
```

O relatório sai em `avaliacao/`: `metricas.json` (as 3 métricas com IC 95%),
`consultas.csv` (cada resultado, com confiança, acerto e legenda) e `roc.png`.
