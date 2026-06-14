# Experimentos

Resultados, métricas e gráficos gerados durante a avaliação do pipeline.

## Organização sugerida

```
05_experimentos/
├── selecao_vlm/      Comparação entre VLMs candidatos (objetivo específico 1)
├── pipeline/         Avaliação do pipeline completo (precisão, recall, latência)
└── integracao_vsm/   Resultados da integração com o protótipo do VSM
```

Cada subpasta de experimento deve conter, idealmente:
- Script/notebook usado para gerar o resultado (referenciar em
  `../03_codigo/`).
- Dados de saída (CSV/JSON) e gráficos.
- Um breve `README.md` descrevendo configuração e principais conclusões.
