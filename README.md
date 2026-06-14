# TCC Mestrado — Busca Semântica em Vídeo via VLM

Projeto de mestrado para desenvolvimento de um pipeline de busca semântica em
gravações de vídeo de longa duração (ex.: 12h), usando um VLM (Vision-Language
Model) de licença permissiva (uso comercial sem royalties), permitindo que o
usuário pesquise por linguagem natural (ex.: "homem de camisa branca e boné
vermelho") e receba os frames correspondentes com o grau de confiança
associado. O resultado final será disponibilizado como microsserviço para
integração com um software VSM desenvolvido em Qt.

## Estrutura do repositório

```
00_proposta/             Projeto de pesquisa / anteprojeto (rascunhos em Markdown)
01_revisao_bibliografica/ Mapeamento de literatura, candidatos a VLM, fichamentos
02_dissertacao/           Documento da dissertação (LaTeX, classe abnTeX2)
03_codigo/                Implementação do pipeline e do microsserviço
04_dados/                 Vídeos de teste, datasets, anotações (não versionar dados grandes/sensíveis)
05_experimentos/          Resultados, métricas, gráficos, logs de avaliação
```

## Status atual

Fase: **início / projeto de pesquisa**. Próximos passos:
1. Consolidar título, problema, objetivos e metodologia em `00_proposta/`.
2. Levantar VLMs candidatos com licença permissiva em `01_revisao_bibliografica/`.
3. Definir cronograma com o orientador.

## Compilando a dissertação (LaTeX/abnTeX2)

Não foi detectada uma instalação de LaTeX nesta máquina. Para compilar
`02_dissertacao/main.tex` localmente, instale o
[MiKTeX](https://miktex.org/) (ele instala pacotes como `abntex2` sob demanda)
ou use o [Overleaf](https://www.overleaf.com/), enviando a pasta
`02_dissertacao/`.
