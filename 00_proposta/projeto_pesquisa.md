# Projeto de Pesquisa (rascunho)

> Documento de trabalho para alinhamento com o orientador. Após consolidado,
> deve ser formatado em ABNT (ver `02_dissertacao/`).

## Título (provisório)

Busca Semântica em Gravações de Vídeo de Longa Duração por Linguagem Natural
Utilizando Modelos de Visão-Linguagem de Código Aberto

## Contextualização

Sistemas de videomonitoramento (VSM) geram grandes volumes de gravações
contínuas (ex.: 12h por câmera). A localização manual de eventos ou pessoas
específicas nessas gravações é demorada e sujeita a erros. Soluções
comerciais de busca semântica em vídeo costumam depender de VLMs
proprietários, cujo licenciamento inviabiliza a incorporação em produtos
comerciais sem repasse de royalties.

## Problema de pesquisa

Como viabilizar, em um sistema de videomonitoramento (VSM desenvolvido em Qt
com arquitetura de microsserviços), uma busca por linguagem natural via chat
(ex.: "homem de camisa branca e boné vermelho") em gravações de vídeo de
longa duração (~12h), retornando os frames correspondentes com o grau de
confiança associado, utilizando exclusivamente modelos de visão-linguagem
(VLMs) com licença que permita uso e comercialização sem custos de
licenciamento/royalties?

## Objetivo geral

Desenvolver e avaliar um pipeline de busca semântica em vídeos de longa
duração baseado em VLM(s) de licença permissiva, capaz de localizar
ocorrências descritas em linguagem natural via chat e retornar os frames
correspondentes com grau de certeza associado, projetado como microsserviço
integrável a um sistema de videomonitoramento (VMS).

## Objetivos específicos

1. Levantar e comparar VLMs open-source com licenças permissivas (ex.: Apache
   2.0, MIT) adequados para reconhecimento de atributos visuais (pessoas,
   vestimentas, cores, objetos) em vídeo.
2. Definir uma estratégia de amostragem/processamento de frames para vídeos
   de longa duração que seja computacionalmente viável.
3. Projetar um pipeline de indexação (embeddings/legendas) e busca por
   similaridade semântica a partir de consultas em linguagem natural.
4. Implementar um microsserviço (API) que receba a consulta via chat e
   retorne os frames correspondentes com timestamp e grau de confiança.
5. Integrar o microsserviço a um protótipo da interface do VSM (Qt) para
   validação ponta a ponta.
6. Avaliar precisão, recall e latência da solução em um conjunto de vídeos de
   teste com anotações de referência (ground truth).

## Justificativa

- **Técnica**: viabilidade de busca semântica em vídeo com modelos abertos
  ainda é pouco consolidada para cenários de vídeo longo (horas) versus
  imagens/clipes curtos, foco da maioria dos benchmarks atuais.
- **Econômica/comercial**: uso de modelos com licença permissiva remove
  barreira de custo de licenciamento para o produto VSM, viabilizando
  comercialização.
- **Aplicada**: resultado integra diretamente a um software em
  desenvolvimento (VSM em Qt/microsserviços), com potencial de uso real.

## Metodologia (esboço)

- Natureza: pesquisa aplicada, com desenvolvimento experimental.
- Revisão bibliográfica: VLMs, retrieval multimodal (texto-imagem/vídeo),
  busca semântica em vídeo de longa duração, sistemas de videomonitoramento.
- Seleção de VLM(s) candidatos considerando: licença (uso comercial sem
  royalties), desempenho em reconhecimento de atributos, custo computacional
  (inferência em CPU/GPU local vs. nuvem).
- Construção do pipeline:
  1. Extração/amostragem de frames do vídeo.
  2. Geração de embeddings e/ou legendas (captions) por frame/cena via VLM.
  3. Indexação vetorial (banco de vetores) dos embeddings.
  4. Recebimento da query em linguagem natural via chat.
  5. Busca por similaridade semântica + ranking por score de confiança.
  6. Retorno dos frames/timestamps correspondentes.
- Implementação como microsserviço (Python/FastAPI), consumido pelo
  aplicativo Qt.
- Avaliação experimental: dataset de vídeos com anotações de
  referência (ground truth) para métricas de precisão, recall e latência.

## Candidatos a VLM (a investigar)

Ver `01_revisao_bibliografica/mapeamento_vlms.md`.

## Restrições e premissas

- O VLM escolhido deve ter licença que permita uso comercial sem pagamento de
  royalties/licenciamento (ex.: Apache 2.0, MIT). Modelos com licenças
  restritivas para uso comercial (ex.: alguns modelos LLaMA-based com
  cláusulas específicas) devem ser avaliados com cautela.
- O resultado deve ser consumível como microsserviço pelo software VSM (Qt),
  ou seja, expor uma API (ex.: REST/gRPC).
- Vídeos de teste de ~12h impõem restrições de custo computacional —
  estratégias de amostragem de frames e processamento em lote/offline devem
  ser consideradas.

## Cronograma (a definir com orientador)

| Etapa | Período (a definir) |
|---|---|
| Revisão bibliográfica e seleção de VLM | |
| Definição do pipeline e protótipo inicial | |
| Implementação do microsserviço | |
| Construção/seleção do dataset de avaliação | |
| Experimentos e avaliação | |
| Integração com VSM (protótipo) | |
| Escrita da dissertação | |
| Qualificação | |
| Defesa | |

## Abertos / pontos para discutir com o orientador

- Definir o programa de pós-graduação, formato exigido (ABNT específico da
  instituição) e prazos.
- Confirmar se haverá acesso a vídeos reais de 12h para teste, ou se será
  necessário compor/simular um dataset.
- Definir critério objetivo de "grau de certeza" (score de similaridade do
  embedding? confiança do VLM? calibração?).
