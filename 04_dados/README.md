# Dados

Vídeos de teste, datasets e anotações (ground truth) usados nos experimentos.

> ⚠️ Vídeos de videomonitoramento podem conter dados pessoais/sensíveis
> (pessoas identificáveis). Antes de armazenar ou versionar:
> - Verificar requisitos de anonimização/consentimento (LGPD).
> - Evitar sincronizar arquivos grandes/sensíveis via OneDrive sem
>   necessidade — considerar armazenamento local separado ou criptografado.

## Organização sugerida

```
04_dados/
├── raw/          Vídeos originais (não versionar / não sincronizar se sensíveis)
├── anotacoes/    Ground truth (descrições, timestamps, bounding boxes)
└── amostras/     Pequenos clipes de exemplo para desenvolvimento
```
