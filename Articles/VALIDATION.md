# Validação da implementação

Verificação realizada em 24/09/2026, Windows, Python 3.12.14, NumPy 2.5.3,
SciPy 1.18.1. O manifesto de cada execução registra as versões efetivamente
usadas. Esta verificação testa a implementação e pontos selecionados; não
certifica convergência científica de toda a malha.

## Testes automatizados

**42 testes passaram**, incluindo os testes existentes do repositório e os
14 testes novos em `tests/test_article_core.py` e `tests/test_article_scan.py`.
Os testes de exemplos existentes requerem backend gráfico não interativo no
ambiente de validação e emitem 31 avisos sobre figuras que não podem ser mostradas.

```bash
MPLBACKEND=Agg python -m pytest -q --disable-warnings
python -m ruff check Articles tests/test_article_core.py tests/test_article_scan.py
```

No PowerShell, use `$env:MPLBACKEND='Agg'` antes do comando pytest. A primeira
tentativa sem essa configuração encontrou ausência de Tcl/Tk no ambiente;
com `Agg`, a suíte completa passou. A verificação de estilo dos arquivos novos
também passou.

As regressões cobrem: beta calculado a partir de S3/T com exatamente duas
avaliações; preservação da opção de quarta ordem; rejeição de stencil inválido;
distinção entre energia liberada e anomalia do traço; refinamento de mínimos;
unidades mHz/Hz na turbulência; preservação de pares campo/temperatura na
interpolação; matching do potencial; domínio logarítmico; broadcasting;
tolerância de nucleação; limites puros; registro de falhas; retomada; trava
contra dois escritores; troca atômica dos detalhes e exportação.

O potencial também foi comparado diretamente ao script da dissertação em
656 avaliações de campo/temperatura, distribuídas entre SM, EFT, potencial
combinado e C=10. A diferença máxima encontrada foi aproximadamente
0.0013 GeV⁴, com as mesmas prescrições e a ressoma gauge ligada. Essa comparação
não importa o script antigo no código de produção.

## Teste com fases e bounces reais

Comando executado, incluindo paralelismo por processos no Windows:

```bash
python -m Articles.collect_data --output Articles/results/validation --m6 1000 1000 5 --C 0 3.35 3.35 --m8 668.740304976422 --no-baselines --beta-check --workers 2
```

Foram registrados dois pontos. O ponto C=0 terminou como
`no_nucleation_found`, sem exceção numérica. Para C=3.35:

| Quantidade | Resultado aproximado |
|---|---:|
| Tc | 77.52170 GeV |
| Tn | 40.73801 GeV |
| S3/Tn - 140 | 0.11741 |
| alpha_trace | 0.332894 |
| beta/H, passo 0.5 GeV | 116.72234 |
| Variação relativa de beta ao usar 0.25 GeV | 0.005795 (0.58%) |
| Frequência do pico acústico | 0.00091335 Hz |

Esse teste satisfaz a tolerância residual de 0.5 e não apresentou avisos
numéricos. O resultado é compatível entre execução serial e paralela. Os
tempos locais foram cerca de 33 s e 76 s para esses dois pontos; não se deve
extrapolar esses tempos para toda a malha.

Executar novamente o mesmo comando reutilizou os dois registros, sem
recalcular os pontos. `--export-only` também foi verificado. Os dados gerados
ficam em `Articles/results/`, excluídos do versionamento; a campanha completa
de 453.906 pontos **não foi executada** nesta validação.

Antes dos resultados de publicação, ainda cabe verificar convergência nas
fronteiras e regiões selecionadas, validade das hipóteses do potencial e dos
templates GW, além da prescrição de detectabilidade escolhida para o artigo.
