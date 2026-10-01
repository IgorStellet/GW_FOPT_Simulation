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

## Revisão para execução no CHE

Revisão local após sincronizar a branch e preservar o commit do usuário.
**47 testes passaram**, com 31 avisos dos exemplos gráficos existentes. O
backend Agg agora é configurado automaticamente nos testes; no Windows,
conftest usa uma pasta exclusiva por execução para evitar a ACL do temporário
global. O cache pytest fica em Articles/results, ignorado pelo Git.

```bash
python -m pytest -q --disable-warnings
python -m ruff check Articles tests/conftest.py tests/test_article_core.py tests/test_article_scan.py tests/test_article_cluster.py
bash -n Articles/cluster/smoke.slurm
bash -n Articles/cluster/production.slurm
```

A suíte passa sem redirecionar manualmente PYTEST_DEBUG_TEMPROOT. Agora há 6
testes de núcleo, 9 de coleta e 4 de cluster, além da suíte preexistente.
As novas regressões verificam cobertura exata das partes, checkpoints fechados
em journal DELETE, recusa de WAL pendente, restauração de observáveis e retomada
após reparar um registro sem aumentar a quantidade de pontos.

| Arquivo/conjunto | Conferência |
|---|---|
| combined_model.py | Leitura das prescrições, testes de massas/domínio e duas chamadas reais |
| collect_data.py | Grade/partes, fases/bounces reais, paralelismo, gravação e retomada |
| cluster_runner.py | Lotes reais de um ponto, cópia persistente e retomada para outra pasta local |
| __init__.py | Importação e análise sintática |
| .gitignore | Resultados continuam excluídos do versionamento |
| README/QUICKSTART/CLUSTER_CHE | Chamadas, unidades, opções e referências conferidas |
| VALIDATION.md | Atualização com evidências e limitações desta revisão |
| test_article_core.py/test_article_scan.py | Testes científicos e de persistência passaram |
| test_article_cluster.py/conftest.py | Checkpoints e preparação do ambiente de testes passaram |
| smoke.slurm/production.slurm | Sintaxe Bash e finais de linha LF conferidos |
| Núcleo alterado anteriormente | Suíte completa e nova comparação dos observáveis reais |

Nova chamada real na grade de dois pontos:

```bash
python -m Articles.collect_data --output Articles/results/cluster_local_smoke --m6 1000 1000 5 --C 0 3.35 3.35 --m8 668.740304976422 --no-baselines --beta-check --workers 2
```

C=0: no_nucleation_found, cerca de 38 s; C=3.35: nucleated, cerca de 81 s,
sem warnings nos dois pontos. Para o ponto nucleado:

| Quantidade | Valor |
|---|---:|
| Tn [GeV] | 40.73800650428223 |
| alpha_energy | 0.458758890225597 |
| alpha_trace | 0.33289431716939344 |
| beta/H | 116.72234483860271 |
| f_sw_peak [Hz] | 0.00091334605711425 |
| h²Omega_sw_peak | 8.061022207395195e-12 |

Também executado cluster_runner --smoke --beta-check --batch-size 1: os mesmos
dois pontos foram calculados em lotes, com cerca de 30 s e 73 s. Checkpoints
foram publicados após cada ponto. A retomada em outro work-dir restaurou dois
registros e executou zero pontos novos. integridade SQLite=ok; alpha_energy,
alpha_trace, beta/H, amplitude e frequência coincidem entre banco, CSV,
detalhes, execução direta, checkpoint e restauração.

O banco ativo local usa WAL; a cópia persistente usa DELETE. Os scripts usam
um nó e uma tarefa com CPUs por tarefa, e a produção divide a grade em 100
partes disjuntas. O sinal antecipado solicita parar após o lote; não garante
concluir um lote antes de SIGKILL. O último lote não publicado pode ser repetido.

**Não houve execução no CHE nem submissão da campanha grande nesta revisão.**
Login/endereço efetivos, Python, partição, quota, scratch e recursos precisam
ser conferidos com a conta disponível. A validação Windows/Bash local não
certifica configuração do scheduler nem convergência física da malha completa.

## Correção da instalação e da chamada do runner — 01/10/2026

Revisão baseada no commit `c79361e`, preservando as alterações do usuário.
O README não foi modificado. A falha do CI era `ModuleNotFoundError: Articles`:
o pacote do artigo não fazia parte da instalação. Foi reproduzida localmente
com o executável `pytest`, mesmo com os testes passando via `python -m pytest`.

A instalação agora inclui `Articles` e `CosmoTransitions`, as duas tabelas
térmicas NPZ e a licença `LICENSE.txt`. Instalação editável, construção de wheel
e instalação do wheel em uma pasta isolada foram verificadas. As importações,
as splines e os hashes dos arquivos instalados funcionaram fora do layout
`ROOT/src`. As entradas `--help`/`--dry-run` funcionaram fora do checkout.
`pip check` não encontrou dependências inconsistentes.

**50 testes passaram** com o executável `pytest` (31 avisos dos exemplos
gráficos). São 6 testes de núcleo, 10 de coleta, 6 de cluster e 28 anteriores.
A saída esperada da rejeição de pastas aninhadas é agora capturada e conferida.
O runner sem argumentos mostra ajuda, retorna zero e não inicia uma campanha;
uma chamada incompleta continua retornando erro de uso. Ruff dos arquivos do
artigo/testes passou; a sintaxe dos três scripts Bash/Slurm foi conferida.

O piloto real foi executado novamente:

```bash
python -m Articles.cluster_runner --smoke --beta-check --batch-size 1 --workers 2 --output Articles/results/cluster_fix_20261001 --work-dir Articles/results/cluster_fix_scratch_20261001
```

C=0: `no_nucleation_found`, 34.6 s; C=3.35: `nucleated`, 77.1 s.
Tn=40.73800650428223 GeV, alpha_energy=0.458758890225597,
alpha_trace=0.33289431716939344, beta/H=116.72234483860271,
beta/H com meio passo=116.04593094770735,
f_sw_peak=0.00091334605711425 Hz e h²Omega_sw_peak=8.061022207395195e-12.
Todos esses valores coincidiram entre SQLite, CSV e detalhes comprimidos.
A integridade foi `ok` no banco local, checkpoint persistente e restauração.
O checkpoint usa DELETE; os bancos locais usam WAL. A retomada em
`cluster_fix_restored_20261001` restaurou os dois pontos sem recalculá-los.

O script `cluster/setup_environment.sh` prepara uma venv nova com Python >=3.11
e instala pip por `ensurepip`, inclusive quando o Python base não tem pip.
Recusa sobrescrever ambientes existentes. O CI passa a conferir esse cenário
em Linux/Python 3.11, além das entradas instaladas em 3.11/3.12/3.13.
O Python 3.11.7 informado pelo usuário no CHE atende ao requisito; falta
confirmar a preparação da venv e o piloto na conta do cluster. Não houve
submissão remota nem execução da campanha completa nesta correção.
