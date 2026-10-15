# Validação v3: continuidade com inconclusivos e CHE — 15/10/2026

## Diagnóstico da execução local do usuário

Foram lidos, sem modificar, os resultados em
C:/Users/igora/PycharmProjects/GW_FOPT_Simulation/Articles/Articles/results/local_adaptive.
O log anexado estava cortado em [21]; os JSON/CSV/detalhes confirmaram a parada:

| m8 | Pontos | Estado | Ponto da interrupção (m6=inf) |
|---|---:|---|---|
| inf | 36 | unresolved | C=3.19, IntegrationError no bounce |
| 840.8964152537145 | 37 | unresolved | C=3.13, IntegrationError no bounce |
| 668.740304976422 | 41 | unresolved | C=3.03, S3/T=117.558 fora da tolerância |

Nenhuma dessas três fatias chegou a m6 finito. A execução era produção, sem
--smoke, com beta_check=False; não foi uma campanha completa nem a aprovação
do piloto. Os poucos pontos decorriam da interrupção estrita em unknown.

A v3 continua nos C/m6 seguintes sem contar unknown como no, após uma tentativa
numérica refinada limitada. O estado final é complete_with_unresolved se
restarem pendências. Uma coluna com unknown não pode encerrar m6 como vazia.
O padrão de três ausências verificadas após semente permanece. Ações negativas
são rejeitadas e registradas, sem alimentar a raiz como uma amostra física.
Não foram modificados arquivos do núcleo nem o potencial/ressoma.

## Verificações automatizadas

```bash
python -m pytest -q Articles/tests tests/test_article_core.py tests/test_article_scan.py tests/test_article_cluster.py -k "not slurm_scratch_selection"
python -m ruff check Articles --exclude Articles/results
```

72 testes Python passaram na seleção local. Os cinco casos que iniciam Git Bash
ficaram fora da seleção Windows, conforme a limitação de sandbox já registrada
abaixo. Isso não é uma execução Bash aprovada. A CI Linux deve validar os
templates na revisão publicada.

Casos novos: unknown entre duas negativas quebra a sequência; desconhecido no
bootstrap não impede encontrar semente; três colunas com unknown não podem
encerrar o eixo antes de uma massa menor nucleada; todos os unknowns respeitam
os bounds; checkpoint restaura e segue outras massas sem repetir erro salvo;
runner publica os dados antes de retornar código 3; diagnóstico distingue zero
colunas finitas de conclusão; refinamento conserva alvo/tolerância S3/T e beta;
falha persistente não é convertida em ausência; ação negativa é preservada e
rejeitada. O modo estrito continua testado com --on-unresolved stop.

O run_local.py foi conferido por execução direta com --dry-run a partir da
pasta Articles, usando --output relativo: as três fatias foram validadas na
raiz correta sem Articles/Articles. Nenhum bounce foi executado nesse check.

## Testes físicos dirigidos

C=1, m6=550 GeV, m8=inf, daisy ligado e beta-check produziu:

- Tn=75.23051156 GeV;
- S3/T=140.05327441, residual 0.05327441 (tolerância 0.5);
- alpha_trace=0.0308918253;
- beta/H=791.513954; com meio passo, 768.289557 (diferença relativa 2.93%).

O ponto foi avaliado de novo, sem reaproveitar um banco. A faixa da dissertação
em C=1 usa c6/f²=3.2..3.45 TeV^-2, equivalente a m6~538..559 GeV, dentro dos
bounds. A figura da dissertação usa daisy desligado, portanto não se promete
reproduzir seus valores numéricos com a prescrição atual. C=1 e C=3.2 agora
pertencem exatamente à grade, além dos passos ancorados em 3.83.

As falhas C=3.19 e 3.13 foram reproduzidas; refinar fases não garante repará-las.
Não são ausências físicas. No teste dirigido com fases refinadas, C=3.03,
m8=668.740304976422 recuperou nucleação aceita, S3/T=139.84143688, mas beta
continuou não resolvido. Isso não autoriza inventar um espectro nesse ponto.
Os testes dirigidos não substituem a campanha completa nem provam ausência de
ilhas desconectadas. Os detalhes reais usados na investigação foram mantidos
em Articles/results/validation_v3 (ignorados pelo Git).

## Execução no CHE

A configuração informada pelo usuário revelou milliways-ext com MaxTime=15 h,
milliways01/03 em mix e milliways02 em down*. O ponto de benchmark levou cerca
de 59 s no Milliways, contra 429.6 s no c003. Não é uma extrapolação de tempo da
campanha. Templates v3: três jobs de uma CPU, piloto 2 h, produção 14 h, sinal
USR1 com 15 minutos de aviso e lote padrão 1. Memória preservada em 4/8 GB.
A partição/nó podem ser sobrescritos na submissão. Procedimento atualizado em
CLUSTER_CHE.md; destino NOVO smoke_v3 e production_v3. Não reutilize manifestos
v2 modificando hashes. A campanha v3 completa não foi executada aqui.

---

## Histórico v2 (não são os comandos nem os estados atuais)

# Validação v2: semente por m6 e gráficos — 13/10/2026

Esta revisão substitui a continuação por linhas de C por colunas de m6.
A primeira nucleação de cada massa é verificada em sua própria busca em C,
inclusive no salto m6=inf -> 2000 GeV, antes da regra de três ausências.
As mudanças de fontes continuam exclusivamente em Articles. O núcleo e
as fórmulas físicas foram preservados.

## Testes automatizados atuais

```bash
python -m pytest -q Articles/tests
python -m pytest -q Articles/tests tests/test_article_core.py tests/test_article_scan.py tests/test_article_cluster.py -k "not slurm_scratch_selection"
python -m ruff check Articles
```

**61 testes passaram** na seleção Python: 34 em Articles/tests e 27 existentes;
Ruff passou. Os cinco testes antigos que iniciam Git Bash foram excluídos do
comando local por sua restrição conhecida no sandbox Windows. Essa exclusão
não constitui uma execução Bash aprovada. A CI Linux mantém a verificação da
suíte existente e dos templates. Ambiente local: Python 3.12.14, NumPy 2.5.4,
SciPy 1.18.1 e Matplotlib 3.11.2.

Os novos casos controlados verificam:

- Primeiro m6 finito exige **sete C sem nucleação antes de encontrar a semente**;
  a busca não para no terceiro. Isso também vale com o limite de passos do piloto.
- Uma mudança abrupta entre duas massas finitas exige outra busca de semente.
- Depois de esgotar C descendente, a busca acima da candidata encontra uma
  faixa deslocada para C maior, sem supor monotonicidade entre massas.
- Cada m8 verifica sua própria semente, independentemente das outras fatias.
- Só depois da semente se usa a regra de três ausências em C; um hit reinicia
  o contador. Uma coluna m6 só é vazia depois de esgotar todos os C da grade.
- A parada em m6 exige três colunas vazias depois de nucleação verificada;
  colunas iniciais vazias não encerram a busca antes da primeira semente.
- Falhas e limites de lote não viram ausências físicas. Retomada preserva
  a ordem, evita repetir avaliações confirmadas e conserva matrizes/estado.

## Piloto local e do cluster com gráficos

O teste real preparado para o usuário é:

```bash
python -m Articles.run_local --smoke --beta-check --output Articles/results/pilot_adaptive_v2
```

Ele calcula m6=inf e 2000 GeV em cada um dos três m8. A busca de semente não
é truncada. Depois dela, o piloto avalia até três passos de C por lado.
`pilot_step_limit` indica uma faixa ainda aberta; não é fronteira física.
A execução real desse piloto v2 completo está a cargo do usuário, antes da
produção. A campanha completa e a submissão remota no CHE não foram iniciadas.

O executor em lotes foi exercitado com um avaliador controlado: publicou o
banco, o estado v2 e os gráficos PNG/SVG; ao restaurar em outra pasta não
repetiu avaliações. Nesse caso a coluna finita só nucleia em C=3.69 após
sete ausências a partir de C=3.83. O teste não é uma nova medição física.

Os mapas mostram nucleação, ausência, inconclusivo e não calculado; marcam
sementes, três ausências e o limite de janela do piloto. Os espectros são
reconstruídos sem bounces: frequências Hz são convertidas para mHz ao chamar
o núcleo, usando alpha_trace assinado, epsilon_turb=0.05 e kappa_coll=0.
Observáveis não resolvidos são omitidos dos espectros com motivo explícito.
A comparação dos detectores usa picos acústicos e curvas PIS, em outro painel;
PIS não é tratada como ruído ponto a ponto nem como SNR total.

PNG/SVG foram gerados e inspecionados visualmente. Os testes conferem que o
SQLite físico não muda ao desenhar, que alpha inválido não vira espectro,
que uma campanha sem observáveis resolvidos produz diagnóstico explícito e
que três fatias podem ser comparadas sem fundir seus bancos. As amostras
espectrais são exportadas para CSV em Hz e selection.json registra a seleção.

Também foram redesenhados espectros dos **dados reais já salvos da v1**, sem
recalcular as transições, para conferir o pós-processamento. Esses pontos não
são apresentados como um scan novo da revisão v2.

A seleção dos imports aceita tanto o pacote instalado como src na raiz do
checkout. Foi conferido --dry-run nos três cenários e o modo pela raiz sem
src no PYTHONPATH, preservando o ajuste recente do usuário para esse uso.
O mesmo núcleo efetivamente importado continua identificado no manifesto.

## Limites e retomada

Use uma pasta nova: a identidade agora é adaptive_v2_verified_columns.
Não reutilize os resultados v1 mudando seu manifesto. O bootstrap pode custar
mais quando não há nucleação, porque verifica todo o eixo C antes de declarar
uma massa vazia. Bounds e --max-points limitam o trabalho; nunca se corta uma
busca de primeira semente apenas por três ausências.

Faixas menores que o passo e ilhas além das podas posteriores à semente
continuam exigindo refinamento dirigido. O piloto não pretende fechar toda
a faixa de C. QUICKSTART.md e CLUSTER_CHE.md descrevem os limites, figuras e
como encaminhar a produção depois da conferência do piloto.

---

# Histórico da v1 (busca por linhas de C)

Os registros abaixo descrevem a versão anterior e não o fluxo ou os comandos
recomendados para o piloto v2 atual.

# Validação do scan adaptativo — 13/10/2026

Esta revisão altera somente `Articles`. Os arquivos do núcleo, o script da
dissertação, o modelo combinado e os testes anteriores foram preservados.
As evidências históricas da grade estão ao final, separadas desta validação.

## Teste 1: lógica, dados e retomada

```bash
python -m pytest -q Articles/tests
python -m pytest -q Articles/tests tests/test_article_core.py tests/test_article_scan.py tests/test_article_cluster.py -k "not slurm_scratch_selection"
python -m ruff check Articles
```

Resultado: **48 testes passaram** na suíte Python selecionada, incluindo
21 testes adaptativos. Ruff passou. Python 3.12.14, NumPy 2.5.4, SciPy 1.18.1.
Os cinco casos antigos `slurm_scratch_selection` foram tentados e bloqueados
pelo sandbox Windows ao iniciar Git Bash (`NtCreateDirectoryObject`, acesso
negado). Por isso foram excluídos da execução Python final; não são cinco
verificações Slurm aprovadas. A suíte completa de exemplos gráficos não foi
validada neste ambiente, que não dispõe de matplotlib no ambiente isolado.

Cobertura adaptativa: três ausências consecutivas em ambas as direções;
reinicialização por hit; três linhas vazias em C; bounds e semente decimal
exata; bootstrap antes da poda em C; duas linhas no piloto; erros inconclusivos;
limite por lote; matrizes; checkpoints fechados e restauração em outra pasta;
três processos com saídas disjuntas; publicação do diagnóstico antes de parar.
Também cobre retry de beta/GW persistentemente não resolvidos: um ponto é
repetido no máximo uma vez por execução do runner, mesmo atravessando lotes.

## Teste 2: fases e bounces reais, três m8

```bash
python -m Articles.run_local --smoke --beta-check --output Articles/results/adaptive_local_pilot
```

Os três processos terminaram com saída 0 e estado `pilot_complete`, com
**duas linhas completas por fatia**, totalizando **32 avaliações reais**:

| m8 [GeV] | Semente verificada C | Linhas completas C | Avaliações, incluindo bootstrap |
|---|---:|---|---:|
| inf | 3.83 | 3.83, 3.81 | 8 |
| 840.8964152537145 | 3.79 | 3.79, 3.77 | 10 |
| 668.740304976422 | 3.71 | 3.71, 3.69 | 14 |

Todas as sementes usam m6=inf. Em cada uma das seis linhas, m6 finito
`2000, 1250, 500` GeV produziu três ausências consecutivas. O fim do eixo
coincide com o terceiro ponto no piloto; a verificação de que um quarto ponto
é efetivamente pulado está no Teste 1. As matrizes contêm respectivamente
2 hits e 6, 8, 12 ausências, incluindo os pontos do bootstrap. Células não
avaliadas permanecem `not_evaluated`, sem valores físicos inventados.
SQLite passou `integrity_check`; contagens de matrizes e banco coincidiram.

Referências numéricas com m6=inf:

| m8 [GeV] | C | Tn [GeV] | beta/H, h=0.5 GeV |
|---|---:|---:|---:|
| inf | 3.83 | 43.9415109 | 204.996110 |
| inf | 3.81 | 50.6138941 | 378.666701 |
| 840.8964152537145 | 3.79 | 39.8045600 | 130.771179 |
| 840.8964152537145 | 3.77 | 48.0934738 | 308.942176 |
| 668.740304976422 | 3.71 | 42.5506616 | 199.046854 |
| 668.740304976422 | 3.69 | 49.3002461 | 334.357882 |

Em m8≈840.90 GeV, C=3.79, `beta_step_sensitivity` marcou os observáveis como
não resolvidos, embora a nucleação tenha sido aceita. O scan manteve o hit,
sem converter a instabilidade de beta em ausência de nucleação. Para o ponto
puro C=3.83, S3/Tn≈139.982752, dentro da tolerância absoluta de 0.5.

O piloto foi executado durante o desenvolvimento; seu manifesto registra os
hashes efetivamente usados. Ajustes posteriores na escolha da semente entre
linhas e no controle de retry têm testes próprios. Não se deve reutilizar
seus diretórios com fontes alteradas, nem editar o fingerprint.

## Executor em lotes

```bash
python -m Articles.cluster_runner --smoke --scenario-index 0 --beta-check --batch-size 2 --output Articles/results/adaptive_runner_pilot --work-dir Articles/results/adaptive_runner_scratch
```

Concluiu oito avaliações reais em quatro lotes e publicou os checkpoints.
A chamada seguinte na mesma execução encontrou zero pontos pendentes.
Esse teste usa o executor do cluster no Windows, sem constituir submissão
Slurm. Os templates atuais usam três tarefas de uma CPU, array 0-2%3, com
pastas m8-0/1/2. O helper de scratch já existente foi preservado.

Com o código final, foi executado também:

```bash
python -m Articles.cluster_runner --scenario-index 0 --C 3.83 3.83 0.02 --m6 2000 2000 5 --beta-check --batch-size 1 --output Articles/results/final_checkpoint_probe --work-dir Articles/results/final_checkpoint_scratch
python -m Articles.cluster_runner --scenario-index 0 --C 3.83 3.83 0.02 --m6 2000 2000 5 --beta-check --batch-size 1 --output Articles/results/final_checkpoint_probe --work-dir Articles/results/final_checkpoint_restored
```

A primeira chamada calculou dois pontos reais em dois lotes, encerrando como
`complete`; a segunda restaurou o checkpoint em outra pasta e calculou zero
pontos. Não houve mudança de observáveis. Esta campanha curta termina nos
bounds e não substitui o piloto de três ausências.

## Regressão real do ponto combinado

```bash
python -m Articles.collect_data --scenario-index 2 --seed-C 3.35 --seed-m6 1000 --C 3.35 3.35 0.02 --m6 1000 1000 5 --beta-check --output Articles/results/final_combined_probe
```

Com o código final, a semente finita m6=1000 GeV, m8=668.740304976422 GeV,
C=3.35 reproduziu Tn=40.7380065043 GeV, alpha_trace=0.3328943172 e
beta/H=116.72234484, sem quality_flags. S3/Tn=140.11741377 foi aceito.
A busca na direção de massa maior também avaliou m6=inf, com nucleação
aceita em Tn=90.2844125271 GeV. A campanha terminou nos bounds.

## Relação com a dissertação e limites

Foi consultado o capítulo 5, seções 5.6–5.7 (pp. impressas 155–168; PDF
163–176), especialmente a janela de nucleação dos mecanismos combinados,
a cooperação de deformações pequenas e a degenerescência de observáveis.
As faixas publicadas na dissertação não são usadas como um resultado físico
universal no novo programa; a semente e os pontos de parada são calculados
com as prescrições atuais.

O scan completo não foi executado. A hipótese de continuação pode perder
ilhas desconectadas ou faixas mais estreitas que os passos escolhidos.
O piloto grosseiro não certifica fronteiras nem convergência física. O teste
real no CHE permanece necessário na conta do usuário antes da produção.

---

# Histórico anterior: grade cartesiana

Os registros abaixo pertencem à implementação anterior (até a revisão
`ee96f52`) e não descrevem o padrão atual. Comandos históricos do runner
podem usar opções que tinham outro significado naquela versão. Para as
chamadas atuais, consulte QUICKSTART.md e CLUSTER_CHE.md.

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

## Compatibilidade das campanhas e temporário do job — 01/10/2026

O erro local de retomada foi associado a uma campanha antiga em
`Articles/results/combined`. O manifesto registra 24/09/2026; a comparação
identificou alteração no coletor e inclusão posterior do runner. A proteção
contra misturar resultados de códigos/ambientes/configurações diferentes foi
mantida. Agora ela lista as diferenças e consulta o banco somente para leitura
antes de configurá-lo para escrita. `--dry-run` também confere essa compatibilidade
e não cria campanha. Os eixos padrão usam floats, assim escrever explicitamente
os mesmos valores na chamada não produz uma incompatibilidade artificial.

O log enviado do CHE para o job **107761** mostra que o Python iniciou, mas a
criação de `/scratch/local/ftovar/...` falhou com PermissionError em `/scratch`.
Os dois templates agora carregam `cluster/job_environment.sh`: usam a escolha
explícita FOPT_SCRATCH_ROOT, ou procuram SLURM_TMPDIR, TMPDIR e /tmp. Criam uma
pasta exclusiva com mktemp e imprimem o caminho. Uma escolha explícita inválida
é rejeitada sem tentar criar sua raiz. Falhas ao preparar pastas na chamada
direta do runner também recebem uma mensagem curta. O guia captura o número
real retornado por sbatch, evitando usar JOB_ID literalmente.

**60 testes passaram**, com 31 avisos dos exemplos anteriores: 6 de núcleo,
13 de coleta, 13 de cluster e 28 anteriores. As regressões novas cobrem banco
incompatível sem alteração dos arquivos/journal, diagnóstico das diferenças,
dry-run sem cálculo/gravação, equivalência das chamadas padrão/explícita,
falta de permissão na pasta de trabalho, escolhas de temporário e recusa de
checkpoint incompatível sem substituir os dados persistentes. Ruff passou;
a sintaxe dos quatro scripts Bash/Slurm foi conferida.

Novo piloto executado:

```bash
python -m Articles.collect_data --output Articles/results/campaign_fix_20261001 --m6 1000 1000 5 --C 0 3.35 3.35 --m8 668.740304976422 --no-baselines --beta-check --workers 2
```

C=0: `no_nucleation_found`, 39.6 s; C=3.35: `nucleated`, 84.6 s.
Tn=40.73800650428223 GeV, alpha_energy=0.458758890225597,
alpha_trace=0.33289431716939344, beta/H=116.72234483860271,
f_sw_peak=0.00091334605711425 Hz e h²Omega_sw_peak=8.061022207395195e-12.
Esses observáveis coincidem entre SQLite, CSV e detalhes comprimidos; os inputs
do espectro e amostras da ação também estão nos detalhes. Integridade SQLite=ok.
Repetir o comando executou zero pontos novos; dry-run aceitou a campanha atual
e rejeitou corretamente uma anterior, listando os arquivos alterados.

O checkpoint foi publicado em `slurm_fix_20261001/smoke` e o próprio smoke.slurm
foi executado localmente com Git Bash, variáveis Slurm simuladas e temporários
preferenciais ausentes. Escolheu /tmp, restaurou os dois pontos, executou zero
novos e publicou o checkpoint. Isso confere o fluxo do script e do runner;
**não é uma submissão Slurm nem uma execução validada no CHE**. É necessário
submeter novamente o piloto atualizado na conta e conferir seus logs antes da
campanha grande. Os resultados locais continuam ignorados pelo Git.
