# CHE: ambiente, piloto, campanha e download

Os comandos abaixo são executados **no CHE**, exceto a última seção, marcada
PowerShell. Copie uma linha inteira por vez. Os dados físicos ficam no CHE;
Git recebe apenas código, testes e documentação.

## 1. Prepare um checkout separado

Se há jobs antigos em execução, mantenha o checkout deles intacto. Uma cópia
separada evita que um job importe arquivos de outra branch durante a rodagem.

```bash
git clone --depth 1 --single-branch --branch codex/article-fopt-combined https://github.com/IgorStellet/GW_FOPT_Simulation.git "$HOME/projects/GW_FOPT_combined_T2"
cd "$HOME/projects/GW_FOPT_combined_T2"
module load python/3.11.7-gcc-12.2.0
bash article_FOPT_combined/cluster/setup.sh
```

O setup cria `article_FOPT_combined/.venv`, instala requirements e executa os
testes locais. Ele usa `venv/ensurepip`, resolvendo a situação em que o Python
do módulo existe mas o comando `python3 -m pip` ainda não existe.

Se a instalação precisar de acesso externo indisponível no login do CHE,
guarde o erro completo; as dependências também podem ser preparadas pelo
administrador em um wheelhouse compatível com Python/arquitetura do cluster.

Se já usou esse checkout no CHE, atualize a branch depois que os jobs
anteriores terminarem. Com o ambiente Python já instalado, a instalação
pode ser reaproveitada.

```bash
cd "$HOME/projects/GW_FOPT_combined_T2"
git pull --ff-only origin codex/article-fopt-combined
```

Em **cada novo login**, defina:

```bash
cd "$HOME/projects/GW_FOPT_combined_T2"
module load python/3.11.7-gcc-12.2.0
export FOPT_PYTHON="$PWD/article_FOPT_combined/.venv/bin/python"
export FOPT_RESULTS_ROOT="$HOME/fopt-combined-T2-v2"
"$FOPT_PYTHON" -c 'import sys,numpy,scipy; print(sys.executable); print("numpy",numpy.__version__,"scipy",scipy.__version__)'
```

`FOPT_RESULTS_ROOT` precisa ser uma pasta persistente autorizada para sua
conta. O exemplo usa seu home, onde a quota precisa comportar os dados.
A revisão da busca térmica exige um manifesto novo para recalcular.
Reserve `fopt-combined-T2-v2` para esse cálculo e use `fopt-combined-T2-v1`
como fonte de dados para rascunhos gráficos, caso queira comparar:

```bash
"$FOPT_PYTHON" -m article_FOPT_combined figures --output "$HOME/fopt-combined-T2-v1/campaign" --destination "$HOME/fopt-combined-T2-v1/figures-replot" --draft
```

## 2. Confira partições e limites

```bash
sinfo -s
sinfo -o "%P %a %l %D %C"
scontrol show partition debug
scontrol show partition milliways-ext
sacctmgr show assoc where user="$USER" format=Cluster,Account,User,Partition,QOS,DefaultQOS
```

`sinfo` mostra o estado das filas e o limite de tempo. `scontrol` detalha os
limites de cada partição; a associação da conta pode impor limites adicionais.
Uma fila visível não implica autorização para toda conta/QOS.

Os templates usam `debug` para o piloto e figuras (2 h) e `milliways-ext`
para coleta (14 h), nomes usados no trabalho anterior. Confirme a autorização
e os limites atuais. Para trocar, acrescente, por exemplo,
`--partition=NOME` ao comando `submit.sh`; isso sobrescreve a diretiva do template.

Partição define quais recursos/fila atenderão o job. `--cpus-per-task`
reserva CPUs; `--mem` reserva memória; `--time` define o limite de duração.
A concorrência do array é o número após `%`: `0-31%16` tem 32 tarefas,
até 16 simultâneas. Diminuir `%16` para `%8` reduz a concorrência e preserva
a divisão dos dados. Mantenha os índices `0-31` e `shard-count=32` coerentes.

Se reduzir a duração do job, ajuste também `FOPT_WALL_SECONDS` para deixar
folga. Por exemplo: para `--time=12:00:00`, use
`export FOPT_WALL_SECONDS=41400` antes de submeter. Esse limite impede lançar
um ponto que não terá tempo de publicar seu checkpoint.

## 3. Piloto físico dos quatro cenários

```bash
"$FOPT_PYTHON" -m article_FOPT_combined init --preset pilot --output "$FOPT_RESULTS_ROOT/pilot"
bash article_FOPT_combined/cluster/submit.sh pilot
```

O array calcula um ponto completo por m8, incluindo m8=400. São testes de
instalação, fases, nucleação, termodinâmica, beta e checkpoint, sem iniciar a
grade grande. O executável é sempre o do ambiente novo, inclusive no nó.

Anote o número mostrado em `Submitted batch job`. Para `JOB=123456`, por exemplo:

```bash
squeue -r -u "$USER"
sacct -j 123456 --format=JobID,State,Elapsed,AllocCPUS,MaxRSS,ExitCode
tail -n 60 article_FOPT_combined/results/logs/pilot-123456_0.out
cat article_FOPT_combined/results/logs/pilot-123456_0.err
```

`PD` significa pendente; `R`, executando. `COMPLETED` em sacct informa que o
processo terminou com sucesso; a validação dos resultados físicos é separada:

```bash
"$FOPT_PYTHON" -m article_FOPT_combined check-pilot --output "$FOPT_RESULTS_ROOT/pilot"
"$FOPT_PYTHON" -m article_FOPT_combined status --output "$FOPT_RESULTS_ROOT/pilot"
```

Para testar uma combinação finita adicional:

```bash
"$FOPT_PYTHON" -m article_FOPT_combined point --output "$FOPT_RESULTS_ROOT/pilot" --m6 1000 --m8 inf --C 2.8 --timeout 900
```

Esse comando deve ser executado em um nó interativo alocado conforme as
regras do CHE, ou adaptado ao template do piloto; evite computação pesada no
nó de login. Uma falha física/numerical fica no JSON do ponto. Timeout e erro
de fases conservam decisão desconhecida e não certificam exclusão.

O piloto também admite uma busca curta de fronteiras com `--max-points 4`.
Ela deve ser feita em um job alocado e serve para observar pausa/retomada;
quatro pontos não bastam para certificar as duas bordas de todos os cenários.

Antes da produção, faça também o piloto das **oito colunas de fronteira**:

```bash
export FOPT_PILOT_STAGE=boundaries
bash article_FOPT_combined/cluster/submit.sh pilot
unset FOPT_PILOT_STAGE
```

Quando os jobs terminarem, consulte `status --output "$FOPT_RESULTS_ROOT/pilot"`.
Cada cenário deve ter duas colunas examinadas no preset padrão. Para um piloto
com área suficiente para conferir os mapas, inclua algumas massas finitas
na criação do piloto, por exemplo `--m6 675 725 25`. Com essa escolha, cada
cenário terá quatro colunas, contando `m6=inf`. Se o orçamento de duas horas
foi atingido, repita a mesma submissão para concluir. Depois rode o piloto dos
pontos internos, ainda no mesmo manifesto:

```bash
export FOPT_PILOT_STAGE=observables
bash article_FOPT_combined/cluster/submit.sh pilot
unset FOPT_PILOT_STAGE
```

Consulte novamente `status`: os pontos planejados devem estar salvos e com
observáveis resolvidos. Assim o piloto cobre também fronteiras, planejamento
do interior e retomada. Sua única massa finita é um teste de fluxo; os mapas
bidimensionais exigem a grade de produção ou outro piloto com mais massas.

## 4. Fronteiras de produção

Depois do piloto aprovado:

```bash
"$FOPT_PYTHON" -m article_FOPT_combined init --preset full --output "$FOPT_RESULTS_ROOT/campaign"
"$FOPT_PYTHON" -m article_FOPT_combined boundaries --output "$FOPT_RESULTS_ROOT/campaign" --dry-run
bash article_FOPT_combined/cluster/submit.sh boundaries
```

O array tem 32 tarefas com uma CPU cada. Há 302 colunas (301 massas finitas
mais m6=inf) em cada um dos quatro m8. A distribuição é determinística e
disjunta. As tarefas usam as colunas anteriores que lhes pertencem como
primeiros palpites, verificando cada testemunha com o novo potencial.

```bash
"$FOPT_PYTHON" -m article_FOPT_combined status --output "$FOPT_RESULTS_ROOT/campaign"
```

Para avançar, cada cenário deve apresentar `columns_done=columns_expected`.
`bounded` significa bordas determinadas; `no_seed_on_grid` significa que
nenhuma faixa foi encontrada até a resolução declarada. `unresolved` pede
recuperação; `upper_range_limit` pede revisar a janela de C; `reentry_detected`
pede revisar a hipótese de uma faixa conectada.

Se o job apenas esgotou seu orçamento, repita a mesma submissão. Os pontos e
colunas completos são lidos do cache. Para retentar as falhas:

```bash
export FOPT_RETRY_FAILED=1
bash article_FOPT_combined/cluster/submit.sh boundaries
unset FOPT_RETRY_FAILED
```

Se as pendências forem timeouts de pontos, a retentativa pode usar
`export FOPT_BOUNDARY_TIMEOUT=600` junto de `FOPT_RETRY_FAILED=1`.
O tempo por ponto é independente da configuração física da campanha.

## 5. Observáveis entre as bordas

Somente depois de todas as colunas terem sido examinadas:

```bash
"$FOPT_PYTHON" -m article_FOPT_combined observables --output "$FOPT_RESULTS_ROOT/campaign" --dry-run
bash article_FOPT_combined/cluster/submit.sh observables
```

São 16 tarefas, até 8 simultâneas, cada uma com dois subprocessos físicos e
duas CPUs reservadas. O padrão calcula nove pontos internos e duas bordas a
cada 25 GeV, além de pontos independentes de validação. Todos os cenários
incluem o controle m6=inf. Ações/fases salvas nas bordas são reaproveitadas.

`status` informa pontos salvos, completos, planejados e ainda ausentes.
Repetir a submissão completa os ausentes. `FOPT_RETRY_FAILED=1` habilita
retentativas. O mesmo ambiente, código físico e manifesto devem ser mantidos
durante a campanha; atualizações gráficas podem ser feitas depois.

## 6. Figuras, auditoria e refinamentos pontuais

Para uma primeira inspeção:

```bash
export FOPT_DRAFT=1
bash article_FOPT_combined/cluster/submit.sh figures
unset FOPT_DRAFT
```

As imagens ficam em `$FOPT_RESULTS_ROOT/figures`. `coverage_audit.json` informa
as lacunas, `point_requests.csv` lista os pontos que precisam de dados e
`interpolation_validation.csv` compara valores interpolados e calculados.
O job de figuras lê apenas os produtos necessários à renderização. A exportação
integral da campanha é uma etapa separada.

Para calcular apenas coordenadas solicitadas, use em um job alocado:

```bash
"$FOPT_PYTHON" -m article_FOPT_combined point --output "$FOPT_RESULTS_ROOT/campaign" --points-file "$FOPT_RESULTS_ROOT/figures/point_requests.csv" --retry-failed --timeout 900
```

Para estreitar só os cruzamentos do limiar LISA, use também, em um job alocado:

```bash
"$FOPT_PYTHON" -m article_FOPT_combined point --output "$FOPT_RESULTS_ROOT/campaign" --points-file "$FOPT_RESULTS_ROOT/figures/lisa_point_requests.csv" --timeout 900
```

A lista é atualizada ao gerar as figuras novamente. Se a auditoria mostrar
que a resolução da grade é insuficiente em muitas colunas, mantenha a
configuração física e submeta:

```bash
export FOPT_REFINE=1
bash article_FOPT_combined/cluster/submit.sh observables
unset FOPT_REFINE
```

O refinamento intercala massas e duplica as divisões em C. Após terminar:

```bash
export FOPT_REPLACE_FIGURES=1
bash article_FOPT_combined/cluster/submit.sh figures
unset FOPT_REPLACE_FIGURES
```

A versão final exige auditoria aprovada. Um erro nessa etapa deixa os
relatórios prontos para orientar o próximo cálculo. `--replace` substitui os
produtos gráficos; os pontos físicos permanecem na pasta de campanha.

## 7. Compacte no CHE e baixe no Windows

Depois de conferir que os PDFs/PNGs foram produzidos:

```bash
"$FOPT_PYTHON" -m zipfile -c "$FOPT_RESULTS_ROOT/figures.zip" "$FOPT_RESULTS_ROOT/figures"
ls -lh "$FOPT_RESULTS_ROOT/figures.zip"
```

No **PowerShell do seu computador**, usando os dados de acesso já empregados:

```powershell
scp -P 13900 ftovar@152.84.248.250:/home/ftovar/fopt-combined-T2-v2/figures.zip "$env:USERPROFILE\Downloads\FOPT-T2-figures-v2.zip"
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\FOPT-T2-figures-v2.zip" -DestinationPath "$env:USERPROFILE\Downloads\FOPT-T2-figures-v2" -Force
explorer "$env:USERPROFILE\Downloads\FOPT-T2-figures-v2"
```

Para exportar também todos os dados, execute no CHE em um job alocado:

```bash
"$FOPT_PYTHON" -m article_FOPT_combined export --output "$FOPT_RESULTS_ROOT/campaign" --zip "$FOPT_RESULTS_ROOT/campaign-complete.zip"
```

Com o ZIP pronto, baixe no PowerShell:

```powershell
scp -P 13900 ftovar@152.84.248.250:/home/ftovar/fopt-combined-T2-v2/campaign-complete.zip "$env:USERPROFILE\Downloads\FOPT-T2-data-v2.zip"
```

Se escolher outro `FOPT_RESULTS_ROOT`, troque o caminho remoto de scp pelo
caminho absoluto mostrado por `ls`. O caractere `$` é necessário para expandir
uma variável no shell; escrever `FOPT_RESULTS_ROOT/...` literalmente procura
uma pasta com esse nome. O ZIP só estará disponível depois de sua criação
ter terminado com sucesso.
