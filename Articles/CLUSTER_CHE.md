# Primeira execução no CHE

Este roteiro prepara o acesso, dois testes e a campanha adaptativa em três fatias de m8.
Não há submissão automática. A produção deve seguir o teste local e o teste
no nó de cálculo.

## 1. Entrar pelo Windows

No PowerShell, segundo o manual de 2024:

```powershell
ssh -p 13900 SEU_USUARIO@152.84.248.250
```

Confirme endereço/porta e a chave do servidor com a administração se necessário.
Na autenticação por senha, os caracteres podem não aparecer enquanto você digita.
Após entrar, os próximos comandos são Linux, executados no CHE. `exit` encerra
o SSH. Não é necessário instalar uma interface gráfica para usar o cluster.

O nó de entrada serve para preparar arquivos, ambientes e submeter jobs. A coleta
com bounces deve executar numa alocação Slurm, não diretamente no nó de entrada.

## 2. Conferir ambiente e recursos

```bash
hostname
pwd
module avail
module list
python3 --version
sinfo -a
sinfo -N
quota -s
df -h /home /scratch/local
```

Se quota ou algum caminho não existir, peça a alternativa ao suporte. O scratch
local é relevante no nó de cálculo; sua presença no nó de entrada não é garantia.
Se Python for anterior a 3.11, procure um módulo Python/Miniconda atual e carregue
o nome realmente disponível com `module load NOME`. Não use o módulo Anaconda2.
A venv deve ser criada com o Python escolhido. Nenhum MPI/GPU é exigido pelo scan.

O manual lista debug com limite de 2h e generic com 15 dias. Existem filas
restritas a grupos, inclusive milliways/cosmoobs. Consulte sinfo e a administração
para saber a sua permissão; o roteiro não pressupõe acesso a essas filas.

## 3. Preparar o repositório e o Python

```bash
mkdir -p "$HOME/projects"
cd "$HOME/projects"
git clone --branch codex/articles-combined-potential-scan https://github.com/IgorStellet/GW_FOPT_Simulation.git
cd GW_FOPT_Simulation
bash Articles/cluster/setup_environment.sh python3
export FOPT_PYTHON="$PWD/.venv-che311/bin/python"
mkdir -p Articles/results/logs
"$FOPT_PYTHON" -m Articles.collect_data --dry-run --output Articles/results/installation-check
bash -n Articles/cluster/smoke.slurm
bash -n Articles/cluster/production.slurm
bash -n Articles/cluster/job_environment.sh
```

Se já houver clone, use-o e confira `git status` antes de atualizar. Instale no
ambiente que será usado pelo job. Não execute pip automaticamente nos nós de
cálculo: eles podem não ter acesso à internet. Preserve clone e ambiente durante
a campanha; alterações em fontes/versões podem invalidar a retomada.

### Python 3.11 disponível, mas sem pip

`python3 -m pip: No module named pip` no Python base não exige instalar outro
Python. O script acima cria uma **nova** `.venv-che311`, instala pip nela com
`ensurepip` e instala o projeto e pytest usando esse mesmo Python. Nenhum pacote
é instalado em `/opt/spack`. Se a pasta já existir, o script para e a preserva;
escolha outro nome, por exemplo:

```bash
bash Articles/cluster/setup_environment.sh python3 .venv-che311-nova
export FOPT_PYTHON="$PWD/.venv-che311-nova/bin/python"
```

Se a preparação parar **depois** de criar o ambiente, não precisa recriá-lo.
Para concluir a instalação na pasta criada, use (ajuste o nome se necessário):

```bash
export FOPT_PYTHON="$PWD/.venv-che311/bin/python"
"$FOPT_PYTHON" --version
"$FOPT_PYTHON" -m ensurepip --upgrade
"$FOPT_PYTHON" -m pip install --upgrade pip
"$FOPT_PYTHON" -m pip install -e . pytest
"$FOPT_PYTHON" -m pip check
```

Se o módulo escolhido não oferecer `venv`/`ensurepip`, a preparação informa isso;
procure outro módulo Python >=3.11 ou peça ao suporte o módulo apropriado.
Carregar Python 3.11 não transforma uma venv antiga em uma venv 3.11. Evite
`pip install` isolado: ele pode pertencer a outro Python. Confira sempre
`"$FOPT_PYTHON" -m pip --version`. O erro de NumPy com versões só até 1.24.4
também pode indicar um índice de pacotes limitado; se persistir no ambiente
novo, registre a versão do Python/pip e consulte o índice autorizado pelo CHE.

Para testar a suíte em um nó, peça uma sessão interativa permitida pela sua conta:

```bash
srun --partition=debug --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=4G --time=00:30:00 --pty bash -l
cd "$HOME/projects/GW_FOPT_Simulation"
"$FOPT_PYTHON" -m pytest -q Articles/tests
exit
```

Se debug não estiver disponível, use a partição confirmada. A execução completa
inclui exemplos numéricos e pode precisar de mais tempo; ajuste após medir.

## 4. Escolher onde salvar

O manual descreve /home com quota soft de 30 GB e hard de 150 GB, com grace
de 15 dias. /share/storage1 é área compartilhada para produtos científicos;
/share/storage2 destina-se principalmente ao grupo CONNIE. Confirme uma pasta
autorizada com espaço para sua campanha. Os detalhes de fases podem ocupar
dezenas de GB no scan grande; meça o piloto antes de extrapolar o armazenamento.

O manual descreve /scratch/local, mas esse caminho pode estar ausente ou sem
permissão no nó alocado, como ocorreu no job 107761. Os templates não presumem
sua existência: criam uma pasta exclusiva no primeiro caminho gravável entre
SLURM_TMPDIR, TMPDIR e /tmp, usando mktemp. Esses caminhos são avaliados dentro
do job, no nó de cálculo. O `.out` informa o diretório efetivamente escolhido.
O SQLite ativo fica nessa pasta temporária; checkpoints vão ao destino persistente.

Para o teste pequeno, uma pasta na sua home é suficiente:

```bash
export FOPT_RESULTS_ROOT="$HOME/fopt-tests"
mkdir -p "$FOPT_RESULTS_ROOT" Articles/results/logs
```

Mantenha `FOPT_PYTHON` apontando para o ambiente escolhido na etapa 3.
Se o Python exige um módulo, exporte também o nome realmente usado:
`export FOPT_PYTHON_MODULE=NOME_DO_MODULO`. Os scripts o carregam explicitamente.
Para escolher outro disco local confirmado no nó, exporte FOPT_SCRATCH_ROOT com
uma pasta **existente e gravável**. Uma escolha explícita inválida faz o job parar
com uma mensagem clara. Para usar a seleção automática:

```bash
unset FOPT_SCRATCH_ROOT
```

Não use uma montagem NFS para o banco ativo. `/tmp` permite o teste pequeno;
antes da campanha grande, confira o espaço do disco no nó de cálculo.

## 5. Os dois testes antes de produzir

Use a branch atualizada e uma pasta nova para esta estratégia. Não reutilize
checkpoints da grade antiga. Confirme partição, tempo e quota na sua conta;
os valores dos templates são exemplos e não comprovam disponibilidade.

**Teste 1: lógica e persistência**, em uma alocação de cálculo:

```bash
srun --partition=debug --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:30:00 --pty bash -l
cd "$HOME/projects/GW_FOPT_Simulation"
"$FOPT_PYTHON" -m pytest -q Articles/tests
exit
```

**Teste 2: piloto real de duas linhas completas por m8:**

```bash
mkdir -p Articles/results/logs
sbatch --export=ALL Articles/cluster/smoke.slurm
```

O template cria um array de **três tarefas**, índices 0, 1, 2, uma por m8.
Cada tarefa roda o teste de lógica e o piloto real. Os resultados ficam em
`$FOPT_RESULTS_ROOT/smoke/m8-0`, `m8-1`, `m8-2`. Cada pasta tem banco, detalhes,
matriz e `adaptive_state.json`. Confira os três estados `pilot_complete` e
as duas linhas completas antes da produção.

O piloto usa m6=`inf, 2000, 1250, 500` GeV. A procura inicial da semente pode
avaliar outros C em m6=inf antes das duas linhas; ela é limitada pelos bounds.
Uma parada `bound` deve ser distinguida da regra de três ausências. A parada
externa de três linhas vazias é exercitada no Teste 1; o piloto tem só duas.

## 6. Produção em três frentes, uma por m8

```bash
"$FOPT_PYTHON" -m Articles.run_local --dry-run --beta-check --output "$FOPT_RESULTS_ROOT/production"
sbatch --export=ALL Articles/cluster/production.slurm
```

`run_local --dry-run` aqui apenas confere as configurações, sem bounces no nó
de entrada. Não execute `run_local` sem dry-run nesse nó.

| Tarefa do array | m8 [GeV] | Destino persistente |
|---|---:|---|
| 0 | inf | production/m8-0 |
| 1 | 840.8964152537145 | production/m8-1 |
| 2 | 668.740304976422 | production/m8-2 |

O array é `0-2%3`: no máximo três jobs simultâneos, cada um com uma CPU,
uma tarefa e um processo de cálculo. Não há MPI. Dentro de cada m8 a busca é
sequencial: as decisões dependem dos resultados anteriores. A distribuição
por m8 não garante speedup exatamente 3, pois os tempos das fatias diferem.
Não use a partição antiga em 100 shards no scan adaptativo.

O padrão tem C em [0,10], passo 0.02, semente 3.83; m6 finito em
[500,2000] GeV, passo 5 GeV, mais inf; regra de três ausências. O piloto usa
outra grade e outra identidade. Para personalizar a produção, altere os
argumentos do runner no template e use outro destino persistente.
Todos os parâmetros da busca estão disponíveis no runner: `--C`, `--m6`,
`--seed-C`, `--seed-m6`, `--patience`, `--scenario-index`, `--retry-failed`.

## 7. Checkpoints, interrupções e retomada

O runner calcula lotes de cinco pontos novos por padrão e fecha o SQLite
antes de publicar. Cada ponto já fica confirmado no banco local imediatamente;
se o nó for encerrado abruptamente, o último lote ainda não publicado pode
precisar ser repetido. Para checkpoints mais frequentes:

```bash
export FOPT_BATCH_SIZE=1
sbatch --export=ALL Articles/cluster/production.slurm
```

Lotes menores aumentam o custo das cópias. O sinal USR1 é encaminhado ao
coordenador; ele termina o lote atual, publica e sai. O aviso de cinco minutos
não garante que um ponto difícil termine a tempo. A retomada usa o último
checkpoint confirmado. Não há submissão automática de jobs novos.

Submeta de novo o mesmo template, sem alterar código, versões ou parâmetros.
Cada tarefa restaura sua fatia em uma pasta local exclusiva nova e reconstrói
as decisões a partir do SQLite, sem repetir pontos confirmados. Para retomar
somente uma fatia:

```bash
sbatch --export=ALL --array=1 Articles/cluster/production.slurm
```

Se houver ponto inconclusivo, a tarefa para e publica o diagnóstico já salvo;
falha não conta como ausência para encerrar uma direção. Consulte os detalhes.
Para repetir uma falha transitória, acrescente `--retry-failed` ao runner e
submeta novamente a fatia. Uma falha persistente para novamente. Mudança na
física, resolução ou fontes exige outra pasta; não remova o manifesto.

Para exercitar o mesmo runner localmente antes de submeter:

```bash
python -m Articles.cluster_runner --smoke --scenario-index 0 --beta-check --batch-size 1 --output Articles/results/runner_pilot --work-dir Articles/results/scratch_pilot_novo
```

`--work-dir` deve estar vazio, ser distinto de `--output` e não estar dentro
dele nem contê-lo. Ao repetir, escolha uma pasta de trabalho nova e mantenha
o destino persistente. Sem argumentos, o runner só mostra ajuda.

## 8. Acompanhar e conferir

Use `squeue -u "$USER"` e os logs `scan-ARRAY_INDICE.out/.err` ou
`smoke-ARRAY_INDICE.out/.err`. Consulte `sacct -j ID --format=JobID,State,Elapsed,ExitCode`
quando o job terminar. O término de uma tarefa não significa que as demais
terminaram; confira cada fatia e seu estado final `complete`.

As matrizes usam `yes`, `no`, `unknown`, `not_evaluated`; `matrix_axes.json`
identifica o m8 de cada pasta. `adaptive_state.json` guarda as sementes, os
três últimos pontos/linhas que fecharam a direção, o primeiro pulado e bounds.
Não interprete pontos pulados como exclusão física. Confira a resolução das
fronteiras e possíveis ilhas com scans direcionados antes do artigo.

O acesso ao CHE e a submissão remota precisam ser feitos na sua conta.
A validação local de checkpoint não constitui uma execução remota no Slurm.
