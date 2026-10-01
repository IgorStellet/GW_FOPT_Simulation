# Primeira execução no CHE

Este roteiro prepara o acesso, um teste de dois pontos e a campanha grande.
Não há submissão automática. A grade grande deve seguir o teste local e o teste
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
"$FOPT_PYTHON" -m Articles.collect_data --dry-run
bash -n Articles/cluster/smoke.slurm
bash -n Articles/cluster/production.slurm
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
"$FOPT_PYTHON" -m pytest -q
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

/scratch/local pertence a cada nó; o manual informa remoção de arquivos antigos
após 30 dias. Não é o destino final dos dados. Os templates usam scratch para
SQLite ativo e copiam checkpoints para o destino persistente.

Para o teste pequeno, uma pasta na sua home é suficiente:

```bash
export FOPT_RESULTS_ROOT="$HOME/fopt-tests"
mkdir -p "$FOPT_RESULTS_ROOT" Articles/results/logs
```

Mantenha `FOPT_PYTHON` apontando para o ambiente escolhido na etapa 3.
Se o Python exige um módulo, exporte também o nome realmente usado:
`export FOPT_PYTHON_MODULE=NOME_DO_MODULO`. Os scripts o carregam explicitamente.
FOPT_SCRATCH_ROOT pode substituir /scratch/local somente por um disco local
confirmado no nó. Não aponte esse scratch para uma montagem NFS.

## 5. Submeter o teste de dois pontos

Na raiz do repositório, prepare as variáveis **na mesma sessão SSH** em que
executará `sbatch`. Elas precisam ser exportadas novamente após sair do SSH e
entrar em outra sessão. `FOPT_RESULTS_ROOT` é a pasta persistente dos resultados;
`FOPT_PYTHON` é o interpretador do ambiente instalado na etapa 3.

```bash
export FOPT_PYTHON="$PWD/.venv-che311/bin/python"
export FOPT_RESULTS_ROOT="$HOME/fopt-tests"
mkdir -p "$FOPT_RESULTS_ROOT" Articles/results/logs
"$FOPT_PYTHON" --version
"$FOPT_PYTHON" -m Articles.collect_data --dry-run
```

Se escolheu outro nome para a venv, ajuste `FOPT_PYTHON`. Se esta pasta de
resultados contém uma campanha de outra versão do código, escolha outra pasta.
Somente após as duas chamadas Python acima funcionarem, submeta:

```bash
sbatch --export=ALL Articles/cluster/smoke.slurm
squeue -u "$USER"
```

`--export=ALL` transmite as variáveis exportadas ao job. O nome do diretório
de resultados é uma escolha do usuário; o script exige essa escolha antes de
iniciar o cálculo. Se aparecer `FOPT_RESULTS_ROOT: ...` no `.err`, o job não
recebeu um valor não vazio e parou antes de chamar Python. Refaça os exports
nesta sessão e faça uma nova submissão: o job anterior já terminou.
Referência: [exportação de ambiente no sbatch](https://slurm.schedmd.com/sbatch.html).

O Slurm informa um JOB_ID. O script reserva um nó, uma tarefa, duas CPUs e 4 GB
por até 2h na fila debug. Ele calcula C=0 e C=3.35 para m6=1000 GeV e
m8=668.740304976422 GeV, com beta-check. O checkpoint é feito a cada ponto.

Monitore substituindo JOB_ID pelo número recebido. Os comandos abaixo retornam
ao prompt e podem ser executados em sequência:

```bash
scontrol show job JOB_ID
tail -n 80 Articles/results/logs/smoke-JOB_ID.out
cat Articles/results/logs/smoke-JOB_ID.err
sacct -j JOB_ID --format=JobID,State,Elapsed,AllocCPUS,MaxRSS,ExitCode
```

Para acompanhar um job em execução continuamente, use separadamente
`tail -f Articles/results/logs/smoke-JOB_ID.out`. Esse comando fica aberto,
inclusive depois de o job terminar; use Ctrl+C para voltar ao prompt antes de
executar outro comando. Ctrl+C encerra apenas tail, não o job.
Para cancelar o job, use `scancel JOB_ID`.
PD significa aguardando recursos; R significa executando. O log mostra checkpoints
publicados, status dos pontos e quantidade confirmada. Confira os CSVs em
`$FOPT_RESULTS_ROOT/smoke` e os valores de referência no QUICKSTART.

## 6. Por que estas diretivas diferem do modelo de 70 tarefas

O coletor usa multiprocessing dentro de um nó. A alocação correta é
`--nodes=1 --ntasks=1 --cpus-per-task=N`, com `--workers N`. Pedir 70 tarefas
não faz o programa distribuir trabalho em 70 processos MPI ou em vários nós.
Um array Slurm permite distribuir partes independentes para vários nós.

SQLite em WAL não deve operar em filesystem de rede. cluster_runner chama o
coletor em lotes no scratch local; quando o coletor fecha o banco e os workers,
copia detalhes/CSV e publica o banco por último. A cópia persistente usa journal
DELETE. Nunca copie manualmente apenas scan.sqlite de um banco WAL ativo.

Referências técnicas: [Slurm sbatch](https://slurm.schedmd.com/sbatch.html),
[SQLite WAL](https://www.sqlite.org/wal.html).

## 7. Campanha grande, depois dos testes

Primeiro confirme quota, partição, memória por worker e tempo observado. Escolha
um destino persistente autorizado para produção, diferente do teste. Por exemplo,
substitua o caminho abaixo pela pasta concedida ao seu usuário/projeto:

```bash
export FOPT_RESULTS_ROOT=/CAMINHO/PERSISTENTE/AUTORIZADO/fopt-campaign
mkdir -p "$FOPT_RESULTS_ROOT" Articles/results/logs
"$FOPT_PYTHON" -m Articles.collect_data --dry-run --shard-count 100 --shard-index 0 --beta-check
sbatch --export=ALL Articles/cluster/production.slurm
```

O template é um ponto de partida: generic, 3 dias, 4 CPUs e 8 GB por job;
array 0–99, no máximo 2 jobs simultâneos (8 CPUs no total). Não é uma medição de
memória suficiente nem uma garantia de terminar em 3 dias. Ajuste recursos depois
do piloto, respeitando as regras da conta. GPU não acelera este código.

As 453.906 coordenadas são repartidas pela posição na grade módulo 100. As partes
0–5 têm 4.540 pontos; 6–99 têm 4.539. Cada ponto pertence exatamente a uma parte;
nenhuma precisão é reduzida pela distribuição. Toda parte recebe a mesma grade,
prescrições e beta-check. As pastas finais são `production/part-0` até `part-99`.

Se mudar a quantidade de partes, ajuste simultaneamente `#SBATCH --array` e
`--shard-count`, e use outra campanha. Trocar apenas a concorrência `%2`, CPUs
ou batch-size não altera a definição física nem a distribuição dos pontos.

Checkpoint padrão a cada 100 pontos novos. Em corte abrupto, resultados do último
lote ainda não publicado podem precisar ser recalculados; checkpoints anteriores
permanecem válidos. O sinal 5 min antes do limite pede parada ao coordenador após
o lote atual. Ele não garante que um lote demorado termine nesse prazo.
O scratch não é removido automaticamente: confira os checkpoints antes de limpeza.

## 8. Retomar e analisar

Para retomar, submeta o mesmo script com o mesmo destino e código/ambiente. Cada
job restaura seu checkpoint para um scratch novo e pula pontos já confirmados.
Para repetir somente partes incompletas, restrinja o array na chamada, por exemplo
`sbatch --array=4,7,12%2 Articles/cluster/production.slurm`. Não altere shard-count.
Uma nova submissão não pode disputar uma parte ainda ativa: a trava rejeita isso.

Job COMPLETED pode significar saída após o aviso de tempo, não que todas as
coordenadas foram investigadas. Compare o total confirmado com o tamanho esperado
da parte. Falhas numéricas também são registros; verifique seus status e flags.
O wrapper não repete falhas automaticamente, pois repetir sempre o mesmo erro
poderia impedir terminar a parte. Revisões físicas devem formar outra campanha.

Para análises, leia os points.csv/transitions.csv de todas as partes, associando
point_id e transition_index. Não leia apenas part-0 nem sobrescreva manifestos
das outras partes. Espectros completos continuam reconstruíveis a partir dos
parâmetros; nenhuma curva completa é gravada pelo template.

Para baixar um teste no PowerShell do seu computador:

```powershell
scp -P 13900 -r SEU_USUARIO@152.84.248.250:/home/SEU_USUARIO/fopt-tests/smoke .
```

Para produção, use o caminho persistente escolhido e planeje a transferência
conforme o volume. Não apague a única cópia dos detalhes ao baixar somente CSVs.

## Resumo das decisões que faltam no primeiro acesso

Nome de usuário/endereço/porta efetivos; módulo Python >=3.11; partição permitida;
pasta persistente e quota; scratch local; memória/tempo do piloto. Com essas
informações e o teste bem-sucedido, os templates tornam a submissão concreta.
