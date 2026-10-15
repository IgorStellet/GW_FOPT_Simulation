# CHE: ambiente, validação e produção em três fatias

Este roteiro usa o estado informado pelo usuário em 15/10/2026. Os templates
v3 usam milliways-ext. A consulta real mostrou MaxTime=15:00:00 e os nós
milliways01/03 parcialmente ocupados; milliways02 estava down*. O manual de
2024 não descreve toda essa configuração. Consulte novamente antes de escolher
recursos. A divergência entre milliways-int no sinfo e o erro do scontrol não
foi resolvida; não se presume acesso/configuração dessa partição.

Todas as fontes modificadas ficam em Articles. Não atualize o checkout nem o
ambiente enquanto houver jobs usando esses arquivos. Uma atualização numérica
exige novo destino: checkpoints v2 não são retomados como v3.

## 1. Ambiente persistente no nó de entrada

Entre pela conexão SSH habitual. Na raiz do checkout:

```bash
cd "$HOME/projects/GW_FOPT_Simulation"
git status --short
git branch --show-current
module list
export FOPT_PYTHON_MODULE=python/3.11.7-gcc-12.2.0
module load "$FOPT_PYTHON_MODULE"
python3 --version
```

A branch é codex/articles-combined-potential-scan. Depois de todos os jobs
anteriores desse checkout terminarem, atualize com git pull --ff-only. Preserve
alterações locais e resultados. Se ainda não tem clone, clone essa branch na
pasta de projetos da sua home.

Uma venv concluída permanece instalada depois do logout. Apenas módulo e
variáveis da sessão precisam ser selecionados novamente. Não reinstale pacotes
em cada login ou nos nós de cálculo. Para uma instalação NOVA, uma única vez:

```bash
bash Articles/cluster/setup_environment.sh python3 .venv-che311
```

O script preserva diretórios existentes. Se a instalação anterior parou depois
de criar a venv, conclua nela usando seu próprio Python e pip, sem apagar nada.
Se ela pertence a um Python antigo, escolha outro nome para uma venv nova.

Em cada nova sessão (ajuste o nome se sua venv for diferente):

```bash
export FOPT_PYTHON="$PWD/.venv-che311/bin/python"
export FOPT_RESULTS_ROOT="$HOME/fopt-articles-v3"
mkdir -p "$FOPT_RESULTS_ROOT" Articles/results/logs
"$FOPT_PYTHON" --version
"$FOPT_PYTHON" -m pip check
unset FOPT_SCRATCH_ROOT
```

O Python do job é esse caminho absoluto; o prefixo (base) no terminal não
substitui essa seleção. O projeto exige Python >=3.11. A venv depende do mesmo
módulo base com que foi criada. bash executa setup_environment.sh; sh não é a
chamada adequada para esse script.

## 2. Recursos e escolha de partição

```bash
sinfo -a
scontrol -a show partition
scontrol show partition milliways-ext
scontrol show node milliways03
type -a sbatch sinfo scontrol
quota -s
```

Mais CPUs no nó não prova maior desempenho por processo. No benchmark do
usuário, o ponto C=3.83, m6=m8=inf, com beta-check levou aproximadamente 59 s no
Milliways, contra 429.6 s em c003. Esse é um resultado de um ponto, não uma
previsão do tempo de toda a campanha.

O array 0-2%3 já usa até TRÊS CPUs simultâneas: três jobs, cada um com uma tarefa
e uma CPU, uma por m8. A busca dentro de m8 é sequencial. Aumentar ntasks ou
cpus-per-task não cria paralelismo adicional nesse código. Não use o modelo
ntasks=70 para um único processo Python. As medições do piloto mostraram menos
de 200 MB por tarefa; mantenha 4 GB no piloto e 8 GB na produção enquanto mede
regiões mais difíceis. Não há MPI/GPU nesta execução.

milliways03 pode ser fixado com --nodelist=milliways03. O template não fixa nó:
permite ao escalonador escolher um disponível na partição. Não selecione um nó
down*. Se quiser comparar milliways01, repita um ponto em destino NOVO e preserve
o mesmo código, versões e parâmetros. Memória adicional não acelera um cálculo
que já cabe na memória reservada.

## 3. Teste 1 e piloto real (teste 2)

O template smoke roda a suíte de lógica/persistência e depois o piloto real em
cada tarefa. Submeta da raiz e selecione o nó medido:

```bash
sbatch --export=ALL --nodelist=milliways03 Articles/cluster/smoke.slurm
squeue -r -u "$USER"
```

O limite do piloto é 2 horas. Para alterar o limite sem editar o template, use
--time=03:00:00 (deve respeitar MaxTime da partição). Os resultados ficam em:

- $FOPT_RESULTS_ROOT/smoke_v3/m8-0
- $FOPT_RESULTS_ROOT/smoke_v3/m8-1
- $FOPT_RESULTS_ROOT/smoke_v3/m8-2

Cada piloto tem duas colunas m6: inf e 2000 GeV. Em toda coluna procura a
primeira nucleação sem corte por três ausências. Depois dela, o piloto limita
a expansão a três passos C por lado. pilot_step_limit é limite de teste e não
fronteira física. O piloto não substitui a produção, nem examina m6~550 GeV.

A v3 tenta recuperar um ponto inconclusivo uma vez. Se não consegue, grava
unknown e continua os outros C/m6. Unknown quebra a sequência de negativas;
uma coluna com unknown nunca conta como uma coluna vazia para encerrar m6.
Nenhuma falha é convertida em no. complete_with_unresolved e
pilot_complete_with_unresolved exigem revisão; os executores retornam código 3
nesse caso depois de publicar os dados e figuras. --on-unresolved stop no
runner/coletor ativa a interrupção estrita opcional.

Depois que as tarefas terminarem:

```bash
"$FOPT_PYTHON" -m Articles.status --output "$FOPT_RESULTS_ROOT/smoke_v3"
"$FOPT_PYTHON" -m Articles.plot_results --output "$FOPT_RESULTS_ROOT/smoke_v3"
```

Confira pilot_complete nas três fatias, as duas massas, sementes verificadas e
os plots. Se houver sufixo _with_unresolved, examine os detalhes mostrados pelo
diagnóstico antes de usar esses pontos. Cada plots/ contém nucleation_map.png,
gw_spectra.png, versões SVG, selection.json e spectrum_samples.csv. O mapa
conjunto é smoke_v3/plots/nucleation_overview.png. Copie as imagens por SCP ou
WinSCP para visualizar no computador. As figuras usam Agg e não abrem janelas.

## 4. Produção e retomada em alocações de 14 horas

Depois de validar a revisão nova:

```bash
export FOPT_BATCH_SIZE=1
"$FOPT_PYTHON" -m Articles.run_local --dry-run --beta-check --output "$FOPT_RESULTS_ROOT/production_v3"
sbatch --export=ALL --nodelist=milliways03 Articles/cluster/production.slurm
```

O template já pede milliways-ext, 14 horas, três tarefas e aviso USR1 de 15
minutos. Isso cabe no MaxTime=15 horas observado. O aviso pede ao coordenador
que termine o lote e publique antes de sair; não garante que um ponto patológico
termine dentro do aviso. Com lote 1, só o ponto em andamento ainda não publicado
pode precisar ser repetido após encerramento abrupto.

| Índice | m8 [GeV] | Destino |
|---|---:|---|
| 0 | inf | production_v3/m8-0 |
| 1 | 840.8964152537145 | production_v3/m8-1 |
| 2 | 668.740304976422 | production_v3/m8-2 |

Mantenha fontes, venv e parâmetros fixos durante a campanha. Ao terminar uma
alocação ainda parcial, repita a mesma submissão com o mesmo destino. Não há
resubmissão automática. Para retomar só uma fatia:

```bash
sbatch --export=ALL --nodelist=milliways03 --array=1 Articles/cluster/production.slurm
```

A conclusão física exige complete nas três fatias; uma saída Slurm COMPLETED
pode corresponder apenas à parada para checkpoint. Um estado final com
inconclusivos guarda a cobertura pendente. Para tentar novamente esses pontos
sem mudar fontes/versões, use --retry-failed no runner do template; uma mudança
numérica exige nova campanha. Não altere/apague o manifesto para forçar retomada.

Os bounds são C=0..10 com passo 0.02 ancorado em 3.83, mais C=1 e C=3.2 exatos;
m6=500..2000 GeV, passo 5, mais inf. Três ausências verificadas consecutivas
fecham uma direção C somente após semente nessa coluna. Três colunas completas
sem hit nem unknown fecham m6, depois de já haver semente. A poda pressupõe uma
faixa conectada na resolução adotada; possíveis ilhas exigem scans dirigidos.

## 5. Logs, métricas e espaço em disco

```bash
squeue -r -u "$USER"
tail -n 60 Articles/results/logs/scan-ID_DO_ARRAY_*.out
tail -n 40 Articles/results/logs/scan-ID_DO_ARRAY_*.err
"$FOPT_PYTHON" -m Articles.status --output "$FOPT_RESULTS_ROOT/production_v3"
```

Use o ID realmente retornado pelo sbatch. Checkpoint publicado mostra a
contagem acumulada; [1] se repete quando o lote tem um ponto. paused_point_budget
é esperado entre lotes e não significa ausência de nucleação.

No CHE houve erro de acesso ao banco do sacct. Isso não é uma falha do cálculo.
Se persistir, reporte ao suporte; não é necessário reinstalar Python. squeue,
scontrol e logs continuam úteis. Para consumo atual, use sstat com os IDs
NUMÉRICOS das tarefas e .batch (não a notação ARRAY_INDICE). Eles aparecem no
nome do temporário no .out ou em scontrol show job ARRAY_INDICE. Exemplo:

```bash
sstat -j ID_NUMERICO.batch --format=JobID,AveCPU,AveRSS,MaxRSS
```

O banco ativo fica no disco local do nó. O helper seleciona SLURM_TMPDIR, TMPDIR
ou /tmp e publica checkpoints fechados com journal DELETE no destino
persistente. Uma escolha manual FOPT_SCRATCH_ROOT deve existir e ser gravável
NO NÓ; nunca use NFS para o banco ativo. Os logs mostram o caminho selecionado.
A home tem quotas: meça tamanho do piloto e espaço livre antes de ampliar.
Para produtos maiores, confirme uma pasta autorizada de /share/storage1 com o
grupo. Nunca armazene a única cópia importante em temporários do nó.
