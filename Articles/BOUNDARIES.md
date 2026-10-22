# Fronteiras primeiro, observáveis depois

Este é o fluxo recomendado para a nova campanha. O fluxo v3 em `run_local.py`,
`collect_data.py`, `cluster_runner.py`, `smoke.slurm` e `production.slurm` foi
preservado. Use destinos novos; nenhuma campanha em andamento foi interrompida
por esta revisão. Todas as alterações ficam em `Articles`.

Retomar a MESMA pasta v3 exige o código anterior. Novos módulos também mudam
os hashes: após atualizar, use o v3 como origem de `--reuse`, não como destino.
Não edite manifestos para forçar compatibilidade.

## O algoritmo e os padrões

- Etapa `boundaries`: somente fases, nucleação e VEVs em `Tn`; não calcula
  termodinâmica de GW, `alpha`, `beta/H`, verificação de meio passo ou espectros.
- Cada ponto tem **300 s**, incluindo imports/cálculo/serialização. O processo
  é encerrado ao atingir o teto, com até quatro segundos adicionais de limpeza.
  Salva `point_timeout / unknown`. **Não há recuperação refinada automática**
  no fluxo novo. O v3 conserva o comportamento anterior, sem esse teto.
- Critério padrão `--criterion strong`: nucleação aceita e
  **`abs(phi_low(Tn))/Tn > 1`**. `--criterion nucleation` busca qualquer
  nucleação aceita. `--strong-threshold` permite outro corte em campanha nova.
- Preserva `C=0..10`, `m6=500..2000 GeV`, passo de 5 GeV, mais `inf`, e os três
  `m8` anteriores. Não aumenta o passo de `m6` nem muda o potencial/ressoma.
- Preserva o alvo físico `S3/T=140`, com tolerância absoluta 0.5.

Em toda nova massa verifica uma semente novamente, começando pela faixa
anterior. Busca com passo 0.1 em C; se não encontra, completa a resolução de
0.02, incluindo bounds e C=1/3.2. **Não aplica a regra de três antes da
semente.** Faixas mais estreitas que essa resolução podem não ser encontradas.

Depois da semente, dá saltos crescentes no interior. Ao encontrar um ponto
externo, volta por bisseção até um bracket dentro/fora de largura **<=0.02 em C**.
Confirma com três pontos externos separados por 0.02, salvo no bound global.
As fronteiras da massa anterior são palpites verificados na nova massa.
Não calcula todos os pontos intermediários.

Um ponto fraco fica como `weak_nucleation`, separado de `no_nucleation_found`.
Uma transição forte aceita basta para incluir o ponto; uma secundária fraca
não o exclui. Candidatas não resolvidas impedem exclusão por força. O VEV é o
mínimo refinado em `Tn` usado pelo próprio núcleo no bounce. Também salva o
salto `delta_phi_over_Tn`; este difere de `vev_over_Tn` se a fase inicial já
for quebrada. O corte não prova detectabilidade e depende da prescrição/gauge:
[critério de força](https://arxiv.org/abs/1101.4665),
[parâmetros de GW](https://arxiv.org/abs/1512.06239).

Uma falha nunca é testemunha externa. Após três falhas consecutivas na direção,
registra a fronteira `unresolved` e passa adiante: cobertura **pendente**, não
exclusão. Se a faixa reaparece durante a confirmação, marca
`nonconnected_samples`. A busca pressupõe uma faixa conectada por coluna;
ilhas distantes exigem verificação dirigida. Três colunas completas sem região
selecionada na grade de sementes podem encerrar `m6`; colunas com unknown não
contam como vazias. `bound_limited` significa região chegando ao bound da busca,
sem certificar uma fronteira física além dele.

## Interromper/preservar o v3 antes de atualizar

Não atualize o checkout/venv enquanto os processos antigos usam esses arquivos.

### Local

Use Ctrl+C no terminal de `Articles.run_local` e aguarde o prompt/encerramento
das três fatias. O ponto em andamento pode ser perdido; os pontos anteriores
confirmados no SQLite ficam preservados. Copie **a pasta inteira**, incluindo
eventuais `scan.sqlite-wal` e `scan.sqlite-shm`; não os apague manualmente.
PowerShell, na raiz do projeto:

```powershell
$old = 'Articles/results/production_adaptive_v3'
$backup = 'Articles/results/production_adaptive_v3_backup_' + (Get-Date -Format 'yyyyMMdd-HHmmss')
Copy-Item -LiteralPath $old -Destination $backup -Recurse
git status --short
git branch --show-current
git pull --ff-only
python -m pip check
```

A branch deve ser `codex/articles-combined-potential-scan`. Preserve alterações
locais se houver conflito; não use reset/clean. Nenhum pacote novo é necessário.

### CHE

Confirme o ID atual. O array dos logs enviados era **108601**:

```bash
squeue -r -u "$USER"
scancel 108601
squeue -r -u "$USER"
```

Use esse cancelamento apenas se ainda for seu array atual. O v3 publica a cada
ponto: um encerramento abrupto pode perder o ponto atual de cada tarefa, sem
perder os checkpoints já publicados. O backup é do destino persistente, não
do `/tmp`. Depois de encerrar os jobs:

```bash
backup_dir="$HOME/fopt-articles-v3/production_v3_backup_$(date +%Y%m%d-%H%M%S)"
cp -a "$HOME/fopt-articles-v3/production_v3" "$backup_dir"
cd "$HOME/projects/GW_FOPT_Simulation"
git status --short
git branch --show-current
git pull --ff-only
export FOPT_PYTHON_MODULE=python/3.11.7-gcc-12.2.0
module load "$FOPT_PYTHON_MODULE"
export FOPT_PYTHON="$PWD/.venv-che311/bin/python"
"$FOPT_PYTHON" -m pip check
export FOPT_RESULTS_ROOT="$HOME/fopt-articles-v4"
export FOPT_REUSE_ROOT="$HOME/fopt-articles-v3/production_v3"
mkdir -p "$FOPT_RESULTS_ROOT" Articles/results/logs
unset FOPT_SCRATCH_ROOT
```

Use o módulo/venv já confirmados no seu ambiente, caso tenham outro nome.
Módulo/variáveis são selecionados em cada sessão; a venv permanece instalada.
Não rode bounces no nó de entrada.

## Teste 1: lógica, timeout e retomada

Localmente:

```powershell
python -m pytest -q Articles/tests
```

O piloto Slurm roda esse teste no nó de cálculo antes da física. Os testes novos
cobrem encerramento real por tempo, saltos abruptos, sementes estreitas,
brackets, confirmação, unknown, retomada, reuso e publicação fechada. Os testes
anteriores do v3 continuam na suíte.

## Teste 2: duas colunas reais e mapas

Localmente:

```powershell
python -m Articles.run_boundaries --smoke --reuse Articles/results/production_adaptive_v3 --output Articles/results/boundaries_pilot_v1
python -m Articles.boundary_status --output Articles/results/boundaries_pilot_v1
python -m Articles.plot_boundaries --output Articles/results/boundaries_pilot_v1
```

O piloto localiza as duas fronteiras reais de `m6=inf` e `2000 GeV` em cada
`m8`; não limita a busca a três passos internos como o piloto v3. Pode levar
dezenas de minutos, dependendo das falhas, mas cada ponto tem teto. Não há
promessa de prazo total fixo para encontrar uma semente.

CHE, depois das exportações anteriores:

```bash
sbatch --export=ALL --nodelist=milliways03 Articles/cluster/boundaries_smoke.slurm
squeue -r -u "$USER"
"$FOPT_PYTHON" -m Articles.boundary_status --output "$FOPT_RESULTS_ROOT/boundaries_pilot_v1"
```

Depois de terminar:

```bash
"$FOPT_PYTHON" -m Articles.plot_boundaries --output "$FOPT_RESULTS_ROOT/boundaries_pilot_v1"
```

São três tarefas, uma CPU/4 GB cada, até 2 h, em `milliways-ext`. Omitir
`--nodelist` permite ao Slurm escolher o nó. O reuso é opcional.

Confira `boundary_state.json`, `boundaries.csv` e `plots/boundary_map.png`
em cada fatia. O comparativo fica em `plots/boundary_overview.png`; há SVGs.
Somente os marcadores são pontos calculados; o sombreado interno é **inferido**.
Brackets pendentes são hachurados. `pilot_complete_with_unresolved` e saída 3
exigem revisão; não equivalem a um piloto inteiramente resolvido.

## Produção: somente fronteiras

Depois de avaliar o piloto local:

```powershell
python -m Articles.run_boundaries --reuse Articles/results/production_adaptive_v3 --reuse Articles/results/boundaries_pilot_v1 --plots --output Articles/results/boundaries_v1
```

Depois de avaliar o piloto do CHE:

```bash
sbatch --export=ALL --nodelist=milliways03 Articles/cluster/boundaries.slurm
```

São três tarefas, uma CPU/8 GB cada, alocações de 14 h. Publica cada ponto,
mantém SQLite ativo no disco local e publica snapshots fechados sem WAL ativo.
O aviso de 15 min pede parada após o ponto limitado pelo orçamento.
Retome repetindo a mesma submissão/destino, com código/ambiente/opções fixos.
Para uma fatia: `sbatch --export=ALL --nodelist=milliways03 --array=1 Articles/cluster/boundaries.slurm`.
Não há ressubmissão automática. Colunas já terminadas, inclusive pendentes,
não são reexploradas na retomada da mesma campanha; `--retry-failed` reabre
a procura. Novos dados importados podem reabrir as massas afetadas.

Os templates leem `FOPT_POINT_TIMEOUT=300`, `FOPT_BOUNDARY_CRITERION=strong` e
`FOPT_C_TOLERANCE=0.02`. Alterá-los exige outro destino. Para outras opções,
consulte `python -m Articles.boundary_scan --help`.

No novo fluxo, Ctrl+C pede parada após o ponto limitado. Um arquivo `STOP` na
raiz da campanha pausa as três fatias; dentro de uma fatia pausa só ela. Remova
somente esse arquivo de controle depois de parar, antes de retomar. O CHE também
aceita USR1 no batch, encaminhado pelo template. `paused` não significa conclusão.

## Segunda etapa: observáveis no interior

Ela nunca começa automaticamente. Usa C com passo 0.02 ancorado em 3.83,
mais referências C=1/3.2, sementes e extremidades internas. Só usa colunas
`bounded`; a parte incerta dos brackets fica fora. Cada ponto verifica novamente
nucleação/observáveis dentro do mesmo teto. Avisos de beta/GW impedem uma
conclusão limpa, mesmo se a nucleação for aceita.

Primeiro, piloto com dois Cs interiores por coluna delimitada:

```powershell
python -m Articles.run_boundaries --stage interior --smoke --boundaries Articles/results/boundaries_pilot_v1 --reuse Articles/results/production_adaptive_v3 --output Articles/results/interior_pilot_v1
```

Esse piloto gera mapas/espectros anteriores com comparação dos detectores,
sem criar espectros para observáveis inconclusivos. Depois, produção:

```powershell
python -m Articles.run_boundaries --stage interior --boundaries Articles/results/boundaries_v1 --reuse Articles/results/production_adaptive_v3 --plots --output Articles/results/interior_v1
```

Os equivalentes no CHE são `Articles/cluster/interior_smoke.slurm` e
`Articles/cluster/interior.slurm`. `FOPT_BOUNDARIES_ROOT` escolhe a origem; os
padrões são `boundaries_pilot_v1` e `boundaries_v1` dentro de `FOPT_RESULTS_ROOT`.
Os destinos são `interior_pilot_v1` e `interior_v1`.

Fronteiras parciais/inconclusivas bloqueiam essa etapa por padrão. Depois de
revisar, `--allow-partial-boundaries` permite também usar o intervalo conservador
entre testemunhas positivas de uma coluna pendente. Exige duas bordas com
testemunhas internas; não inclui a faixa incerta externa. Registra colunas
parciais/omitidas e mantém saída 3. No template: `FOPT_ALLOW_PARTIAL_BOUNDARIES=1`.
Isso não resolve nem exclui as bordas pendentes. Colunas comprovadamente vazias
na grade de sementes são exclusões do recorte, não pendências.

## Reuso e interpretação

`--reuse` somente lê a origem. Exige mesmos hashes de núcleo/modelo, versões
NumPy/SciPy e configurações físicas relevantes. Se incompatível, não apague
manifestos: investigue ou omita o reuso para recalcular em destino novo. Cada
registro importado conserva origem/fingerprint. Falhas antigas não entram como
negativas. Unknown de outra campanha de fronteiras com o mesmo orçamento é
conservado como unknown, sem repetir o custo automaticamente. Na segunda etapa,
pontos sem espectro válido são recalculados. A origem DB+WAL é copiada por
leitura de arquivos e aberta apenas numa cópia local, sem criar locks/SHM na
origem; se ela mudar durante a cópia, o reuso é recusado.

`--max-points N` pausa após N avaliações novas; retome sem o limite.
`--retry-failed` permite repetir unknown revisitados pela estratégia uma vez
por execução, com o mesmo orçamento e
sem refinamento automático. Alterar o orçamento exige destino novo, podendo
reutilizar os resultados confiáveis anteriores.

`complete` significa fim da estratégia nas hipóteses/resolução declaradas, não
prova ausência de ilhas. Estados `_with_unresolved` e saída 3 conservam
pendências. Mapas/CSV distinguem corte de força, ausência encontrada, timeout e
faixa inferida; nenhuma falha numérica prova indetectabilidade.

No Slurm, saída 3 pode aparecer como FAILED mesmo quando a busca terminou
com pendências registradas; consulte o estado científico antes de interpretar
a saída do escalonador. Uma falha de infraestrutura também pode gerar FAILED.
