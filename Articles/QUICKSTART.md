# Início rápido: scan adaptativo do artigo

Execute os comandos na raiz do repositório, com Python >=3.11.
A física, as unidades, a notação `m6`/`m8` e os dados de GW foram preservados.
O padrão agora é seguir a faixa com nucleação, sem percorrer toda a grade.

```bash
python -m pip install -e . pytest
python -m Articles.run_local --dry-run
```

## Os dois testes antes da produção

**Teste 1 — regra de parada e retomada, sem bounces reais:**

```bash
python -m pytest -q Articles/tests
```

Confere três ausências consecutivas nas duas direções de m6; reinício do
contador após nucleação; três linhas de C vazias; limites; semente deslocada;
falhas numéricas; matrizes e retomada por lotes, inclusive em outra pasta.

**Teste 2 — piloto real, duas linhas de C por cenário de m8:**

```bash
python -m Articles.run_local --smoke --beta-check --output Articles/results/pilot_adaptive
```

Inicia três processos independentes. Cada um verifica a semente, calcula a
linha em que ela nucleia e a linha seguinte abaixo. O piloto usa m6 finito
`2000, 1250, 500` GeV, além de `inf`, para exercitar três ausências sem centenas
de bounces. Essa grade grosseira serve somente ao teste. A busca inicial da
semente pode calcular outros C em m6=inf; portanto duas linhas completas não
significam somente dois pontos, nem somente dois C avaliados no bootstrap.
O teste controlado acima verifica a parada externa de três linhas vazias;
um piloto limitado a duas linhas não pode verificá-la por si só.

Para testar apenas uma fatia, use:

```bash
python -m Articles.collect_data --smoke --scenario-index 0 --beta-check --output Articles/results/pilot_m8_0
```

Confira `adaptive_state.json`: estado `pilot_complete`, duas linhas completas,
semente verificada e as paradas em m6. Um motivo `bound` indica que a grade
acabou antes de três ausências; não equivale a uma fronteira fechada. Uma
nucleação com beta/GW não resolvidos continua sendo `yes` para o scan, mas
exige revisão dos observáveis antes da análise física.

## Produção local, três frentes

Depois dos testes, use uma pasta nova:

```bash
python -m Articles.run_local --beta-check --output Articles/results/production_adaptive
```

| Índice | m8 [GeV] | Pasta |
|---|---:|---|
| 0 | inf | m8-0 |
| 1 | 840.8964152537145 | m8-1 |
| 2 | 668.740304976422 | m8-2 |

Cada processo percorre sua fatia sequencialmente: os próximos pontos dependem
dos resultados anteriores. A separação por m8 permite três tarefas simultâneas;
o ganho real depende do custo de cada fatia e dos recursos disponíveis.
Não use `--workers 3` nem divida a sequência adaptativa em shards.
Para uma única fatia de produção:

```bash
python -m Articles.collect_data --scenario-index 1 --beta-check --output Articles/results/production_m8_1
```

`collect_data` sem seleção percorre os três m8 sequencialmente na mesma pasta.
O botão Run do PyCharm deve usar o módulo `Articles.run_local` ou
`Articles.collect_data`, raiz do repositório como diretório e o mesmo Python
em que foi instalado `-e .`. Os argumentos começam em `--output`, sem `python -m`.

## Parâmetros e funcionamento

| Opção do coletor | Padrão | Uso |
|---|---|---|
| --strategy | adaptive | `grid` mantém a grade antiga para verificações |
| --seed-C | 3.83 | Semente inicial; verificada em cada m8 |
| --seed-m6 | inf | Operador de dimensão seis inicialmente desligado |
| --C MIN MAX STEP | 0 10 0.02 | Bounds e passo ancorado na semente |
| --m6 MIN MAX STEP | 500 2000 5 | Grade finita de massas, em GeV; inf é incluído |
| --patience | 3 | Ausências consecutivas para fechar uma direção |
| --scenario-index | todos | Seleciona 0, 1 ou 2 dos m8 padrão |
| --m8 VALOR ... | três cenários | Cenários personalizados; incompatível com scenario-index |
| --max-points N | sem limite | Limita pontos novos/reprocessados por chamada |
| --smoke | desligado | Duas linhas completas e grade grosseira de m6 |
| --dry-run | desligado | Confere configuração e compatibilidade, sem bounces |
| --retry-failed | desligado | Repete falhas/observáveis não resolvidos visitados |
| --beta-check | desligado | Confere beta com h/2, preservando h=0.5 GeV |

Os demais controles físicos do coletor continuam disponíveis: `--T-min`,
`--T-max`, `--phi-max`, `--n-phi`, `--n-T-seeds`, `--action-tolerance` e
`--beta-step`. `--workers`, `--shard-count`, `--shard-index` e `--no-baselines`
pertencem à execução `--strategy grid`. O launcher local expõe somente as
opções comuns mostradas em `python -m Articles.run_local --help`.

Diminuir m6 aumenta o operador de dimensão seis. Não se usa m6=0, nem se
introduz coeficiente negativo. O passo de 5 GeV em m6 corresponde a um passo
não uniforme no coeficiente 1/m6². O salto entre inf e 2000 GeV também faz
parte da resolução declarada. C segue 3.83, 3.85, ... para cima, e
3.81, 3.79, ... para baixo; os bounds 0 e 10 são incluídos se alcançados.

Se a semente em m6 declarado não nucleia, o bootstrap diminui C até verificar
uma semente ou atingir C_min. A regra de três linhas vazias passa a valer
somente após essa verificação. Cada linha explora m6 nas duas direções;
a mediana dos pontos com nucleação serve de semente para a próxima linha.
Após três linhas completas consecutivas sem nucleação, termina essa direção
em C. As duas direções de C têm contadores e sementes independentes.

A busca pressupõe uma faixa que possa ser acompanhada nessa resolução.
Ilhas além de três ausências, ou faixas que se deslocam mais que a janela
pesquisada entre linhas, podem ser perdidas. Pontos pulados ficam como
`not_evaluated`, nunca como exclusões físicas. Verifique fronteiras e ilhas
com refinamentos `--strategy grid` antes de resultados de publicação.

## Dados e retomada

Cada pasta contém `scan.sqlite`, `manifest.json`, `points.csv`,
`transitions.csv`, detalhes comprimidos, `adaptive_state.json`, matrizes CSV
e `matrix_axes.json`. Na execução de três frentes há uma matriz por pasta;
os índices das matrizes são locais, e `matrix_axes.json` identifica o m8.

As células são `yes` (nucleação aceita), `no` (ausência encontrada),
`unknown` (falha/inconclusivo) ou `not_evaluated`. `adaptive_state.json`
registra os últimos três m6/C sem nucleação e o primeiro valor pulado.
Beta/GW inconclusivos com nucleação aceita não encerram a faixa. Falhas
numéricas ou nucleação fora da tolerância interrompem a campanha, com o
ponto salvo, para não podar a região a partir de informação inconclusiva.

Repita o mesmo comando para retomar. O banco reconstrói a sequência sem
recalcular pontos confirmados. Código, ambiente e configuração devem coincidir
com o manifesto. Não reutilize as pastas da grade antiga, nem as do piloto
na produção. Para outra configuração, escolha outra pasta; não edite hashes.
Para repetir uma falha transitória, acrescente `--retry-failed`. Se a falha
persistir, o comando para novamente, sem um ciclo infinito.

Cluster e checkpoints em disco local: [CLUSTER_CHE.md](CLUSTER_CHE.md).
Modelo e interpretação dos dados: [README.md](README.md).
