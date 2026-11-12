# Início rápido: scan adaptativo do artigo

Para a campanha nova, siga [BOUNDARIES.md](BOUNDARIES.md): backup/parada do v3,
dois testes, mapas de fronteiras e etapa interior separada, com teto de 300 s
por ponto. Os comandos abaixo documentam o fluxo completo v3 preservado.

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

Confere a busca sem poda antes da primeira nucleação em cada m6/m8, saltos
abruptos da faixa, três ausências em C depois da semente, reinício do contador,
três colunas m6 vazias somente após esgotar C, bounds, erros, retomada e plots.

**Teste 2 — piloto real com visualização, duas massas m6 por cenário de m8:**

```bash
python -m Articles.run_local --smoke --beta-check --output Articles/results/pilot_adaptive_v3
```

Inicia três processos independentes. Em cada m8, calcula a coluna m6=inf e
em seguida m6=2000 GeV. **Cada coluna procura sua própria primeira nucleação
em C, sem a regra dos três pontos nessa procura**, inclusive no primeiro
salto de inf para massa finita. Depois da semente, o piloto avalia no máximo
três passos de C para cada lado. A grade finita grosseira declarada é
2000, 1250, 500 GeV, mas o limite de duas colunas deixa as duas últimas
massas não calculadas. O piloto não é uma medição completa da faixa.

Uma parada `pilot_step_limit` indica o limite de teste, sem excluir a região
além dele. `consecutive_no_nucleation` guarda as três ausências após a semente;
`bound` indica que o eixo acabou. A busca inicial da semente pode exigir muitos
C; o limite de três passos do piloto só vale **depois** de encontrá-la.

Para testar apenas uma fatia:

```bash
python -m Articles.collect_data --smoke --scenario-index 0 --beta-check --output Articles/results/pilot_m8_0_v3
```

Confira `adaptive_state.json`: estado `pilot_complete`, duas colunas em
`slices[].columns`, suas sementes `verified_seed_C` e os motivos de parada.

### Quais imagens abrir

Em cada pasta `m8-0`, `m8-1`, `m8-2`, a subpasta `plots/` contém:

| Arquivo | O que mostra |
|---|---|
| nucleation_map.png / .svg | C por m6, sementes verificadas e as três ausências que encerraram uma direção |
| gw_spectra.png / .svg | Até três C selecionados, espectros e comparação dos picos acústicos com LISA, DECIGO e BBO |
| selection.json | Pontos escolhidos, observáveis rejeitados, eficiências e versão gráfica |
| spectrum_samples.csv | Frequências em Hz e componentes acústica, turbulência, colisões e total |

O launcher também salva `pilot_adaptive_v3/plots/nucleation_overview.png`,
comparando as três fatias. Verde: nucleação aceita; laranja: ausência encontrada;
roxo: inconclusivo; cinza: não calculado. Círculo preto: semente; cruzes: três
ausências de parada. Um losango vermelho sobre um hit indica observáveis sob
revisão, sem apagar a nucleação aceita.

Os espectros são reconstruídos dos parâmetros salvos, sem novos bounces.
Somente transições aceitas com observáveis resolvidos entram no painel de GW;
se não houver nenhuma, a imagem informa isso explicitamente. A seleção cobre
C baixo/intermediário/alto dentre os disponíveis; em C empatado prefere m6
finito. Não usa valor absoluto de alpha para esconder sinal inválido.

As curvas disponíveis no núcleo são **PIS do canal acústico**, sensibilidades
integradas de pico. Por isso o painel direito compara os picos acústicos com
elas; o painel esquerdo mostra os espectros total e acústico. Cruzamentos do
espectro total com PIS não são tratados como SNR ou garantia de detecção.
A turbulência usa epsilon=0.05 e colisões kappa=0, conforme o manifesto.

Para redesenhar imagens a partir de uma campanha já salva:

```bash
python -m Articles.plot_results --output Articles/results/pilot_adaptive_v3
```

Isso não retoma o scan nem altera os dados físicos. Também aceita uma única
pasta `m8-N`. `--plots` permite gerar figuras fora do piloto; `--no-plots`
desliga o desenho automático no coletor. O runner desenha uma vez ao publicar
seu checkpoint final, inclusive diagnóstico parcial quando possível.

## Produção local, três frentes

Depois dos testes, use uma pasta nova:

```bash
python -m Articles.run_local --beta-check --output Articles/results/production_adaptive_v3
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
python -m Articles.collect_data --scenario-index 1 --beta-check --output Articles/results/production_m8_1_v3
```

`collect_data` sem seleção percorre os três m8 sequencialmente na mesma pasta.
O botão Run do PyCharm deve usar o módulo `Articles.run_local` ou
`Articles.collect_data`, raiz do repositório como diretório e o mesmo Python
em que foi instalado `-e .`. Os argumentos começam em `--output`, sem `python -m`.

## Parâmetros e funcionamento

| Opção do coletor | Padrão | Uso |
|---|---|---|
| --strategy | adaptive | `grid` mantém a grade antiga para verificações |
| --seed-C | 3.83 | C candidato inicial; confirmado independentemente em cada m6/m8 |
| --seed-m6 | inf | Operador de dimensão seis inicialmente desligado |
| --C MIN MAX STEP | 0 10 0.02 | Bounds e passo ancorado na semente |
| --m6 MIN MAX STEP | 500 2000 5 | Grade finita de massas, em GeV; inf é incluído |
| --patience | 3 | Ausências consecutivas para fechar uma direção |
| --scenario-index | todos | Seleciona 0, 1 ou 2 dos m8 padrão |
| --m8 VALOR ... | três cenários | Cenários personalizados; incompatível com scenario-index |
| --max-points N | sem limite | Limita pontos novos/reprocessados por chamada |
| --smoke | desligado | Duas colunas m6; busca da semente sem corte; até três passos C por lado depois dela; plots |
| --plots / --no-plots | automático no piloto | Controla a geração de figuras do coletor |
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

Em cada novo m6, o programa parte do maior C nucleado na coluna anterior
(3.83 inicialmente) e desce C até localizar nucleação, sem cortar após três
ausências. Se chega a C_min sem semente, confere também os C acima da
candidata até C_max, para não assumir que a faixa sempre se desloca para baixo.
Uma coluna só é declarada vazia depois de esgotar a grade de C inteira.

**Somente depois da semente verificada nessa mesma massa**, explora C para
cima e para baixo; três ausências consecutivas encerram cada direção, com
contadores independentes. Em seguida passa ao próximo m6 e repete a busca da
semente. Três colunas m6 consecutivas completamente vazias encerram uma direção
em m6, depois de já haver nucleação verificada. Antes da primeira semente de
uma direção não se aplica essa poda. Os bounds sempre limitam o trabalho.

A busca inicial pode aumentar o custo quando não há nucleação, porque exige
conferir todos os C antes de declarar a coluna vazia. O programa não promete
um número fixo de avaliações ou tempo de execução. `--max-points` permite
limitar uma chamada e retomá-la depois, inclusive no meio do bootstrap.

O salto abrupto da faixa entre massas não dispara mais a antiga parada de
três C antes de uma semente. Ainda existem limites da amostragem: faixas mais
estreitas que o passo de C podem ser perdidas, assim como ilhas além de três
ausências após a semente ou além de três colunas vazias. Pontos pulados ficam
`not_evaluated`; refinamentos `--strategy grid` podem conferir essas regiões.

## Dados e retomada

Cada pasta contém `scan.sqlite`, `manifest.json`, `points.csv`,
`transitions.csv`, detalhes comprimidos, `adaptive_state.json`, matrizes CSV
e `matrix_axes.json`. Na execução de três frentes há uma matriz por pasta;
os índices das matrizes são locais, e `matrix_axes.json` identifica o m8.

As células são `yes` (nucleação aceita), `no` (ausência encontrada),
`unknown` (falha/inconclusivo) ou `not_evaluated`. `adaptive_state.json`
registra o bootstrap de cada m6, as três ausências em C, colunas vazias
e o primeiro valor pulado. Os limites do piloto ficam explícitos.
Beta/GW inconclusivos com nucleação aceita não encerram a faixa. Falhas
numéricas ou nucleação fora da tolerância ficam unknown após a tentativa
limitada de recuperação. A busca continua sem usar esses pontos para podar.
O modo --on-unresolved stop mantém a interrupção estrita opcional.

Repita o mesmo comando para retomar. O banco reconstrói a sequência sem
recalcular pontos confirmados. Código, ambiente e configuração devem coincidir
com o manifesto. Esta versão tem identidade `adaptive_v3_audited_unknowns`. Não reutilize
pastas da versão anterior, nem as do piloto na produção. Use uma pasta nova,
como `pilot_adaptive_v3`, preservando os resultados antigos. Para outra configuração, escolha outra pasta; não edite hashes.
Para repetir uma falha transitória, acrescente `--retry-failed`. Se a falha
persistir, permanece unknown e a busca segue, sem um ciclo infinito de retries.

Cluster e checkpoints em disco local: [CLUSTER_CHE.md](CLUSTER_CHE.md).
Modelo e interpretação dos dados: [README.md](README.md).


## Encerramento e pontos inconclusivos (v3)

```bash
python -m Articles.status --output Articles/results/production_adaptive_v3
python -m Articles.status --output Articles/results/production_adaptive_v3 --json
```

O estado mostra se foi campanha completa, piloto, limite de pontos ou falha.
A v3 tenta recuperar um ponto inconclusivo uma vez, sem relaxar S3/T. Se o erro
persistir, registra unknown e continua os outros C/m6. Unknown interrompe a
contagem de ausências; uma coluna com unknown não pode fechar o eixo m6 como
coluna vazia. O estado final ganha _with_unresolved e os executores retornam 3
quando há revisão pendente. Esse código não apaga o scan nem indica que toda
sua busca parou. O modo estrito opcional é --on-unresolved stop.

A grade mantém o passo C=0.02 ancorado em 3.83, acrescentando C=1 e C=3.2 como
cortes de referência exatos. A faixa c6/f²=3.2..3.45 TeV^-2 da dissertação em
C=1 corresponde a m6 aproximadamente 538..559 GeV, dentro de 500..2000. A
figura da dissertação é sem daisy; esta campanha continua com daisy ligado.
O piloto de duas massas não pretende mapear essa região de m6 baixo.

run_local sem argumentos executa produção; --smoke executa o piloto. A saída
padrão é Articles/results/production_adaptive_v3, ancorada na raiz do checkout
mesmo ao iniciar pelo botão Run. O resumo final informa o estado de cada m8.
Não reutilize manifestos/dados v2 como v3: todas as mudanças estão em Articles,
mas seus hashes numéricos fazem parte da identidade da campanha.
