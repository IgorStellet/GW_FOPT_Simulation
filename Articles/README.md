# Dados para o artigo sobre mecanismos combinados de FOPT

**Novo fluxo recomendado:** [fronteiras primeiro, interior depois](BOUNDARIES.md).
Use `run_boundaries.py` para procurar fronteiras com orçamento por ponto e corte
de força opcional. O fluxo completo v3 abaixo continua disponível. Preserve e
pare campanhas em andamento antes de atualizar o checkout; use destinos novos.

Primeira execução: [guia rápido](QUICKSTART.md). Primeiro acesso e submissão:
[roteiro do CHE](CLUSTER_CHE.md), com templates em [cluster/](cluster/).

Esta pasta reúne a coleta de dados do artigo. O ponto de entrada é
[`collect_data.py`](collect_data.py): ele percorre o espaço de parâmetros,
identifica fases e transições, calcula os parâmetros termodinâmicos e grava os
resultados. O piloto também produz mapas e espectros diagnósticos. Não é necessário repetir
o cálculo de tunelamento para reconstruir os espectros a partir dos parâmetros
salvos.

O cálculo reutiliza os módulos de [`src/CosmoTransitions`](../src/CosmoTransitions)
para o potencial térmico, rastreamento de fases, bounce, derivadas e ondas
gravitacionais. A camada do artigo organiza o modelo, a execução e a persistência;

## 1. O modelo e suas unidades

A parte polinomial do potencial é parametrizada diretamente por duas escalas de
massa positivas:

$$
V_0(\phi)=-\frac{\mu^2}{2}\phi^2+\frac{\lambda}{4}\phi^4
          +\frac{\phi^6}{8m_6^2}+\frac{\phi^8}{16m_8^4}.
$$

Assim, `m6` e `m8` são dados em **GeV**. Diminuir uma dessas massas aumenta a
deformação correspondente. A entrada `inf` desliga o operador; uma massa igual a
zero não faz isso.

A normalização segue as equações (421)–(425), pp. 133–134 da dissertação. Na
notação anterior, os coeficientes que multiplicavam os operadores eram
$c_6/f^2=m_6^{-2}$ e $c_8/f^4=m_8^{-4}$. Isso explica os três valores
iniciais abaixo; a execução usa diretamente as massas, sem conversão entre
parametrizações:

| Cenário | `m8` [GeV] | Correspondência na dissertação, com `f = 1000 GeV` |
|---|---:|---:|
| Sem operador de dimensão oito | `inf` | `c8 = 0` |
| Primeiro cenário com dimensão oito | `840.8964152537145` | `c8 = 2` |
| Segundo cenário com dimensão oito | `668.740304976422` | `c8 = 5` |

O setor de medida funcional usa $\Lambda=1000$ GeV e a contribuição logarítmica
renormalizada da equação (428), p. 135. Os termos são combinados conforme a
equação (444), p. 156, preservando as condições locais do vácuo eletrofraco a
temperatura zero. A motivação científica e a discussão de degenerescências estão
nas seções 5.6–5.7. As páginas indicadas são as páginas impressas da dissertação.

Para `C > 0`, o argumento do logaritmo exige
$\phi<\Lambda/\sqrt C$. O limite de campo configurado não autoriza ultrapassar
essa fronteira.

## 2. Prescrições físicas e numéricas

O scan fixa a **ressoma térmica  ligada**.
Não há comparação entre opções de ressoma nesta coleta. 

A nucleação é procurada pelo critério $S_3(T_n)/T_n\simeq140$, com tolerância
absoluta padrão **0.5 no valor de `S3/T`**. Essa tolerância não significa uma
incerteza de 0.5 GeV na temperatura. O intervalo inicial de busca é de 1 a 250 GeV.

O cálculo de $\beta/H$ usa uma derivada central de segunda ordem, com duas
avaliações da ação:

$$
\frac{\beta}{H}=T_n\frac{F(T_n+h)-F(T_n-h)}{2h},
\qquad F(T)=\frac{S_3(T)}{T}.
$$

O passo padrão é `h = 0.5 GeV`. As amostras da ação são preservadas para que a
derivada possa ser auditada. A opção `--beta-check` calcula também o stencil com
`h/2`, mantendo a ordem dois, para avaliar a sensibilidade ao passo. O coletor não
suaviza a dependência de beta no espaço de parâmetros nem substitui valores por
um ajuste polinomial.

A intensidade usada nas previsões de GW é `alpha_trace`, com a diferença entre
as fases mantendo seu sinal. `alpha_energy` também é armazenada. Valores
incompatíveis com as hipóteses do template não devem ser corrigidos tomando seu
módulo. A configuração completa das previsões fica registrada com os dados.

## 3. Busca adaptativa e primeira execução

O padrão é `--strategy adaptive`, implementado em
[adaptive_scan.py](adaptive_scan.py). Parte de C=3.83 e m6=inf para cada m8,
verifica uma semente e acompanha a faixa com nucleação. Todos os cálculos de
cada ponto continuam sendo os do coletor e do núcleo existentes.

1. Em cada m6, inclusive o primeiro finito após inf, desce C a partir da
   candidata herdada até encontrar uma nucleação verificada. Nessa busca não
   se usa a regra de três ausências.
2. Se não encontra semente até C_min, confere também os C acima da candidata
   até C_max, sem pressupor monotonicidade. Só declara uma coluna vazia depois
   de esgotar seu eixo C inteiro sem nucleação. Inconclusivos são auditados e não contam como ausência.
3. Depois da semente confirmada **nessa coluna**, explora C nas duas direções;
   três ausências consecutivas encerram cada direção, com reinício após um hit.
4. Passa ao próximo m6 e verifica outra semente, reutilizando o maior C
   nucleado como candidata. Três colunas m6 completamente vazias consecutivas
   encerram uma direção em m6 somente após haver nucleação verificada.
5. O piloto calcula duas colunas, m6=inf e 2000 GeV, com a procura de semente
   sem corte. Depois de cada semente limita C a três passos por lado e gera
   mapas/espectros. `pilot_step_limit` não é uma fronteira física.

A massa inf desliga o operador; as massas finitas continuam positivas. Não
se pesquisa acoplamento negativo. O passo de m6 é uniforme em GeV, portanto
não é uniforme em 1/m6². O padrão finito é 500..2000 GeV, passo 5 GeV;
C é limitado a [0,10], passo 0.02, ancorado em 3.83. Incluem-se os extremos
se a busca os alcançar. Os três m8 permanecem os da tabela anterior.

O salto entre massas não depende mais de uma semente apenas herdada.
A hipótese restante é uma faixa conectada de C depois de uma semente, na resolução escolhida.
A regra dos três pontos não prova ausência de ilhas desconectadas. Regiões
puladas são explicitamente não avaliadas; fronteiras, mudança de resolução
e possíveis ilhas precisam de verificações direcionadas para publicação.

Instalação, testes e produção local em três processos independentes:

```bash
python -m pip install -e . pytest
python -m pytest -q Articles/tests
python -m Articles.run_local --dry-run
python -m Articles.run_local --smoke --beta-check --output Articles/results/pilot_adaptive_v3
python -m Articles.run_local --beta-check --output Articles/results/production_adaptive_v3
```

Execute produção depois de conferir os dois testes. Cada processo calcula
um m8 e usa sua própria pasta. `collect_data` também aceita
`--scenario-index 0`, `1` ou `2`; sem essa opção calcula os três em sequência.
Cada fatia é sequencial porque depende das decisões anteriores. O ganho de
três tarefas simultâneas depende do balanceamento entre as fatias.

A grade antiga continua disponível com `--strategy grid`. Ela deve ser uma
escolha explícita. Seu padrão histórico tinha 453.906 pontos; isso deixou de
ser a quantidade de avaliações do scan padrão. O teto da grade adaptativa
pode ser grande, mas ela calcula somente o que a busca solicita.

## 4. Configuração da campanha

[QUICKSTART.md](QUICKSTART.md) documenta todas as opções principais e exemplos.
`--seed-C`, `--seed-m6`, `--patience`, `--C` e `--m6` controlam a busca.
`--scenario-index` seleciona um dos três m8; `--m8` permite cenários próprios.
`--max-points` é um limite por chamada, independente da identidade científica
da campanha. `--dry-run` confere a identidade sem criar dados.

Os parâmetros físicos e tolerâncias continuam em `Settings` e nas opções
do coletor. `--beta-check` preserva a comparação h/h2. `--workers`, shards e
`--no-baselines` continuam disponíveis somente para `--strategy grid`.
O adaptativo sempre inclui m6=inf, mas não calcula automaticamente todos os
C=0 nem acrescenta m8=inf a uma lista personalizada: os valores não visitados
ficam não avaliados. Referências puras adicionais podem ser calculadas pela
grade explícita em outra pasta.

## 5. Arquivos e retomada

| Arquivo | Conteúdo |
|---|---|
| scan.sqlite | Pontos e transições, com commit por ponto; fonte da retomada |
| manifest.json | Modelo, configuração, versões e hashes das fontes |
| points.csv / transitions.csv | Diagnósticos e observáveis preservados do coletor |
| details/*.json.gz | Fases, transições críticas, ações, avisos e erros |
| adaptive_state.json | Bootstrap, sementes verificadas por coluna m6 e evidência das paradas |
| plots/*.png / .svg | Mapas do piloto, espectros e picos comparados às PIS |
| plots/selection.json / spectrum_samples.csv | Seleção, exclusões e componentes espectrais em Hz |
| matrix_m8_N.csv | Matriz C por m6: yes, no, unknown ou not_evaluated |
| matrix_axes.json | Eixos e correspondência entre N e m8, por pasta |

`scan_decision` é independente do status dos observáveis: uma FOPT aceita
continua `yes` mesmo que beta ou GW estejam não resolvidos. Uma falha numérica ou residual de nucleação não aceito é `unknown`. Após uma
recuperação numérica limitada, o scan padrão continua nos próximos C/m6, com
registro salvo. Um unknown quebra a sequência de ausências consecutivas.
`--on-unresolved stop` mantém a interrupção estrita opcional. Só `no_first_order_found` e `no_nucleation_found` contam como
ausência para a poda. Os últimos três pontos de C ou colunas de m6 que justificam cada
parada são salvos; atingir um bound é registrado separadamente.

Repita o mesmo comando para retomar. A sequência é reconstruída a partir dos
resultados gravados, sem repetir bounces confirmados. `--retry-failed` tenta
reparar falhas/observáveis não resolvidos visitados, uma vez por chamada.
Código, versões, parâmetros ou estratégia diferentes exigem outra pasta.
A identidade agora é `adaptive_v3_audited_unknowns`; use outra pasta para
esta revisão e preserve a campanha anterior. A v3 inclui os cortes exatos C=1 e
C=3.2, além da grade ancorada em 3.83 e dos extremos 0 e 10. Não reutilize dados do piloto na produção.
`--export-only` atualiza points/transitions CSV a partir do banco; as matrizes
adaptativas e o estado são regenerados ao retomar a busca.

No cluster, [cluster_runner.py](cluster_runner.py) conserva o SQLite ativo no
disco local e publica checkpoints fechados, com journal DELETE no destino
persistente. [CLUSTER_CHE.md](CLUSTER_CHE.md) explica os três jobs, um por m8.

Os gráficos não resolvem bounces novamente. `gw_spectra` separa o espectro
à esquerda e a comparação dos picos acústicos com os detectores à direita.
As curvas existentes são PIS de canal acústico; não se interpretam cruzamentos
delas com o espectro total como SNR. Observáveis não resolvidos são omitidos dos
espectros, com motivo em `selection.json`. Detalhes e comandos para redesenhar:
[QUICKSTART.md](QUICKSTART.md).

## 6. Como interpretar as tabelas

O mapa do artigo deve separar pelo menos: transição encontrada com nucleação,
ausência de transição encontrada, ausência de raiz de nucleação no intervalo
investigado e falha de cálculo. **Uma falha numérica nunca é um ponto não
detectável.** Tampouco a ausência de uma raiz no intervalo escolhido prova que a
transição não nucleia em nenhuma temperatura.

Preserve os identificadores dos pontos ao cruzar tabelas. O critério aproximado `S3/T = 140`
não constitui, por si só, um cálculo de percolação ou de conclusão da transição.

As colunas de frequência do coletor são expressas em **Hz**. As funções de
[`gravitational_Waves.py`](../src/CosmoTransitions/gravitational_Waves.py),
incluindo `gw_omega_total_h2`, recebem frequências em **mHz**. Ao reconstruir um
espectro com uma malha em Hz, passe `f_mHz = 1000 * f_Hz`. As densidades
espectrais são `h² Ω`, adimensionais. Temperaturas e campos estão em GeV; o
potencial está em GeV⁴; `S3` está em GeV; `S3/T`, alpha e beta/H são
adimensionais.

A razão entre o pico acústico e a sensibilidade integrada de pico acústica
(PIS) é um **indicador sob as hipóteses dessas curvas**. Ela não é um cálculo
geral do SNR do espectro total nem uma garantia de detecção. O segundo código
deverá declarar o detector, a curva, o critério e as hipóteses ao produzir regiões
classificadas como detectáveis. Os parâmetros e diagnósticos salvos permitem
rever essa classificação sem resolver novamente todos os bounces.

As colunas `LISA_sw_PIS_ratio`, `DECIGO_sw_PIS_ratio` e `BBO_sw_PIS_ratio` guardam
essas razões, sem aplicar um limiar de detecção arbitrário. Os ajustes PIS são
os de [Schmitz, arXiv:2002.04615](https://arxiv.org/abs/2002.04615).
Ao reconstruir o espectro total, use também `spectrum_defaults` do manifesto:
`epsilon_turb=0.05` e `kappa_coll=0`, além de `v_w=1`, `g_star=106.75`,
`alpha=alpha_trace` e `T_star=Tn`. Passe esses argumentos explicitamente:
o padrão histórico de turbulência do núcleo usa outra eficiência. A velocidade
da parede e as eficiências são hipóteses fenomenológicas; não foram determinadas
dinamicamente pelo potencial.

Os estados salvos em `points.csv` são:

| `status` | Interpretação |
|---|---|
| `nucleated` | Histórico contém FOPT nucleada e os observáveis solicitados foram calculados |
| `no_first_order_found` | Não foi localizada FOPT crítica nem nucleada na busca realizada |
| `no_nucleation_found` | Há FOPT crítica, mas nenhuma FOPT nucleada foi localizada no histórico |
| `observables_unresolved` | Há candidato a FOPT, mas residual, stencil ou observáveis exigem revisão; consultar `quality_flags` |
| `numerical_failure` | A execução do ponto falhou; consultar erro e detalhes |

Cruze `points.csv` e `transitions.csv` por `point_id`; cada transição tem seu
`transition_index`. Colunas vazias são valores não obtidos, não zeros físicos.
Massas desligadas aparecem como `inf` (string em JSON). `ew_is_lowest_sampled`
e `ew_below_origin` permitem selecionar pontos com vácuo eletrofraco adequado
**na janela investigada**. O sinalizador `strong_by_field_ratio` corresponde à
convenção operacional `abs(phi_low-phi_high)/Tn >= 1`, não a um cálculo de
washout por sphalerons. O histórico segue um caminho de resfriamento do núcleo;
as demais degenerescências críticas ficam nos detalhes.

## Organização e relação com o núcleo

Mapa e inventário das funções: [CORE_MAP.md](CORE_MAP.md).

| Arquivo | Responsabilidade nesta coleta |
|---|---|
| `Articles/combined_model.py` | Define somente a física específica do potencial e diagnósticos em T=0 |
| `Articles/adaptive_scan.py` | Semente verificada por m6, faixas de C, paradas e matrizes |
| `Articles/plot_results.py` | Mapas, reconstrução de espectros e comparação de picos/PIS sem bounces |
| `Articles/core_imports.py` | Mesmo núcleo instalado ou a partir de src no checkout |
| `Articles/run_local.py` | Três processos independentes, um por m8 |
| `Articles/collect_data.py` | Coordena a estratégia, chama o núcleo e salva os resultados sem produzir figuras |
| `src/CosmoTransitions/__init__.py` | Expõe a interface pública do pacote |
| `finiteT.py` | Integrais térmicas e splines de Jb/Jf |
| `helper_functions.py` | Pesos de diferenças finitas, gradiente, Hessiana, interpolação e integradores |
| `generic_potential.py` | Adaptação de formato, derivadas térmicas e coordenação de fases/transições |
| `transitionFinder.py` | Mínimos, histórias de fases, temperaturas críticas e nucleação |
| `tunneling1D.py` | Bounce O(3), integração radial e ação euclidiana |
| `gravitational_Waves.py` | Termodinâmica, beta/H, espectros e ajustes PIS |

Esta revisão altera somente Articles; o núcleo permanece intacto. Não importamos
o script da dissertação. As alterações já existentes na branch no núcleo
acrescentam `order=2` sem remover o padrão anterior `order=4`, refinam os mínimos
antes de calcular a ação, explicitam as duas convenções de alpha e corrigem
unidades em turbulência/saídas Hz e a ordenação de fases recebidas fora de ordem.
As splines térmicas mantêm os domínios e continuações reais do núcleo existente;
o tratamento CW de Goldstones preserva os contratermos de passo finito da tese.
Essas prescrições também precisam ser consideradas ao interpretar os resultados.

Os testes e a execução de validação estão descritos em [VALIDATION.md](VALIDATION.md).

## 7. Refinamento para resultados de publicação

Uma única resolução e a regra de três ausências não certificam convergência. Primeiro localize as regiões de interesse; depois
recalcule fronteiras, pontos representativos e regiões com beta instável em
outra pasta. Por exemplo:

```bash
python -m Articles.collect_data --strategy grid --output Articles/results/refinement --m6 750 850 2 --C 3.0 3.4 0.005 --m8 inf --beta-check
```

Compare os resultados com variações do passo em temperatura, resolução da busca
de fases e tolerâncias. A derivada de ordem dois reduz o número de avaliações e
muda a sensibilidade ao ruído; ela não elimina automaticamente o erro numérico.
Evite interpolar através de falhas ou tratar regiões não investigadas como
fisicamente excluídas. Os benchmarks da dissertação, Apêndice E, são referências
úteis, considerando as mudanças de ressoma e de derivada e o arredondamento das
tabelas.

Para os gráficos do artigo, a estrutura dos dados permite comparar o plano
`(C, m6)` entre cenários de `m8`, sobrepor referências puras, estudar
super-resfriamento, alpha e beta/H, reconstruir espectros selecionados e procurar
pontos com previsões semelhantes. A interpretação das degenerescências continua
dependendo das hipóteses físicas registradas em cada campanha.


## Diagnóstico de encerramento e recuperação v3

```bash
python -m Articles.status --output Articles/results/production_adaptive_v3
```

O diagnóstico lê JSON/CSV, sem calcular bounces nem modificar a campanha.
Mostra estado, número de pontos e colunas m6 finitas, intervalo C calculado,
paradas em massa e caminhos dos detalhes inconclusivos. `complete` e
`pilot_complete` significam término sem pontos inconclusivos na busca visitada.
`complete_with_unresolved` e `pilot_complete_with_unresolved` significam que a
busca terminou, mas contém pontos a revisar; não certificam ausência nesses
pontos. O launcher e o runner retornam código 3 nesse caso, depois de salvar os
resultados e figuras. `paused_point_budget` é um checkpoint parcial normal.

A recuperação automática é limitada a uma tentativa: duplica a grade de campo,
aumenta as sementes térmicas, reduz os passos do rastreamento e as tolerâncias
numéricas de mínimos/raiz. Conserva potencial, solver de bounce, ressoma,
alvo/tolerância S3/T e passo de beta. As duas tentativas e seus parâmetros ficam
nos detalhes; falhas persistentes continuam unknown. `--recovery-attempts 0` no
coletor desliga essa tentativa. Ações negativas são rejeitadas e preservadas nos diagnósticos, sem alimentar a raiz como uma amostra física.

Inconclusivos não viram no, não fecham uma fronteira e não tornam uma coluna
vazia. A busca segue sem nova avaliação infinita do mesmo erro. As três
negativas de parada devem ser consecutivas, sem unknown entre elas. Antes da
primeira semente continua a busca completa dentro dos bounds, mesmo havendo
falhas. Uma execução com pontos inconclusivos exige revisão/refinamento para
publicação; a regra de três ausências também continua sujeita à hipótese de
uma faixa conectada e à resolução adotada.

O PyCharm pode iniciar run_local.py diretamente. Caminhos relativos de saída
são resolvidos pela raiz do repositório, evitando Articles/Articles/results;
os subprocessos também usam essa raiz. O modo PILOTO/PRODUÇÃO e os estados das
três fatias aparecem explicitamente no terminal. Executar sem --smoke inicia
produção. Use uma pasta nova ao mudar de v2 para v3.
