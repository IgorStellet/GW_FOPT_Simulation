# Como os arquivos se relacionam

O fluxo começa em `__main__.py`. `init` salva a configuração e o manifesto.
`boundaries` pede apenas nucleação a `execution`, que inicia um processo de
`physics`. Esse processo constrói `model.Potential` e entrega suas derivadas
a `thermal_history`, que usa a ponte `core` para o núcleo em `src`.
`storage` publica o resultado antes da próxima
coordenada. `observables` monta a grade entre as bordas, usando o mesmo
avaliador. `figures` lê os checkpoints e renderiza os produtos gráficos.

```text
boundaries   ──► observables  ──► figures + detectors
     │                │                ▲
     └─► execution ◄─┘                │
              │                      storage
              └─► physics ─────────────┘
                    ├─► model
                    └─► thermal_history ──► core/src
```

## Responsabilidade de cada arquivo

| Arquivo | Responsabilidade |
|---|---|
| `model.py` | Única definição do potencial, massas e anéis |
| `core.py` | Ponte para src e adaptações locais de fases, beta e unidades |
| `config.py` | Parâmetros da campanha e validação das opções |
| `physics.py` | Avaliação física de uma coordenada e extração dos observáveis |
| `thermal_history.py` | Fases, coexistência e cruzamentos da ação no resfriamento |
| `execution.py` | Timeout, subprocessos, orçamento, cache e retomada |
| `storage.py` | Manifesto, locks, JSONs atômicos e CSVs |
| `boundaries.py` | Sementes, continuação em massa, caminhadas e brackets |
| `observables.py` | Plano da faixa inteira, validação e distribuição dos pontos |
| `detectors.py` | PIS acústico e PLIS LISA, com unidades explícitas |
| `figures.py` | Geometria local, auditoria, benchmarks e as seis figuras |
| `__main__.py` | Comandos e opções públicas; ponto de entrada do usuário |
| `__init__.py` | Identifica o pacote, sem executar uma campanha ao importar |
| `tests/test_model.py` | Matching em v, multiplicidades de W, Z e top, anéis e domínio |
| `tests/test_physics.py` | Curvaturas/fases, stencil, aceitação e definições termodinâmicas |
| `tests/test_thermal_history.py` | Suporte da fase simétrica, cruzamentos internos e falhas entre sondas |
| `tests/test_pipeline.py` | Fronteiras, sharding, cache, retomada e timeout real |
| `tests/test_figures.py` | Máscaras locais, sensibilidades e geração integral das figuras com fixture sintética |
| `tests/conftest.py` | Temporários exclusivos dos testes no Windows/CHE |
| `cluster/setup.sh` | Ambiente dedicado e testes antes do primeiro piloto |
| `cluster/environment.sh` | Módulo, Python, destino persistente e temporários de cada job |
| `cluster/submit.sh` | Valida o estágio, cria logs e chama sbatch |
| `cluster/pilot.slurm` | Piloto de pontos, fronteiras ou interior, escolhido por FOPT_PILOT_STAGE |
| `cluster/boundaries.slurm` | Array de produção das bordas |
| `cluster/observables.slurm` | Array de produção dos observáveis |
| `cluster/figures.slurm` | Auditoria, tabelas gráficas e figuras no cluster |
| `requirements.txt` | Dependências do fluxo; instalação sem modificar a raiz |
| `pytest.ini` | Descoberta e cache dos testes somente desta pasta |
| `ruff.toml` | Estilo/lint local, sem mudar as regras da main |
| `.gitignore` | Mantém resultados, ambiente e temporários fora do Git |
| `.gitattributes` | Garante finais de linha LF nos scripts do Linux |

Os documentos têm objetivos separados: README é a entrada rápida; PHYSICS
explica as equações; CLUSTER traz os comandos do CHE; MIGRATION registra o
escopo das branches; VALIDATION relata as verificações realizadas. Este
arquivo acompanha a leitura das funções.

## Percurso de um ponto

1. `Evaluator` verifica se o resultado já existe e se foi pedida retentativa.
2. `Budget` reserva tempo e uma unidade do limite de pontos novos.
3. O processo isolado recebe parâmetros, escolhas numéricas e, se houver,
   a história salva da borda.
4. `Potential` prepara as constantes e os contratermos. A checagem de vácuo
   a T=0 e da fase inicial pode encerrar o ponto com um diagnóstico explícito.
5. `thermal_history` encontra o suporte da origem por sua curvatura e usa o
   núcleo para traçar os mínimos quebrados. As splines representam os ramos
   dentro dos intervalos traçados. A busca consulta a ação durante a
   coexistência e escolhe o cruzamento no resfriamento. O campo do bounce
   continua avaliado com o potencial completo.
6. A etapa de bordas termina aqui. Na etapa de observáveis, os mínimos são
   refinados, alpha é calculado e são pedidas duas ações por estimativa de beta, com controle de estabilidade.
7. As falhas de beta deixam a nucleação e alpha disponíveis no registro.
   Uma recuperação de fases mais finas pode ser tentada, mantendo potencial,
   prescrição de beta e tolerância da ação.
8. O JSON do ponto é publicado atomicamente e a tentativa é preservada.
   Só depois o coordenador pede outra coordenada.

## Como ler as bordas e o interior

Uma borda certificada tem `inside_C`, com uma transição forte aceita, e
`outside_C`, com decisão externa calculada. `width_C` é sua incerteza de
bracketing. Falha numérica recebe `unknown`, nunca o papel de `outside`.

Depois de encontrar uma semente, a busca assume uma faixa conectada. A
continuação em massa fornece apenas palpites; todas as novas coordenadas
são avaliadas. Testes além de cada saída podem detectar reentradas; o
relatório registra esse caso em vez de apresentar uma única faixa certificada.
A procura sem semente cobre C até `seed_resolution`; faixas menores que essa
resolução podem ser perdidas.

O interior usa u entre zero e um e N pontos igualmente espaçados em u. Como
a largura em C muda com m6, a grade acompanha a janela física. Pontos de
validação em C e em massas intermediárias ficam fora do interpolador usado
na auditoria. Dados adicionais na mesma coluna podem enriquecer os mapas
depois da verificação. Cada triângulo requer três vértices válidos e colunas
adjacentes da subgrade.
`--refine` intercala massas e frações de C, preservando os pontos anteriores.

## Troca de potencial

`thermal_history.build_history` recebe `V`, `gradV`, `hessV` e
`dgradV_dT` pela interface de derivadas, além da janela de campo, da configuração
numérica e do critério de nucleação. Serve para potenciais radiais pares de
um campo cujo resfriamento começa na origem. `physics.derivatives_for` monta
essa interface a partir de `Vtot(X,T)`. A definição de massas, o domínio e
os diagnósticos específicos continuam no modelo. Um potencial com outras
coordenadas exige ajustar a parametrização e os diagnósticos antes da coleta.

O módulo térmico procura intervalos estáveis da origem por amostragem da
curvatura. Para cada par de ramos coexistentes, consulta a ação de cima para
baixo e investiga mínimos locais se as sondas ficarem acima do alvo. Um
bounce que falha durante o fechamento do bracket leva a sondas vizinhas.
Uma falha após uma transição resolvida fica registrada na história parcial.

## Função por função

Links e finalidades das funções desta edição, incluindo funções internas.

### __main__.py

| Função/método | Finalidade |
|---|---|
| [parser](__main__.py#L29) | Declara somente opções usadas pelo fluxo atual; --help explica cada comando. |
| [init_campaign](__main__.py#L149) | Traduz o preset e as poucas opções públicas em uma configuração verificável. |
| [status](__main__.py#L195) | Resumo por m8; complete_with_unresolved nunca é anunciado como validação científica. |
| [execute](__main__.py#L238) | Abre uma campanha compatível, executa o estágio e publica tabelas ao terminar. |
| [main](__main__.py#L327) | Despacha os cinco estágios; erros de configuração têm saída curta e código 2. |

### boundaries.py

| Função/método | Finalidade |
|---|---|
| [coordinate](boundaries.py#L20) | Arredondamento apenas da coordenada de busca, para reutilizar pontos exatos. |
| [search_axis](boundaries.py#L25) | Inclui ambos os extremos mesmo quando o intervalo não é múltiplo de step. |
| [column_path](boundaries.py#L30) | Um checkpoint independente por coluna de massa e cenário. |
| [find_column](boundaries.py#L35) | Localiza uma faixa, retornando limites certificados ou incertezas explícitas. |
| [find_column.ask](boundaries.py#L45) | Consulta/cacheia uma coordenada de C e registra sua decisão física. |
| [find_column.edge](boundaries.py#L58) | Caminha a partir da semente e contrai o bracket de uma das duas bordas. |
| [assigned_columns](boundaries.py#L215) | Distribui cenários/colunas sem sobreposição; count=4 dá um m8 por job. |
| [run](boundaries.py#L225) | Retoma colunas completas e publica cada nova coluna assim que termina. |

### config.py

| Função/método | Finalidade |
|---|---|
| [axis](config.py#L10) | Grade decimal inclusiva, com extremos separados por um número inteiro de passos. |
| [Numerics.validate](config.py#L46) | Função interna que mantém a operação local descrita no bloco de código. |
| [Campaign.validate](config.py#L90) | Função interna que mantém a operação local descrita no bloco de código. |
| [Campaign.masses](config.py#L111) | Inclui o controle m6=inf; percorre massas finitas de 2000 para 500 GeV. |
| [Campaign.observable_masses](config.py#L115) | Subgrade determinística, incluindo ambos os extremos e o controle puro. |
| [campaign_from_dict](config.py#L122) | Reconstrói tipos a partir do manifesto JSON estrito, onde inf é uma string. |

### core.py

| Função/método | Finalidade |
|---|---|
| [ordered_phase_inputs](core.py#L29) | Ordena juntos T, X e dX/dT antes do construtor original de Phase. |
| [ordered_phase_inputs.initialize](core.py#L37) | Ordena conjuntamente os arrays da fase antes de chamar o construtor original. |
| [phase_from_record](core.py#L48) | Reconstrói a spline de uma fase salva para enriquecer uma borda já calculada. |
| [RefinedPhase.__init__](core.py#L61) | Inicializa os parâmetros, as dependências e os caches descritos pela classe. |
| [RefinedPhase.__getattr__](core.py#L65) | Delega atributos que não foram adaptados à Phase original do núcleo. |
| [RefinedPhase.valAt](core.py#L68) | Retorna um mínimo refinado em T escalar; derivadas usam a spline original. |
| [calculator](core.py#L84) | Usa o calculador original com mínimos refinados e cache por temperatura. |
| [central_beta](core.py#L92) | Beta/H com dois bounces, erro de truncamento O(step²), sem mudar step. |
| [sound_spectrum](core.py#L106) | h²Omega_sw: interface em Hz, convertendo para os mHz usados no src. |
| [beta_at_transition](core.py#L118) | Ordem dois com dois bounces novos; stencil central ou unilateral admissível. |

### thermal_history.py

| Função | Finalidade |
|---|---|
| [symmetric_phases](thermal_history.py#L22) | Constrói as fases da origem nos intervalos de curvatura positiva. |
| [trace_phases](thermal_history.py#L56) | Combina fases quebradas traçadas pelo núcleo com a origem exata. |
| [highest_crossing](thermal_history.py#L142) | Procura o cruzamento de S3/T no resfriamento e recupera brackets com sondas vizinhas. |
| [find_transition](thermal_history.py#L244) | Restringe o bounce à coexistência de duas fases críticas. |
| [build_history](thermal_history.py#L282) | Organiza a sequência térmica e preserva transições anteriores se houver uma falha posterior. |

### detectors.py

| Função/método | Finalidade |
|---|---|
| [pis_h2](detectors.py#L27) | Amplitude acústica de pico necessária para o limiar declarado. |
| [acoustic_snr](detectors.py#L35) | SNR do canal acústico, sob a forma espectral do src usada pelos ajustes PIS. |
| [lisa_noise_h2](detectors.py#L44) | Ruído efetivo em h²Omega, dois canais, braços de 2.5 Gm e foreground de 4 anos. |
| [_plis_coefficients](detectors.py#L69) | Integra uma vez cada lei de potência, formando o envelope de sensibilidade. |
| [lisa_plis_h2](detectors.py#L81) | Curva visual para espectros; a seleção quantitativa continua usando PIS acústico. |

### execution.py

| Função/método | Finalidade |
|---|---|
| [Budget.__init__](execution.py#L28) | Inicializa os parâmetros, as dependências e os caches descritos pela classe. |
| [Budget.reserve](execution.py#L38) | Reserva um ponto apenas se houver tempo para publicá-lo dentro do job. |
| [Budget.install_signals](execution.py#L50) | Solicita parada controlada; o ponto atual termina dentro de seu timeout. |
| [Budget.install_signals.stop](execution.py#L53) | Marca uma parada solicitada, deixando o ponto em execução terminar. |
| [failed_record](execution.py#L62) | Falha de execução com decisão desconhecida, distinta de ausência de nucleação. |
| [bounded_evaluation](execution.py#L75) | Timeout inclui imports e recuperação; mata o filho e preserva a cauda do log. |
| [Evaluator.__init__](execution.py#L154) | Inicializa os parâmetros, as dependências e os caches descritos pela classe. |
| [Evaluator.__call__](execution.py#L161) | Publica o resultado antes de entregá-lo ao algoritmo que pediu o ponto. |

### figures.py

| Função/método | Finalidade |
|---|---|
| [mass_label](figures.py#L35) | Representação LaTeX curta dos cenários, incluindo operador desligado. |
| [number](figures.py#L40) | Retorna número finito ou None; valores ausentes continuam ausentes nos mapas. |
| [valid_value](figures.py#L45) | Uma transição selecionada pode ter Tn válido e beta pendente, separadamente. |
| [flatten_records](figures.py#L60) | Lê os resultados mais recentes de observáveis e calcula SNRs sem novos bounces. |
| [Surface.triangulation](figures.py#L92) | Materializa a geometria Matplotlib apenas quando há área calculada suficiente. |
| [case_samples](figures.py#L99) | Identifica a grade e as amostras extras no interior de cada coluna. |
| [supported_cell](figures.py#L121) | Seleciona a diagonal e os triângulos sustentados por vértices válidos. |
| [build_surface](figures.py#L155) | Constrói o mapa local por campo, preservando regiões sem suporte calculado. |
| [lisa_refinement_requests](figures.py#L204) | Pede novos pontos no meio de cruzamentos LISA ainda largos. |
| [coverage_report](figures.py#L238) | Audita bordas, dados ausentes e validação independente da interpolação em C. |
| [boundary_segments](figures.py#L372) | Separa sequências adjacentes com suporte para a borda ou faixa. |
| [draw_boundary_guides](figures.py#L407) | Interpola bordas válidas e indica a faixa entre testemunhas. |
| [match_benchmarks](figures.py#L485) | Seleciona sinais próximos em log(f_peak,Omega_peak), priorizando massas menores. |
| [Writer.__init__](figures.py#L545) | Inicializa os parâmetros, as dependências e os caches descritos pela classe. |
| [Writer.save](figures.py#L572) | Marca rascunhos e salva duas representações da mesma figura. |
| [draw_windows](figures.py#L589) | Figura 1: compensação de mecanismos, com faixas contínuas e sem pontos. |
| [contour](figures.py#L637) | Desenha somente níveis presentes na área interpolada e calculada. |
| [shared_limits](figures.py#L648) | Mesmas escalas de cor em todos os atlas; nenhuma seleção por percentis. |
| [draw_atlas](figures.py#L672) | Mapas Tn, alpha, beta/H e v(Tc)/Tc, com o contorno acústico LISA. |
| [draw_detectors](figures.py#L753) | Figura 5: regiões detectáveis mais isolinhas de alpha e beta/H. |
| [draw_spectra](figures.py#L850) | Figura 6: espectros com PLIS LISA acima e picos/PIS dos detectores abaixo. |
| [render](figures.py#L957) | Audita primeiro; publicação requer suporte completo, rascunho permite diagnosticar. |

### model.py

| Função/método | Finalidade |
|---|---|
| [Parameters.__post_init__](model.py#L29) | Normaliza as coordenadas e valida o domínio físico dos parâmetros. |
| [Potential.__init__](model.py#L52) | Fixa o matching em v, os acoplamentos e o coeficiente de Coleman–Weinberg. |
| [Potential.domain_limit](model.py#L66) | Limite estrito /phi/<Lambda/sqrt(C); infinito em C=0. |
| [Potential.tree](model.py#L70) | V0=-mu² phi²/2+lambda phi⁴/4+phi⁶/(8m6²)+phi⁸/(16m8⁴). |
| [Potential.masses_squared](model.py#L75) | Massas de W, Z e top, em GeV². |
| [Potential.coleman_weinberg](model.py#L84) | Termo renormalizado de W, Z e top com contratermos analíticos em v. |
| [Potential.measure](model.py#L93) | Medida logarítmica com primeira/segunda derivadas subtraídas em v. |
| [Potential.thermal](model.py#L105) | Jb/Jf do src por spline em m²/T²; conserva as massas não ressomadas. |
| [Potential.longitudinal_masses_squared](model.py#L125) | W_L e autovalores neutros Z_L/gamma_L da matriz térmica eletrofraca. |
| [Potential.gauge_rings](model.py#L135) | Dois modos W_L, um Z_L e um gamma_L; modos transversais permanecem nus. |
| [Potential.__call__](model.py#L145) | Soma física única; todas as derivadas do núcleo recebem esta mesma soma. |
| [Potential.diagnostics](model.py#L155) | Registra normalização e condição necessária de vácuo eletrofraco a T=0. |

### observables.py

| Função/método | Finalidade |
|---|---|
| [sampling_masses](observables.py#L21) | Refinamento insere colunas intermediárias já delimitadas, preservando a grade anterior. |
| [plan](observables.py#L36) | Lista coordenadas determinísticas; não avalia potencial nem altera resultados. |
| [run](observables.py#L122) | Executa subprocessos limitados em paralelo; somente os coordenadores usam threads. |

### physics.py

| Função/método | Finalidade |
|---|---|
| [NucleationCriterion.__init__](physics.py#L35) | Inicializa os parâmetros, as dependências e os caches descritos pela classe. |
| [NucleationCriterion.__call__](physics.py#L38) | Executa a operação principal do objeto com seus parâmetros preparados. |
| [BoundedLog.write](physics.py#L54) | Função interna que mantém a operação local descrita no bloco de código. |
| [derivatives_for](physics.py#L60) | Derivadas do potencial completo, incluindo a dependência térmica dos anéis. |
| [build_history](physics.py#L85) | Rastreia fases e nucleação pelo pipeline do src, na janela declarada. |
| [initial_phase_check](physics.py#L94) | Verifica a fase inicial antes de gastar tempo rastreando uma história inadequada. |
| [pack_history](physics.py#L116) | Salva fases e transições leves para reutilizar as bordas na segunda etapa. |
| [unpack_history](physics.py#L141) | Recupera a história exata de um ponto de borda, sem procurar Tn novamente. |
| [nucleation_record](physics.py#L149) | Extrai ação, campos, Tc e força, sem calcular alpha ou derivadas de ações. |
| [beta_record](physics.py#L193) | Escolhe um passo admissível e verifica estabilidade até convergir ou atingir o piso. |
| [thermodynamic_record](physics.py#L263) | Enriquece uma transição aceita, preservando alpha mesmo se beta falhar. |
| [classify](physics.py#L350) | Uma transição forte aceita seleciona o ponto; falha permanece unknown. |
| [evaluate_once](physics.py#L362) | Uma tentativa física; todo erro é um registro explícito, nunca uma exclusão. |
| [evaluate](physics.py#L460) | Até duas tentativas; refina fases mantendo potencial e prescrições numéricas. |
| [evaluate.score](physics.py#L485) | Prioriza observáveis completos, decisão física resolvida e dados já disponíveis. |
| [main](physics.py#L500) | Entrada privada do processo isolado; os comandos de usuário estão em __main__.py. |

### storage.py

| Função/método | Finalidade |
|---|---|
| [safe](storage.py#L34) | JSON estrito: inf identifica operador desligado; NaN/-inf tornam-se null. |
| [canonical](storage.py#L47) | Serialização determinística usada nos hashes e nos arquivos salvos. |
| [timestamp](storage.py#L52) | Horário UTC explícito para checkpoints, tentativas e auditoria. |
| [point_id](storage.py#L57) | Identificador das coordenadas; a identidade física está no manifesto da pasta. |
| [parameters_from_dict](storage.py#L62) | Converte apenas as três coordenadas físicas de um registro salvo. |
| [atomic_json](storage.py#L67) | Escreve, sincroniza e publica no mesmo filesystem, preservando o arquivo anterior em falhas. |
| [read_json](storage.py#L83) | Lê um produto JSON; erros de integridade permanecem visíveis ao usuário. |
| [physics_identity](storage.py#L88) | Hash apenas do código físico, seus dados spline e versões numéricas. |
| [initialize](storage.py#L111) | Cria uma campanha nova ou verifica compatibilidade integral de uma existente. |
| [load_campaign](storage.py#L157) | Abre uma campanha; os leitores gráficos podem dispensar comparação com o código atual. |
| [task_lock](storage.py#L169) | Lock de sistema operacional; é liberado em crash, sem apagar checkpoints. |
| [point_path](storage.py#L199) | Caminho de um resultado; estágios não sobrescrevem os resultados uns dos outros. |
| [save_point](storage.py#L204) | Preserva cada tentativa e publica o último resultado do estágio. |
| [point_records](storage.py#L219) | Itera os resultados físicos mais recentes, sem executar cálculos. |
| [write_csv](storage.py#L225) | Publica CSV UTF-8 com união de colunas; estruturas ficam como JSON em células. |
| [export_tables](storage.py#L249) | Exporta pontos, todas as transições e fronteiras; dispensa refazer a física. |
| [archive_campaign](storage.py#L275) | ZIP atômico dos dados publicados, sem locks de execução nem arquivos temporários. |
