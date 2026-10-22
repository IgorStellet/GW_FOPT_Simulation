# Figuras do artigo no CHE

Este fluxo lê os resultados que estão no cluster, grava figuras PDF/SVG/PNG e
tabelas de auditoria em outra pasta. Nenhum resultado precisa ir para o GitHub.
O comando `render` não executa fases, bounces ou derivadas. `plan` escolhe
coordenadas; só `points` inicia cálculos físicos adicionais.

O código novo está em `Articles/figures/`. Os módulos Python de coleta v4 e o
núcleo não foram alterados. Esse subdiretório não entra no glob de fontes do
manifesto v4; acrescentá-lo não invalida a identidade das campanhas existentes.
O novo subpacote também é incluído no wheel. Não atualize bibliotecas para
desenhar: use a mesma venv da coleta.

## O que podemos concluir dos resultados atuais

A leitura fornecida em 22/10/2026 tem 29.843 pontos, nas três fatias terminadas
como `complete_with_unresolved`. Isso significa fim da estratégia, com
pendências conservadas, não fronteiras inteiramente resolvidas. Há 127, 156 e
183 colunas delimitadas, respectivamente. Algumas outras colunas são vazias
na grade de sementes; as inconclusivas continuam distintas dessas exclusões.

O critério usado foi `abs(phi_low(Tn))/Tn > 1`, com aceitação
`abs(S3/Tn - 140) <= 0.5`. O mapa de fronteiras não pode ser relabelado como um
corte em `v(Tc)/Tc`. O novo leitor recupera o VEV crítico do registro da
transição ou de uma única transição crítica compatível nos detalhes. Se faltar
ou houver ambiguidade, o campo fica ausente. Não extrapola fases até Tc.

O valor C~4.14 do modelo puro de medida a Lambda=1000 GeV é referência de
degenerescência a T=0, não uma condição universal do potencial combinado.
Operadores adicionais mudam V(v,0)-V(0,0). O diagnóstico desse valor já salvo
é desenhado separadamente. A figura não troca estabilidade do vácuo por
nucleação: são condições diferentes. O zoom padrão C=0..4.3 só muda os eixos;
a cobertura C=0..10 e os pontos selecionados fora do zoom são exportados.

## Conjunto de figuras e a pergunta de cada uma

| Arquivo | Uso no artigo |
|---|---|
| `01_parameter_windows` | Janela em m6 físico e C para os três m8; figuras principais |
| `02_mechanism_compensation` | Sobreposição em (1 TeV/m6)^2: quanto C é compensado pelo operador polinomial |
| `03_observables_m8_0/1/2` | Painéis Tn, alpha_trace, beta/H e v(Tc)/Tc; inspirados nos mapas de Chala et al. |
| `04_strength_at_criticality` | Contornos 0.5, 1, 1.5, 2, 3 de v(Tc)/Tc, onde os dados permitem |
| `05_detector_windows` | Intervalos amostrados que passam o limiar PIS acústico de LISA, DECIGO e BBO |
| `06_gw_benchmarks` | Poucos espectros e, em painel próprio, picos acústicos versus PIS dos detectores |
| `07_correlations_and_degeneracy` | alpha vs beta/H e coincidências de picos entre cenários; candidatos, não inferência estatística |
| `08_verified_cooperation` | Pontos combinados selecionados cujos dois controles puros efetivamente calculados falham no corte de força |
| `A1_calculated_coverage` | Cobertura completa C=0..10 e falhas numéricas, para o apêndice/auditoria |
| `A2_strength_at_nucleation` | v(Tn)/Tn, para comparar com o corte crítico |
| `A3_supercooling` | 1-Tn/Tc: associação entre super-resfriamento, alpha e sinal |
| `A4_zero_temperature_vacua` | V(v,0)-V(0,0), usando diagnósticos já salvos; zero não garante mínimo global |

As figuras estão em inglês e usam fontes incorporadas nos PDFs, texto editável
nos SVGs e PNGs de 300 dpi. Cada observável usa a mesma escala de cores entre
cenários. Os casos infinitos não recebem uma distância
arbitrária no eixo de massas: na primeira figura são referências horizontais;
na segunda são o ponto zero da força do operador. Não se liga esse ponto aos
primeiros m6 finitos através de uma região não investigada.

Faixas só ligam colunas adjacentes delimitadas, a 5 GeV. Brackets verificados
são laranja. Nas bordas inconclusivas, aparecem apenas testemunhas internas;
uma falha não é prolongada até C=0 ou C=10. Os mapas de valores só interpolam
triângulos apoiados em dados válidos: qualquer vértice sem valor, conflito
entre fontes ou lacuna grande deixa um buraco. Regiões brancas não significam
ausência de FOPT ou de detectabilidade.

## 1. Atualizar apenas a análise e conferir a chamada

Na raiz do checkout do CHE, um comando por vez. A revisão acrescenta o
subpacote, documentação, testes e templates; preserva os módulos numéricos.

```bash
cd "$HOME/projects/GW_FOPT_Simulation"
```

```bash
git status --short
```

```bash
git pull --ff-only
```

```bash
export FOPT_PYTHON_MODULE=python/3.11.7-gcc-12.2.0 FOPT_PYTHON="$HOME/projects/GW_FOPT_Simulation/.venv-che311/bin/python" FOPT_RESULTS_ROOT="$HOME/fopt-articles-v4"
```

```bash
module load "$FOPT_PYTHON_MODULE"
```

```bash
"$FOPT_PYTHON" -m Articles.figures --help
```

As importações em uma instalação editável já apontam para o pacote Articles.
Não é necessário reinstalar NumPy/SciPy ou criar outra venv. Se o comando não
encontrar o módulo, confira branch, atualização e caminho do interpretador.

## 2. Desenhar primeiro com o que já foi salvo

O job de desenho copia snapshots para um temporário local; não cria WAL/SHM
nas fontes. Além de preservar o banco, verifica compatibilidade de
núcleo/modelo/versões. Pode levar tempo para ler detalhes comprimidos.

```bash
mkdir -p Articles/results/logs
```

```bash
sbatch --export=ALL --nodelist=milliways03 Articles/cluster/figures.slurm
```

Pede uma CPU/8 GB por até 2 h. O resultado padrão é
`$FOPT_RESULTS_ROOT/article_figures_v1`. O log é `figures-NUMERO.out/.err`.
O desenho direto equivalente, para usar numa alocação interativa, é:

```bash
"$FOPT_PYTHON" -m Articles.figures render --boundaries "$FOPT_RESULTS_ROOT/boundaries_v1" --output "$FOPT_RESULTS_ROOT/article_figures_v1" --bundle
```

Para o controle visual, leia `figure_report.json` e `CAPTIONS.md`. Ausência de
alpha/beta produz painéis explicitamente indisponíveis. As fronteiras v4
calculam nucleação/VEVs, não GW em todos os pontos. Nenhum alpha ou beta é
deduzido apenas de Tn. Registros reutilizados com GW válido podem ser usados.

## 3. Obter somente os observáveis que faltam

Escolha um plano pequeno, inicialmente três massas e três Cs por cenário,
com controles puros. Só são propostos pontos sem observáveis resolvidos.

```bash
"$FOPT_PYTHON" -m Articles.figures plan --boundaries "$FOPT_RESULTS_ROOT/boundaries_v1" --output "$FOPT_RESULTS_ROOT/article_figures_v1/observables_plan.json" --mass-samples 3 --C-samples 3 --include-controls
```

O arquivo lista exatamente as coordenadas e o motivo de cada cálculo. Não é
sobrescrito se já existir: um plano iniciado é imutável. Com zero pontos,
nenhuma execução adicional é necessária. Escolher poucas massas/valores C
fornece benchmarks e cortes; não preenche automaticamente um mapa denso.

Para calcular exclusivamente um ponto explícito (ordem m6, C, m8):

```bash
"$FOPT_PYTHON" -m Articles.figures plan --boundaries "$FOPT_RESULTS_ROOT/boundaries_v1" --output "$FOPT_RESULTS_ROOT/article_figures_v1/single_point.json" --point 1000 3.35 668.740304976422
```

`--point` é repetível e substitui a amostragem automática. Os parâmetros estão
diretamente em m6/m8; `inf` desliga o operador. Para usar esse plano no template,
exporte FOPT_POINT_PLAN com seu caminho e escolha um destino novo. Não troque
o plano dentro de um destino já iniciado.

Primeiro, execute no máximo um ponto por fatia para testar o fluxo:

```bash
export FOPT_MAX_POINTS=1
```

```bash
sbatch --export=ALL --nodelist=milliways03 Articles/cluster/figure_points.slurm
```

São três tarefas, uma CPU/8 GB cada, até 14 h. O padrão é 900 s por ponto,
beta de ordem 2, comparação com meio passo e nenhuma recuperação automática.
Os limites numéricos e o modelo vêm do manifesto original. Os resultados vão
para `article_points_v1/m8-0/1/2`; as fronteiras não são modificadas. O
checkpoint fechado é publicado depois de cada ponto. Um sinal de tempo pede
parada após o ponto atual. Confira os logs `paper-points-ARRAY_INDICE.out/.err`.

Depois de conferir o piloto, retome o mesmo plano/destino sem o limite:

```bash
unset FOPT_MAX_POINTS
```

```bash
sbatch --export=ALL --nodelist=milliways03 Articles/cluster/figure_points.slurm
```

Um estado `paused` é retomável. `complete_with_unresolved` exige revisão;
somente `FOPT_RETRY_SELECTED=1` solicita repetir os inconclusivos na próxima
execução. FOPT_SELECTED_TIMEOUT muda o orçamento: exige destino novo em
FOPT_SELECTED_ROOT. Os pontos válidos antigos continuam na fonte.

Para mapas densos, use a etapa interior já existente ou um plano com mais
massas e Cs, após medir o tempo. Isso é o primeiro cálculo de alpha/beta desses
pontos; não exige repetir a busca de fronteiras. Não se pode obter um campo
contínuo confiável desses observáveis a partir de alguns poucos espectros.

## 4. Atualizar figuras com os novos pontos

```bash
export FOPT_SELECTED_ROOT="$FOPT_RESULTS_ROOT/article_points_v1"
```

```bash
sbatch --export=ALL --nodelist=milliways03 Articles/cluster/figures.slurm
```

Para incluir também uma campanha interior, exporte FOPT_OBSERVABLES_ROOT com
seu caminho. O desenho direto aceita `--observables PASTA` repetidamente.
Fontes compatíveis são unidas somente em memória. Uma falha nova não apaga
um valor resolvido. Se duas fontes discordam entre seleção e exclusão na
mesma coordenada, a coordenada fica inconclusiva e é registrada no relatório.
Uma avaliação isolada nova não reescreve/certifica os brackets da campanha
original: eles conservam suas pendências até uma revisão das fronteiras.

O ponto real m6=1000 GeV, C=3.35, m8=668.740304976422 GeV foi calculado e
reaberto por esse fluxo na validação local. O relatório de testes e valores
está em [FIGURES_VALIDATION.md](FIGURES_VALIDATION.md). Esse teste verifica o
fluxo, mas não substitui a leitura dos resultados reais do CHE.

## Critério dos detectores e controles puros

Usamos os ajustes PIS acústicos já presentes no núcleo, normalizados a 1 ano
de Schmitz (2021): `rho_sw = sqrt(t_obs/yr) * Omega_sw_peak/Omega_PIS_s`.
O padrão é 4 anos e rho_sw>=10. São hipóteses explícitas de comparação,
ajustáveis por FOPT_OBSERVATION_YEARS e FOPT_SNR_THRESHOLD. As curvas usam as
configurações de detector do artigo de referência, não uma atualização dos
projetos experimentais para 2026. Não incluem SNR total, correlações com
turbulência ou um modelo novo de ruído/foreground.

Sensibilidade PIS não é ruído pontual: os espectros ficam em um painel e seus
picos versus PIS em outro. As faixas dos detectores só conectam Cs vizinhos
com GW resolvido e distância <=0.041. São mantidos intervalos desconectados;
não se presume que todo C acima de uma única borda seja detectável.

Cooperação na figura 08 exige comparações efetivas: mesmo ponto combinado,
controle C=0 com os mesmos m6/m8 e controle m6=m8=inf no mesmo C. Desligar só
m6 em um cenário de m8 finito não fornece o modelo puro de medida. Controles
ausentes/falhos não provam cooperação. A figura testa a força da transição,
não invisibilidade em colisores nem ausência de GW dos controles.

## Arquivos para conferir e baixar

Além das figuras, são gravados `points_analysis.csv`, `all_saved_transitions.csv`,
`boundary_brackets.csv`, `benchmarks.csv`, `spectrum_samples.csv`,
`degeneracy_candidates.csv`, `selected_outside_zoom.csv` e `figure_report.json`.
Eles registram fontes, hashes, política de transição primária, unidades,
escolhas PIS, lacunas e conflitos. As legendas científicas ficam em CAPTIONS.md.

`review_bundle.zip` contém PNGs e tabelas/relatório, sem SQLite ou detalhes
completos. Para baixar no PowerShell do COMPUTADOR, com o endereço/porta da
conexão já usada neste projeto:

```powershell
scp -P 13900 ftovar@152.84.248.250:/home/ftovar/fopt-articles-v4/article_figures_v1/review_bundle.zip "$env:USERPROFILE\Downloads\review_bundle.zip"
```

O pacote pode ser anexado ao chat para conferir as figuras com os dados reais.
Nada nessa transferência usa o GitHub. Para produção editorial, os PDFs/SVGs
permanecem na pasta do cluster e podem ser copiados pelo mesmo método.

## Referências e limites de interpretação

- Berganholi et al., JHEP 07 (2025) 129: https://arxiv.org/abs/2502.09593
- Chala, Krause e Nardini, JHEP 07 (2018) 062: https://arxiv.org/abs/1802.02168
- Schmitz, JHEP 01 (2021) 097: https://arxiv.org/abs/2002.04615

As faixas deste projeto usam outro conjunto de prescrições e coordenadas;
não se copiam os números de contorno/exclusão daqueles artigos. Candidatos
próximos em frequência/amplitude não demonstram degenerescência do espectro
inteiro ou de uma inferência com dados. O relatório identifica essa limitação.
