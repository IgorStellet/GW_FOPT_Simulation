# Bordas, alpha/beta e alcance do LISA

Este é um fluxo adicional. Os comandos antigos `Articles.figures` e as
campanhas boundaries_v1 continuam disponíveis. O novo comando é
`python -m Articles.figures.frontiers`. Ele escreve outros destinos e usa o
mesmo potencial, ressoma, critério de nucleação e derivada de ordem 2.
Não atualize NumPy/SciPy da venv para executar esta análise.

## O que os arquivos recebidos mostram

Os CSVs de 22/10/2026 contêm 29.843 pontos, 165 com GW resolvido e 7.600
razões críticas disponíveis. Os 165 GW estão concentrados em m6=inf/2000,
não distribuídos pela janela. Há também 59 registros com alpha calculado e
beta pendente. O novo plot aproveita alpha separadamente.

As duas testemunhas em C por massa totalizam 1.754 registros de borda;
somente três já têm GW resolvido. Portanto o plano completo, com passo de
5 GeV em m6 e sem amostras interiores, pede 1.751 coordenadas novas. Esse
número é conferido novamente ao criar o plano no CHE, a partir dos bancos.

**Duas bordas em C não garantem os extremos de alpha ou beta/H no interior.**
O programa chama essas quantidades de valores nas bordas. A figura E03 e a
tabela interior_outside_edge_range.csv permitem testar a hipótese com
valores interiores disponíveis. `--interior-checks 1` acrescenta um ponto
interior por massa se desejado; o padrão zero atende à coleta só nas bordas.

## As sete figuras novas

| Figura | Conteúdo |
|---|---|
| E01_clean_transition_window | Somente pontos verdes calculados, duas faixas estreitas nas bordas e referência numérica v/T=1 |
| E02_alpha_beta_at_edges | Alpha e beta/H vs m6, em cada m8; círculos/borda inferior, triângulos/superior; cor indica C |
| E03_alpha_beta_along_C | Alpha e beta/H vs C, para m6=1000/1500 GeV em cada m8 |
| E04_lisa_parameter_window | Região amostrada com SNR acústico do LISA acima do limiar, preservando buracos e componentes |
| E05_GW_pure_measure | Menor/maior C detectável efetivamente calculado, com m6=m8=inf |
| E06_GW_m6_1000 | Espectros dos extremos detectáveis amostrados em C, nos três m8, com m6=1000 GeV |
| E07_GW_m6_1500 | Mesma comparação, com m6=1500 GeV |

E01 não cria linhas verticais entre borda inferior e superior. Também quebra
ligações quando C salta mais de 0.08 entre massas vizinhas: um ponto
isolado continua visível, sem fabricar um dente na curva. A faixa hachurada
é local à borda; não cobre toda a janela. Não desenha os x nem marcadores
roxos. As pendências continuam nas tabelas e na cobertura v1.

As figuras de Tn e super-resfriamento anteriores permanecem úteis e são
reproduzidas pelo comando original. Os PDFs/SVGs são vetoriais, com pontos
densos rasterizados; os PNGs padrão têm 300 dpi.

## O significado da referência v/T=1

A seleção original foi |phi_low(Tn)|/Tn>1, não v(Tc)/Tc>1. O programa encontra
cruzamentos do nível 1 entre dois valores salvos e exporta seu bracket em C,
IDs de pontos e temperatura de referência. Não atravessa falhas nem lacunas
grandes. `--strength-reference Tc` muda a referência desenhada, sem mudar a
seleção original. `--strength-reference none` a omite.

Não há uma expressão algébrica exata simples para essa linha no potencial
completo, que inclui CW, logaritmo de medida, funções térmicas e ressoma.
A condição crítica envolve simultaneamente estacionariedade e degenerescência
dos mínimos; a condição em Tn também depende do bounce S3/T=140. Uma linha de
uma aproximação polinomial de alta temperatura seria outro cálculo. Por isso
a legenda diz **numérica**, e não analítica/teórica exata. Regiões sem
cruzamentos salvos não recebem uma curva inventada.

## Por que o passo de beta muda nesta coleta

O teste de uma borda real (m6=1000, m8=668.740304976422 GeV, C=1.887272842575)
encontrou Tn=121.20251 e Tc=121.72739 GeV. O passo fixo de 0.5 GeV quase chega
a Tc. Beta/H foi 4.63e6 com h=0.5 e 8.64e4 com h=0.25: o cheque de meio passo
rejeita corretamente o resultado. No mesmo histórico de fases, h=0.05/0.025
deu 5.687e4/5.540e4, diferença relativa de 2.6%.

O padrão NOVO, `--beta-policy margin`, usa

    h = min(h_max, max(0.005 GeV, 0.1 * margem_metastavel))

A margem é a menor distância de Tn às extremidades de coexistência das fases,
incluindo Tc como limite superior. h_max vem de Settings.beta_step da origem
(0.5 GeV no v4). Longe das extremidades o passo continua 0.5. O cálculo usa o
recorder original, a derivada central de ordem 2 e o mesmo teste h versus h/2
(tolerância relativa de 20%). Se o piso não couber no intervalo ou a
convergência falhar, beta continua inconclusivo. Não há derivada unilateral,
extrapolação de fases ou troca de sinal para forçar um resultado.

Cada transição guarda beta_step_GeV, beta_step_policy,
beta_maximum_step_GeV, beta_metastable_margin_GeV e a comparação com meio
passo. O manifesto registra a política. `--beta-policy fixed` usa o passo
fixo original; `--beta-step VALOR` altera h_max no plano novo. Não altere um
plano iniciado: crie outro arquivo/destino para outra prescrição.

## 1. Atualizar e preparar o terminal CHE

Um comando por bloco, na branch codex/articles-combined-potential-scan:

```bash
cd "$HOME/projects/GW_FOPT_Simulation" && git pull --ff-only
```

```bash
export FOPT_PYTHON_MODULE=python/3.11.7-gcc-12.2.0 FOPT_PYTHON="$HOME/projects/GW_FOPT_Simulation/.venv-che311/bin/python" FOPT_RESULTS_ROOT="$HOME/fopt-articles-v4"
```

```bash
module load "$FOPT_PYTHON_MODULE" && mkdir -p Articles/results/logs
```

## 2. Plano de alpha/beta somente nas bordas

```bash
"$FOPT_PYTHON" -m Articles.figures.frontiers plan-edges --boundaries "$FOPT_RESULTS_ROOT/boundaries_v1" --output "$FOPT_RESULTS_ROOT/frontier_plans/edges_v1.json" --mass-step 5
```

O plano é imutável e lista as coordenadas exatas. Colunas inconclusivas
contribuem com as testemunhas positivas realmente calculadas, identificadas
como não certificadas. O plano não refaz a busca das fronteiras.

São quatro partes por m8, total de 12 tarefas com bancos independentes. O
template executa até três simultaneamente, uma CPU/8 GB por tarefa e 14 h.
Um sinal 30 min antes do fim permite concluir/publicar o ponto limitado a
900 s. Não é necessário executar as 12 partes ao mesmo tempo.

Primeiro teste de um ponto por cenário, nas tarefas 0, 4 e 8:

```bash
FOPT_MAX_POINTS=1 sbatch --export=ALL --array=0,4,8 --nodelist=milliways03 Articles/cluster/edge_observables.slurm
```

```bash
"$FOPT_PYTHON" -m Articles.figures.frontiers status --output "$FOPT_RESULTS_ROOT/edge_observables_v1"
```

Depois de conferir os logs edges-ARRAY_INDICE.out/.err, execute todas as
partes. O piloto será reutilizado, sem recalcular pontos concluídos:

```bash
unset FOPT_MAX_POINTS
```

```bash
sbatch --export=ALL --nodelist=milliways03 Articles/cluster/edge_observables.slurm
```

Ao atingir o tempo/limite de pontos, o estado é paused. Reenvie o mesmo
template com o mesmo plano/destino para continuar. Não dispare duas cópias
da mesma parte simultaneamente: há bloqueio de escritor. complete_with_unresolved
significa que algumas coordenadas precisam de diagnóstico; o código de saída
3 informa isso ao Slurm. FOPT_FRONTIER_RETRY_FAILED=1 solicita repetir também
beta/GW pendente com nucleação aceita. O padrão não repete essas falhas.

## 3. Plano para o limiar do LISA

Depois da etapa de bordas, crie este plano, que incorpora seus dados:

```bash
"$FOPT_PYTHON" -m Articles.figures.frontiers plan-lisa --boundaries "$FOPT_RESULTS_ROOT/boundaries_v1" --observables "$FOPT_RESULTS_ROOT/edge_observables_v1" --output "$FOPT_RESULTS_ROOT/frontier_plans/lisa_v1.json" --mass-step 100 --benchmark-masses 1000 1500 --C-step 0.05 --C-tolerance 0.01
```

O padrão investiga cortes em m6 a cada 100 GeV, mais os extremos disponíveis,
1000/1500 e inf. Com os CSVs recebidos, a prévia tem 51 cortes e cerca de
1.611 coordenadas iniciais, antes do reuso e refinamento. A contagem final
pode mudar após completar observáveis. O plano inclui as testemunhas de
borda e o vizinho externo quando a distância é <=0.2 em C; não termina
automaticamente no maior ponto positivo. Não explora novamente C=0..10.

Primeiro faça um piloto e confira o estado:

```bash
FOPT_MAX_POINTS=1 sbatch --export=ALL --array=0,4,8 --nodelist=milliways03 Articles/cluster/lisa_reach.slurm
```

```bash
"$FOPT_PYTHON" -m Articles.figures.frontiers status --output "$FOPT_RESULTS_ROOT/lisa_reach_v1"
```

Depois execute o conjunto completo:

```bash
sbatch --export=ALL --nodelist=milliways03 Articles/cluster/lisa_reach.slurm
```

Dados GW compatíveis são reutilizados; os demais pontos são avaliados pelo
mesmo núcleo. O passo de C localiza todos os componentes VISÍVEIS nessa
resolução. Cada mudança detectável/não detectável é refinada até 0.01. Um
unknown interrompe o bracket e a faixa. Não se prova ausência de uma ilha
mais estreita que 0.05 nem se preenche o espaço entre massas não calculadas.
Reduzir os passos pede um plano/destino novos. Esta etapa é necessária para
uma faixa LISA: duas avaliações nas bordas físicas não localizam seu limiar.

O critério é o mesmo PIS acústico usado anteriormente:

    rho_sw = sqrt(t_obs/ano) * [h² Omega_sw,peak / h² Omega_PIS_s(f_sw,peak)]

Padrão: 4 anos, rho_sw>=10. A sensibilidade é avaliada na frequência DO
PONTO, não no mínimo global da curva do LISA. As curvas são os ajustes
históricos de Schmitz (2021), não novos projetos instrumentais de 2026.
Não é SNR total de colisões+turbulência+som. A classificação usa o canal
acústico; o espectro total é mostrado como informação complementar.

## 4. Gerar as figuras novas no cluster

O template encontra automaticamente os destinos novos se existirem:

```bash
sbatch --export=ALL --nodelist=milliways03 Articles/cluster/frontier_figures.slurm
```

O resultado padrão é article_frontiers_v1, separado de article_figures_v1.
Para desenhar antes dos cálculos adicionais, o mesmo comando usa apenas os
dados existentes e indica os painéis pendentes. Para ajustar a chamada,
`render --help` lista zoom, massas, formatos, DPI, tempo, limiar e lacunas.

Os painéis de espectros têm total/som em cima e picos versus PIS embaixo.
Assim o leitor vê os detectores sem confundir PIS com ruído espectral
pontual. A legenda mostra C e SNR. Os extremos são os menores/maiores C
detectáveis AMOSTRADOS, com IDs e estado da busca; não são extremos globais
analíticos nem a descoberta garantida de uma experiência.

Para revisar localmente só os CSVs já recebidos:

```powershell
python -m Articles.figures.frontiers render --exports "C:\Users\Igora\OneDrive\Documentos\Trabalho e Estudos\fopt-figures-articles" --output "Articles/results/frontier-local-review" --bundle
```

Nenhum cálculo físico ocorre nessa chamada. O interpretador deve ser o da
instalação do projeto. O leitor preserva campos ausentes e booleanos False.

## Arquivos salvos e transferência

PDF/SVG/PNG das sete figuras, edge_observables.csv, strength_reference.csv,
lisa_profiles.json, lisa_endpoint_benchmarks.csv, spectrum_samples.csv,
interior_outside_edge_range.csv, frontier_report.json e CAPTIONS.md. Os
arquivos incluem coordenadas, passos usados, hipóteses, origem e hashes.
O ZIP compacto não inclui SQLite, detalhes nem os CSVs originais completos.

No PowerShell do computador, depois do job de figuras terminar:

```powershell
scp -P 13900 ftovar@152.84.248.250:/home/ftovar/fopt-articles-v4/article_frontiers_v1/review_bundle_frontiers.zip "$env:USERPROFILE\Downloads\FOPT-frontiers-v1.zip"
```

Os destinos podem ser alterados por FOPT_EDGE_DATA_ROOT, FOPT_LISA_DATA_ROOT e
FOPT_FRONTIER_FIGURES_ROOT. Os planos por FOPT_EDGE_PLAN/FOPT_LISA_PLAN.
FOPT_FRONTIER_TIMEOUT muda o limite por ponto e requer destino novo.
FOPT_FRONTIER_SHARDS muda a partição e requer também outro destino e um
array coerente: 0 a 3*shards-1. Não mude isso numa campanha iniciada.

Referência do critério PIS: https://arxiv.org/abs/2002.04615. A implementação
preserva os coletores/modelo/núcleo v4; os novos módulos ficam em
Articles/figures/frontiers, fora do glob de identidade numérica v4.
