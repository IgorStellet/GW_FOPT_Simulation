# FOPT com mecanismos combinados

Este diretório contém a campanha atual do artigo. O fluxo é único:
**potencial → fronteiras → observáveis no interior → seis figuras → exportação**.
Execute os comandos a partir da raiz de `GW_FOPT_Simulation`.

O potencial soma os operadores `phi^6/(8 m6^2)` e `phi^8/(16 m8^4)`, a medida
logarítmica de parâmetro `C`, Coleman–Weinberg e correções térmicas. A ressoma
Arnold–Espinosa inclui Higgs, três Goldstones e os modos longitudinais
eletrofracos. As autoenergias escalares conservam os termos **T²**. `Jb` e `Jf`
são calculadas por **spline em m²/T²** pelo núcleo do repositório.

## Leia conforme sua necessidade

| Documento | Para que serve |
|---|---|
| [PHYSICS.md](PHYSICS.md) | Equações, aproximações, critérios e limites físicos |
| [CLUSTER.md](CLUSTER.md) | Instalação no CHE, piloto, submissões, retomada e download |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Responsabilidade de cada arquivo e de cada função |
| [MIGRATION.md](MIGRATION.md) | Comparação das branches e o que mudou nesta campanha |
| [VALIDATION.md](VALIDATION.md) | Testes realmente executados e pilotos físicos |

## O que ajustar

Escolha a configuração uma vez, no comando `init`. O manifesto fica salvo
em `--output/manifest.json`; todos os estágios seguintes leem esse arquivo.

| Opção de `init` | Padrão de produção | Significado |
|---|---:|---|
| `--m6 MIN MAX STEP` | `500 2000 5` | Grade de fronteiras, em GeV; também inclui `m6=inf` |
| `--m8 VALORES...` | `inf 840.8964152537145 668.740304976422 400` | Quatro cenários, em GeV |
| `--C MIN MAX` | `0 4.2` | Janela de busca; o código sinaliza uma borda que toque o teto |
| `--C-tolerance` | `0.02` | Largura máxima do bracket dentro/fora de cada borda |
| `--interior-points N` | `9` | N pontos internos por faixa, além das duas bordas internas |
| `--interior-stride K` | `5` | Observáveis a cada K colunas: 25 GeV no padrão |
| `--beta-step` | `0.5` | Passo máximo em temperatura, em GeV, para beta de ordem dois |
| `--fixed-beta-step` | desligado | Exige o passo solicitado mesmo em intervalos estreitos |
| `--action-tolerance` | `0.5` | Aceitação `abs(S3/T-140)<=tolerância` |
| `--T-range MIN MAX` | `1 200` | Janela de temperaturas rastreada, em GeV |
| `--n-phi` | `1200` | Resolução inicial da busca de mínimos |

`inf` desliga o operador correspondente. Menor `m6` ou `m8` significa maior
coeficiente do operador. Lambda=1000 GeV, v=246 GeV, massas do SM, prescrição
de contratermos, ressoma, `g*=106.75` e `v_w=1` estão fixos e documentados em
PHYSICS.md. Uma alteração física exige outra campanha, com outro manifesto.

As opções de execução `--workers`, `--timeout`, `--wall-seconds`,
`--max-points`, `--scratch` e `--shard-*` controlam recursos e distribuição;
elas permitem retomar a mesma campanha. O padrão verifica beta em h e h/2
até estabilidade de 20%, com piso de 0.01 GeV, mantendo a estimativa mais fina.
Perto dos extremos, o passo inicial também é ajustado à coexistência.
Passos, stencils e resultados da checagem ficam no JSON. A checagem
não acrescenta curvas às figuras. `--fixed-beta-step` usa uma única estimativa
no passo pedido, sem esse controle adaptativo. `--retry-failed` retenta resultados
inconclusivos. O timeout limita o processo inteiro, incluindo recuperação.

## Instalação e primeiro teste local

Use Python **3.11 ou posterior**. A instalação deste fluxo usa o arquivo
de dependências abaixo; `pip install -e .` na raiz é dispensável.

```powershell
python -m venv article_FOPT_combined/.venv
.\article_FOPT_combined\.venv\Scripts\python.exe -m pip install -r article_FOPT_combined/requirements.txt
.\article_FOPT_combined\.venv\Scripts\python.exe -m pytest -c article_FOPT_combined/pytest.ini article_FOPT_combined/tests -q
.\article_FOPT_combined\.venv\Scripts\python.exe -m article_FOPT_combined init --preset pilot --output article_FOPT_combined/results/pilot
.\article_FOPT_combined\.venv\Scripts\python.exe -m article_FOPT_combined point --output article_FOPT_combined/results/pilot --m6 inf --m8 inf --C 3.5
```

O último comando calcula um ponto físico completo. Repeti-lo lê o checkpoint
salvo. Para testar o coordenador de fronteiras em poucos pontos novos:

```powershell
.\article_FOPT_combined\.venv\Scripts\python.exe -m article_FOPT_combined boundaries --output article_FOPT_combined/results/pilot --max-points 4
.\article_FOPT_combined\.venv\Scripts\python.exe -m article_FOPT_combined status --output article_FOPT_combined/results/pilot
```

O limite de quatro pontos é uma **pausa de piloto**, sem completar as bordas.
Retome com a mesma chamada; retire `--max-points` para concluir o estágio.
No Linux, use o Python do ambiente virtual em vez do executável `Scripts`.

No PyCharm, crie uma configuração **Module name = `article_FOPT_combined`**,
com a raiz do repositório como Working directory, o ambiente novo como
interpretador e, por exemplo, `point --output article_FOPT_combined/results/pilot
--m6 inf --m8 inf --C 3.5` em Parameters. `model.py` e `core.py` são bibliotecas;
o ponto de entrada para rodar a campanha é o módulo acima.

## Produção em cinco comandos

Aqui `python` representa o Python do ambiente preparado. Para uma campanha
nova, rode `init --preset full`. O padrão tem **1208 colunas de fronteiras**
e até **2728 pontos-base de observáveis**, mais pontos de validação, quando
todas as colunas tiverem faixa selecionada. A quantidade real depende dos dados.

```bash
python -m article_FOPT_combined init --preset full --output article_FOPT_combined/results/campaign
python -m article_FOPT_combined boundaries --output article_FOPT_combined/results/campaign
python -m article_FOPT_combined observables --output article_FOPT_combined/results/campaign --workers 2
python -m article_FOPT_combined figures --output article_FOPT_combined/results/campaign --destination article_FOPT_combined/results/figures
python -m article_FOPT_combined export --output article_FOPT_combined/results/campaign --zip article_FOPT_combined/results/campaign.zip
```

No CHE, use os arrays de CLUSTER.md em lugar das chamadas seriais de coleta.
Verifique `status` entre os estágios. Um job que termina seu tempo de trabalho
pode deixar uma campanha pausada com checkpoints íntegros.

## O que fica salvo

| Produto | Conteúdo |
|---|---|
| `manifest.json` | Configuração, aproximação, unidades, versões e hashes do código físico |
| `points/nucleation/*.json` | Ação, Tn, Tc, campos, seleção e fases reutilizáveis |
| `points/observables/*.json` | Tudo acima, alpha nas duas convenções, beta, amostras do stencil, pico e sensibilidades |
| `attempts/` | Histórico de cada avaliação externa; recuperação de resolução também fica dentro do registro |
| `boundaries/m8-*/column-*.json` | Duas bordas, testemunhas dentro/fora, largura dos brackets e pendências |
| `plans/` | Coordenadas pedidas para o interior e validação |
| `tables/points.csv` | Resumo de cada ponto por estágio |
| `tables/transitions.csv` | Todas as transições salvas, inclusive as que não foram selecionadas |
| `tables/boundaries.csv` | Fronteiras e diagnóstico de sua determinação |

**Alpha e beta são calculados na etapa `observables` ou em `point`.** A etapa
`boundaries` procura nucleação e força, sem pagar o custo desses observáveis.
O espectro completo é reconstruído dos parâmetros salvos; `figures` também
exporta `spectrum_samples.csv` para os benchmarks mostrados.

## As seis figuras

1. Faixas sobrepostas de compensação no plano `(1 TeV/m6)^2,C`.
2. Atlas `Tn, alpha, beta/H, v(Tc)/Tc` para o primeiro m8.
3. O mesmo atlas para o segundo m8.
4. O mesmo atlas para o terceiro m8.
5. Faixas LISA/DECIGO/BBO com isolinhas de alpha e beta.
6. Dois conjuntos de espectros próximos entre modelos, com LISA PLIS acima
   e picos/PIS dos detectores abaixo.

O quarto m8 recebe um atlas suplementar; outra figura suplementar sobrepõe
as janelas dos quatro cenários. Para trocar o cenário principal,
use, por exemplo, `figures --m8 inf 840.8964152537145 400` junto às opções
obrigatórias. Os dados permanecem iguais. PDFs, PNGs, legendas em inglês,
tabela de benchmarks e relatórios de suporte acompanham as figuras.

## Cobertura e refinamento

Os pontos internos são `C_low+j*(C_high-C_low)/(N+1)`, `j=1..N`, entre
testemunhas com transição forte aceita. Pontos adicionais, fora da grade
gráfica, verificam a interpolação em C e em massas intermediárias.

`figures` primeiro escreve `coverage_audit.json`, `point_requests.csv` e
`interpolation_validation.csv`. A versão final exige cobertura e validação;
`--draft` gera imagens de diagnóstico com marca de rascunho.

Para completar pontos pendentes, use `point --points-file CAMINHO/point_requests.csv
--retry-failed`, com o mesmo `--output`. Se a interpolação variar demais,
`observables --refine` intercala colunas de massa e dobra a divisão em C,
reutilizando as coordenadas anteriores. As referências de erro são 0.15 dex
para alpha/beta/SNR e 10% para Tn e v(Tc)/Tc; são critérios gráficos de validação,
distintos da incerteza física do modelo.

O código conserva lacunas que ainda pedem cálculo. Um limite analítico
presumido ou um preenchimento gráfico não substitui esses dados.
Se um ponto interno calculado fica fora da seleção, a auditoria registra
`selection_contradictions`: é preciso verificar a classificação e a hipótese
de faixa conectada. Repetir o cálculo pode confirmar uma exclusão física;
ela não deve ser apagada pela interpolação.
