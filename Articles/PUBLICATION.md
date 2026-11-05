# As seis figuras principais do artigo

O comando novo é `python -m Articles.figures.publication`. Ele somente lê
dados; não calcula fases, ações ou derivadas. Os comandos anteriores e as
campanhas boundaries/edge_observables/lisa_reach continuam disponíveis.

## O que é produzido

1. **Fig01**: três janelas sobrepostas em C versus `(1 TeV/m6)^2`.
2. **Fig02–04**: Tn, alpha de traço, beta/Hn e v(Tc)/Tc para cada m8, com
   escalas compartilhadas e contorno do alcance acústico do LISA.
3. **Fig05**: três linhas/m8 e duas colunas/observável, combinando as faixas
   dos detectores com isolinhas de log10(alpha) e log10(beta/Hn).
4. **Fig06**: trios com sinais próximos, primeiro a m6 fixo, depois permitindo
   massas diferentes. Tabela 1 informa todas as coordenadas e observáveis.

PDF/SVG são as versões vetoriais para o manuscrito; PNG tem 300 dpi. A pasta
também contém `index.html`, legendas em inglês, guia em português, tabelas
CSV/LaTeX, relatório com hashes e um ZIP completo. Dados não vão ao GitHub.

## Gerar no CHE

Depois de atualizar a branch, preserve as mesmas versões da venv. Prepare
FOPT_RESULTS_ROOT/FOPT_PYTHON_MODULE/FOPT_PYTHON como nas etapas anteriores.
Pare/aguarde todos os jobs que escrevem nos dados de origem.

```bash
sbatch --export=ALL --nodelist=milliways03 Articles/cluster/publication_figures.slurm
```

Destino padrão: `$FOPT_RESULTS_ROOT/article_publication_v1`. O template usa
os bancos completos, incluindo os pontos interiores; não precisa transferir
CSV para o cluster. Se existir `publication_refinement_v1`, inclui esse
suplemento. Fontes ainda em execução são rejeitadas.

Não sobrescreve uma edição existente por padrão. Para outro destino:

```bash
FOPT_PUBLICATION_ROOT="$FOPT_RESULTS_ROOT/article_publication_v2" sbatch --export=ALL --nodelist=milliways03 Articles/cluster/publication_figures.slurm
```

## Reproduzir com os arquivos recebidos no computador

O ZIP de bordas da segunda leva **não** continha a tabela termodinâmica
completa do interior. A chamada abaixo exige também o CSV/ZIP completo,
conferindo sua quantidade de pontos e os registros de borda contra o relatório.

```powershell
python -m Articles.figures.publication --exports "C:\Users\Igora\OneDrive\Documentos\Trabalho e Estudos\fopt-figures-articles" --frontiers "C:\Users\Igora\Downloads\FOPT-frontiers-v1.zip" --full-table "C:\Users\Igora\Downloads\FOPT-all-points-v2.zip" --output "Articles/results/article_publication_v1"
```

Use o Python da instalação do projeto. `--help` lista limites de eixos,
formatos, resolução e configuração de detectores. `--preview` admite uma
transferência incompleta, mas marca todas as figuras como prévias. Não deve
ser usado para produzir a versão final.

## Interpolação e precisão

- A seleção original continua sendo nucleação aceita perto de S3/T=140 e
  `abs(phi_low(Tn))/Tn>1`. Não se troca esse corte por v(Tc)/Tc.
- Uma testemunha interna é uma restrição unilateral, e não uma medição
  pontual de fronteira. As guias padrão minimizam curvatura dentro dos
  brackets externos certificados e das restrições internas. Depois se usa
  PCHIP entre massas contíguas. Limites, amostras e deslocamentos ajustados
  ficam em `parameter_windows.csv`. `--boundary-guide raw` desliga o ajuste.
- A guia preenchida **não prova** que todo seu interior nucleia. Não há
  extrapolação entre m6=2000 e infinito, nem preenchimento de colunas ausentes.
- Os mapas usam triângulos locais: no máximo 100.01 GeV em massa e 0.081
  em C. Essas resoluções correspondem aos cortes adicionais e à amostragem
  disponível. Avaliações fracassadas entram como nós inválidos; lacunas não
  são excluídas antes de triangular. Campos nunca calculados não viram zero.
- Alpha válido é mantido quando apenas beta falha. Beta com flags de
  qualidade não entra nos mapas, espectros ou SNR.
- O contorno tracejado vinho repete o LISA com o beta(h/2) já salvo. Os
  benchmarks passam o limiar com ambos os passos; as bandas claras dos
  espectros mostram essa sensibilidade numérica, sem serem intervalos de
  confiança. `detector_step_sensitivity.csv` lista mudanças de classificação.
  A Tabela 2 conserva intervalos separados para h e h/2. O critério nominal
  da campanha não é sobrescrito.
- Detectabilidade: canal acústico, PIS já usado no projeto, quatro anos e
  SNR>=10 por padrão. DECIGO/BBO são pós-processados nos mesmos pontos;
  seus limites não herdam o refinamento específico do LISA em C.
- Sinais próximos não demonstram degenerescência estatística. A seleção
  minimiza distância no plano log10(f_peak), log10(Omega_peak), e a tabela
  registra a diferença RMS dos espectros acústicos na banda relevante.

## Verificação pontual de beta, sem nova campanha grande

`refinement_requests.csv` lista lacunas próximas de pontos LISA detectáveis.
O renderizador não repete falhas automaticamente. O comando a seguir lê os
bancos originais e escreve um plano imutável apenas para essas coordenadas:

```bash
"$FOPT_PYTHON" -m Articles.figures.publication.supplement --boundaries "$FOPT_RESULTS_ROOT/boundaries_v1" --observables "$FOPT_RESULTS_ROOT/edge_observables_v1" --reach "$FOPT_RESULTS_ROOT/lisa_reach_v1" --beta-step 1 --output "$FOPT_RESULTS_ROOT/frontier_plans/publication_check_v1.json"
```

O passo máximo de beta é 1 GeV nessa verificação, limitado pela margem
metastável como antes. Permanecem a diferença central de ordem 2 e o teste
h versus h/2, com a mesma tolerância relativa. A ideia é verificar se um
passo maior reduz cancelamento/ruído; a melhora não é presumida. Todos os
outros ajustes vêm do manifesto original. Não é uma alteração dos dados v1.

Submeta esse plano com o executor já existente, em **outro destino**:

```bash
unset FOPT_MAX_POINTS
```

```bash
FOPT_FRONTIER_SHARDS=1 FOPT_EDGE_PLAN="$FOPT_RESULTS_ROOT/frontier_plans/publication_check_v1.json" FOPT_EDGE_DATA_ROOT="$FOPT_RESULTS_ROOT/publication_refinement_v1" sbatch --export=ALL --array=0-2%3 --nodelist=milliways03 Articles/cluster/edge_observables.slurm
```

```bash
"$FOPT_PYTHON" -m Articles.figures.frontiers status --output "$FOPT_RESULTS_ROOT/publication_refinement_v1"
```

Depois que os três jobs encerrarem, gere outra edição das figuras. Se alguma
derivada continuar inconclusiva, preserve essa informação: repetir indefinidamente
ou preencher a falha por interpolação não estabelece sua convergência.

## Baixar os arquivos, no PowerShell do Windows

```powershell
scp -P 13900 ftovar@152.84.248.250:/home/ftovar/fopt-articles-v4/article_publication_v1/article_figures_complete.zip "$env:USERPROFILE\Downloads\FOPT-article-publication-v1.zip"
```

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\FOPT-article-publication-v1.zip" -DestinationPath "$env:USERPROFILE\Downloads\FOPT-article-publication-v1"
```

```powershell
Invoke-Item "$env:USERPROFILE\Downloads\FOPT-article-publication-v1\index.html"
```
