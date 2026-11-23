# Branch nova e preservação da campanha anterior

Base desta organização: `main`, commit `0b4e5ab`.
Branch anterior preservada: `codex/articles-combined-potential-scan`,
commit `ab0bd5e` na comparação inicial.
Branch nova: `codex/article-fopt-combined`.

## Resposta à pergunta sobre o diff antigo

O diff antigo para main tinha 99 arquivos e **não estava restrito a Articles**.
Os 13 caminhos externos eram:

```text
.gitattributes
.github/workflows/ci.yml
.gitignore
conftest.py
docs/Thesis/Thesis_Main_Code/results_runs/c6c8_pure/run05_beta_best_fit_values.csv
docs/Thesis/Thesis_Main_Code/results_runs/glauber_pure/run03_beta_best_fit_values.csv
pyproject.toml
src/CosmoTransitions/generic_potential.py
src/CosmoTransitions/gravitational_Waves.py
src/CosmoTransitions/transitionFinder.py
tests/test_article_cluster.py
tests/test_article_core.py
tests/test_article_scan.py
```

A branch nova nasceu diretamente de main. Seus arquivos versionados novos
ficam todos em `article_FOPT_combined/`. Os arquivos da tese, o núcleo, o
pyproject e os testes da main permanecem iguais à base. As alterações do
usuário e a campanha antiga continuam na branch anterior; não houve reset,
reescrita de histórico, remoção dos resultados ou force-push.

Para conferir o escopo antes de integrar:

```bash
git diff --name-only origin/main...HEAD
git diff --exit-code origin/main...HEAD -- src docs tests pyproject.toml .github
```

Resultados ignorados de checkouts anteriores podem permanecer no disco
local. Eles não fazem parte do diff nem são carregados pelo novo fluxo.

## Compatibilidade com o núcleo da main

| Necessidade | Solução dentro da pasta nova |
|---|---|
| Ordenação incorreta de campos em Phase com temperaturas reversas | `core.ordered_phase_inputs` ordena os três arrays antes do construtor; restaura o método ao sair |
| Beta de ordem dois | Reutiliza `_S3_at_T` do calculador e `fd_weights_1d` do helper |
| Mínimos refinados para ações/termodinâmica | `RefinedPhase` adapta `valAt`, delegando os demais atributos à fase original |
| Alpha com sinal e duas convenções | Registro explícito a partir do potencial e derivadas do src |
| Frequências históricas do src em mHz | Interface do artigo em Hz; conversão centralizada |
| Instalação na raiz e caminhos de pacotes | Ambiente dedicado com `requirements.txt`; execução da raiz do checkout, sem editable install |
| Testes e temporários Windows | `pytest.ini` local e diretório exclusivo em `.work/pytest/` |

Esses adaptadores evitam transportar uma segunda cópia do núcleo. As
correções gerais em src podem ser integradas futuramente em uma manutenção
separada, com seus próprios testes. O fluxo novo já funciona com o núcleo da
base indicada e registra o hash dos arquivos efetivamente utilizados.

## Potencial e coleta de produção

| Edição | Termos diretos de um loop | Anéis |
|---|---|---|
| Primeira campanha em `Articles` | Higgs, três Goldstones, W, Z e top | Três modos efetivos de gauge |
| Edição T² | Higgs, três Goldstones, W, Z e top | Escalares com autoenergias hard T² e gauge longitudinal com mistura neutra |
| Produção do artigo | W, Z e top, com contratermos analíticos em v | Gauge longitudinal com mistura neutra |

A produção conserva os operadores polinomiais e a medida renormalizada.
O manifesto identifica a prescrição e permite retomar checkpoints da mesma
campanha. A mudança nos termos de loop exige recalcular as fronteiras.

A busca revista avalia a ação durante a coexistência das fases. Os
observáveis são coletados entre as bordas e usam beta de ordem dois,
com ajuste de passo em intervalos estreitos. A classificação instrumental
usa o canal acústico e sua sensibilidade integrada de pico.

## Organização da entrega

Há um único potencial, um único algoritmo de bordas e uma única amostragem
interior. As seis figuras compartilham o mesmo leitor e a mesma geometria.
Os quatro cenários são coletados desde o início. Três recebem as figuras
principais; o quarto recebe o atlas suplementar e pode substituir outro
cenário na apresentação sem novos cálculos físicos.

O CI global existente em main permanece fora deste diff. Os testes deste
artigo têm comando independente em README.md e são executados também pelo
setup do CHE; sua inclusão no workflow global é uma etapa de integração da
main. A pasta de resultados e o ambiente virtual ficam fora do Git.
