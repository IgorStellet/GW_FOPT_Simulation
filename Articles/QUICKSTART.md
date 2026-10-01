# Guia rápido: coleta de dados do artigo

O programa percorre (m6, C) para cenários de m8, procura fases, Tc e nucleação,
calcula alpha_energy, alpha_trace, beta/H e o pico acústico de GW. Salva dados
para análise posterior; não produz figuras. Execute na raiz do repositório.

## Modelo fixado

$$V_{\rm eff}=V_{\rm tree}+V_{\rm CW}+V_{\rm CT}+V_{\rm medida}+V_T+V_{\rm ring},$$
$$V_{\rm tree}=-\mu^2\phi^2/2+\lambda\phi^4/4+\phi^6/(8m_6^2)+\phi^8/(16m_8^4).$$

m6/m8, campo e temperatura em GeV; C adimensional. `inf` desliga um operador.
lambda e mu² mantêm v=246 GeV e mh=125 GeV. A medida é logarítmica, com
subtração local em v e domínio |phi|<Lambda/sqrt(C). Lambda=1000 GeV.
CW usa Q=v e contratermos finitos; massas dos loops vêm do setor polinomial.
Ressoma gauge simplificada da tese sempre ligada. GW usa alpha_trace,
Tstar=Tn, g*=106.75 e vw=1. Detalhes das equações: [README](README.md).

## Primeira chamada

Python >=3.11, no mesmo ambiente usado pelo terminal/editor:

```bash
python -m pip install -e .
python -m pip install pytest
python -m pytest -q
python -m Articles.collect_data --dry-run
```

Grade de **dois pontos**: C=0 e 3.35, um cenário de massas; inclua beta-check:

```bash
python -m Articles.collect_data --output Articles/results/local_smoke --m6 1000 1000 5 --C 0 3.35 3.35 --m8 668.740304976422 --no-baselines --beta-check --workers 2
```

Referência anterior: C=0 sem nucleação encontrada; C=3.35 com Tn~40.738 GeV,
alpha_trace~0.332894 e beta/H~116.722. Confira status e quality_flags.
Repetir o mesmo comando retoma a campanha; para mudar a configuração, use
outra pasta. Versões diferentes podem alterar resultados e a compatibilidade.

## O que pode ser ajustado na chamada

| Opção | Padrão | O que faz |
|---|---|---|
| `--m6 MIN MAX STEP` | `500 2000 5` | Eixo inclusivo de massas, GeV |
| `--C MIN MAX STEP` | `0 10 0.02` | Eixo inclusivo de C |
| `--m8 VALOR ...` | `inf 840.8964152537145 668.740304976422` | Cenários discretos, GeV |
| `--output PASTA` | `Articles/results/combined` | Campanha e retomada |
| `--workers N` | `1` | Processos em um único nó |
| `--max-points N` | sem limite | Primeiros N pontos pendentes, não amostra aleatória |
| `--dry-run` | desligado | Inspeciona a grade sem bounces |
| `--no-baselines` | desligado | Não acrescenta C=0/m6=inf/m8=inf |
| `--retry-failed` | desligado | Repete numerical_failure/observables_unresolved |
| `--export-only` | desligado | Atualiza CSVs a partir do banco |
| `--action-tolerance` | `0.5` | Faixa adimensional em S3/T-140 |
| `--beta-step` | `0.5` | Passo térmico de beta, GeV; ordem sempre 2 |
| `--beta-check` | desligado | Compara também h/2; mantém resultado principal com h |
| `--T-min`, `--T-max` | `1`, `250` | Intervalo térmico, GeV |
| `--phi-max` | `1000` | Janela de campo, limitada também pelo polo |
| `--n-phi`, `--n-T-seeds` | `1200`, `5` | Busca inicial em campo/temperatura |
| `--shard-index`, `--shard-count` | `0`, `1` | Parte disjunta da grade; índice começa em zero |

Use ponto decimal; intervalos devem ser múltiplos inteiros dos passos.
As referências puras usam massas infinitas. O padrão com referências tem
453.906 pontos. Shards diferentes precisam de pastas diferentes e da mesma grade.

## O que não tem opção de terminal

Em `Settings`: g*, vw, alvo 140, limiar forte 1, tolerância relativa de beta 0.2,
minima_phitol, x_eps/T_eps, deltaX_target, root_T_tolerance/root_maxiter e ordem
das derivadas do potencial (4, distinta da ordem 2 de beta). Podem ser alterados
na API/código, com outra campanha; não são opções de collect_data na chamada.
Lambda é fixado pelo gerador. Ressoma, CW/Goldstones, massas dos loops e esquema
de medida exigem alteração explícita do modelo; não há switches para alterná-los.

## Onde estão os dados

`scan.sqlite`: principal e retomada; `points.csv`: todos os pontos e diagnósticos;
`transitions.csv`: alpha_energy/trace, beta/H, Tc/Tn, campos, ação, pico GW e PIS;
`details/*.json.gz`: fases e amostras da ação; `manifest.json`: configuração.
Salva o **pico acústico**, não o espectro completo; os inputs permitem reconstruí-lo.
Falha/célula vazia não é zero nem ausência física de FOPT. Status nucleated não
garante seleção de vácuo/percolação. Frequências CSV em Hz; núcleo recebe mHz.
Dados em results não são enviados automaticamente ao GitHub.

Cluster: [primeiro acesso, teste e produção no CHE](CLUSTER_CHE.md).
