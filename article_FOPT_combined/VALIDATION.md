# Validação da campanha do artigo

## Potencial e testes

A produção usa os termos diretos de W, Z e top e os anéis longitudinais
de gauge. Os testes do modelo verificam o matching de $v$ e da
curvatura a $T=0$, o Coleman–Weinberg renormalizado, a contagem dos
modos térmicos, a matriz neutra e a subtração dos termos cúbicos.
No limite da medida pura, o teste confirma o cruzamento de
$V(v,0)-V(0,0)$ entre $C=4.13$ e $C=4.15$, como esperado a partir
da Eq. (20) de [Berganholi et al.](https://arxiv.org/html/2502.09593v1).

Os testes do fluxo verificam checkpoints, retomada, rejeição de um
manifesto incompatível, stencils de beta e renderização das seis figuras
e dos suplementos com dados sintéticos. Execute na raiz do checkout:

```bash
python -m pytest -c article_FOPT_combined/pytest.ini article_FOPT_combined/tests -q
```

## Piloto físico local

Os quatro pontos usados por `cluster/pilot.slurm` foram calculados pelo
comando `point` e aprovados por `check-pilot`. Todos têm $m_6=\infty$,
nucleação forte aceita e observáveis completos. O maior residual absoluto
de $S_3/T_n$ foi $0.390$, dentro da tolerância de $0.5$.

| m8 [GeV] | C | Tc [GeV] | Tn [GeV] | S3/Tn | v(Tn)/Tn | alpha_trace | beta/Hn |
|---:|---:|---:|---:|---:|---:|---:|---:|
| inf | 3.5 | 103.9610 | 97.3842 | 140.2158 | 2.2290 | 0.01164 | 3615.47 |
| 840.8964 | 3.4 | 105.4000 | 99.6118 | 140.1505 | 2.1458 | 0.01057 | 4199.62 |
| 668.7403 | 3.3 | 104.3084 | 98.1928 | 140.0255 | 2.1969 | 0.01116 | 3926.56 |
| 400 | 0 | 77.4832 | 40.4062 | 139.6103 | 6.0744 | 0.29483 | 186.57 |

Em uma coordenada da busca de fronteiras com $m_6=1200$ GeV,
$m_8=\infty$ e $C=2.9527777778$, o código obteve
$T_n=110.1405$ GeV e $S_3/T_n=140.0047$. A segunda etapa reutilizou
a história salva e obteve $\alpha_{\rm tr}=0.006907$ e
$\beta/H_n=7827.75$, com o teste de estabilidade aprovado.

A cobertura da grade de produção e os resultados das figuras serão
avaliados com os dados do CHE. Os pilotos do cluster, a retomada dos jobs
e a auditoria das figuras estão em [CLUSTER.md](CLUSTER.md).

Os números e testes da edição com anéis escalares T² estão documentados
no [commit daquela edição](https://github.com/IgorStellet/GW_FOPT_Simulation/blob/cfc04927b7f122806e59d2562e8c2fb7a60d9633/article_FOPT_combined/VALIDATION.md).
