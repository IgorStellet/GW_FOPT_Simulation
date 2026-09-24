# Dados para o artigo sobre mecanismos combinados de FOPT

Esta pasta reúne a coleta de dados do artigo. O ponto de entrada é
[`collect_data.py`](collect_data.py): ele percorre o espaço de parâmetros,
identifica fases e transições, calcula os parâmetros termodinâmicos e grava os
resultados. Os gráficos ficam para uma etapa posterior. Não é necessário repetir
o cálculo de tunelamento para reconstruir os espectros a partir dos parâmetros
salvos.

O cálculo reutiliza os módulos de [`src/CosmoTransitions`](../src/CosmoTransitions)
para o potencial térmico, rastreamento de fases, bounce, derivadas e ondas
gravitacionais. A camada do artigo organiza o modelo, a execução e a persistência;
não constitui outra implementação desses métodos.

## 1. O modelo e suas unidades

A parte polinomial do potencial é parametrizada diretamente por duas escalas de
massa positivas:

\[
V_0(\phi)=-\frac{\mu^2}{2}\phi^2+\frac{\lambda}{4}\phi^4
          +\frac{\phi^6}{8m_6^2}+\frac{\phi^8}{16m_8^4}.
\]

Assim, `m6` e `m8` são dados em **GeV**. Diminuir uma dessas massas aumenta a
deformação correspondente. A entrada `inf` desliga o operador; uma massa igual a
zero não faz isso. Não há um parâmetro independente `f` no modelo do artigo.

A normalização segue as equações (421)–(425), pp. 133–134 da dissertação. Na
notação anterior, os coeficientes que multiplicavam os operadores eram
\(c_6/f^2=m_6^{-2}\) e \(c_8/f^4=m_8^{-4}\). Isso explica os três valores
iniciais abaixo; a execução usa diretamente as massas, sem conversão entre
parametrizações:

| Cenário | `m8` [GeV] | Correspondência na dissertação, com `f = 1000 GeV` |
|---|---:|---:|
| Sem operador de dimensão oito | `inf` | `c8 = 0` |
| Primeiro cenário com dimensão oito | `840.8964152537145` | `c8 = 2` |
| Segundo cenário com dimensão oito | `668.740304976422` | `c8 = 5` |

O setor de medida funcional usa \(\Lambda=1000\) GeV e a contribuição logarítmica
renormalizada da equação (428), p. 135. Os termos são combinados conforme a
equação (444), p. 156, preservando as condições locais do vácuo eletrofraco a
temperatura zero. A motivação científica e a discussão de degenerescências estão
nas seções 5.6–5.7. As páginas indicadas são as páginas impressas da dissertação.

Para `C > 0`, o argumento do logaritmo exige
\(\phi<\Lambda/\sqrt C\). O limite de campo configurado não autoriza ultrapassar
essa fronteira. `m6` e `m8` são escalas dos coeficientes efetivos; identificá-las
com o cutoff de uma teoria ultravioleta exige uma hipótese física adicional.

## 2. Prescrições físicas e numéricas

O scan fixa a **ressoma térmica gauge simplificada usada na dissertação, ligada**.
Não há comparação entre opções de ressoma nesta coleta. Essa prescrição não é a
ressoma completa de Arnold–Espinosa; os resultados devem ser identificados pela
prescrição efetivamente implementada.

A nucleação é procurada pelo critério \(S_3(T_n)/T_n\simeq140\), com tolerância
absoluta padrão **0.5 no valor de `S3/T`**. Essa tolerância não significa uma
incerteza de 0.5 GeV na temperatura. O intervalo inicial de busca é de 1 a 250 GeV.

O cálculo de \(\beta/H\) usa uma derivada central de segunda ordem, com duas
avaliações da ação:

\[
\frac{\beta}{H}=T_n\frac{F(T_n+h)-F(T_n-h)}{2h},
\qquad F(T)=\frac{S_3(T)}{T}.
\]

O passo padrão é `h = 0.5 GeV`. As amostras da ação são preservadas para que a
derivada possa ser auditada. A opção `--beta-check` calcula também o stencil com
`h/2`, mantendo a ordem dois, para avaliar a sensibilidade ao passo. O coletor não
suaviza a dependência de beta no espaço de parâmetros nem substitui valores por
um ajuste polinomial.

A intensidade usada nas previsões de GW é `alpha_trace`, com a diferença entre
as fases mantendo seu sinal. `alpha_energy` também é armazenada. Valores
incompatíveis com as hipóteses do template não devem ser corrigidos tomando seu
módulo. A configuração completa das previsões fica registrada com os dados.

## 3. Instalação e primeira execução

Use Python 3.11 ou posterior e execute os comandos a partir da raiz do
repositório. Instale o projeto no ambiente Python que será usado no scan:

```bash
python -m pip install -e .
python -m Articles.collect_data --help
```

Comece conferindo a malha, sem calcular bounces:

```bash
python -m Articles.collect_data --dry-run
```

A configuração padrão tem 301 valores finitos de `m6`, 501 valores de `C` e
três cenários de `m8`. Com as referências em `m6 = inf`, são **453.906 pontos**:
`(301 + 1) * 501 * 3`. É uma campanha grande; o comando de inspeção não estima
seu tempo com base em um ponto representativo. O custo varia bastante entre
pontos, sobretudo próximo a fronteiras de transição.

Para exercitar uma malha pequena, use uma pasta separada:

```bash
python -m Articles.collect_data --output Articles/results/smoke --m6 950 1000 50 --C 3.2 3.3 0.1 --m8 inf --no-baselines --max-points 2
```

Esse exemplo limita o trabalho; não é um teste de convergência nem uma validação
física do scan completo. A existência de FOPT depende dos parâmetros e das
prescrições, não da finalidade do comando.

A campanha padrão é iniciada com:

```bash
python -m Articles.collect_data --output Articles/results/combined
```

Para usar processos independentes, acrescente, por exemplo, `--workers 2`.
O padrão é um processo. Ajuste o número à memória e aos núcleos disponíveis;
mais processos não alteram a malha nem dispensam a convergência de cada ponto.

## 4. Opções principais

Os intervalos de `m6` e `C` recebem mínimo, máximo e passo. A malha padrão inclui
os extremos. Para os números fracionários da linha de comando, use ponto decimal.

| Opção | Padrão | Finalidade |
|---|---|---|
| `--output PASTA` | `Articles/results/combined` | Pasta de uma campanha |
| `--m6 MIN MAX STEP` | `500 2000 5` | Escalas de dimensão seis, em GeV |
| `--C MIN MAX STEP` | `0 10 0.02` | Intensidade adimensional da medida funcional |
| `--m8 VALOR ...` | `inf 840.8964152537145 668.740304976422` | Cenários de dimensão oito, em GeV |
| `--workers N` | `1` | Número de processos de cálculo |
| `--dry-run` | desligada | Inspecionar a campanha antes do cálculo |
| `--max-points N` | sem limite | Limitar a quantidade de pontos a executar |
| `--no-baselines` | desligada | Não acrescentar `m6 = inf`, `C = 0` nem o cenário `m8 = inf` à grade solicitada |
| `--retry-failed` | desligada | Tentar novamente pontos com falha de cálculo |
| `--export-only` | desligada | Atualizar as tabelas exportadas sem calcular pontos |
| `--action-tolerance VALOR` | `0.5` | Tolerância absoluta em `S3/T - 140` |
| `--beta-step VALOR` | `0.5` | Passo da derivada em temperatura, em GeV |
| `--beta-check` | desligada | Comparar derivadas com passos `h` e `h/2` |
| `--T-min VALOR` | `1` | Limite inferior de temperatura, em GeV |
| `--T-max VALOR` | `250` | Limite superior de temperatura, em GeV |
| `--phi-max VALOR` | `1000` | Limite numérico de campo, em GeV, sujeito ao domínio do modelo |
| `--n-phi N` | `1200` | Resolução inicial da busca em campo |
| `--n-T-seeds N` | `5` | Número de temperaturas iniciais para localizar fases |

As referências adicionais em `m6 = inf` são calculadas para os três valores de
`m8`. Apenas a combinação `m6 = inf`, `m8 = inf` representa o setor de medida
funcional puro. `C = 0` fornece o setor polinomial. Na malha padrão, `C = 0` já
está incluído. Sem as referências adicionais, o padrão tem 452.403 pontos.

## 5. Arquivos produzidos e retomada

| Arquivo | Conteúdo e uso |
|---|---|
| `scan.sqlite` | Base principal, com gravação transacional por ponto; controla a retomada |
| `manifest.json` | Configurações, versões e hashes das fontes usadas no cálculo |
| `points.csv` | Uma linha por ponto executado, incluindo os que não produziram FOPT e os que falharam |
| `transitions.csv` | Resultados das transições identificadas; um ponto pode ter mais de uma transição |
| `details/<prefixo>/<id>_<hash>.json.gz` | Histórico das fases, transições críticas, amostras da ação, avisos e registro resumido da execução |

Os CSVs são exportações para análise. A base SQLite é o registro usado para
determinar o que já foi calculado. Não apague a base para atualizar uma tabela:
use `--export-only` com a configuração da campanha. Os arquivos comprimidos
contêm os diagnósticos que não cabem adequadamente em colunas escalares. SQLite
é atualizado a cada ponto; CSVs são atualizados ao terminar a execução, inclusive
na saída tratada por interrupção. Uma interrupção forçada pode deixar os CSVs
desatualizados; `--export-only` recupera a exportação a partir do banco.

Para retomar, execute novamente o mesmo comando e a mesma pasta de saída. Os
pontos já registrados são reaproveitados. Um ponto que estava em cálculo no
momento de uma interrupção pode precisar ser calculado novamente. Use
`--retry-failed` quando quiser repetir falhas registradas, preservando a distinção
entre falha numérica e resultado físico.

Não misture resultados de prescrições ou resoluções diferentes na mesma pasta.
Mantenha o comando e o manifesto de cada campanha; a verificação da configuração
e das fontes protege a retomada contra combinações incompatíveis.

## 6. Como interpretar as tabelas

O mapa do artigo deve separar pelo menos: transição encontrada com nucleação,
ausência de transição encontrada, ausência de raiz de nucleação no intervalo
investigado e falha de cálculo. **Uma falha numérica nunca é um ponto não
detectável.** Tampouco a ausência de uma raiz no intervalo escolhido prova que a
transição não nucleia em nenhuma temperatura.

Preserve os identificadores dos pontos ao cruzar tabelas. Não reduza
automaticamente um ponto a uma única transição: a informação de fases e de
transições deve orientar a seleção física. O critério aproximado `S3/T = 140`
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

| Arquivo | Responsabilidade nesta coleta |
|---|---|
| `Articles/combined_model.py` | Define somente a física específica do potencial e diagnósticos em T=0 |
| `Articles/collect_data.py` | Configura a malha, chama o núcleo e salva os resultados sem produzir figuras |
| `src/CosmoTransitions/__init__.py` | Expõe a interface pública do pacote |
| `finiteT.py` | Integrais térmicas e splines de Jb/Jf |
| `helper_functions.py` | Pesos de diferenças finitas, gradiente, Hessiana, interpolação e integradores |
| `generic_potential.py` | Adaptação de formato, derivadas térmicas e coordenação de fases/transições |
| `transitionFinder.py` | Mínimos, histórias de fases, temperaturas críticas e nucleação |
| `tunneling1D.py` | Bounce O(3), integração radial e ação euclidiana |
| `gravitational_Waves.py` | Termodinâmica, beta/H, espectros e ajustes PIS |

Não importamos o script da dissertação. As alterações pequenas no núcleo
acrescentam `order=2` sem remover o padrão anterior `order=4`, refinam os mínimos
antes de calcular a ação, explicitam as duas convenções de alpha e corrigem
unidades em turbulência/saídas Hz e a ordenação de fases recebidas fora de ordem.
As splines térmicas mantêm os domínios e continuações reais do núcleo existente;
o tratamento CW de Goldstones preserva os contratermos de passo finito da tese.
Essas prescrições também precisam ser consideradas ao interpretar os resultados.

Os testes e a execução de validação estão descritos em [VALIDATION.md](VALIDATION.md).

## 7. Refinamento para resultados de publicação

A malha fina facilita investigar faixas estreitas, mas uma única resolução não
certifica convergência. Primeiro localize as regiões de interesse; depois
recalcule fronteiras, pontos representativos e regiões com beta instável em
outra pasta. Por exemplo:

```bash
python -m Articles.collect_data --output Articles/results/refinement --m6 750 850 2 --C 3.0 3.4 0.005 --m8 inf --beta-check
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
