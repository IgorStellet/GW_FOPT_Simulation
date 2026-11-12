# Revisão das seis figuras: coleta e uso no CHE

Esta revisão mantém o potencial, a ressoma, o critério de nucleação e os dados v1.
Os resultados novos ficam em outras pastas. As figuras finais devem ser geradas
depois de conferir a cobertura da coleta; uma mudança de desenho sozinha deixa
as lacunas físicas e numéricas em aberto.

## O diagnóstico dos arquivos recebidos

O pacote `FOPT-article-publication-v1.zip` contém 31.214 coordenadas e 3.125
conjuntos válidos para GW. Há **3.271 testemunhas selecionadas com modo
`nucleation_only` que nunca tiveram alpha calculado**. Esses pontos têm nucleação;
o campo em branco dos mapas de alpha/beta reflete a coleta parcial. Outros 117
pontos selecionados têm alpha, mas beta rejeitado pelo teste de passo ou por um
stencil que sai do intervalo de metastabilidade; 13 ainda precisam dos campos
críticos. Existem também falhas na busca
de fases e intervalos sem vizinhos suficientemente próximos para interpolar.

| m8 [GeV] | Selecionados sem alpha | Selecionados com alpha e beta pendente |
|---|---:|---:|
| infinito | 1.085 | 51 |
| 840,896 | 1.123 | 34 |
| 668,740 | 1.063 | 32 |

Os 13 pontos com campos críticos pendentes se distribuem em 2, 7 e 4 nos três
cenários. Use `selected_diagnostics` da auditoria para conferir o seu pacote.

A máscara dos mapas conserva as falhas e usa alcance local de até 100,01 GeV
em m6 e 0,081 em C. Por isso, ela pode deixar branco mesmo entre pontos com Tn
salvo. A nova coleta resolve observáveis nesse interior em vez de aumentar o
alcance da interpolação.

Na amostra válida com 500 <= m6 < 600 GeV, o maior SNR acústico do LISA é
3,48, 3,66 e 3,76 para os três m8, abaixo do limiar 10. Entre 600 e 700 GeV já
existem amostras detectáveis, com máximos de 37,3, 33,5 e 25,3. Esses são
extremos **amostrados**. O desaparecimento da faixa perto de massas pequenas
precisa ser conferido com a coleta fina junto à borda superior. Uma transição
forte por v/T não determina sozinha a detectabilidade: amplitude e frequência
dependem também de alpha, beta/H e Tn.

## O que a coleta v2 faz

`Articles.figures.publication.coverage` cria um plano imutável com:

- observáveis ausentes nos pontos selecionados existentes;
- reavaliação das buscas sem solução ou das falhas dentro da guia de janela;
- cortes em m6 a cada 50 GeV, reduzidos a 25 GeV até 900 GeV;
- espaçamento em C de até 0,04 no interior e amostras adicionais a 0,005–0,08
  da guia superior, onde a faixa do LISA pode ser estreita;
- controles com m6 infinito já selecionados e ainda incompletos.

Com os arquivos recebidos, isso gera **11.478 coordenadas**: 3.748, 3.750 e
3.980 nos três cenários. O plano criado no CHE é a referência final se os seus
arquivos tiverem recebido outros resultados. Pontos completos são preservados.
O piloto escolhe quatro dificuldades por cenário, totalizando 12 coordenadas.

Cada avaliação usa o potencial e as rotinas existentes de fases, bounce e
termodinâmica. Uma busca sem solução é repetida uma vez com maior resolução de
fases. A nucleação mantém |S3/T - 140| <= 0,5; a derivada de beta mantém ordem 2.
O passo máximo é testado na ordem **0,5; 1; 0,25 GeV**, sempre limitado pela
metastabilidade. A primeira avaliação que passa os critérios de qualidade é
usada; a escolha independe do SNR. Tentativas rejeitadas, passos efetivos,
comparação h/2, configurações de recuperação e erros ficam nos detalhes.
Falhas persistentes continuam classificadas como inconclusivas.

São guardados Tn, Tc, campos das fases, v(Tn)/Tn, v(Tc)/Tc, alpha, beta/H,
beta(h/2), dados para reconstruir os espectros e diagnósticos. Os SNRs dos
detectores e os espectros são reconstruídos dessas quantidades pelo leitor,
usando as mesmas rotinas GW da campanha. Cada ponto tem JSON comprimido;
cada parte tem SQLite, CSVs, manifesto e checkpoint persistentes.

Uma testemunha nova de bounce aceito pode superar uma busca anterior que não
encontrou nucleação. Essa recuperação fica explicitamente registrada em
`recovered_search_non_detections.csv`. Divergências entre duas transições
aceitas permanecem como conflitos a investigar. Os arquivos antigos ficam
intactos.

## Comandos: uma linha inteira de cada vez, no CHE

Entre no repositório e atualize a branch. `--ff-only` preserva alterações locais
e interrompe a atualização se as histórias divergirem. Nesse caso, confira o
estado antes de prosseguir; mantenha a campanha com sua versão consistente.

```bash
cd "$HOME/projects/GW_FOPT_Simulation" && git pull --ff-only origin codex/articles-combined-potential-scan
```

Carregue o mesmo ambiente já usado. Esta revisão usa as dependências existentes.

```bash
module load python/3.11.7-gcc-12.2.0 && export FOPT_PYTHON="$HOME/projects/GW_FOPT_Simulation/.venv-che311/bin/python" FOPT_RESULTS_ROOT="$HOME/fopt-articles-v4" FOPT_PYTHON_MODULE="python/3.11.7-gcc-12.2.0"
```

Crie os planos, uma única vez. O comando lê os bancos originais e o suplemento
`publication_refinement_v1`, caso exista; não calcula pontos no nó de login.

```bash
bash Articles/cluster/coverage_v2.sh plan
```

Submeta o piloto: três tarefas, um núcleo por tarefa, até duas horas.

```bash
bash Articles/cluster/coverage_v2.sh pilot
```

Veja a fila e os checkpoints. Durante o piloto, avalie também os logs
`Articles/results/logs/coverage-JOBID_TAREFA.out` e `.err`.

```bash
bash Articles/cluster/coverage_v2.sh status
```

**Depois do piloto**, este comando exige que todas as suas avaliações terminem
e que cada cenário tenha ao menos um conjunto completo de observáveis.
Ele verifica as identidades e submete 12 partes, com até três tarefas simultâneas,
14 horas por tarefa e orçamento de 900 s por ponto. Os resultados completos do
piloto são reutilizados. O tempo total depende do piloto e da fila: 11 mil
avaliações completas podem demandar dias, mesmo com reaproveitamento dos pontos
antigos. Não se trata de repetir a campanha inteira de 31 mil pontos.

```bash
bash Articles/cluster/coverage_v2.sh submit
```

O aviso de tempo do Slurm pede uma parada entre pontos e publica o checkpoint.
Se `status` mostrar `paused`, **aguarde o job sair da fila** e execute `submit`
novamente: as coordenadas concluídas são retomadas do checkpoint. Mantenha o
plano, os quatro shards por cenário, o orçamento e as versões dos pacotes.
O estado `complete_with_unresolved` significa que todas as coordenadas daquela
parte foram tentadas e algumas seguem inconclusivas. Ele exige auditoria;
um job com exit code zero, sozinho, não certifica observáveis completos.

Com todos os jobs encerrados, compare os motivos pendentes com os da primeira
auditoria. O relatório fica em `coverage_audit_v2/coverage_audit.json`.

```bash
bash Articles/cluster/coverage_v2.sh audit
```

Gere a nova edição das seis figuras, mantendo `article_publication_v1` intacta:

```bash
bash Articles/cluster/coverage_v2.sh figures
```

A pasta nova é `$HOME/fopt-articles-v4/article_publication_v2`. A geração exige
fontes fora do estado `running` e rejeita conflitos físicos ainda abertos. O ZIP
é criado **depois** das seis figuras, legendas e tabelas. Confira o encerramento:

```bash
tail -n 40 Articles/results/logs/publication-v2-*.out Articles/results/logs/publication-v2-*.err
```

Antes de baixar:

```bash
ls -lh "$HOME/fopt-articles-v4/article_publication_v2/article_figures_complete.zip"
```

No PowerShell **do seu computador**, execute esta única linha:

```powershell
scp -P 13900 ftovar@152.84.248.250:/home/ftovar/fopt-articles-v4/article_publication_v2/article_figures_complete.zip "$env:USERPROFILE\Downloads\FOPT-article-publication-v2.zip"
```

Para descompactar e abrir a galeria no computador:

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\FOPT-article-publication-v2.zip" -DestinationPath "$env:USERPROFILE\Downloads\FOPT-article-publication-v2"; Start-Process "$env:USERPROFILE\Downloads\FOPT-article-publication-v2\index.html"
```

## Mudanças nas figuras e critério de fechamento

- Fig. 1 mantém a comparação das três janelas sem marcadores e sem linhas
  verticais. As bordas continuam sendo guias sustentadas pelas testemunhas.
- Figs. 2–4 mantêm os quatro observáveis e o limiar do LISA. A linha de
  `xi_c = v(Tc)/Tc = 1` tem halo branco, rótulo e legenda. Ela só aparece onde
  valores calculados delimitam o nível 1; o potencial térmico completo requer
  um cálculo numérico dessas fases.
- Fig. 5 conserva as faixas dos detectores e as isolinhas de log10(alpha) e
  log10(beta/H), com os novos observáveis do interior.
- Fig. 6 inclui a **PLIS do LISA nos espectros completos** e mantém as **PIS no
  plano dos picos**. A PLIS usa o ruído de referência de Robson, Cornish e Liu,
  com seu ajuste de foreground galáctico de quatro anos. A classificação
  continua baseada no canal acústico/PIS histórico, permitindo comparação
  consistente com a campanha original. A sobreposição à PLIS é uma referência
  visual, não uma reclassificação dos sinais.

As curvas h/2 saem da edição principal. O teste continua nos dados e pode ser
mostrado com `--diagnostics`. Ele é uma comparação da derivada nos passos h e
h/2, sem significado de intervalo de confiança experimental.

Os benchmarks passam o limiar nos dois passos. O grupo A escolhe a menor massa
admissível de 700, 750, 800, 850, 900 e 1000 GeV; B procura picos próximos entre
600 e 800 GeV. A tolerância de distância dos picos continua 0,16 dex. As opções
`--benchmark-masses` e `--benchmark-range` permitem mudar essas faixas de forma
declarada. Com os dados v1, A usa m6=800 GeV e C=3,2401; 3,1681; 3,0453. A
coleta nova pode encontrar trios admissíveis em massas menores. O grupo B já
encontra m6=635, 645 e 680 GeV, com C=2,7290; 2,6715; 2,6769 e SNR próximo de
15 nos três cenários. As distâncias dos picos ficam abaixo de 0,014 dex.

Para fechar as figuras, confira `unresolved_selected_observables.csv`,
`interpolation_audit.csv`, `transition_topology_flags.csv` e o relatório JSON.
Uma falha persistente deve ser investigada pontualmente; o desenho conserva
essa lacuna até existir suporte numérico. Alpha/beta medidos nas bordas
também não garantem extremos globais dos observáveis no interior.

## Etapa posterior: m8=400 GeV

A coleta principal permanece com os três cenários atuais. O caso de 400 GeV
deve começar por uma conferência do vácuo a T=0 com o potencial completo:
diminuir m8 aumenta o termo phi^8/(16 m8^4) e pode retirar o vácuo eletrofraco
global de parte da janela antiga. O diagnóstico de árvore já dá

`V_tree(v)-V_tree(0) = -mh^2 v^2/8 + v^6/(8 m6^2) + 3 v^8/(16 m8^4)`.

Essa expressão é uma prévia; CW e a medida continuam na conferência completa.
Usar a janela anterior sem essa checagem pode desperdiçar a nova campanha.
Após resolver a cobertura principal, os resultados desse diagnóstico orientarão
um plano separado de fronteiras e observáveis para m8=400. A escolha entre
três e quatro cenários nas figuras permanece aberta.

Referências instrumentais: [Schmitz, PIS](https://arxiv.org/abs/2002.04615);
[Robson, Cornish e Liu, ruído LISA](https://arxiv.org/abs/1803.01944).
