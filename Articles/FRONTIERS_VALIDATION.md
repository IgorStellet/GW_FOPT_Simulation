# Validação da revisão de bordas - 22/10/2026

Base: 5e7b943b6edebb2c5e7f56a19750c24bb1bcfd84. Nenhum módulo Python v4 de
coleta, modelo, núcleo ou da análise de figuras anterior foi alterado.
Os dados recebidos foram lidos das exportações locais, sem copiar bancos do
CHE e sem enviar resultados de pesquisa ao GitHub.

## Auditoria dos dados reais recebidos

| m8 [GeV] | Pontos | Selecionados pelo corte forte | GW resolvido | Massas com GW |
|---|---:|---:|---:|---|
| inf | 8450 | 1751 | 62 | inf, 2000 |
| 840.8964152537145 | 9604 | 1748 | 32 | inf |
| 668.740304976422 | 11789 | 1722 | 71 | inf, 2000 |

São 29.843 pontos, 165 com GW e 7.600 razões críticas. Outros 59 registros
têm alpha calculado, com beta pendente. Nas testemunhas de borda (1.754),
somente nove têm alpha e três beta utilizáveis; o plano básico pede 1.751
coordenadas novas. Esses são valores nas bordas EM C, não extremos globais
demonstrados dos observáveis.

O gerador foi executado sobre os CSVs reais: sete figuras em PDF/SVG/PNG,
tabelas, relatório e ZIP compacto. PDFs renderizados com Poppler e
inspecionados visualmente. Os campos indisponíveis permanecem identificados,
sem preenchimento artificial. Há 372 cruzamentos numéricos de v(Tn)/Tn=1 e
80 de v(Tc)/Tc=1 no suporte permitido pelos dados; nenhum foi tratado como
expressão analítica. As fontes são identificadas por hashes no relatório.

Os extremos LISA AMOSTRADOS do caso puro nos dados recebidos são C=3.81 e
3.83, com rho_sw aproximadamente 10.2 e 107 para quatro anos. Isso não prova
que sejam os extremos exatos; a nova etapa de alcance é que refina brackets.

## Diagnóstico e integração física das duas bordas

Massas: m6=1000 GeV, m8=668.740304976422 GeV. Coordenadas de C retiradas das
testemunhas reais dos CSVs; a avaliação abaixo é uma integração local do
fluxo, não uma execução remota ou substituição da campanha do CHE.

| Quantidade | Borda inferior | Borda superior |
|---|---:|---:|
| C | 1.887272842575 | 3.335729461101 |
| Tn [GeV] | 121.20251081746953 | 46.093714146940606 |
| Tc [GeV] | 121.72739382076185 | 78.32315109922742 |
| alpha_trace | 0.0023424721703981387 | 0.20458545379644066 |
| beta/Hn | 57005.81575681097 | 188.48318125897114 |
| h efetivo [GeV] | 0.052488300329231664 | 0.5 |
| beta/Hn com h/2 | 55694.52929948274 | 162.41958889849013 |
| Mudança relativa | 0.02300267858497503 | 0.13828073245787534 |
| Tempo [s] | 90.0 | 75.2 |
| Resultado | nucleated, sem flags | nucleated, sem flags |

Na borda inferior, o passo fixo 0.5 produziu 4.626e6 e seu meio passo
8.636e4, diferença de 98%. A redução foi investigada no mesmo histórico
salvo, reconstruindo os objetos Phase do núcleo e usando sua própria
GravitationalWaveCalculator: h=0.1, 0.05, 0.025, 0.0125 produziu
58965.60, 56870.00, 55404.32 e 54138.61. Não se escreveu outra derivada.
O novo worker ajusta h pela margem local antes de chamar transition_record
original. A integração completa acima confirmou o ganho. Na borda superior
a margem era ampla e o passo permaneceu inalterado.

A retomada do mesmo plano/destino em outro diretório local terminou com
`complete: 0 pontos novos`. Os checkpoints foram publicados fechados.

## Piloto físico do alcance

Um plano local de um corte reutilizou os dois GW anteriores e avaliou
somente um novo C=2.128682278996, sob limite de 300 s e max-points=1.
Terminou paused após 41.3 s do ponto, preservando o checkpoint. A prévia
de retomada confirmou compatibilidade sem novos cálculos.

Esse novo ponto ficou inconclusivo: o núcleo não satisfez o residual de
nucleação e Tn estava na extremidade da fase. O registro conservou
nucleation_residual_outside_tolerance/beta_stencil_outside_metastable_interval;
não foi transformado em SNR=0 nem exclusão. A política de passo não resolve
falhas independentes da procura de fases ou nucleação. Elas continuam
exigindo diagnóstico localizado, sem repetir uma campanha inteira.

## Testes e preservação

- Suíte completa: 181 passaram e um teste específico de Linux foi ignorado
  no Windows. Inclui todos os testes antigos e 21 casos novos.
- Ruff passou. Os quatro templates/helpers novos passaram em bash -n.
- Regressões: intervalo inconclusivo, ausência de conexão vertical entre
  bordas, referência v/T com ramos e buracos, alpha com beta pendente,
  contraexemplo a extremos globais nas bordas, partições disjuntas,
  múltiplos intervalos LISA, falha interrompendo bracket, passo adaptado
  delegando ao recorder original, timeout como dado, reuso sem repetir GW,
  retomada e repetição explícita de observáveis pendentes.
- Sete figuras em três formatos foram verificadas nos testes; arquivos de
  entrada foram comparados por hash antes/depois. Booleanos False e campos
  ausentes dos CSVs são preservados.
- A formatação final dos módulos novos foi conferida por igualdade das
  árvores sintáticas Python, sem mudar o comportamento testado.
- Identidade v4 `provenance(Settings(), {})` conservada:
  d75a47c6b881b9a5b17045e114e4d683826cb5df43523723c0d9049fba5bcf66.

Os arquivos de validação/figuras ficam sob Articles/results, ignorado pelo
Git. As campanhas reais adicionais ainda precisam ser executadas pelo
usuário no CHE. Não houve acesso nem submissão remota nesta validação.
