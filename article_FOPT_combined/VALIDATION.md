# Validação da edição T²

## Revisão da busca e das figuras

Em 23 de novembro de 2026, a busca foi revista com o mesmo `model.py` e
o mesmo núcleo. Três pontos registrados como `no_nucleation_found`
no primeiro export T² passaram a ter nucleação forte e observáveis completos:

| m6 [GeV] | m8 [GeV] | C | Tn [GeV] | S3/Tn | alpha_trace | beta/Hn |
|---:|---:|---:|---:|---:|---:|---:|
| 850 | inf | 3.0035813493 | 86.6732 | 140.0751 | 0.01843 | 2037.91 |
| 1725 | inf | 3.5005087623 | 81.4376 | 139.9799 | 0.02375 | 1883.75 |
| 1525 | 400 | 0.5029582352 | 63.8744 | 139.7756 | 0.05441 | 490.49 |

O ponto de 850 GeV também foi verificado pelo comando `point`, incluindo
subprocesso isolado e checkpoint. Os testes exercitam a recuperação de
cruzamentos internos, uma ação inconclusiva durante a busca e a preservação
de transições resolvidas quando uma etapa posterior falha.

O controle puro com C=2.895 forneceu Tn=124.9709 GeV e v(Tn)/Tn=1.0651.
A busca numa etapa posterior do resfriamento ficou inconclusiva e está
registrada junto da transição já resolvida. Esse ponto levou cerca de 348 s
no ambiente local, o que motiva conferir o orçamento no piloto do CHE.

Os 2.323 registros exportados da v1 foram usados para regenerar as seis
figuras e os suplementos em modo rascunho. Com esses mesmos dados, a área
triangulada com suporte de beta nas Figuras 2 a 4 cresceu cerca de 16%.
Os 203 observáveis incompletos e as 781 colunas com pendências continuam
identificados pela auditoria. A reconstrução gráfica desses dados testa
a apresentação. Os mapas da campanha corrigida dependem da nova coleta.

## Validação da primeira edição

Data: 14 de novembro de 2026.

## Verificações executadas

- **50 testes automatizados**: massas e multiplicidades, limites do SM,
  autoenergias por Hessiana O(4) independente, matriz longitudinal, domínio,
  entrada correta das splines, fases em ordem reversa, beta e estabilidade,
  sinais termodinâmicos, fronteiras, sharding, cache, timeout real, exclusão
  barata do vácuo e exportação ZIP com lock ativo no Windows.
- Lint e formatação de todos os Python deste diretório.
- Sintaxe Bash dos scripts de setup, submissão e dos quatro templates Slurm.
- Geração integral de seis PDFs/PNGs principais e dois suplementos com uma
  fixture sintética identificada como DRAFT; inspeção visual de atlas e GW.
- Exportação real de nove pontos físicos e teste de integridade do ZIP;
  locks e temporários ficam fora do arquivo.
- Comparação do diff com main: nenhuma alteração em src, docs da tese,
  testes da main, pyproject ou workflow global.

Comando dos testes:

```bash
python -m pytest -c article_FOPT_combined/pytest.ini article_FOPT_combined/tests -q
```

## Pilotos físicos reais

Os nove pontos abaixo foram resolvidos com o potencial desta edição,
T entre 1 e 200 GeV, os quatro m8, a tolerância de ação 0.5 e o controle
adaptativo de beta. Todos têm nucleação forte aceita e observáveis completos.
Os dados de tabela são resultados calculados; a fixture gráfica acima é
separada e serve somente para verificar o desenho.

| m6 [GeV] | m8 [GeV] | C | Tn [GeV] | Tc [GeV] | alpha_trace | beta/Hn | h usado [GeV] | mudança no último par de passos |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| inf | 400.0 | 0 | 85.327525 | 96.579143 | 0.01817399 | 1870.02 | 0.25 | 0.425% |
| 1000.0 | 668.740304976422 | 2.5 | 109.240924 | 112.248011 | 0.006553474 | 8788.04 | 0.25 | 3.230% |
| inf | 668.740304976422 | 3.3 | 95.202509 | 100.770298 | 0.01279206 | 4454.73 | 0.25 | 1.516% |
| inf | 840.8964152537145 | 3.4 | 94.689289 | 100.254535 | 0.01321072 | 4463.95 | 0.25 | 1.260% |
| 1000.0 | inf | 2.8 | 107.610394 | 110.945862 | 0.007262245 | 8229.85 | 0.25 | 2.092% |
| inf | inf | 3 | 120.604111 | 121.316337 | 0.003307679 | 47310.9 | 0.0923368 | 9.157% |
| inf | inf | 3.5 | 91.499064 | 97.952735 | 0.0152664 | 3729.68 | 0.25 | 1.450% |
| inf | inf | 3.8 | 54.000067 | 78.950933 | 0.1095903 | 413.174 | 0.25 | 3.972% |
| inf | inf | 3.83 | 44.697937 | 76.584617 | 0.2261424 | 209.144 | 0.25 | 0.932% |

Maior residual de nucleação nesses testes: **0.400051**, dentro de 0.5.
O ponto C=3, m6=m8=inf ficou sem beta na versão de passo fixo porque o
stencil de 0.5 GeV cruzava a coexistência. O algoritmo validado recuperou
esse observável com passo 0.092337 GeV e estabilidade
de aproximadamente 9.2% no último par de estimativas. Isso testa uma causa
real de lacunas, em vez de apenas esconder o espaço em branco no desenho.

O ponto C=3.83, m6=m8=inf fornece SNR acústico LISA
**91.075**, em quatro anos, acima do limiar 10.
Também foram testados pontos não detectáveis; a classificação não é imposta
a todos os pontos com nucleação.

O código de pré-checagem térmica examinou adicionalmente 160 combinações
coarse de m6,m8,C. Das 119 que satisfaziam V(v,0)<V(0,0), nenhuma favoreceu
um campo alto em T=200 GeV nessa malha de diagnóstico. Isso não é um scan
de nucleação nem uma prova analítica global. Em m8=400,C=0,m6=inf, o
contraste entre T=250 e T=200 motivou o teto padrão e seu teste de regressão.

Fingerprint físico/configuração do piloto:
`fa7647d7d3238091adb83c72326de6a467468c235ab524774ec0dd51c970d714`.

Ambiente local: Windows, Python 3.12.14;
NumPy 2.5.3, SciPy 1.18.1,
Matplotlib 3.11.2. O manifesto registra
as versões efetivas em cada ambiente. Os testes no CHE devem ser repetidos
com seu Python 3.11 e os pacotes que forem instalados ali.

## Limites do que foi validado

Não houve acesso ao CHE nem submissão de jobs desta branch nesta sessão.
A grade de 1208 colunas e os mapas científicos finais ainda dependem dos
pilotos de fronteiras/interior e da campanha no cluster. Os comandos estão
em CLUSTER.md, incluindo o uso do mesmo template do piloto em três etapas.

Esses testes verificam implementação, fluxo e alguns pontos representativos.
Eles não estabelecem precisão física uniforme para a grade inteira. As
aproximações hard de alta temperatura, a prescrição infravermelha de passo
fixo a T=0, a continuação das splines e a hipótese de uma faixa conectada
estão explicitadas em PHYSICS.md e ARCHITECTURE.md. O alvo de nucleação
também não substitui uma análise de percolação/completação.

A versão gráfica final exige sua auditoria de cobertura e interpolação,
inclusive de SNR dos detectores. Pontos pendentes ou instabilidade não
recebem valores interpolados como se tivessem sido resolvidos.
