# Validação das figuras - 22/10/2026

Base: commit 17967a86b4da497a1224d472b90c3c53ce1ef183 da branch
codex/articles-combined-potential-scan. Não foram alterados os módulos
numéricos v4, o modelo combinado ou o núcleo. Nenhuma campanha real do CHE
foi copiada para este checkout ou para o GitHub.

## Testes e instalação

- `python -m pytest -q tests Articles/tests`: 160 passaram, um foi ignorado no
  Windows (verificação Bash específica de Linux). A CI também executa agora
  Articles/tests, que antes não estava incluído no comando de testes.
- Ruff passou em Articles e nos três testes de artigo da pasta tests.
- Os dois templates novos passaram na verificação real de sintaxe Bash.
- Wheel construído sem instalar/atualizar as dependências numéricas da venv.
  Foram conferidos os seis módulos Articles/figures e a importação/chamada de
  ajuda a partir de uma pasta isolada, fora da raiz do checkout.
- Testes renderizaram 14 figuras, cada uma em PDF/SVG/PNG, com dados
  sintéticos explicitamente identificados. Também foram inspecionadas
  visualmente versões PDF renderizadas em PNG: janelas, painéis de observáveis
  e espectros. Uma prévia de estilo cobre toda a extensão m6=500..2000 GeV;
  seus dados sintéticos não são resultados científicos.

As regressões verificam: fontes sem alteração byte a byte; distinção Tc/Tn e
recusa de associação crítica ambígua; ausência de observáveis GW inventados;
fator sqrt(tempo) e conversão Hz/mHz no critério PIS; buracos de interpolação;
benchmarks colineares; intervalos detectáveis desconectados; conflitos entre
fontes; plano imutável e pontos explícitos; checkpoint fechado e retomada sem
recalcular pontos já confirmados; preservação da identidade numérica v4.

O primeiro teste sob a restrição de execução do ambiente do assistente teve
cinco falhas de inicialização do Git Bash (NtCreateDirectoryObject: acesso
negado), não falhas do helper Slurm. A suíte completa foi executada com essa
restrição removida e passou. Temporários dos testes ficaram no projeto.

## Ponto físico completo

Executado pelo comando `Articles.figures points`, com teto de 300 s, beta de
ordem 2, comparação com meio passo e recovery_attempts=0:

| Quantidade | Resultado |
|---|---:|
| m6 [GeV] | 1000 |
| m8 [GeV] | 668.740304976422 |
| C | 3.35 |
| Tempo do ponto [s] | 76.2 |
| Tn [GeV] | 40.73800650428223 |
| Tc [GeV] | 77.52169530019351 |
| alpha_trace | 0.33289431716939344 |
| beta/Hn | 116.72234483860271 |
| v(Tn)/Tn | 6.000400580275763 |
| v(Tc)/Tc | 2.952086690854706 |
| f_sw,peak [Hz] | 0.00091334605711425 |
| h² Omega_sw,peak | 8.061022207395195e-12 |
| rho_sw LISA, 4 anos (PIS histórico) | 331.50063168947787 |

O ponto terminou nucleated/selecionado, com GW resolvido. v(Tc) foi lido do
registro crítico salvo. O comando render criou as figuras, tabelas e pacote
de revisão desse ponto sem recalcular a física. A retomada em outro diretório
local publicou `complete: 0 pontos calculados` e preservou o dado confirmado.
Os números são saída do modelo/prescrições atuais, não medição experimental.

## Preservação das campanhas e limites

A identidade de referência `provenance(Settings(), {})` permaneceu
`d75a47c6b881b9a5b17045e114e4d683826cb5df43523723c0d9049fba5bcf66`.
O novo código mora num subpacote que não entra no glob Articles/*.py usado
pela campanha v4. O leitor reutiliza o snapshot local seguro, para não criar
WAL/SHM na origem. Os testes comparam todos os arquivos de entrada antes e
depois do desenho.

Ainda é necessário executar o job de figuras no CHE e revisar as imagens
com os 29.843 pontos reais. As contagens foram informadas pelo usuário; não
representam uma inspeção local dos bancos. As 1.459 ocorrências inconclusivas
reportadas pelo boundary_status e as colunas sem brackets completos não se
transformam em exclusões. Os mapas GW podem continuar esparsos até a primeira
avaliação de alpha/beta nos pontos adicionais escolhidos.

Esta validação não confirma acesso, submissão nem execução no cluster.
