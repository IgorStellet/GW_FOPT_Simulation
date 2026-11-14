# Definição física desta campanha

## Potencial e convenções

Em GeV, com `H†H=phi²/2`,

\[
V_0=-\frac{\mu^2\phi^2}{2}+\frac{\lambda\phi^4}{4}
 +\frac{\phi^6}{8m_6^2}+\frac{\phi^8}{16m_8^4}.
\]

\[
\lambda=\frac{m_h^2}{2v^2}-\frac{3v^2}{2m_6^2}-\frac{3v^4}{2m_8^4},\qquad
\mu^2=\lambda v^2+\frac{3v^4}{4m_6^2}+\frac{v^6}{2m_8^4}.
\]

Fixamos `v=246`, `mh=125`, `mW=80.36`, `mZ=91.19`, `mt=173.1` e
`Lambda=1000`. Os acoplamentos são `g=2mW/v`, `g'²=4mZ²/v²-g²`,
`yt=sqrt(2)mt/v`. A parametrização usa apenas m6 e m8; `inf` desliga o operador.

As curvaturas escalares são

\[
M_h^2=-\mu^2+3\lambda\phi^2+\frac{15\phi^4}{4m_6^2}+\frac{7\phi^6}{2m_8^4},
\qquad
M_G^2=-\mu^2+\lambda\phi^2+\frac{3\phi^4}{4m_6^2}+\frac{\phi^6}{2m_8^4}.
\]

Higgs tem multiplicidade 1; Goldstones têm multiplicidade 3. W, Z e top
usam `g²phi²/4`, `(g²+g'²)phi²/4` e `yt²phi²/2`, respectivamente.

A medida, tratada como contribuição de origem de um loop, é

\[
V_{\rm medida}=-\frac{\Lambda^4}{8\pi^2}
\left[\log(1-t)+\frac{(1-2t_0)t+t^2/2}{(1-t_0)^2}\right],\quad
t=\frac{C\phi^2}{\Lambda^2},\quad t_0=\frac{Cv^2}{\Lambda^2}.
\]

Seu domínio é estritamente `|phi|<Lambda/sqrt(C)`. Primeira e segunda
derivadas da medida se anulam em v. Seus vértices permanecem fora dos loops
térmicos nesta contagem perturbativa; promovê-la a interação dominante seria
outra prescrição, exigindo rever massas e vértices consistentemente.

## Um loop e a prescrição a temperatura zero

\[
V_{\rm CW}=\sum_i\frac{n_iM_i^4}{64\pi^2}
\left[\log\frac{|M_i^2|}{v^2}-c_i\right].
\]

Usamos `(n_h,n_G,n_W,n_Z,n_t)=(1,3,6,3,-12)`, `c=3/2` para escalares/top
e `c=5/6` para gauge. O módulo no log seleciona a parte real da continuação.
Um piso de `1e-300` protege a avaliação de `log(0)`; o limite do produto
`M^4 log M²` é zero. Isso não remove a singularidade de sua segunda derivada.

Os contratermos finitos são `-delta_mu² phi²/2+delta_lambda phi⁴/4`.
Eles conservam a prescrição operacional anterior: gradiente e Hessiana de CW
com diferenças de ordem dois e **passo fixo de 0.246 GeV** em v. Se A e B são
essas duas derivadas, `delta_lambda=(A/v-B)/(2v²)` e
`delta_mu²=A/v+delta_lambda*v²`.

Essa é uma regularização numérica declarada da curvatura de Goldstone.
**Ela não equivale a uma renormalização completa com matching à massa de polo.**
Sua sensibilidade deve ser parte do estudo de incertezas antes de atribuir
precisão física aos resultados. A correção térmica desta versão mantém essa
escolha separada e explícita, permitindo comparações controladas.

## Termos térmicos e ressoma escolhida

O potencial total é

\[
V=V_0+V_{\rm CW}+V_{\rm CT}+V_{\rm medida}+V_T+V_{\rm ring}.
\]

`V_T` usa as funções **completas por spline** do src, com massas originais,
incluindo Higgs e Goldstones. O argumento é `M²/T²`, inclusive para curvaturas
negativas. `Jf` do núcleo já contém o sinal fermiônico; sua multiplicidade
térmica é `+12`. Os coeficientes hard T² abaixo **não são somados novamente
a V_T**: servem para construir as massas nos anéis.

O backend herdado usa a spline bosônica em `-3.72402637<=M²/T²<=1410`
e a fermiônica em `-6.82200203<=M²/T²<=1350`. Abaixo do limite inferior,
conserva o valor de borda; acima do superior, retorna zero. Essa convenção
permanece a do src, sem uma nova malha térmica neste diretório. Em forte
super-resfriamento, o uso dessa continuação e o domínio da aproximação hard
merecem estudo próprio; a comparação de passos de beta não mede esse erro físico.

Definindo

\[
c_{\rm th}=\frac{3g^2+g'^2}{16}+\frac{y_t^2}{4}+\frac{\lambda}{2},
\]

conservamos as contribuições hard de um loop, dependentes do campo:

\[
\Pi_h=T^2\left[c_{\rm th}+\frac{3\phi^2}{m_6^2}+\frac{25\phi^4}{4m_8^4}\right],
\qquad
\Pi_G=T^2\left[c_{\rm th}+\frac{\phi^2}{m_6^2}+\frac{5\phi^4}{4m_8^4}\right].
\]

Elas são as curvaturas radial/transversal da parcela hard T². A parte escalar
vem de `T²*(Mh²+3MG²)/24`. Na origem, os quatro modos recebem a mesma correção;
o limite `m6,m8 -> inf` recupera a expressão usual do SM.

Arnold–Espinosa adiciona

\[
V_{{\rm ring},s}=-\frac{T}{12\pi}\operatorname{Re}
\left[(M_h^2+\Pi_h)^{3/2}-(M_h^2)^{3/2}
 +3\{(M_G^2+\Pi_G)^{3/2}-(M_G^2)^{3/2}\}\right].
\]

`real_cubic` implementa `Re[(M²+i0)^(3/2)]`: zero para M² negativo. As massas
originais negativas continuam presentes nas funções térmicas e em CW; apenas
se escolhe a parte real do termo cúbico. O módulo elevado a `3/2` seria outra
função e produziria um potencial diferente.

Os termos flower hard T⁴/T⁶ ficam fora desta versão. A escolha é
**um loop mais anéis, com autoenergias escalares hard T²**, e não um matching
SMEFT completo a duas/três ordens de loops ou a toda ordem inversa de massa.
Os operadores superiores exigem uma contagem perturbativa própria para esse
avanço. O uso de integrais completas em V_T não torna exata a aproximação de
alta temperatura empregada nas autoenergias.

No setor de gauge usamos `Pi_W=11g²T²/6`, `Pi_B=11g'²T²/6` e diagonalizamos

\[
\overline{\mathcal M}_L^2=
\frac{\phi^2}{4}\begin{pmatrix}g^2&-gg'\\-gg'&g'^2\end{pmatrix}
+\begin{pmatrix}\Pi_W&0\\0&\Pi_B\end{pmatrix}.
\]

O anel soma `2*(W_L res³-W³)+(Z_L res³-Z³)+gamma_L res³`. Modos transversais
permanecem nus nessa ordem. Esse tratamento também substitui a antiga
aproximação de três modos efetivos de gauge. Fermions não têm anel de modo
zero, embora yt contribua para Pi_h/Pi_G.

## Seleção e observáveis

A campanha busca uma história iniciada na fase simétrica, na janela
`1<=T<=200 GeV`. Exigimos a condição necessária `V(v,0)<V(0,0)` para selecionar
o vácuo eletrofraco final. Esse teste barato evita bounces em pontos que já
falham nessa condição. Ele não certifica estabilidade global fora da janela
de campos examinada. O valor de `V(v,0)-V(0,0)` fica salvo.

Antes de rastrear fases, também examinamos o mínimo do potencial em T_max.
Se ele favorece um campo alto, o ponto recebe decisão desconhecida e um
diagnóstico, em vez de ser anunciado como ausência de nucleação. O piloto
com m8=400, m6=inf, C=0 mostrou um mínimo em phi~431.5 GeV em T=250 GeV,
enquanto a origem é o mínimo na mesma janela em T=200 GeV. Isso motivou o
teto padrão de 200 GeV e a checagem explícita da fase inicial. Um mínimo desse
tipo exige examinar a validade da aproximação hard de alta temperatura em
campos grandes; seu aparecimento não prova não restauração física de simetria.

Uma nucleação é aceita se `|S3(Tn)/Tn-140|<=0.5`. A região selecionada também
exige `|phi_low(Tn)|/Tn>1`. Salva-se separadamente o salto entre os dois campos.
Esse é um critério operacional de força, com as limitações de dependência de
gauge do potencial perturbativo. A temperatura crítica e a razão `v(Tc)/Tc`
são obtidas das fases degeneradas calculadas, e não de uma linha analítica
universal tomada do modelo com apenas C.

O alvo 140 adota a aproximação de nucleação em um Universo dominado por
radiação. Esta campanha não integra a probabilidade de percolação/conclusão
da transição; em super-resfriamento forte, essa é uma verificação física
adicional. `observables_complete` significa que as grandezas solicitadas
foram calculadas e passaram pelos critérios numéricos, não que a conclusão
cosmológica da transição foi demonstrada.

Com Delta sempre falso menos verdadeiro e `rho_rad=pi²*g* Tn⁴/30`,

\[
\alpha_{\rm energia}=\frac{\Delta V-T_n\Delta(\partial_TV)}{\rho_{\rm rad}},\qquad
\alpha_{\rm tr}=\frac{\Delta V-(T_n/4)\Delta(\partial_TV)}{\rho_{\rm rad}}.
\]

As derivadas são parciais a campo fixo. Ambas as convenções, seus sinais e
os valores intermediários são armazenados. O espectro usa alpha_trace.

\[
\frac{\beta}{H_n}=T_n\frac{d(S_3/T)}{dT}\bigg|_{T_n}.
\]

O padrão é a diferença central com dois bounces em `Tn+-0.5 GeV`. Se ela cruza
a coexistência, usamos um stencil unilateral de ordem dois: para trás,
`(3F0-4F_-1+F_-2)/(2h)`; para frente, `(-3F0+4F_+1-F_+2)/(2h)`.
F0 reaproveita a ação de nucleação; são dois bounces novos. Se nenhum stencil
cabe com o máximo solicitado, o padrão adapta o passo para 80% da maior
abertura admissível. O modo adaptativo padrão compara h e h/2 em todos os
pontos, com piso de 0.01 GeV e tolerância relativa de 20%, mantendo o valor
mais fino do par estável. Cada estimativa continua sendo de ordem dois; o
teste de estabilidade não a transforma em uma derivada de ordem quatro. Esses
limites, o passo pedido/usado, os stencils e todas as ações ficam registrados.
Sem estabilidade, beta continua pendente; a nucleação e alpha são preservados.
`init --fixed-beta-step` desativa a adaptação e usa uma única estimativa de
dois bounces novos no passo pedido, quando admissível. Essa validação
numérica fica fora das figuras do artigo.

## Ondas gravitacionais e detectores

Esta versão apresenta o **canal acústico**, calculado pelas funções do src,
com suas eficiências e supressão padrão, `v_w=1`, `g*=106.75`, `T*=Tn`.
Os ajustes do núcleo usam mHz; todos os arquivos do artigo usam Hz. A ponte
converte as frequências uma vez. A campanha registra pico e parâmetros para
reconstruir o espectro, em lugar de gravar milhares de frequências por ponto.

As sensibilidades PIS acústicas de Schmitz são normalizadas em 1 ano e SNR=1.
Para a apresentação de 4 anos, `rho_sw=ratio_1yr*sqrt(4)`. O padrão de seleção
instrumental é `rho_sw>=10`. A frequência do pico também entra: comparar só
a amplitude ao mínimo da curva LISA não define a região detectável.

A PLIS LISA no painel de espectros é uma referência visual, calculada com o
ruído de Robson et al., braços de 2.5 Gm e foreground de 4 anos. A classificação
instrumental usa PIS e a forma acústica adotada. Isso não equivale ao SNR de
um espectro total incluindo turbulência/colisões, nem a uma análise de
inferência de parâmetros. A proximidade de dois picos é um candidato a
coincidência de sinais, não uma demonstração de degenerescência experimental.

## Referências

- [Delaunay, Grojean e Wells (2008)](https://arxiv.org/html/0711.2511v2): massas térmicas e anéis em uma extensão com operador sextico.
- [de Vries et al. (2018), apêndice B](https://arxiv.org/html/1710.04061v2): termo radial dependente do campo do operador de dimensão seis.
- [Chala, Krause e Nardini (2018)](https://arxiv.org/html/1802.02168v2): normalizações e curvaturas de dimensão seis/oito.
- [Curtin, Meade e Ramani (2018)](https://doi.org/10.1140/epjc/s10052-018-6268-0): aproximações e limites da ressoma térmica.
- [Croon et al. (2021)](https://doi.org/10.1007/JHEP04(2021)055): contagem perturbativa, flores e incertezas além da prescrição desta campanha.
- [Schmitz (2021)](https://arxiv.org/html/2002.04615v3): sensibilidades integradas de pico, tempo e SNR.
- [Robson, Cornish e Liu (2019)](https://arxiv.org/abs/1803.01944): modelo de ruído LISA.
