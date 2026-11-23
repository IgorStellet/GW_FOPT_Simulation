# Definição física da campanha do artigo

## Potencial e convenções

O campo radial do Higgs satisfaz `H†H=phi²/2`. Em unidades de GeV, usamos

$$
V_{\rm eff}(\phi,T)=V_{\rm EFT}(\phi)+\Delta V_{\rm meas}(\phi)
+V_{\rm CW}^{\rm ren}(\phi)+\Delta V_T(\phi,T)+V_{\rm ring}^{L}(\phi,T).
$$

$$
V_{\rm EFT}(\phi)=-\frac{\mu^2\phi^2}{2}+\frac{\lambda\phi^4}{4}
 +\frac{\phi^6}{8m_6^2}+\frac{\phi^8}{16m_8^4},\quad
\lambda=\frac{m_h^2}{2v^2}-\frac{3v^2}{2m_6^2}-\frac{3v^4}{2m_8^4},\quad
\mu^2=\lambda v^2+\frac{3v^4}{4m_6^2}+\frac{v^6}{2m_8^4}.
$$

Essas relações seguem de $V_{\rm EFT}'(v)=0$ e $V_{\rm EFT}''(v)=m_h^2$.
Fixamos `v=246`, `mh=125`, `mW=80.36`, `mZ=91.19`, `mt=173.1` e
`Lambda=1000`. Os acoplamentos são `g=2mW/v`, `g'²=4mZ²/v²-g²` e
`yt=sqrt(2)mt/v`. As massas `m6` e `m8` são as escalas de supressão
definidas por $m_6^{-2}=c_6/f^2$ e $m_8^{-4}=c_8/f^4$.
`inf` desliga o respectivo operador.

A contribuição renormalizada da medida, conforme o artigo de Berganholi et al., é

$$
\Delta V_{\rm meas}=-\frac{\Lambda^4}{8\pi^2}
\left[\log(1-t)+\frac{(1-2t_0)t+t^2/2}{(1-t_0)^2}\right],\qquad
t=\frac{C\phi^2}{\Lambda^2},\quad t_0=\frac{Cv^2}{\Lambda^2}.
$$

Suas duas primeiras derivadas se anulam em $v$. Para $C>0$, o domínio é
$|\phi|<\Lambda/\sqrt C$. Para $C=0$, a contribuição da medida é zero.

## Coleman–Weinberg e matching a temperatura zero

Os termos diretos de um loop são de $W,Z,t$, em gauge de Landau. O Higgs
radial é o campo do potencial e do bounce. Os determinantes de Higgs e
Goldstones e os anéis escalares ficam fora da aproximação adotada.

$$
M_W^2(\phi)=\frac{g^2\phi^2}{4},\qquad
M_Z^2(\phi)=\frac{(g^2+g'^2)\phi^2}{4},\qquad
M_t^2(\phi)=\frac{y_t^2\phi^2}{2}.
$$

Os contratermos $-\delta\mu^2\phi^2/2+\delta\lambda\phi^4/4$ são somados
analiticamente ao Coleman–Weinberg. Como $M_i^2\propto\phi^2$, fixar
$V_{\rm CW}^{\rm ren\,\prime}(v)=V_{\rm CW}^{\rm ren\,\prime\prime}(v)=0$
dá

$$
V_{\rm CW}^{\rm ren}=
\frac{6m_W^4+3m_Z^4-12m_t^4}{64\pi^2v^4}
\left[\phi^4\left(\log\frac{\phi^2}{v^2}-\frac32\right)+2v^2\phi^2\right].
$$

O termo logarítmico tem limite zero em $\phi=0$. O matching fixa a
curvatura local em $m_h^2$ no modelo efetivo. As referências de
Croon et al. discutem a omissão dos loops do escalar no bounce sob
uma hierarquia de escalas. A escolha aqui é uma aproximação para a coleta.

## Correções térmicas e anéis de gauge

$$
\Delta V_T=\frac{T^4}{2\pi^2}\left[
  6J_B(M_W^2/T^2)+3J_B(M_Z^2/T^2)-12J_F(M_t^2/T^2)\right].
$$

Aqui $J_{B,F}(y)=\int_0^\infty dx\,x^2
\log(1\mp e^{-\sqrt{x^2+y}})$, com o sinal superior para bósons.
O código usa as funções completas por spline em $M^2/T^2$.
O `Jf` do src já inclui o sinal fermiônico, por isso sua multiplicidade
na implementação é $+12$.

Na ressoma de Arnold–Espinosa, usamos as massas de Debye do plasma do
SM, $\Pi_W=11g^2T^2/6$ e $\Pi_B=11g'^2T^2/6$, que incluem o
dupleto na polarização de gauge. Para os modos longitudinais carregados,
$X_W=M_W^2+\Pi_W$. Para os neutros, escrevemos

$$
a=\frac{g^2\phi^2}{4}+\Pi_W,\qquad d=\frac{g'^2\phi^2}{4}+\Pi_B,\qquad
b=-\frac{gg'\phi^2}{4},\qquad X_\pm=\frac{a+d\pm\sqrt{(a-d)^2+4b^2}}2.
$$

$$
V_{\rm ring}^{L}=-\frac{T}{12\pi}\left[
  2\bigl(X_W^{3/2}-(M_W^2)^{3/2}\bigr)
  +X_+^{3/2}-(M_Z^2)^{3/2}+X_-^{3/2}
\right].
$$

Os fatores contam $W_L^+$, $W_L^-$ e os dois modos neutros da
matriz de $W_L^3$ e $B_L$. A subtração remove os cúbicos que já
estão em $\Delta V_T$. As funções térmicas usam massas originais. A
aproximação de alta temperatura entra nas massas de Debye dos anéis.

## Seleção e observáveis

A campanha busca uma história iniciada na fase simétrica, na janela
`1<=T<=200 GeV`. Exigimos a condição necessária `V(v,0)<V(0,0)` para selecionar
o vácuo eletrofraco final. Esse teste barato evita bounces em pontos que já
falham nessa condição. Ele não certifica estabilidade global fora da janela
de campos examinada. O valor de `V(v,0)-V(0,0)` fica salvo.

A busca preserva a origem nas temperaturas em que ela é um mínimo local.
Uma fase inicial diferente da simétrica recebe decisão desconhecida e
diagnóstico. A janela de campo chega a 1000 GeV ou, para C positivo, a
0.98 Lambda/sqrt(C), quando esse limite é menor. Entre fases coexistentes,
a busca avalia a ação durante o resfriamento e procura o primeiro cruzamento
do alvo encontrado.

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

$$
\alpha_{\rm energia}=\frac{\Delta V-T_n\Delta(\partial_TV)}{\rho_{\rm rad}},\qquad
\alpha_{\rm tr}=\frac{\Delta V-(T_n/4)\Delta(\partial_TV)}{\rho_{\rm rad}}.
$$

As derivadas são parciais a campo fixo. Ambas as convenções, seus sinais e
os valores intermediários são armazenados. O espectro usa alpha_trace.

$$
\frac{\beta}{H_n}=T_n\frac{d(S_3/T)}{dT}\bigg|_{T_n}.
$$

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

- [Chala, Krause e Nardini (2018)](https://arxiv.org/abs/1802.02168) para as normalizações dos operadores de dimensão seis e oito.
- [Berganholi et al. (2025)](https://arxiv.org/abs/2502.09593) para a medida funcional e o Coleman–Weinberg renormalizado.
- [Arnold e Espinosa (1993)](https://arxiv.org/abs/hep-ph/9212235) para a ressoma dos anéis.
- [Croon et al. (2021)](https://arxiv.org/abs/2009.10080) para as massas longitudinais e os limites das aproximações no bounce.
- [Schmitz (2021)](https://arxiv.org/abs/2002.04615) para sensibilidades integradas de pico, tempo e SNR.
- [Robson, Cornish e Liu (2019)](https://arxiv.org/abs/1803.01944) para o modelo de ruído LISA.
