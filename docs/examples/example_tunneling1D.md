# Tunneling 1D - All that you need to know

This page visa te ensinar apenas aquilo que é necessário para que alguem possa utilizar e chamar as funções que encontram
a solução de bounce no espaço de 1 campo (1D).

Para aqueles que não estão interessados em ver como cada função funciona separadamente e exemplos claros e cada subfunção
do módulo, esse é o lugar. Entretanto, caso haja dúvidas sobre o funcionamento de uma função espécífica ou queira olhar
uma função específica recomendo olhar a pasta [modules/tunneling1D](/docs/modules/tunneling1D).


## Single Field Instaton

### Introduction and minimal params
Nesse módulo queremos resolver a equação diferencial: 


$$\frac{d^2 \phi}{dr^2}+\frac{\alpha}{r}\frac{d\phi}{dr} = \frac{dV}{d\phi} $$

É importante frizar que essa equação vem de uma transformação do laplaciano para transformada esférica e de assumir uma 
simetria esférica, além disso, mais importante ainda, essa equação já possui uma inversão do potencial que aparece ao
fazer a rotação de wick no eixo temporal, ou seja, passando $t\rightarrow -i\tau$, de modo que obtemos a equação,
$\nabla^2 \phi = V'(\phi)$ [com $\nabla$ sendo 4D], com $\rho\equiv \sqrt{\tau^2+x^2+y^2+z^2}$.


Observação: Essa é uma aproximação semiclássica e é válida para o tunelamento!, governando apeans a parte onde o campo
transiciona (isso deve ser descrito quanticamente/semiclássico pois são caminhos inacessíveis classicamente). Após o 
tunelamento voltamos a descrever de forma clássica, de modo que em t=0, o que "vemos" no espaço é o perfil $\phi_b(r)$
descrito pela solução de bounce $\phi_b$ encontrada!. Esse ponto é bastante crucial, podemos imaginal uma linha temporal
de $\tau \rightarrow \infty$ até $\tau=0$ temos a descrição semiclássica feita aqui (lembrando qeu $\tau$ crescendo
são altas temperaturas logo quanto maior $\tau$ estamos caminhando para trás no universo, indo para o ínicio). Em $\tau$
igual a 0 voltamos para o tempo real $t=0$ e descrevemos classicamente, com as seguintes equações:


$$\ddot{\phi} -\nabla^2 \phi =\Box \phi = -V'(\phi) \qquad \dot{\phi}=0; \quad \phi(x,t=0)=\phi_{bounce}(r)  $$


Por fim, $\alpha$ é o coeficiente de atrito, dado por $d-1$, onde d é o número de dimensões do problema. 
Para temperaturas muito antas a dimensão temporal deixa de aparecer (isso está associado a grandes oscilações), de modo 
que temos $T\gg 0 \rightarrow \alpha=2 | (O(3))$; $T\rightarrow 0 | \alpha = 3|(O(4))$. Geralmente a transição ocorre em
temperaturas altas de modo que é definido como padrão $\alpha=2$ na função.

Dito essa introdução mínima, podemos responder, o que precisamos fornecer minimamente para o código funcionar (3 parâmetros):

* $\phi_{True}$ - Valor do campo no mínimo real
* $\phi_{meta}$ - Valor do campo no mínimo metainstável 
* $V(\phi)$ - Valor do potêncial sob o qual o campo está submetido

Se tiver analitacamente o valor de $V'$ ou $V''$ também pode ser fornecido, assim como de $\alpha$ se diferente de 2.

```python
import numpy as np
from CosmoTransitions.tunneling1D import SingleFieldInstanton

# Quartic potential with two minima
def V(phi):  return 0.25*phi**4 - 0.49*phi**3 + 0.235*phi**2

# Instantiate the solver (use builtin derivatives)
inst = SingleFieldInstanton(
    phi_absMin=1.0,
    phi_metaMin=0.0,
    V=V,
    alpha=2,          # O(3) symmetry
)
```

### About contorno counditions

Esperamos que a solução de bounce respeite as seguintes condições: A solução de bounce deve sair do repouso e caminhar 
até o falso vácuo parando em cima do mesmo. 

É importante visualizar em mente que estamos observando o potencial invertido, invertendo o potencial podemos imaginar
algo clássico, de modo que o caminho natural é sair em algum valor do topo, perto de $\phi_{true}$ e chegar no topo menor
$\phi_{meta}$ sem velocidade, precisamos achar o ponto que soltando a bolinha em $V$ entre $\phi_{true}$ e $\phi_{meta}$
chegue em $\phi_{meta}$ parado, é claro que assumimos a existência de uma única solução!. 

Desse modo, as condições de contorno podem ser resumidas em:

* $\phi'(0)=0$
* $\phi'(\infty) = 0$
* $\phi(\infty) = \phi_{meta}$

Note que nenhuma dessas condições impõe nada sobre $\phi_0 \equiv \phi(0)$, de modo que o campo "incial" em $\tau$
(Podemos imaginar que é o começo do campo real, ou para nós, onde termina a solução após o bounce), pode ser qualquer valor
entre o vácuo verdadeiro e o vácuo meta instável.

E como o vácuo atinge o verdadeiro após a transição ? Após a transição quem governará a dinâmica será a clássica, dado pela
equação com a assinatura -1, de modo que a descrição do relaxamento do campo até $\phi_{true}$ após o bounce é dado pela 
dinâmica clássica, após a nucleação da bolha. É claro que na maioria dos casos queremos que todo o campo vá para $\phi_{true}$
recuperando o que observamos no universo (até agora).

* Pq $\phi'_0 = 0$? Lemrando que ' representa a derivada radial. Temos simetria radial de modo que
tiramos automaticamente que $\phi'_0=0$. Além disso, o termo de fricção $\alpha \phi'/r$ fica singular
a menos que $\phi_0$ tenda a 0 também quando r fica pequeno.
* No tempo real isso é refletido com $\dot{\phi}(x,t=0)=0$ (derivada temporal).
* Note que para $t>0$ podemos ter $\dot{\phi}\neq 0$, mas $\phi'=0$ para todo t, visto que mantemos a simetria radial.


Com isso fechamos a ideia geral da teoeria e podemos partir para outros parâmetros importantes encontrado pelo código.


### Other imporants params in $V(\phi)$ | Lot 2 of the code

Temos dois pontos importantes em $V(\phi)$. 

Invertendo o potencial e pensando classicamente, podemos perceber rapidamente que existe um ponto mínimo do qual
$\phi_0$ pode sair para atingir $\phi_{meta}$. Esse ponto é quando o potencial atinge o mesmo valor de $V(\phi_{meta})$ 
novamente (após a barreira). Nesse ponto, definido por $\phi_{bar}$ o potencial se encontra com a energia mínima necessária
para poder subir o poço de potencial (como está invertido esse poço é a barrreira) e atingir $\phi_{meta}$. Dessa forma,
temos uma região clara de busca pela condição incial certa, definida por:

$$V(\phi_{bar}) \geq V(\phi_0) \geq V(\phi_{True}) $$

$\phi_{bar}$ pode ser interpretado como um ponto de retorno se a partícula fosse soltada de $\phi_{meta}$ nesse
potencial invertido. 

Outro ponto importante é $\phi_{top}$. Esse ponto, nada mais é do que o $V(\phi)$ máximo, o topo da barreira de potencial
Além desse ponto ser utilizado para encontrar o $\phi_{bar}$ dentro do código, também é importante para deinir a
"grossura"/ "espessura " da parede de potencial.

O que isso significa exatamente? Imagine que próximo ao $\phi_{top}$ temos:

$$ V(\phi) = V_{top}-\frac{1}{2} k^2 (\phi-\phi_{top})^2$$

Temos portanto, $k=1/\sqrt{V''(\phi_{top})}\equiv r_{scale}$. Essa é uma medida de quanto está separado
o campo $\phi_0$ do campo $\phi_{meta}$ na solução encontrada e serve para definir muito dos limites numéricos do código.
Na solução de bounce isso ficará mais claro. Esperamos que em thin_wall seja pequeno, visto que a transição ocorre abrupdamente
enquanto para thickwall seja mais grossa, correspondendo aos nomes de fato.

Observação: Para thick wall $V''(\phi_{top})$ pode ser quase 0, de modo que podemos fazer uma aproximação
isolando k na equação e calculando num ponto próximo a $\phi_{top}$.


### 1. Visualização de todos os pontos.


[Colocar gráfico + discussões]


###  Near solutions and initial conditions| Lot 3

Para qualquer valor de $\phi(r)$ próximo a algum $\phi_0$ arbitrário podemos reescrever
a EDO de forma aproximada por:

$$\phi(r)-\phi_0 = \frac{dV}{d2V}\Bigg[\Gamma(\nu+1)\Big(\tfrac{t}{2}\Big)^{-\nu} I_\nu(t)-1\Bigg]$$

Obs: Coorrigi o meeting das funções (corte para $10^{-5}$). 

Essa função é valida sempre que o campo $\phi(r)$ está próximo ao $\phi_0$ escolhido, independetemente se r é grande ou pequeno
Isso é importante de entender para conseguirmos entender melhor as funções futuras.

Logo, podemos dizer que essa expansão é válida em torno de um ponto onde ${\phi}'\approx 0$.
e podemos levar r até onde quisermos se a derivada continuar próxima a 0 (continua válida para r grande);

Isso deve ser utilizado apenas em torno de $\phi_0$, é assumido intermante que r=0 de $\phi_0$ expandido logo recomendado
utilizar apenas nessa região ou reparametrizar se necessário utilizar em outras.


Como há uma singularidade em r=0, precisamos começar nossa integração um pouco distante desse raio. 
Dessa forma precisamos definir um $\Delta \phi_{cutoff}$ (lower bound) para começar a integração. 

Temos que dar um chute de $\phi_0$ também (isso é tratado intermanete mas é importante evidenciar).

$\Delta \phi$ é utilizado para o chute de $\phi_0$ de modo que $\phi_0 = \phi_{True}+\Delta \phi$

Esse valor pode ser qualquer um desde que $\phi_0$ fique dentro dos limites de $\phi_{bar}$ e $\phi_{true}$.

O programa intermanete avalia $\phi(r) $ na solução exata de $\phi_0$ em $r_{min}$ (queremos o menor possível)

Se $|\phi(r_{min})-\phi_{true}|> \Delta \phi_{cutoff}$ Começamos da li, se não aumentamos o raio até que $\phi(r)$ essa 
equação seja satisfeita. 

A ordem é: Chutamos um valor inicial do campo, encontramos o valor de um pouco longe de 0, verificamos se o cutoff é satisfeito
se não aumenta o valor do raio, mantendo o mesmo $\phi_0$ até a condição ser possivelmente satisfeita. Caso não seja possível teremos
que tentar outro chute para $\phi_0$ (tratado intermante no código).

A princípio a regra é: $\Delta \phi_{cutoff}$ deve ser o menor possível para que não comecemos
a integraçlão longe de $\phi_{True}$ Isso excluiria uma região que pode ser encontrada a solução!.

Isso deve ser ainda mais levado em consideração em thinwall, pois esperamos que a condição inciial seja próxima de $\phi_{true}$



[Colocar gráficos de erro com cutoff muito grande + códigos]


Se o interior sobe sua solução então quer dizer que deveria ser menor o cutoff,
uma vez que, após encontrada o raio da solução esperamos que o $\phi'_0 =0$ ou seja, podemos sempre
expandir para "trás" utilizando a função aproximada, que não deve subir e sim ser um plato em $\phi_0$ para r menor.
No gráfico acima vemos essse erro na curva laranja.


[Gráfico com tudo certo e os respectivos prints]



### ODE/Integrate & Saveprofile

Na ODE temos $\dot{\phi}=y; \ddot{\phi}=V'(\phi)- \alpha/r \dot{\phi}$

Dado as condições iniciais e a EDO, podemos integrar utiliando o RKCK5 para encontrar uma solução tal que:

$$|\phi(\infty)-\phi_{metamin}|< \epsilon_{\phi}$$

$$|\phi'(\infty)| < \epsilon_{d\phi} $$

O que o integrate vai fazer é integrar da condição incial achada até um dos seguintes erros acontecer:

1. Convergir respeitando as condiçoes acima, devolvendo r,$\phi$ e $\phi'$ que acontece.
2. $\phi'(r)>0$ (Undershooting). Como o campo deve caminha até o falso vácuo esperamos que a derivada seja negativa sempre
se a derivada ficar positiva quer dizer que a fricção ganhou e não houve energia suficiente para chegar ao falso vácuo e o campo começou a voltar.
Devolve $r,\phi,\phi'$ onde $\phi'(r)=0$
3. $\phi-\phi_{meta}<0$ (overshoot). Dentro de algum passo o campo passo o vácuo falso, ou seja, muita energia foi dada e ele não
parou em repouso no falso vácuo, como é esperado. Devolve ponto onde $\phi=\phi_{meta}$


Nos casos de overshooting e undershooting, o código altera o valor inicial do campo até ser encontrado a solução exata.

Após encontrado a condição inciial exata, utilizamos o integrate and save profile para encontrar o perfil do campo, salvando tods osvalores de $\phi$
para um conjunto dado de raios (R arraay). Interpola cubicamente entre os raios dados se o passo do RKCK5 for maior que a diff entre os raios.


### Find profile

Essa é a parte mais importante do código, onde alguns parâmetros que são definidos intermanemente talvez sejam bons de serem alterados.

Primeiro temos que $r_{min} = 10^{-4} r_{scale}$ raio mínimo próximo ao vácuo verdadeiro que é procurado a solução.

`phi_tol`: Default é $10^{-4}$ usado para encontrar as tolerânciais relativas $\epsilon_\phi$ e $\epsilon_{d\phi}$ que 
a solução deve respeitar.

`cuttoff:` $10^{-2} \cdot |\phi_{meta}-\phi_{true}|$.

Como $\phi_0$ não é conhecido a priori chutamos com:

$$\phi_0 = \phi_{true} +e^{-x}(\phi_{meta}-\phi_{true}) $$

x grande $\rightarrow$ próximo ao vácuo verdadeiro. x pequeno $\rightarrow$ próximo ao falso vácuo.
Começa o guess, com $x$ tal que $\phi_0=\phi_{bar}$, visto que $\phi_{0} \geq \phi_{bar}$.

Tenta achar r que permita tal $\phi$ se não achar aumenta x (mais próximo a $\phi_{true}$)

Integra até convergir ou undershooting/oveershooting. Se a fricção parou antes $\phi$ (overshooting)
estamos muit perto de $\phi_{true}$ e portanto diminui x ($x_{max}$ vira x, começa em $\infty$) e o novo x é a média do 
do $x_{max}$ com $x_{min}$.

Para overshooitng (energia demais) o x estava muito próximo do falso vácuo (termo de fricção fraco demais) então coloca o novo bound
$x_{min} = x$ e o x testado é a média do mínimo com o máximo.

Esse processo ocorre até convergir.

Após encontrado a soluçção, cria $R=r_0$ encontrado pelas condições iniciais, até $r_f$ com n_points.
Note que $r_0$ pode ser bem maior que 0 (sobretudo em thin wall) pois o campo pode ficar "parado" até poder decair.

Preenchemos a região "interior" de $r<$r_0$ utilizando a aproximação, pois, a princípio, temos $\dot{\phi}\approx 0$
nessa região de forma que podemos utilizar a aproximação exata, preenchendo a bolha internamente.

Boas práticas: Verificar se $\dot{\phi}(r_{min})\approx 0$



[4 plots pensados para verificarmos a solução.]


### Action and $\beta$

