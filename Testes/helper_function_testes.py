from CosmoTranstions_2 import set_default_args, monotonic_indices, clamp_val, rkqs, _rkck
import numpy as np
import matplotlib.pyplot as plt

#---------------- Primeira sequência de testes e modificações ------------------------

"""
Objetivo da função set_default_args é modificar os valores padrão de parâmetros de uma função
seja alterando a função original ou criando um wrapper
"""

# Exemplo de utilizações:

# exemplo 1: posição + keyword-only ( A direita do * apenas funções que podem ser passadas por nome, e.g., d=20 e não f(20) )
def f(a, b=2, c=3, *, d=4):
    return a, b, c, d

print(f(10))

# in-place
set_default_args(f, b=20, d=40) # Muda os parâmetros da função anterior para b=20 e d=40 | Utilizando o inplace muda a função original para a nova desejada

print(f(10))


# non-inplace
def g(a, b=2, c=3, *, d=4):
    return a, b, c, d

g2 = set_default_args(g, inplace=False, b=99, d=111) # Cria uma nova função com parâmetros base diferentes da original

print(g(1))
print(g2(1))

# exemplo 2: Erros quando parametros não existem ou sem default
def h(a, b, c=3):
    return a, b, c

try:
    set_default_args(h, x=1) # Sem variável correspondente
except ValueError as e:
    print("Erro: ", e)

try:
    set_default_args(h, b=10)  # b não tem default
except ValueError as e:
    print("Erro: ", e) # Parâmetro sem valor inicial correspondente


"""
Objetivo da função motonic_indices é fornecer os índices relativos aos parâmetros crescentes de uma lista. Ex de utilidade: Se a função tiver um ou outro ponto que
foge ao monoticamente crescente esse ponto é retirado, podendo ajudar no caso de pequenos desvios indesejados
"""

x =  [1,2,3,-1, 20,19,50] # Exemplo com um valor quebrado no meio, crescente
y = []
for i in monotonic_indices(x):
    y.append(x[i])

print(y)

k =  [50,19,20,-1, 3,2,1] # Exemplo com um valor quebrado no meio, decrescente
print(monotonic_indices(k)) # índices em ordem descrescente

"""
Objetivo da função clamp_val é transformar os valores de uma lista que não se encontram entre um dado intervalo [a,b], para se encontrarem no dado intervalo. 
Caso o valor seja maior transforma no máximo (b) e caso seja menor transforma no mínimo (a)
Isso pode ser útil para retirar resultados não físicos de simulações/cálculos.
"""

x =  [1,2,3,-1, 20,19,50]

y = clamp_val(x, a=1,b=20)

print(y)


#---------------- Segunda sequência de testes e modificações ------------------------

"""
Objetivo da função rkqs é utilizar o método de runge-kutta de 5 ordem com erro adaptativo para calcular as soluções de EDO
quer exigem integração das funções, abaixo seguem dois exemplos simples
"""

# Testes de Numerical integrals

def f(y, t):
    return y  # dy/dt = y → solução exata y = exp(t)

y0 = np.array([1.0])
t0 = 0.0
dydt0 = f(y0, t0)

result = rkqs(y0, dydt0, t0, f, dt_try=0.1, epsfrac=1e-6, epsabs=1e-9)
print(result)

# Deve dar Delta_y ~ 0.105 (pois exp(0.1)-1 ≈ 0.105).
#-------------------------------------------------

# Teste com oscilador harmônico

def harmonic_oscillator(y, t, omega):
    """
    Derivative for the simple harmonic oscillator.
    y[0] = position x
    y[1] = velocity v
    """
    x, v = y
    dxdt = v
    dvdt = -omega**2 * x
    return np.array([dxdt, dvdt])

def integrate_oscillator(y0, t0, t_end, dt_try, omega, epsfrac=1e-6, epsabs=1e-9):
    """
    Integrates the harmonic oscillator using rkqs.
    """
    y = np.array(y0, dtype=float)
    t = t0
    dt = dt_try

    positions = [y[0]]
    velocities = [y[1]]
    times = [t]

    while t < t_end:
        dydt = harmonic_oscillator(y, t, omega)
        dy, dt_used, dt_next = rkqs(
            y, dydt, t, harmonic_oscillator,
            dt, epsfrac, epsabs, args=(omega,)
        )
        # Update solution
        y = y + dy
        t = t + dt_used

        positions.append(y[0])
        velocities.append(y[1])
        times.append(t)

        dt = dt_next  # use adaptive step

    return np.array(times), np.array(positions), np.array(velocities)

# Parameters
omega = 1.0       # natural frequency
y0 = [1.0, 0.0]   # initial position=1, velocity=0
t0, t_end = 0.0, 20.0
dt_try = 0.1

# Run integration
times, positions, velocities = integrate_oscillator(y0, t0, t_end, dt_try, omega)

# Plot results
plt.figure(figsize=(10,5))
plt.plot(times, positions, label="Position x(t)")
plt.plot(times, velocities, label="Velocity v(t)")
plt.xlabel("Time t")
plt.ylabel("State")
plt.legend()
plt.grid(True)
plt.title("Harmonic Oscillator (Runge-Kutta with adaptive step size)")
plt.show()

#---------------- Terceira sequência de testes e modificações ------------------------
