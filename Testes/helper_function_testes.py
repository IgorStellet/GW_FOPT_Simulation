from CosmoTranstions_2 import set_default_args, monotonic_indices, clamp_val

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


#---------------- Primeira sequência de testes e modificações ------------------------