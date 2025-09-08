from helper_functions import set_default_args

# exemplo 1: posição + keyword-only
def f(a, b=2, c=3, *, d=4):
    return a, b, c, d

# in-place
set_default_args(f, b=20, d=40)
assert f(1) == (1, 20, 3, 40)

# non-inplace
def g(a, b=2, c=3, *, d=4):
    return a, b, c, d

g2 = set_default_args(g, inplace=False, b=99, d=111)
assert g(1) == (1, 2, 3, 4)   # original permanece igual
assert g2(1) == (1, 99, 3, 111)

# exemplo 2: erro quando parametro não existe ou sem default
def h(a, b, c=3):
    return a, b, c

try:
    set_default_args(h, x=1)
except ValueError as e:
    print("ok: ", e)

try:
    set_default_args(h, b=10)  # b não tem default
except ValueError as e:
    print("ok: ", e)
