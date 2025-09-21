from src.CosmoTransitions import (set_default_args, monotonic_indices, clamp_val)
#---------------- First round of tests and modifications ------------------------
#########################
# Miscellaneous functions
#########################

print("---------- TESTS MISCELLANEOUS FUNCTIONS ---------")
"""
The purpose of set_default_args is to modify the default parameter values of a function,
either by changing the original function or by creating a wrapper.
"""

# Usage examples:

# example 1: positional + keyword-only (to the right of * only keyword-only params can be passed by name, e.g., d=20 and not f(20))
def f(a, b=2, c=3, *, d=4):
    return a, b, c, d

print(f(10))

# in-place
set_default_args(f, b=20, d=40)  # Changes the previous function's defaults to b=20 and d=40 | Using inplace mutates the original function to the desired new defaults

print(f(10))


# non-inplace
def g(a, b=2, c=3, *, d=4):
    return a, b, c, d

g2 = set_default_args(g, inplace=False, b=99, d=111)  # Creates a new function with different base defaults from the original

print(g(1))
print(g2(1))

# example 2: Errors when parameters do not exist or lack a default
def h(a, b, c=3):
    return a, b, c

try:
    set_default_args(h, x=1)  # No corresponding parameter
except ValueError as e:
    print("Error: ", e)

try:
    set_default_args(h, b=10)  # 'b' has no default
except ValueError as e:
    print("Error: ", e)  # Parameter without an initial default value


"""
The purpose of monotonic_indices is to provide the indices of elements that form a strictly increasing subsequence.
Use case: if the sequence has one or another point that breaks strict monotonic increase, that point is removed, which can help with small unwanted deviations.
"""

x = [1, 2, 3, -1, 20, 19, 50]  # Example with a broken value in the middle, overall increasing
y = []
for i in monotonic_indices(x):
    y.append(x[i])

print(y)

k = [50, 19, 20, -1, 3, 2, 1]  # Example with a broken value in the middle, overall decreasing
print(monotonic_indices(k))  # indices mapped back for the decreasing case

"""
The purpose of clamp_val is to transform the values of a list that are not within a given interval [a, b]
so that they fall within that interval. If a value is larger, it becomes the maximum (b); if smaller, it becomes the minimum (a).
This can be useful to remove non-physical results from simulations/calculations.
"""

x = [1, 2, 3, -1, 20, 19, 50]

y = clamp_val(x, a=1, b=20)

print(y)
