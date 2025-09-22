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
print("---- TESTS set_default_args ----")
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
print("---- TESTS monotonic_indices ----")
import numpy as np

# 1) Clean a mostly-increasing sequence with one bad spike
x = [1, 2, 3, -1, 20, 19, 50]  # overall increasing, but has local decreases
idx = monotonic_indices(x)
x_clean = [x[i] for i in idx]
print(idx)      # e.g., [0, 1, 2, 4, 6]
print(x_clean)  #     -> [1, 2, 3, 20, 50]  (strictly increasing, kept endpoints)

# 2) Sequence that overall decreases: indices are mapped back correctly
k = [50, 19, 20, -1, 3, 2, 1]
idx_k = monotonic_indices(k)
k_sub = [k[i] for i in idx_k]
print(idx_k)    # indices valid in the ORIGINAL (decreasing) orientation
print(k_sub)    # subsequence from start to end that is strictly increasing after internal handling

# 3) Pre-conditioning before derivatives (deriv14, deriv23, deriv1n)
# Suppose x is supposed to be increasing, but isn’t strictly so due to noise.
x = np.array([0.0, 0.1, 0.21, 0.20, 0.4, 0.5])  # small non-monotonic blip at 0.21 -> 0.20
y = np.sin(x)

idx = monotonic_indices(x)
x_mono = x[idx]
y_mono = y[idx]

# Now x_mono is strictly increasing and safe for derivative/interpolation routines.

"""
The purpose of clamp_val is to transform the values of a list that are not within a given interval [a, b]
so that they fall within that interval. If a value is larger, it becomes the maximum (b); if smaller, it becomes the minimum (a).
This can be useful to remove non-physical results from simulations/calculations.
"""
print("---- TESTS clamp_val ----")
# 1) Simple clamping with scalar bounds
x = [1, 2, 3, -1, 20, 19, 50]
y = clamp_val(x, a=1, b=20)
print(y)  # -> [ 1  2  3  1 20 19 20]

# 2) Bounds given in reverse order (still works)
print(clamp_val([1, 5, 10], 8, 3))  # -> [3 5 8]

# 3) Array bounds with broadcasting (per-column limits)
X = np.array([[ -5.0, 0.2,  9.0],
              [  0.5, 2.5, 11.0]])
low  = np.array([0.0, 0.0,  1.0])   # shape (3,)
high = np.array([1.0, 2.0, 10.0])   # shape (3,)
print(clamp_val(X, low, high))
# -> [[0.  0.2 9. ]
#     [0.5 2.  10.]]

# 4) Prevent non-physical negatives before a sqrt
data = np.array([1e-6, -1e-8, 4.0])
safe = clamp_val(data, 0.0, np.inf)
print(np.sqrt(safe))  # well-defined
