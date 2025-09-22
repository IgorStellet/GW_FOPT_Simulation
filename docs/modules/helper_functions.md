# Miscellaneous functions

---
## `set_default_args`
### Signature

```python
set_default_args(func: Callable, inplace: bool = True, **kwargs) -> Callable
```

### Purpose
Update the *default* values of a function’s parameters without changing its external behavior for explicit arguments.
This is handy when a top-level API calls deeply nested functions but you want to tweak defaults of those inner functions without touching every call site.
When placing the desired function you can create a new one (wrapper) or keep the existing one with the new default parameters.
> ⚠️ Unlike `functools.partial`, this does **not** bind/force arguments at call time; it only changes what the function uses **when an argument is omitted** by the caller.

### Parameters, returns and Raises
**Parameters**
- func: the function (or unbound method) whose defaults will be updated.

- inplace:
  - True — mutate func in place by editing __defaults__ (positional/positional-or-keyword) and __kwdefaults__ (keyword-only).

  - False — return a wrapper that applies the new defaults but leaves the original func untouched; the wrapper’s signature is updated to show the new defaults.

- **kwargs: mapping of parameter_name=new_default_value.

**Returns**
- If inplace=True: the same (possibly bound) object you passed (so existing references still work).

- If inplace=False: a new callable wrapper with updated signature.

### Raises

- TypeError if func is not callable.

- ValueError if a passed name does not exist in func’s signature.

- ValueError if a passed name exists but does not have a default (you cannot create a default where none exists).

- ValueError if you try to target *args or **kwargs (variadics have no defaults).


### When to use and Examples
Use when you control a pipeline with deep calls and want different defaults globally for a nested function.
Or when you want to expose a variant of a function with new defaults while keeping the original intact (use inplace=False).

see [tests/helper_functions/Miscellaneouys](/tests/helper_functions/Miscellaneous_functions.py) for more
**Examples**
```python
# 1) Positional + keyword-only defaults (in-place)
def f(a, b=2, c=3, *, d=4):
    return a, b, c, d

print(f(10))           # -> (10, 2, 3, 4)

# Mutate the original defaults
set_default_args(f, b=20, d=40)

print(f(10))           # -> (10, 20, 3, 40)

#2) Non in-place: return a wrapper with update defaults
def g(a, b=2, c=3, *, d=4):
    return a, b, c, d

g2 = set_default_args(g, inplace=False, b=99, d=111)

print(g(1))            # -> (1, 2, 3, 4)   (original unchanged)
print(g2(1))           # -> (1, 99, 3, 111) (wrapper uses new defaults)

#3) Exapected errors (bad names / no default present)
def h(a, b, c=3):
    return a, b, c

try:
    set_default_args(h, x=1)          # No parameter `x` in `h`
except ValueError as e:
    print("Error:", e)

try:
    set_default_args(h, b=10)         # `b` exists but has NO default in the signature
except ValueError as e:
    print("Error:", e)
####################################
```
---

## `monotonic_indices`
### Signature

```python
monotonic_indices(x: array_like) -> np.ndarray
```

### Purpose
Return the indices of a **strictly increasing** subsequence of x, always including the first and last elements.
This is useful to "repair" a nearly-monotonic grid(e.g., a few spurious downward spikes)
without sorting or regridding—a common need before calling routines that require 
strictly increasing coordinates (e.g., numerical differentiation or interpolation). 
This function **removes** non-increasing points between the start and end of the array

### Parameters, returns and Raises
**Parameters**
- `x`: (array_like): Input 1D sequence (Numpy array).

**Returns**
- `np.ndaary` of shape (m,): **Indices** I such that `x[I]` is strictly increasing and `I[0]==0`, `I[-1]==len(x)-1` if x is increasing and the opposite if x is decreasing

### Raises / Assumptions

- If the overall trend is decreasing (`x[0] > x[-1]`), the function works by reversing internally and then mapping the indices back
- assumes `len(x)>=1` and that x is a ndarray.

### When to use and Examples
- Before calling finite difference derivatives which require strictly monotonic coordinate arrays.
- Before interpolation on grids that should be increasing but contain small gliches
- To quickly visualize or compute on a clean, mnotone subset of a noise 1D grid withoud sorting or re-sampling

see the full test script in [tests/helper_functions/Miscellaneouys](/tests/helper_functions/Miscellaneous_functions.py) for more

**Examples**
```python
# 1) Clean a mostly-increasing sequence with one bad spike
x = [1, 2, 3, -1, 20, 19, 50]  # overall increasing, but has local decreases
idx = monotonic_indices(x)
x_clean = [x[i] for i in idx]
print(idx)      # e.g., [0, 1, 2, 4, 6]
print(x_clean)  #     -> [1, 2, 3, 20, 50]  (strictly increasing, kept endpoints)

# 3) Pre-conditioning before derivatives (deriv14, deriv23, deriv1n)
# Suppose x is supposed to be increasing, but isn’t strictly so due to noise.
x = np.array([0.0, 0.1, 0.21, 0.20, 0.4, 0.5])  # small non-monotonic blip at 0.21 -> 0.20
y = np.sin(x)

idx = monotonic_indices(x)
x_mono = x[idx]
y_mono = y[idx]
# Now x_mono is strictly increasing and safe for derivative/interpolation routines.
####################################
```
---
## `clamp_val`
### Signature

```python
clamp_val(x: np.array, a: int, b: int) -> np.ndarray
```

### Purpose
Force (or "clip") all values of `x` to lie inside the **closed interval** `[min(a,b), max[a,b]]`
This is useful to eliminate **non-physical** or **unstable** values (e.g., negative densities, probabilites outside `[0,1]`, or arguments to `log/sqrt` that must stay positive)

### Parameters, returns and Raises
**Parameters**
- `x` (array_like): Values to clamp. Any shape; will be returned as a NumPy array
- `a,b`(int or array): Lower and upper bounds. Can be a scalar or a array. Bounds can be given in any order

**Returns**
- `np.ndaary`: Array with the same shape as `x`, where every entry is clipped to `[a,b]` interval. 

### Raises

- No custom exceptions.

### When to use and Examples
- When you want to enforce **physical constraints**, like negative densities, probabilites outside `[0,1]`, or arguments to `log/sqrt` that must stay positive
- To avoid numerical pathologies in interative algorithms
- Sanitizing imputs for pltting or interpolation routines

see the full test script in [tests/helper_functions/Miscellaneouys](/tests/helper_functions/Miscellaneous_functions.py) for more

**Examples**
```python
# 1) Simple clamping with scalar bounds
x = [1, 2, 3, -1, 20, 19, 50]
y = clamp_val(x, a=1, b=20)
print(y)  # -> [ 1  2  3  1 20 19 20]

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
```

# Numerical Integration functions

---
## `_rkck`
### Signature

```python
_rkck(y: np.ndarray, dydt: np.ndarray, t: float,f: Callable, dt: float, args: tuple = ()) -> Tuple[np.ndarray, np.ndarray]
```

### Purpose
Perform **one** embedded Runge-Kutta **Cash-Karp** step (the classic 5th-order method with a 4th-order ebedded estimate)

### Parameters, returns and Raises
**Parameters**
- y  (array_like): Current state at time `t`.
- dydt (array_like): Derivative `dy/dt` at (y, t), usually `f(y, t)`.
- t (float): Current integration variable (independent variable).
- f (callable): Derivative function with signature `f(y, t, *args)`.
- dt (float): Step size.
- args (tuple, optional) Extra positional arguments for f (usualy defaults)


**Returns**
- `dyout` (ndarray): The 5th-order **increment** Δy to update the state: y_next ≈ y + dyout.
- `yerr` (ndarray): An estimate of the **local truncation error** for the increment (the difference between 5th and 4th order formulas)

### Raises

- No custom exceptions.

### When to use 
This function is used by the function below (`rkqs`) to solve the desired ODE and therefore is not called directly. 
(That's why there are no examples of it)

--- 
## `rkqs`
### Signature

```python
rkqs(y: np.ndarray, dydt: np.ndarray, t: float, f: callable, dt_try: float, epsfrac: float, epsabs: float, args: tuple = ()) -> _rkqs_rval
```

### Purpose
Perform **one adpative step** of the (RKCK) method **with error control**
It's the acceptance/rejection + step control wrapper around the low-leve `_rkck` step.
The function has larger steps when the error is very small and decreases the step when the error becomes large. 
This ensures accuracy and speed in solving the integral.

### Parameters, returns and Raises
**Parameters**
- Same as _rkck (y, dydt, t, f, dt and args) plus:
- dt_try  (float): Initial step-size guess 
- epsfrac (float): **Relative** error tolerance (dimensionless)
- epsabs (float): **Absolute** error tolerance (same units as `y`)


**Returns**
- named tuple `_rkqs_rval[Delta)y, Delta_t, dtxt]`
- `Delta_y ` (ndarray): 5th-order **increment** to advance the state: y_next ≈ y + Delta_y. 
- `Delta_t` (float): The actual step size used on the accepted step (may be smaller than dt_try).
- `dtxt` (float): A **suggested** step size for the next call (based on the observed error)

### Raises

- `IntegrationError`: If the step size underflows numerically (i.e., `t+dt==t`) after repeated shrinking)

### How to use and Examples

- The error is handled as follows:
 - With the relative error `epsfrac` evaluate `denom = y * epsfrac`
 - With the absolute error `epsabs` evaluate `err_ratio = |yerr| / max(denom, epsabs)`
 - Then take errmax = max(err_ratio) and `if errmax <1` accept pace. Otherwise compute a reduced dt until accepted
 - After acceptance, propose dtnxt (x5 if the step was extremely accurate and reduced if the error was not so accurate)

- Start with `epsfrac ~ 1e-6` (relative) and `epsfrac ~ 1e-9` (absolute) for typical float64 runs.
- There is no predefined formula for errors, just trial and error with the desired ODE


see the full test script in [tests/helper_functions/Numerical_integration](/tests/helper_functions/Numerical_integration.py) for more
There are one more image and two more prints examples there.
**Examples**
```python
# Minimal usage pattern (pseudo-code)
y = y0.copy()
t = t0
dt = dt_try
while t < t_end:
    dydt = f(y, t, *args)
    Delta_y, Delta_t, dtnxt = rkqs(y, dydt, t, f, dt, epsfrac, epsabs, args)
    y += Delta_y
    t += Delta_t
    dt = dtnxt

```
- **Harmonic oscillator test images**  
![Harmonic oscillator state](/docs/assets/helper_functions/integration_1.png)


![Adaptive step sizes](/docs/assets/helper_functions/integration_2.png)

---


## `NOME DA FUNÇÃO`
### Signature

```python
FUNÇÃO 
```

### Purpose
ESCREVER PARA QUE SERVE E SUAS UTILIZADDES

### Parameters, returns and Raises
**Parameters**
- `x`: PARÂMETROS DA FUNÇÃO

**Returns**
- `np.ndaary`=0` O QUE ELA RETORNA 

### Raises

- O QUE ELA DA DE RAISE OU ASSUME

### When to use and Examples
- QUANDO USAR OU QUANDO É USADO

see the full test script in [tests/helper_functions/Miscellaneouys](/tests/helper_functions/Miscellaneous_functions.py) for more

**Examples**
```python
ALGUNS DOS EXEMPLOS FEITOS
```
--- 