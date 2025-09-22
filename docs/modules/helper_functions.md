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