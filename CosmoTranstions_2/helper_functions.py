# New version of helper_functions

import numpy as np
import inspect
import functools
from typing import Callable, Any, Dict


# -------------------------------------------------------------
# Miscellaneous functions - Functions to help others in general



def set_default_args(func: Callable, inplace: bool = True, **kwargs) -> Callable:
    """
     Update the default values of parameters of `func`.

     Parameters
     ----------
     func : Callable
         The function (or unbound method) whose defaults will be updated.
     inplace : bool, optional (default=True)
         If True, modifies the given function (changes __defaults__ / __kwdefaults__).
         If False, returns a *wrapper* that applies the defaults without touching `func`.
     **kwargs
         Mapping parameter_name=new_default_value

     Behavior / Notes
     ----------------
     - Supports both positional parameters with defaults and keyword-only parameters.
     - Raises ValueError if:
         * a name passed in kwargs does not exist in the function’s signature.
         * the parameter exists but does NOT have a default value (cannot replace what doesn’t exist).
     - For *args and **kwargs parameters, defaults cannot be set.
     - For bound methods, pass the underlying function (Class.method) or use __func__.

     Returns
     -------
     Callable
         The modified function itself (if inplace=True) or a wrapper (if inplace=False).

     Example
     -------
     >>> def f(a, b=2, c=3, *, d=4):
     ...     return a, b, c, d
     >>> set_default_args(f, b=20, d=40)
     >>> f(1)
     (1, 20, 3, 40)

     >>> g = set_default_args(f, inplace=False, b=99)
     >>> g(1)
     (1, 99, 3, 40)
     """
    if not callable(func):
        raise TypeError("`func` must be callable")

    # If the user passed a bound method (e.g. instance.method), get the actual function
    original_obj = func
    is_bound_method = hasattr(func, "__func__") and inspect.ismethod(func)
    if is_bound_method:
        func = func.__func__

    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    param_map: Dict[str, inspect.Parameter] = {p.name: p for p in params}

    # Validate names and if they have default
    for name in kwargs:
        if name not in param_map:
            raise ValueError(f"Function '{func.__name__}' doesn't have '{name}' parameter")
        if param_map[name].default is inspect._empty:
            raise ValueError(f"Parameter '{name}' doesn't have any default to be changed")

    # Separate updates by parameter type
    pos_updates: Dict[str, Any] = {}
    kwonly_updates: Dict[str, Any] = {}
    for name, val in kwargs.items():
        kind = param_map[name].kind
        if kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            pos_updates[name] = val
        elif kind == inspect.Parameter.KEYWORD_ONLY:
            kwonly_updates[name] = val
        else:
            # VAR_POSITIONAL or VAR_KEYWORD
            raise ValueError(f"It's not possible to give a default for the parameter of the type {kind} ('{name}')")

    # --- In-place update: handle __defaults__ and __kwdefaults__ ---
    def _apply_inplace(f: Callable) -> Callable:
        # keyword-only defaults (dict)
        current_kwdefaults = getattr(f, "__kwdefaults__", None) or {}
        current_kwdefaults = dict(current_kwdefaults)  # cópia mutável
        current_kwdefaults.update(kwonly_updates)
        if current_kwdefaults:
            f.__kwdefaults__ = current_kwdefaults
        else:
            # if it is empty, remove it to avoid residue
            if hasattr(f, "__kwdefaults__"):
                f.__kwdefaults__ = None

        # positional defaults: build list in order of positional parameters that have default
        pos_params = [p for p in params
                      if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        pos_defaults_params = [p for p in pos_params if p.default is not inspect._empty]

        if pos_defaults_params:
            new_defaults = []
            for p in pos_defaults_params:
                if p.name in pos_updates:
                    new_defaults.append(pos_updates[p.name])
                else:
                    new_defaults.append(p.default)
            f.__defaults__ = tuple(new_defaults)
        else:
            # there are no positional defaults -> ensure attribute None
            f.__defaults__ = None

        return f

    # --- Non-inplace: create wrapper that applies defaults without touching the original func ---
    def _make_wrapper(f: Callable) -> Callable:
        # prepare new default values for signing
        new_params = []
        for p in params:
            if p.name in kwargs:
                # replace only the default (do not change the kind)
                new_params.append(p.replace(default=kwargs[p.name]))
            else:
                new_params.append(p)
        new_sig = sig.replace(parameters=new_params)

        @functools.wraps(f)
        def wrapper(*a, **kw):
            # Ensure defaults are applied if not passed by the caller
            for name, val in kwargs.items():
                if name not in kw:
                    kw[name] = val
            return f(*a, **kw)

        # adjust wrapper signature to reflect new defaults
        wrapper.__signature__ = new_sig

        # also set __kwdefaults__ for compatibility (optional)
        kwdefaults_for_wrapper = {k: kwargs[k] for k in kwonly_updates} if kwonly_updates else {}
        if kwdefaults_for_wrapper:
            wrapper.__kwdefaults__ = kwdefaults_for_wrapper

        return wrapper

    if inplace:
        # apply directly to the function (or underlying function, if it is a method)
        modified = _apply_inplace(func)
        # if it is a bound method, return the bound method original_obj (which already references the modified function)
        return original_obj if is_bound_method else modified
    else:
        return _make_wrapper(func)



def monotonic_indices(x):
    """
    Return the indices of a strictly increasing subsequence of `x`
    starting at the first element and ending at the last element.

    Parameters
    ----------
    x : array_like
        Input sequence.

    Returns
    -------
    ndarray
        Indices of elements forming a monotonic increasing subsequence.

    Notes
    -----
    - If the sequence decreases from start to end, it is temporarily reversed
      and indices are mapped back before returning.
    - Guarantees inclusion of the first and last index.
    """
    x = np.asarray(x)

    # If sequence decreases overall, reverse it to simplify logic
    if x[0] > x[-1]:
        x = x[::-1]
        reversed_order = True
    else:
        reversed_order = False

    indices = [0]
    for i in range(1, len(x) - 1):
        if x[i] > x[indices[-1]] and x[i] < x[-1]:
            indices.append(i)
    indices.append(len(x) - 1)

    indices = np.array(indices)
    if reversed_order:
        indices = (len(x) - 1) - indices

    return indices


def clamp_val(x, a, b):
    """
    Clamp values of `x` to be within the closed interval [a, b].

    Parameters
    ----------
    x : array_like
        Input values.
    a, b : array_like
        Interval bounds. Can be scalars or arrays, must be broadcastable with `x`.

    Returns
    -------
    ndarray
        Values of `x` clipped to lie between min(a, b) and max(a, b).

    Examples
    --------
    >>> clamp_val([1, 5, 10], 3, 8)
    array([3, 5, 8])
    >>> clamp_val([1, 5, 10], 8, 3)  # reversed bounds
    array([3, 5, 8])
    """
    x = np.asarray(x)
    lower = np.minimum(a, b)
    upper = np.maximum(a, b)
    return np.clip(x, lower, upper)


# -------------------------------------------------------------
# Numerical integration - Functions to evaluate the integrals