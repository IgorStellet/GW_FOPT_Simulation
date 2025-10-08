# Single Field Instanton

This document describes the **single-field instanton** solver and the public API that underpins it. The implementation follows 
the overshoot/undershoot shooting method and supports both **thin-wall** and **thick-wall** regimes with robust numerical defaults.

---

## Module overview

`tunneling1D` provides tools to compute O(α+1)–symmetric bounce (instanton) solutions in one field dimension. In the absence of gravity, the equation of motion for a radial profile `phi(r)` reads

$$\frac{d^2\phi}{dr^2} + \frac{\alpha}{r}\frac{d\phi}{dr} = \frac{dV}{d\phi}$$

with boundary conditions

$$\phi '(0)=0 \qquad \phi(\infty)=\phi_{\rm metaMin}$$

The **primary entry point** is the class `SingleFieldInstanton`, which:

* validates the potential and basic inputs,
* supplies numerically stable derivative approximations when the user does not provide them,
* detects a characteristic scale and the barrier location,
* and exposes high-level routines to find the **profile** and the **action**.

---

## Behavioral guarantees & compatibility

* **Scalar/array semantics.** All derivative helpers accept scalars or NumPy arrays and **preserve shapes**.
* **Backward compatibility.** Names and signatures are preserved. If you supply custom `dV`/`d2V`, the class uses them everywhere.
* **Numerical robustness.** Defaults favor stability in thin-wall and stiff potentials; the step size never collapses to zero even if the two minima coincide numerically.
* **Clear errors.** Missing metastability or barriers raise `PotentialError` with informative messages.

---

## Quick usage example

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
    phi_eps=1e-3,     # relative FD step (safe floor applied automatically)
    fd_order=4,       # 4th-order differences
    validate=True,
)

# prof = inst.findProfile()
# S    = inst.findAction(prof)
```

---

## Reproducibility notes

* The FD step is `h = max(phi_eps * |Δphi|, fd_eps_min, auto_floor)`, where `auto_floor ≈ sqrt(machine_eps) × scale`.
* Built-in derivative formulas are **central differences** and do not cache potential evaluations across calls; users may optimize their `V` if desired.

---

## Class: `SingleFieldInstanton`

Compute properties of a single-field instanton via the overshoot/undershoot method. Most users will only call `findProfile` and `findAction`, 
but the constructor and derivative helpers are documented here because they define the **potential interface** and core **validations**.

### Signature

```python
class SingleFieldInstanton:
    def __init__(
        self,
        phi_absMin: float,
        phi_metaMin: float,
        V: Callable[[float | np.ndarray], float | np.ndarray],
        dV: Callable | None = None,
        d2V: Callable | None = None,
        phi_eps: float = 1e-3,
        alpha: int | float = 2,
        phi_bar: float | None = None,
        rscale: float | None = None,
        *,
        fd_order: int = 4,
        fd_eps_min: float | None = None,
        validate: bool = True,
    )
```

### Purpose

Create a solver instance for a **single scalar field** in a given potential `V(phi)`, with optional user-supplied derivatives. 
The instance owns all numerical settings (finite-difference step/accuracy, friction `α`, barrier point, characteristic radius scale) required by later calls.

### Parameters

* `phi_absMin` *(float)*
  Field value of the **stable** (true) vacuum.

* `phi_metaMin` *(float)*
  Field value of the **metastable** (false) vacuum.

* `V` *(callable)*
  Potential `V(phi)`. It may accept scalars or NumPy arrays (vectorized); scalar support is sufficient.

* `dV`, `d2V` *(callable | None)*
  Optional first and second derivatives. If provided, they **override** the built-in finite-difference routines. Either or both may be supplied.

* `phi_eps` *(float, default `1e-3`)*
  **Relative** finite-difference step used by the built-in `dV`/`d2V`. The absolute step is
  
$$h = \texttt{phi_eps}\times |\phi_{\rm metaMin}-\phi_{\rm absMin}|$$

  A safe lower floor is applied automatically (see *Notes*).

* `alpha` *(int | float, default `2`)*
  Friction coefficient in the ODE. For O(α+1) symmetry, `alpha` equals the spatial dimension.

* `phi_bar` *(float | None)*
  Field value at the **edge of the barrier**, defined by `V(phi_bar) = V(phi_metaMin)`. If `None`, it is found by `findBarrierLocation()`.

* `rscale` *(float | None)*
  Characteristic radial scale. If `None`, it is found by `findRScale()`.

* `fd_order` *{2, 4}, keyword-only, default `4`*
  Order of the built-in central finite differences (ignored if the user passes `dV`/`d2V`).

* `fd_eps_min` *(float | None), keyword-only*
  Absolute lower bound on the FD step `h`. If `None`, a safe value of order `sqrt(machine_eps) × scale` is used.

* `validate` *(bool, default `True`), keyword-only*
  Perform basic sanity checks and emit helpful errors/warnings.

### Raises

* `PotentialError("V(phi_metaMin) <= V(phi_absMin); tunneling cannot occur.", "stable, not metastable")`
  If the potential is not metastable.

* `PotentialError(...)`
  If barrier finding or scale detection fails (e.g., no barrier, negative curvature fit).

### Notes

* **Thin-wall acceleration.** When minima are nearly degenerate, the solver starts integration near the wall using a local quadratic solution, making the search efficient even for extremely thin walls.

* **FD step safety.** The absolute step `h` is computed from the **minima separation**; if the separation is tiny (or zero due to user inputs), a machine-precision-aware **floor** is applied to prevent catastrophic cancellation.

* **Overrides.** If you pass custom `dV`/`d2V`, the solver will use those everywhere (including the high-accuracy helper `dV_from_absMin`).

* **Caching.** The constructor caches `V(phi_absMin)`, `V(phi_metaMin)` and basic deltas; downstream methods reuse them.

---

## Lot SF-1 — Potential interface & validations

This lot modernizes the **potential interface** and core **validations**, and provides the built-in derivatives. 
The public names are unchanged for backward compatibility.

### `__init__`

See the class section above for the full signature and behavior. Implementation highlights:

* **Metastability check:** require `V(phi_metaMin) > V(phi_absMin)`.
* **Robust FD step:** `h = max(phi_eps * |Δphi|, fd_eps_min or auto_floor)`, where `Δphi = phi_metaMin - phi_absMin` and `auto_floor ~ sqrt(machine_eps) × scale`.
* **User overrides respected:** any provided `dV`/`d2V` replace the built-ins transparently.
* **Barrier/scale:** compute `phi_bar` and `rscale` if not supplied; accept user input but warn if `V(phi_bar) ≉ V(phi_metaMin)`.

---

### `dV(phi)`

#### Signature

```python
dV(phi: float | np.ndarray) -> float | np.ndarray
```

#### Purpose

Built-in finite-difference approximation to ( V'(\phi) ). Works with scalars or arrays (broadcasted). 
If the user supplied a custom `dV` at construction, that override is used instead.

#### Scheme

* **Order 4 (default)**
  $$V'(\phi)\approx\frac{V(\phi-2h)-8V(\phi-h)+8V(\phi+h)-V(\phi+2h)}{12h}$$

* **Order 2 (fallback)**
  $$V'(\phi)\approx\frac{V(\phi+h)-V(\phi-h)}{2h}$$

with h =  absolute FD step defined in `__init__`.

#### Parameters, Returns and Raises

* **Parameters:** `phi` *(float | array-like)* points where to evaluate.
* **Returns:** same shape as `phi`.
* **Raises:** only propagates exceptions from user `V` (non-finite, etc.).

#### Notes

* The routine is **side-effect free** and **vectorization-friendly**.
* The step `h` is absolute; see `__init__` for how it is chosen safely.

---

### `dV_from_absMin(delta_phi)`

#### Signature

```python
dV_from_absMin(delta_phi: float) -> float
```

#### Purpose

High-accuracy derivative at $(\phi=\phi_{\rm absMin}+\delta\phi)$. Near the minimum, direct finite differences can lose precision;
we therefore **blend** a Taylor estimate using (V'') with the FD estimate.

#### Method

* Let $(\phi=\phi_{\rm absMin}+\delta\phi)$
* Compute `dV_fd = dV(phi)` (built-in or user override).
* Compute `dV_lin = d2V(phi) * delta_phi`.
* Blend with a smooth weight
  
$$w = \exp\Big[-\big(\delta\phi/h\big)^2\Big],\qquad h=\text{FD step}$$
  Return $(w\cdot dV_{\rm lin} + (1-w)dV_{\rm fd})$.

#### Parameters, Returns and Raises

* **Parameters:** `delta_phi` *(float)* offset from the absolute minimum.
* **Returns:** *(float)* blended derivative at `phi_absMin + delta_phi`.
* **Raises:** Propagates only if user-supplied `dV`/`d2V` fail.

#### Notes

* Guarantees **`dV_from_absMin(0.0) = 0.0`** to within round-off (through the blend).
* Uses the same `h` configured in `__init__`.

---

### `d2V(phi)`

#### Signature

```python
d2V(phi: float | np.ndarray) -> float | np.ndarray
```

#### Purpose

Built-in finite-difference approximation to ( V''(\phi) ). Works with scalars or arrays (broadcasted). 
If the user supplied a custom `d2V`, that takes precedence.

#### Scheme

* **Order 4 (default)**
  $$V''(\phi)\approx\frac{-V(\phi-2h)+16V(\phi-h)-30V(\phi)+16V(\phi+h)-V(\phi+2h)}{12h^2}$$

* **Order 2 (fallback)**
  $$V''(\phi)\approx\frac{V(\phi+h)-2V(\phi)+V(\phi-h)}{h^2}$$

#### Parameters, Returns and Raises

* **Parameters:** `phi` *(float | array-like)*.
* **Returns:** same shape as `phi`.
* **Raises:** only propagates exceptions from `V`.

#### Notes

* Uses the same safe absolute step `h` as `dV`.
* Vectorized; preserves input shape (scalar-in → scalar-out).

---


