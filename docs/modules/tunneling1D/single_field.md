# Single Field Instanton

This document describes the **single-field instanton** solver and the public API that underpins it. The implementation follows 
the overshoot/undershoot shooting method and supports both **thin-wall** and **thick-wall** regimes with robust numerical defaults.
see the [example](examples_single_field.md) page of this class if you want to understand more of each function!.
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
This happens because near the true minimum V'$(\phi)$ it's expected to be exactly zero, therefore it's relevant to evaluate with an
alternative Method, i.e., `V'(\phi) \approx V''(\phi_{\rm absMin}) (\phi-\phi_{\rm absMin})`

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

## Lot SF-2 — Barrier & scales

This lot modernizes how we **locate the barrier** that separates the metastable (false) vacuum from the absolute (true) vacuum and 
how we estimate a **characteristic radial scale** for the instanton solution. 
Both functions keep their legacy names and public behavior, but the numerics and diagnostics are stronger and more transparent.

---

### `findBarrierLocation`

#### Signature

```python
findBarrierLocation(self) -> float
```

#### Purpose

Return the **edge of the barrier** (`phi_bar`) defined implicitly by

$$V(\phi_{\mathrm{bar}}) = V(\phi_{\mathrm{metaMin}})$$

on the **downhill** side of the barrier when moving from the metastable minimum toward the absolute minimum.
This is the crossing point at which the field “leaves” the false-vacuum plateau as it rolls toward the true vacuum.

#### Behavior & algorithm (what’s new)

* We first **locate the barrier top** (`phi_top`) — the maximizer of (V) — in the open interval between the two minima via a bounded 1D search (robust even if the potential has gentle wiggles).
* We then solve for the **downhill crossing** of $(G(\phi)=V(\phi)-V(\phi_{\rm metaMin}))$ between `phi_top` and the absolute minimum using a **bracketed Brent root**.
* On success, we return `phi_bar`. We also cache diagnostics in `self._barrier_info`:

  * `phi_bar`, `phi_top`,
  * `V_top_minus_Vmeta = V(phi_top) - V(phi_metaMin)`,
  * `V_meta`, `V_abs`,
  * `interval = (min(phi_metaMin, phi_absMin), max(...))`.

This approach eliminates fragile reliance on strict monotonicity and makes error messages precise.

#### Parameters, returns and Raises

**Parameters**

*None (uses the instance state).*

**Returns**

* `float`: `phi_bar` such that $(V(\phi_{\rm bar})=V(\phi_{\rm metaMin}))$.

**Raises**

* `PotentialError("…", "no barrier")`: if the barrier top cannot be located inside the interval, or the barrier height is non-positive (no barrier).
* `PotentialError("…", "stable, not metastable")`: defensively raised if $(V(\phi_{\rm metaMin}) \le V(\phi_{\rm absMin}))$.

#### Notes

* The method is **purely 1D** and makes no smoothness assumptions beyond what the line search and root solver require (continuous (V)).
* The cached `self._barrier_info` lets you reuse `phi_top` and heights in later analysis/plots without recomputation.

---

### `findRScale`

#### Signature

```python
findRScale(self) -> float
```

#### Purpose

Estimate a **characteristic radial scale** $(r_{\rm scale})$ for the instanton solution. 
This sets physical step sizes for ODE integration and helps select plotting ranges.

#### Physical meaning of the scale

Near the **barrier top** the Euclidean EoM linearizes to

$$\phi'' + \frac{\alpha}{r}\phi' \simeq V''(\phi_{\rm top}) \bigl(\phi - \phi_{\rm top}\bigr)$$

If the top were strictly quadratic with curvature $(V''(\phi_{\rm top})<0)$, a naive local scale is $(r_{\rm curv}\sim 1/\sqrt{\lvert V''(\phi_{\rm top})\rvert})$.
However, many relevant potentials in cosmology/phase transitions have **flat-topped** barriers $((V''(\phi_{\rm top})\approx 0))$, making $(r_{\rm curv})$ blow up 
even when tunneling is well-defined.

To remain stable and **legacy-compatible**, we adopt a **cubic surrogate** for the barrier shape that matches:

* a **maximum** at the top $((\phi_{\rm top}))$,
* a **minimum** at the metastable vacuum $((\phi_{\rm metaMin}))$,

which yields the robust scale

$$\boxed{
r_{\rm scale} = r_{\rm cubic} =
\frac{\bigl|\phi_{\rm top}-\phi_{\rm metaMin}\bigr|}
{\sqrt{6[V(\phi_{\rm top})-V(\phi_{\rm metaMin})]}} }$$

This remains finite on flat tops and empirically tracks the small-oscillation period scale up to an $(\mathcal{O}(1))$
factor — ideal for setting numerics.

#### Behavior & algorithm (what’s new)

* Reuses `findBarrierLocation` to validate the barrier and obtain `phi_top` and its height above the false vacuum.
* Computes and **returns** the legacy **cubic** scale $(r_{\rm cubic})$ (unchanged public behavior).
* Also computes an optional **curvature** diagnostic $(r_{\rm curv} = 1/\sqrt{-V''(\phi_{\rm top})})$ when 
$(V''(\phi_{\rm top})<0)$ (otherwise (+\infty)).
* Caches diagnostics in `self._scale_info`:

  * `phi_top`, `V_top_minus_Vmeta`, `xtop = phi_top - phi_metaMin`,
  * `rscale_cubic`, `rscale_curv`, `d2V_top`.

#### Parameters, returns and Raises

**Parameters**

*None (uses the instance state).*

**Returns**

* `float`: $(r_{\rm scale} = r_{\rm cubic})$, used elsewhere for step sizes and domain lengths.

**Raises**

* `PotentialError("…", "no barrier")`: if the barrier fails validation (no interior top or non-positive height).

#### Notes

* Returning the cubic scale preserves the **legacy interface** and numerical behavior.
* The diagnostics are helpful for **thin-wall detection**, performance tuning, and sanity plots.

---

Here’s the continuation of **`single_field.md`** for the next block.

---

## Lot SF-3 — Quadratic local solution & initial conditions

This block implements the *local* (near-center) analytic control we use to (i) evaluate the field in a small neighborhood of the bubble center and
(ii) generate safe initial conditions away from the (r=0) singular point of the radial equation.
The two public members covered here are:

* `exactSolution(r, phi0, dV, d2V)`
* `initialConditions(delta_phi0, rmin, delta_phi_cutoff)`

Both are methods of `SingleFieldInstanton`.

### Physical background (why Bessel functions appear)

The Euclidean bounce equation for a spherically symmetric profile in $((\alpha+1))$ spatial dimensions is

$$\phi''(r) + \frac{\alpha}{r}\phi'(r) = V'(\phi)\qquad r\ge 0$$

Near a chosen point $(\phi_0)$ (typically very close to the true minimum $(\phi_{\rm absMin})$ when constructing thin-wall bounces), 
Taylor-expand the potential to quadratic order:

$$V'(\phi) \approx dV + d2V(\phi-\phi_0); \quad dV = V'(\phi_0);\quad d2V = V''(\phi_0)$$

Let $(\delta\phi(r)=\phi(r)-\phi_0)$. Then

$$\delta\phi'' + \frac{\alpha}{r}\delta\phi' - d2V\delta\phi = dV$$

Shift out the constant drive with $(\delta\phi=\psi + dV/d2V)$ (when $(d2V\neq 0)$), to obtain the homogeneous equation

$$\psi'' + \frac{\alpha}{r}\psi' - d2V\psi = 0$$

whose regular solution at the origin is expressed in terms of Bessel functions. Defining

$$\nu \equiv \frac{\alpha-1}{2}\qquad ,\beta \equiv \sqrt{|d2V|},\qquad t \equiv \beta r$$

the *regular* solution is

* **Stable curvature** ((d2V>0), harmonic well):

$$\phi(r)-\phi_0 = \frac{dV}{d2V}\Bigg[\Gamma(\nu+1)\Big(\tfrac{t}{2}\Big)^{-\nu} I_\nu(t)-1\Bigg]$$

* **Unstable curvature** ((d2V<0), inverted well, e.g. near the barrier top): replace $(I_\nu \to J_\nu)$.

Regularity at the origin enforces $(\phi'(0)=0)$ for any $(\alpha\ge 0)$.

If **(d2V=0)** (flat curvature), the ODE reduces to a constant drive and the *exact* regular solution is the polynomial

$$\phi(r)=\phi_0 + \frac{dV}{2(\alpha+1)}r^2\qquad \phi'(r)=\frac{dV}{\alpha+1}r$$

These closed forms are what the code evaluates, with numerically stable branches for small/large arguments.

---

### `exactSolution`

#### Signature

```python
exactSolution(r: float, phi0: float, dV: float, d2V: float) -> exactSolution_rval
# exactSolution_rval = namedtuple("exactSolution_rval", "phi dphi")
```

#### Purpose

Compute the **regular** local solution $((\phi(r),\phi'(r)))$ at radius (r) assuming a quadratic expansion of the potential around $(\phi_0)$.
This is used to:

* accurately probe the profile near the origin,
* build safe, physically consistent initial conditions for the global ODE solver.

#### Key definitions (appearing in formulas and code)

* $(\alpha)$: friction power in the radial term, i.e. spacetime dimension minus 1.
* $(\nu=(\alpha-1)/2)$: effective Bessel index fixed by the radial Laplacian.
* $(\beta=\sqrt{|d2V|})$ and $(t=\beta r)$: scale & argument controlling oscillatory/exponential behavior.
* Regularity: $(\phi'(0)=0)$ (enforced exactly by the implementation).

#### Parameters

* `r` (`float`): radius (≥ 0). At `r==0` the method returns `(phi0, 0.0)` exactly.
* `phi0` (`float`): expansion point for the quadratic model.
* `dV` (`float`): $(V'(\phi_0))$.
* `d2V` (`float`): $(V''(\phi_0))$.

#### Returns

* `exactSolution_rval(phi, dphi)`: field and radial derivative at radius `r`.

#### Implementation notes

* **Flat curvature** (`d2V==0`): uses the exact polynomial solution above (no Bessel calls).
* **Small argument** (`t = beta*r ≤ 1e-2`): uses a **short even-power series** up to (t^6), which is well-conditioned and avoids any division by `r`.
* **General case**: uses the Bessel/modified-Bessel forms, with overflow/underflow warnings suppressed locally; the combined expressions are finite.
* Input validation ensures all inputs are finite and `r≥0`.

#### Physical interpretation

* (d2V>0): local **restoring force**; near a true minimum the solution is “massive” and grows as $(I_\nu(t))$ but regularized to match $(\phi'(0)=0)$.
* (d2V<0): local **tachyonic/inverted** curvature, relevant near the barrier top; the solution is oscillatory via $(J_\nu)$.
* (d2V=0): locally flat—driven purely by the constant slope (dV); the profile starts quadratically from the center.

---

### `initialConditions`

#### Signature

```python
initialConditions(delta_phi0: float, rmin: float, delta_phi_cutoff: float) -> initialConditions_rval
# initialConditions_rval = namedtuple("initialConditions_rval", "r0 phi dphi")
```

#### Purpose

Choose **where** to start integrating the full ODE (away from the (r=0) singularity) and with **which values** $((\phi,\phi'))$,
using the local quadratic solution as a high-accuracy guide. 
The goal is to start just outside the bubble center yet already sufficiently displaced from the true minimum to keep the 
overshoot/undershoot search efficient and stable.

#### Inputs & meaning

* `delta_phi0`: desired central offset $( \phi(0)-\phi_{\rm absMin} )$. In thin-wall cases this can be *very* small.
* `rmin`: the **smallest** radius allowed for starting the global integration (relative to `rscale` in higher-level code).
* `delta_phi_cutoff`: the target magnitude of the field offset at the starting radius $(r_0)$:
  $(|\phi(r_0)-\phi_{\rm absMin}| > |\delta\phi_{\rm cutoff}|)$.

#### Strategy (what the code does)

1. Construct $(\phi_0 = \phi_{\rm absMin} + \delta\phi_0)$ and compute $(dV=V'(\phi_0))$, $(d2V=V''(\phi_0))$.
2. Use `exactSolution` at `rmin`.

   * If $(|\phi(r_{\min})-\phi_{\rm absMin}| > |\delta\phi_{\rm cutoff}|)$, **start there**.
   * If the field is moving the **wrong way** (sign of $(\phi'(r_{\min}))$ opposite to $(\delta\phi_0))$, **start there** as well; increasing (r) won’t fix the direction.
3. Otherwise, **geometrically increase** (r) (×10 each step) and re-evaluate with `exactSolution` until the cutoff is exceeded (this brackets the crossing).
4. Solve for the exact $(r_0)$ by a 1D root find on
   $(f(r)=|\phi(r)-\phi_{\rm absMin}|-|\delta\phi_{\rm cutoff}|)$.
5. Return $((r_0,\phi(r_0),\phi'(r_0)))$ as a named tuple.

If the geometric search fails to bracket the crossing (pathological potential/settings), a clear `IntegrationError` is raised.

#### Returns

* `initialConditions_rval(r0, phi, dphi)`: starting radius and values to feed the global integrator.

#### Notes & guidance

* This method is agnostic to the global wall shape; it only relies on the **local** quadratic model, which is accurate near the true minimum where $(r_0)$ lives in thin-wall cases.
* Choosing a too large `delta_phi_cutoff` may degrade accuracy (starting too far from the regime where the quadratic model is excellent). Too small can slow down the shoot or underflow numerics. The default policy used upstream balances these effects.
* The named-tuple interface is intentional—downstream code can unpack by name or position.

---

### Common assumptions (for both functions)

* **Spherical symmetry** and **regularity at the origin** $((\phi'(0)=0))$.
* The **principal** branches for $(\sqrt{\cdot})$ and Bessel functions are used.
* $(\alpha\ge 0)$ (physical cases); nonetheless, the formulas are coded generically.
* All inputs are finite; non-finite inputs raise a value error upfront.

---

### Numerical stability & accuracy

* **Small-argument regime**: the (t)-series keeps terms through $(t^6)$, which is more than enough for $(t\lesssim 10^{-2})$ in double precision.
* **Flat curvature**: handled by an **exact polynomial**; no Bessel calls or divisions-by-(r).
* **Overflow/underflow**: benign warnings inside SciPy’s Bessel routines are silenced locally, and the combinations used are finite by construction.
* **Deterministic regularity**: at `r==0`, `exactSolution` returns `(phi0, 0.0)` exactly.

---

### Quick reference (symbols)

* $(\alpha)$: friction power in the radial Laplacian; equals spacetime dimension minus 1.
* $(\nu=(\alpha-1)/2)$: Bessel index.
* $(dV=V'(\phi_0))$, $(d2V=V''(\phi_0))$: local slope & curvature of the potential.
* $(\beta=\sqrt{|d2V|})$, $(t=\beta r)$: scale and argument for Bessel functions.
* $(\phi_{\rm absMin})$, $(\phi_{\rm metaMin})$: true and false vacuum field values (set in the class constructor).
* `delta_phi0`, `delta_phi_cutoff`: user-level displacements used to place the start of the integration.

---

## Lot SF-4 — ODE core (equation, adaptive driver, sampler)

This lot contains the numerical engine that integrates the bounce equation once the potential interface and scales are known. It provides:

* the **equation of motion** in first-order form;
* an **adaptive step driver** that marches the solution and classifies the outcome as *overshoot*, *undershoot*, or *converged*;
* a **sampler** that fills a user-chosen radial grid with a smooth profile using cubic Hermite interpolation between accepted RK steps;
* a small **tolerance normalizer** to make error controls explicit and robust.

Throughout, we solve the radial Euclidean EOM for a single field,

$$\frac{d^2\phi}{dr^2}+\frac{\alpha}{r}\frac{d\phi}{dr}= \frac{dV}{d\phi}(\phi)$$

where $(\alpha)$ is the “friction” coefficient (commonly $(\alpha=2)$ for (O(3)) finite-temperature bounces and $(\alpha=3)$ for $(O(4))$ zero-temperature bounces). 
The $(\alpha/r)$ term comes from the radial Laplacian in $((\alpha+1))$ dimensions.

---

### `_normalize_tolerances` (internal helper)

**Signature**

```python
@staticmethod
_normalize_tolerances(epsfrac, epsabs) -> tuple[float, float, float, float]
```

**Purpose**

Accepts *either* scalars *or* 2-component arrays for the relative/absolute tolerances and returns:

1. `ef_scalar` – a single relative tolerance passed to the RK stepper (strictest across components);
2. `ea_scalar` – a single absolute tolerance for the RK stepper (strictest across components);
3. `eps_phi`   – absolute threshold (3× `epsabs` for $(\phi)$) used by our *event/convergence* tests;
4. `eps_dphi`  – absolute threshold (3× `epsabs` for $(d\phi)$) used likewise.

This keeps the step controller simple (scalar tolerances) while still giving per-component, physically meaningful stopping criteria.

**Notes**

* If `epsabs` is scalar, both $(\phi)$ and $(d\phi)$ use the same $(3\times)$ threshold; with a 2-vector, they may differ.
* The factor **3** in `eps_phi`, `eps_dphi` mirrors the legacy “within ~3× absolute tol ⇒ good enough” convention used elsewhere in this module.

---

### `equationOfMotion`

**Signature**

```python
equationOfMotion(y: np.ndarray, r: float) -> np.ndarray
```

**Purpose**

Right-hand side of the first-order system for $(y=[\phi, \dot\phi])$ $(dot ≡ (d/dr))$:

$$\dot{\phi} = y_1\qquad \dot{y}_1 = \frac{dV}{d\phi}(\phi) - \frac{\alpha}{r} y_1$$

**Implementation details**

* To guard against accidental calls at (r=0) (which should not happen in production—integrations always start at (r>0)), we replace (r) by a tiny positive `r_eff` if needed so the friction term stays finite.
* Uses the user-provided/derived `self.dV(phi)`.

**Physics notes**

The $( \alpha/r )$ “friction” originates from the radial Laplacian in $((\alpha+1))$ Euclidean dimensions.
Regular bounce solutions satisfy $( d\phi/dr = \mathcal{O}(r) )$ as $( r\to 0 )$, so the product $( (\alpha/r)d\phi )$ remains finite.

---

### `integrateProfile`

**Signature**

```python
integrateProfile(
    r0: float,
    y0: array_like,      # [phi(r0), dphi(r0)]
    dr0: float,
    epsfrac, epsabs,     # scalar or 2-vector tolerances
    drmin: float,
    rmax: float,
    *eqn_args
) -> namedtuple("integrateProfile_rval", "r y convergence_type")
```

**What it does**

Advances the ODE solution from $((r_0,y_0))$ using an **adaptive Cash–Karp RK5(4)** stepper (`rkqs`) until one of three conditions is met:

1. **converged** – both $(|\phi-\phi_{\rm metaMin}|<\epsilon_{\phi})$ and $(|d\phi|<\epsilon_{d\phi})$;
2. **overshoot** – within a step the field crosses $(\phi_{\rm metaMin})$;
3. **undershoot** – within a step the field “turns back” (sign of $(d\phi)$ indicates motion away from the target).

In (2) and (3), we locate the event **inside the last accepted step** by **cubic Hermite interpolation** (our `cubicInterpFunction`), and refine with a bracketing root find (`scipy.optimize.brentq`).
If bracketing fails (rare, degenerate slope), we fall back to the point minimizing the relevant magnitude (either $(|\phi-\phi_{\rm metaMin}|)$ or $(|d\phi|))$ on a small uniform subgrid of the step.

**Inputs & error control**

* `epsfrac`, `epsabs` can be scalars or 2-vectors. They are normalized via `_normalize_tolerances`:

  * `ef_scalar`, `ea_scalar` go to `rkqs`;
  * `eps_phi`, `eps_dphi` define the *event/convergence* thresholds.
* `drmin` prevents step underflow; if an accepted step requests `dr<drmin`, we abort with a clear `IntegrationError`.
* `rmax` limits the total travelled distance; we abort if $(r > r_0 + r_{\max})$.

**Direction logic (overshoot vs. undershoot)**

We define a sign `ysign` that encodes “where the target is” relative to the current motion:

* If we start noticeably away from the target, `ysign = sign(phi - phi_metaMin)`.
* If we start essentially on target, we use `ysign = -sign(dphi)` so that moving *away* is treated as an undershoot and a crossing back is an overshoot.

Given `ysign`, we classify a step by looking at:

* **undershoot** if `dphi * ysign > +eps_dphi` (slope keeps pushing further from target);
* **overshoot** if `(phi - phi_metaMin) * ysign < -eps_phi` (crossing the target).

**Return value**

A named tuple with:

* `r` – the final radius (event location or the last time where convergence was satisfied),
* `y` – the final state $([\phi, d\phi])$ at that radius,
* `convergence_type` – one of `"converged"`, `"overshoot"`, `"undershoot"`.

**Why cubic Hermite interpolation?**

We know $((y_0, dy/dr|*{r_0}))$ and $((y_1, dy/dr|*{r_1}))$ for both ends of a step. 
The cubic Hermite (a.k.a. piecewise cubic with end slopes) reconstructs a smooth in-step curve that respects both values and slopes, 
giving accurate, monotone-friendly event localization without taking extra ODE mini-steps.

**Typical usage**

`findProfile` uses this method in a bisection-like loop over the shooting parameter, reading only the outcome (“over/under/converged”) and the precise event radius. 
After the best initial condition is found, it calls `integrateAndSaveProfile` to produce a full, nicely sampled wall profile.

---

### `integrateAndSaveProfile`

**Signature**

```python
integrateAndSaveProfile(
    R: array_like,       # monotonically increasing radii
    y0: array_like,      # [phi(R[0]), dphi(R[0])]
    dr: float,
    epsfrac, epsabs,     # scalar or 2-vector tolerances
    drmin: float,
    *eqn_args
) -> namedtuple("Profile1D", "R Phi dPhi Rerr")
```

**Purpose**

Integrate the ODE once more **and fill a user-specified radial grid** (R) with $(\phi(R_i))$ and $(d\phi/dR(R_i))$. 
This is the second (“sampling”) pass typically used *after* the shooting has determined the correct initial condition and outer radius.

**How it works**

* Uses the same adaptive RK driver as `integrateProfile`.
* Between accepted RK step endpoints $((r_0,y_0))$ and $((r_1,y_1))$, it evaluates the **same cubic Hermite interpolant** and writes samples for all `R[i] ∈ (r0, r1]`.
* If a proposed accepted step would have `dr < drmin`, it **clamps** to `drmin`, records `Rerr` on first occurrence (the radius where step clamping first became necessary), and continues so the output arrays are always fully populated.

**Outputs**

* `R` – the input grid, echoed back;
* `Phi` – values $(\phi(R_i))$;
* `dPhi` – values $(d\phi/dR(R_i))$;
* `Rerr` – `None` if every accepted step satisfied `dr ≥ drmin`; otherwise the **first** radius where clamping was applied.

**Notes**

* This routine does **not** attempt to classify events (over/under/converged). That logic belongs in `integrateProfile`. Here the goal is to *sample* a known good solution.

---

### Practical guidance

* **Tolerances.** A good starting point mirrors the legacy defaults used in the higher-level driver:

  * `epsfrac = [phitol, phitol]`,
  * `epsabs = [|Δφ|·phitol, |Δφ|/rscale · phitol]`,
    with `phitol ~ 1e-4` and `Δφ = φ_metaMin − φ_absMin`. You can also pass scalars.
* **Initial step.** Set `dr0 ~ rmin` (the same “small” radius where we start, coming from Lot SF-2’s `rscale`).
* **Limits.** Choose `rmax` comfortably above the expected wall thickness (often `~O(10)·rscale)`); choose `drmin` at least a few orders of magnitude below the smallest features you want to resolve.
* **Performance.** The cubic Hermite interpolation avoids tiny corrective micro-steps for event localization, which keeps the driver fast while maintaining smooth, physically sensible crossings.

---

### Failure modes and messages

* `IntegrationError("... exceeded rmax ...")` – the profile did not settle/cross within the allowed domain; revisit `rmax` or the shooting parameter.
* `IntegrationError("... step underflow ...")` – the stepper kept asking for `dr < drmin` to meet tolerances; loosen tolerances or increase `drmin` cautiously.
* Value errors guard obvious API misuse (non-finite `y0`, wrong shapes, non-monotonic `R`, etc.).

---

### Summary

Lot SF-4 equips the `SingleFieldInstanton` class with a clean, robust integrator:

* a **physically faithful ODE** (with friction and a safe $(r\to 0)$ guard),
* an **adaptive, tolerance-driven stepper** with explicit convergence semantics (over/under/converged),
* and a **high-quality sampler** that turns accepted steps into smooth profiles on any grid.

These pieces are deliberately modular: subclasses (e.g., constant friction walls) reuse the same machinery by passing extra arguments to `equationOfMotion` via `*eqn_args`, while keeping the numerics identical.

---

## Lot SF-5 — Profile search (overshoot/undershoot)

**Goal.** Find the full bounce profile $( \phi(r) )$ by *shooting* on the unknown center value $( \phi(0) )$. 
We adjust a scalar parameter (x) so that the outward integration converges onto the false (metastable) vacuum as $( r\to\infty )$. 
The search uses classic **overshoot/undershoot** bracketing and a final dense sampling pass.

---

### `SingleFieldInstanton.findProfile(...)`

**Signature (unchanged).**

```python
findProfile(
    xguess=None, xtol=1e-4, phitol=1e-4,
    thinCutoff=0.01, npoints=500, rmin=1e-4, rmax=1e4,
    max_interior_pts=None
) -> Profile1D  # (R, Phi, dPhi, Rerr)
```

#### What problem this solves

We need the solution of

$$\phi''(r) + \frac{\alpha}{r}\phi'(r) = V'(\phi)$$

that starts at the true minimum $((\phi\approx\phi_{\rm absMin}))$ near (r=0) and asymptotes to the false minimum $((\phi\to\phi_{\rm metaMin}))$ for large (r). 
The correct central value $(\phi(0))$ is *not* known a priori; we determine it by shoot-and-correct.

#### The shooting parameter (x)

Instead of varying $(\phi(0))$ directly, we vary

$$\boxed{\phi(0) \equiv \phi_{\rm absMin}+ e^{-x}\big(\phi_{\rm metaMin}-\phi_{\rm absMin}\big)}$$
  so that:

- **Small** (x) → $(\phi(0))$ close to the *false* minimum (high potential energy) → dynamics tend to **overshoot** across $(\phi_{\rm metaMin})$.
- **Large** (x) → $(\phi(0))$ close to the *true* minimum (low energy) → dynamics turn around before reaching $(\phi_{\rm metaMin})$ (**undershoot**).

If `xguess` is not provided, we choose a sensible default by placing $(\phi(0))$ near the barrier “edge” $(\phi_{\rm bar})$, i.e.

$$x_{\rm init} \approx -\ln\left(
\frac{\phi_{\rm bar}-\phi_{\rm absMin}}{\phi_{\rm metaMin}-\phi_{\rm absMin}}
\right).$$

#### Radii and tolerances (numerics)

* We scale all radii with the characteristic `rscale` (Lot SF-2).

  * `rmin * rscale`: the **starting radius** guess and initial stepsize.
  * `drmin = 0.01 * rmin * rscale`: **minimum** allowed stepsize for the RK driver.
  * `rmax * rscale`: **maximum** travel distance from the start.
* Error controls for the adaptive RK driver (Lot SF-4):

  * `phitol` sets both **relative** (`epsfrac=[phitol, phitol]`) and **absolute** tolerances
    
$$epsabs = \big[\texttt{phitol}\cdot |\Delta\phi|, \texttt{phitol}\cdot |\Delta\phi|/\texttt{rscale}\big]$$
    for $([\phi,\phi'])$.
  * The driver uses a strict scalar for per-step control and per-component thresholds for event detection and convergence.

#### Step-by-step algorithm

1. **Map $(x\to\Delta\phi_0)$.**
   $(\Delta\phi_0 = e^{-x}(\phi_{\rm metaMin}-\phi_{\rm absMin}))$.

2. **Practical initial surface at $(r_0>0)$.**
   Call `initialConditions(Δφ0, rmin*rscale, thinCutoff*|Δφ|)`.
   This uses the local quadratic solution (Lot SF-3) to pick $(r_0)$ such that
   $(|\phi(r_0)-\phi_{\rm absMin}| \approx \texttt{thinCutoff}\cdot|\Delta\phi|)$
   and returns$ ((r_0,\phi(r_0),\phi'(r_0)))$.
   *Intuition:* for thin walls, start integration near the wall (not at the exact center), which stabilizes shooting.

3. **Trial integration and event classification.**
   Run `integrateProfile(r0, y0, ...)` (Lot SF-4). Three outcomes:

   * `"converged"`: $( |\phi-\phi_{\rm metaMin}| )$ and $( |\phi'| )$ are within tolerance.
   * `"overshoot"`: within the last step, $(\phi)$ **crossed** $(\phi_{\rm metaMin})$;
     we locate the crossing by **cubic Hermite** interpolation (with consistent slopes).
   * `"undershoot"`: the field **turns around** before reaching $(\phi_{\rm metaMin})$
     (detected via $(\phi'=0 )$), again located by cubic interpolation.

4. **Bracket in (x) and bisect.**
   Maintain $([x_{\min}, x_{\max}])$ such that:

   * undershoot ⇒ $(x_{\min} \leftarrow x)$ (we were too close to the true minimum);
   * overshoot ⇒ ($x_{\max} \leftarrow x)$ (we were too close to the false minimum).
     If no upper bound yet, expand geometrically (`xincrease ≈ 5`).
     Once bracketing exists, **bisect** until $(x_{\max}-x_{\min}<\texttt{xtol})$ or we get `"converged"`.

5. **Final dense pass (returned profile).**
   With the last valid $((r_0,y_0))$ and end radius $(r_f)$, build a uniform array
   `R = linspace(r0, rf, npoints)` and call
   `integrateAndSaveProfile(R, y0, ...)`.
   This integrates adaptively **and** fills every `R[i]` by cubic Hermite interpolation between accepted RK steps. The return object is:

   * `R`: radii,
   * `Phi`: (\phi(R)),
   * `dPhi`: (\phi'(R)),
   * `Rerr`: first radius where the step would have fallen below `drmin` (if it happened), else `None`.

6. **(Optional) Interior points $(0 \le r < r_0)$.**
   If `max_interior_pts` is not zero, we synthesize points in the **bubble interior** using the analytic *local* solution (`exactSolution`) derived in Lot SF-3. We place up to `max_interior_pts` points on a non-uniform grid that lands exactly on (r=0) and $(r=r_0)$, then concatenate interior and integrated segments.

   * If `max_interior_pts=None`, we default to `npoints//2`.
   * Set it to `0` to skip interior fill.

#### Convergence / failure modes (and what to tweak)

* **Iteration cap reached** while bracketing in (x): increase `rmax`, relax `phitol`, or widen `thinCutoff` so the initial surface is farther from the center (easier shooting).
* **Step underflow** (`dr < drmin`) in the driver: increase `rmin` (hence `drmin`) or relax `phitol`.
* **Both trials on the same side** (can't bracket): try a smaller/larger `xguess`, or increase `xincrease` (implicitly done inside); extreme thin-wall cases often benefit from a larger `thinCutoff` (e.g. `0.05–0.2`).

#### Physical interpretation

* In the **thin-wall** limit (almost degenerate minima), the correct (x) is *large*: $(\phi(0))$ sits very close to the true vacuum and the wall is narrow. Small changes in (x) produce large changes in outcome—hence the careful bracketing.
* In the **thick-wall** regime, the search is gentler: the field starts farther from the true minimum, and both overshoot/undershoot are easier to bracket.

#### Output guarantees

* The returned profile always corresponds to the **last successful integration** over $([r_0,r_f])$, sampled at `npoints`. If an interior segment is synthesized, the profile begins at (r=0); otherwise it begins at $(r=r_0>0)$ (typical for thin walls).
* `Rerr` is purely diagnostic: if not `None`, it marks the first place where the integrator had to clamp a too-small step to `drmin`; the profile remains valid.

---

**In short:** `findProfile` wraps three building blocks developed in the previous lots—(i) a stable *local* start (`initialConditions`), 
(ii) an event-aware, tolerance-controlled driver (`integrateProfile`),
and (iii) a dense sampler (`integrateAndSaveProfile`)—into a robust overshoot/undershoot search in the single scalar parameter (x).

---

## Lot SF-6 — Action & post-processing

**Goal.** 
Given a converged bounce profile $((R,\Phi(R),\Phi'(R)))$, compute the **Euclidean action** and provide **diagnostics** 
and **geometric scales** that are useful for interpretation and downstream phenomenology.
This lot modernizes the legacy `findAction` and adds practical post-processing helpers.

**Context & notation**

* We work in $(d=\alpha+1)$ radial dimensions (i.e., $(\alpha=d-1)$ in the ODE).
* The unit $(\alpha)$-sphere area is

$$\Omega_\alpha \equiv \frac{2\pi^{(\alpha+1)/2}}{\Gamma!\big((\alpha+1)/2\big)}$$

* A profile is the named tuple returned by `findProfile`: `profile.R`, `profile.Phi`, `profile.dPhi` (and `profile.Rerr`).
* We subtract the false-vacuum energy $(V(\phi_{\rm meta}))$ so the action density is anchored at the metastable vacuum.

---

### What’s new vs. the legacy

* **New**: `actionBreakdown(profile)` — splits (S) into kinetic/potential/“interior bulk” pieces and returns per-radius **densities** for plotting and checks.
* **New**: `wallDiagnostics(profile, frac=(0.1,0.9))` — estimates **wall position** and **thickness** directly from the profile (levels in $(\phi)$ and peak $(|\Phi'|)$).
* **New**: `betaEff(profile, method=...)` — **proxies** for an inverse length/time scale $(\beta_{\rm eff})$ (`"rscale"`, `"curvature"`, `"wall"`).
* **Improved**: `evenlySpacedPhi(...)` — robust $(\phi)$-space resampling with monotonicity enforcement and endpoint padding (zero slopes at vacua).

---

### Physics background (why these formulas)

The action (with the false-vacuum constant removed) is

$$S = \int_{r_0}^{\infty}\Big[\tfrac12(\partial_r\phi)^2+\big(V(\phi)-V(\phi_{\rm meta})\big)\Big]  r^\alpha dr\Omega_\alpha +$$

$$+\underbrace{ \int_{0}^{r_0} \big(V(\phi(r_0)) - V(\phi_{\rm meta})\big),d^dr }_{\text{“interior bulk”}}$$

because thin-wall integrations start at $(r=r_0>0)$. Regularity implies$ (\phi'(r)\sim\mathcal{O}(r))$,
so the **gradient** contribution from $([0,r_0])$ is negligible,
but the **potential** offset must be accounted for via the (d)-ball volume:

$${\rm Vol}_d(r_0)=\frac{\pi^{d/2}}{\Gamma(d/2+1)}r_0^d$$

This matches the original legacy semantics while making the computation explicit and numerically stable.

---

### `findAction(profile)`

**What it computes.**
The scalar Euclidean action (S) using

$$S_{\rm line}=\int_{r_0}^{\infty}\left[\tfrac12\Phi'(r)^2 + V(\Phi(r))-V(\phi_{\rm meta})\right] r^\alpha dr\Omega_\alpha$$

$$\Delta S_{\rm interior}={\rm Vol}*d(r_0)\big[V(\Phi(r_0))-V(\phi*{\rm meta})\big]$$

and returns $(S=S_{\rm line}+\Delta S_{\rm interior})$.

**Inputs.**
`profile` (from `findProfile`): arrays `R`, `Phi`, `dPhi` must be 1D, same length $(\ge 2)$.

**Output.**
A single `float` — the Euclidean action.

**Why it matters.**
(S) controls the (zero-temperature) tunneling rate prefactor $( \Gamma \propto e^{-S} )$ (up to determinants).
In thermal problems $(S_3/T)$ plays the analogous role; 
our formulation and helpers are designed to interface cleanly with those workflows later.

**Numerical notes.**

* Uses Simpson integration on the line contribution with the correct geometric weight $(r^\alpha\Omega_\alpha)$.
* Adds the interior potential-only bulk term if $(r_0>0)$.

---

### `evenlySpacedPhi(phi, dphi, npoints=100, k=1, fixAbs=True)`

**What it does.**
Resamples $((\phi(r), \phi'(r)))$ onto a **uniform grid in $(\phi)$**,
returning arrays $((\phi_i, \phi'_i))$ with $(\phi_i)$ equally spaced.

**Why it’s useful.**

* Many diagnostics/plots are more readable vs **field value** than vs radius (e.g., comparing kinetic vs potential terms along the wall).
* Makes it easy to overlay different profiles in the same $(\phi)$-space.

**Key options.**

* `fixAbs=True` pads endpoints to **exactly** $((\phi_{\rm abs},\phi_{\rm meta}))$ with zero slopes — physically correct for regular instantons.
* Enforces **monotonic $(\phi)$** (drops tiny backtracks before spline fitting) to avoid oscillatory spline artifacts.
* `k=1` (linear) is very robust; `k=3` gives smooth derivatives when the data are clean.

**Output.**
`phi2`, `dphi2` (both 1D arrays with length `npoints`).

---

### `actionBreakdown(profile)`  — **New**

**What it returns.**
A named tuple with:

* `S_total`: same value as `findAction(profile)`.
* `S_kin`, `S_pot`: line integrals of the kinetic and potential pieces **separately**.
* `S_interior`: the interior bulk correction.
* Copies of `r`, `phi`, `dphi`.
* `density`: a dict with arrays

  * `density["kin"]` = $tfrac12\Phi'^2 r^\alpha \Omega_\alpha$,
  * `density["pot"]` = $(V(\Phi)-V_{\rm meta}) r^\alpha \Omega_\alpha$
  * `density["tot"] = density["kin"] + density["pot"]`.

**Why it’s helpful.**

* Lets you **plot** where the action is accumulated (e.g., most weight sits in the wall).
* Makes **sanity checks** straightforward (e.g., verify positivity of densities, small contribution far from the wall, etc.).

**Caveat.**
`density["tot"]` covers only the **line** part; the interior bulk is a single additive scalar, not a distributed density.

---

### `wallDiagnostics(profile, frac=(0.1, 0.9))`  — **New**

**Idea.**
Characterize the **wall geometry** directly from the profile.

**Definitions.**

* Let $(\Delta\phi \equiv \phi_{\rm meta}-\phi_{\rm abs})$.
* Define field levels

$$\phi_{\rm lo}=\phi_{\rm abs}+f_{\rm lo}\Delta\phi$$

$$\phi_{\rm hi}= \phi_{\rm abs}+f_{\rm hi}\Delta\phi  $$

$$(\phi_{\rm mid}=\tfrac12(\phi_{\rm abs}+\phi_{\rm meta})) $$

* Invert the (monotonic) profile to get radii $(r(\phi))$;

**What it returns.**

* `r_peak`: radius where $(|\Phi'|)$ is maximal (often used as “wall center”).
* `r_mid`: radius where $(\phi=\phi_{\rm mid})$.
* `r_lo`, `r_hi`: radii at the chosen fractional field levels.
* `thickness` = $|r_{\rm hi} - r_{\rm lo}|$.
* The field levels `phi_lo`, `phi_hi`.

**Why it’s useful.**

* Provides a **coordinate-free** estimate of wall thickness and location.
* Enables simple comparisons between thin- and thick-wall regimes.

---

### `betaEff(profile, method="rscale")`  — **New**

**Purpose.**
Return a **proxy** for the inverse timescale/length $(\beta_{\rm eff})$ often used for order-of-magnitude reasoning. 
This is **not** the cosmological $(\beta \equiv -d(S_3/T)/dt)$; computing that requires a (T)-dependent potential and $(S_3(T)/T)$.

**Methods.**

* `"rscale"`: $(\beta_{\rm eff} = 1/\texttt{rscale})$.
  Always defined; `rscale` was obtained in Lot SF-2 from barrier geometry.
* `"curvature"`: $(\beta_{\rm eff} = \sqrt{|V''(\phi_{\rm top})|})$.
  Uses the second derivative at the barrier top; agrees with $(1/\texttt{rscale})$ up to $(\mathcal{O}(1))$ in many models.
* `"wall"`: $(\beta_{\rm eff}=1/\texttt{thickness})$.
  Thickness from `wallDiagnostics`; simple geometric proxy tied to the wall width.

**When to use which.**

* If you just need **one number**: `"rscale"`.
* If the **barrier curvature** is physically meaningful in your model: `"curvature"`.
* If your analysis hinges on **wall width** (e.g., friction effects): `"wall"`.

---

### Practical guidance & pitfalls

* **Profiles starting at $(r_0>0)$ (thin walls):**
  Expect a **nonzero interior bulk** potential term. The kinetic part from $([0,r_0])$ is suppressed by regularity.
* **Units & scaling:**
  (S) is dimensionless if (r) and (V) are in consistent units (as in the usual bounce conventions). All geometric addends preserve dimensional consistency.
* **Monotonicity in (\phi):**
  Small numerical back-and-forth in (\phi(r)) can spoil interpolation in (\phi)-space. The resampler removes those via a monotonic-indices filter before spline fitting.
* **Interpreting densities:**
  Most of the action density typically **localizes in the wall**; plotting `density["kin"]` and `density["pot"]` helps verify this and diagnose integration issues.

---

