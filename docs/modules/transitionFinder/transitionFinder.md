# `Transition Finder` 

Tools to study **finite-temperature cosmological phase transitions**.

This module is the bridge between:

- the **microscopic potential** `V(φ, T)` and its derivatives, and  
- the **macroscopic transition history**: which phase is realized at which
  temperature, and when bubbles nucleate between phases.

Conceptually:

- `traceMinimum` + `Phase` (Block A) handle the **geometric / kinematic** part:
  tracing minima as functions of temperature.
- Later blocks (B, C, …) add:
  - phase-network construction (`traceMultiMin`, `removeRedundantPhases`, …),
  - tunneling and nucleation (`tunnelFromPhase`, `findAllTransitions`, …),
  - thermodynamic and GW-oriented quantities (e.g. `S(T)`, `dS/dT`, α, β/H, etc.)

This document tracks the implementation **block by block** as we modernize the
original CosmoTransitions codebase.

---

## Block A – Minimum tracing and `Phase` objects (Phase Tracing)

Block A is about one thing:

> Given a potential `V(x, T)`, follow a **specific minimum** `x_min(T)` as you
> change temperature, and package it as an object with interpolation utilities.

It introduces two core pieces:

- `traceMinimum`: low-level numerical routine that walks a minimum in `(x, T)`
  space.
- `Phase`: a high-level container that stores `x_min(T)` and exposes a spline
  interface, plus book-keeping for second-order transitions.

These are the building blocks that later blocks will use to:

- build a **graph of phases** (nodes = Phase, edges = transitions),
- compute where tunneling can occur,
- scan over temperatures without re-minimizing from scratch every time.

---

### `traceMinimum`

#### Signature

```text
def traceMinimum(
    f: Callable[[npt.NDArray[np.float64], float], float],
    d2f_dxdt: Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]],
    d2f_dx2: Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]],
    x0: npt.ArrayLike,
    t0: float,
    tstop: float,
    dtstart: float,
    deltaX_target: float,
    dtabsMax: float = 20.0,
    dtfracMax: float = 0.25,
    dtmin: float = 1e-3,
    deltaX_tol: float = 1.2,
    minratio: float = 1e-2,
) -> TraceMinimumResult:
    
```
where

```text
class TraceMinimumResult(NamedTuple):
    X: npt.NDArray[np.float64]
    T: npt.NDArray[np.float64]
    dXdT: npt.NDArray[np.float64]
    overX: npt.NDArray[np.float64]
    overT: float
```

#### Conceptual picture

We want to track a **local minimum** of `f(x, T)` as temperature changes:

* `x` is a vector of field values (`R^N` for N fields).
* `T` is the temperature parameter.

At a minimum `x_min(T)` we have:

$$
\frac{\partial f}{\partial x}(x_{\min}(T), T) = 0.
$$

Differentiating w.r.t. `T`:

$$
\frac{d}{dT}\left(\frac{\partial f}{\partial x}\right)
= \frac{\partial^2 f}{\partial x^2} \frac{dx_{\min}}{dT}+
\frac{\partial}{\partial T}\left(\frac{\partial f}{\partial x}\right)=0
$$

so the derivative of the minimum position obeys

$$
H \cdot \frac{dx_{\min}}{dT} = - \frac{\partial}{\partial T}
\left(\frac{\partial f}{\partial x}\right),
$$

where `H = ∂²f/∂x²` is the Hessian. `traceMinimum` integrates this implicit
ODE for `x_min(T)`, while **re-minimizing** at each step to avoid numerical
drift.

The result is:

* a discrete set of temperatures `T[i]`,
* corresponding minima `X[i, :] ≈ x_min(T[i])`,
* and an estimate of `dx_min/dT` at each point, `dXdT[i, :]`.

When the Hessian becomes singular or develops a negative mode, the minimum is
about to **turn into a saddle** → we interpret this as the **end of the phase**
or approaching a transition; the algorithm stops and reports `overX`, `overT`.

#### Arguments (high-level)

* `f(x, T)`
  Scalar objective function. In physics applications this is usually
  the **effective potential** `V(φ, T)` (possibly shifted by a constant).

* `d2f_dxdt(x, T)`
  Returns `∂/∂T (∂f/∂x)`, a 1D array of size `Ndim`. You typically build this
  from the gradient of `f` with respect to `x` using the helper tools (see
  below).

* `d2f_dx2(x, T)`
  Returns the Hessian `∂²f/∂x²`, shape `(Ndim, Ndim)`.

* `x0`, `t0`
  Starting point on the minimum: `x0` should be **already near a minimum** of
  `f(·, t0)`.

* `tstop`, `dtstart`
  Target temperature and initial step size. The sign of `dtstart` controls the
  direction:

  * `dtstart > 0` → trace from `t0` **upwards** toward `tstop`;
  * `dtstart < 0` → trace from `t0` **downwards**.

* `deltaX_target`
  Target displacement in field space per **accepted** step. The algorithm
  adapts `ΔT` so that the error in `x` stays around this scale.

* `dtabsMax`, `dtfracMax`, `dtmin`, `deltaX_tol`, `minratio`
  Numerical safety parameters:

  * `dtabsMax`, `dtfracMax` put an **upper bound** on `|ΔT|`;
  * `dtmin` is the **minimum allowed step** before we declare defeat;
  * `deltaX_tol` rescales the acceptable error in field space;
  * `minratio` controls when the Hessian is considered to have developed
    a problematic zero/negative mode.

#### Return values

`TraceMinimumResult` with:

* `X` : `(n_steps, Ndim)` minima along the traced path.
* `T` : `(n_steps,)` temperatures.
* `dXdT` : `(n_steps, Ndim)` derivative of the minimum w.r.t. T.
* `overX`, `overT` : last “problematic” point encountered:

  * typically a saddle or the point where the minimum ceases to exist;
  * this is used in higher-level code to detect end-of-phase / phase mergers.

#### Relationship to helper functions

The old CosmoTransitions code used to compute finite differences **inside**
`traceMinimum`. In the updated design, we delegate this to the general-purpose
derivative tools in `helper_functions`:

* `gradientFunction` to get `∂f/∂x`,
* `hessianFunction` to get `∂²f/∂x²`,
* finite-difference in T for `d2f_dxdt` if needed.

This separation has two advantages:

1. `traceMinimum` stays focused on **tracing** and **error control**, not on
   how derivatives are computed.
2. Users can plug in **analytic derivatives** whenever they have them.

Later blocks (phase tracing and tunneling) will wrap this into more convenient
APIs so that you rarely need to call `traceMinimum` directly.

---

### `Phase` class

#### Purpose

`Phase` is a **container for one minimum** of the potential. It takes the raw
arrays from `traceMinimum` and adds:

* spline interpolation `T → x_min(T)` via `valAt(T)`,
* book-keeping for connections to other phases (second-order transitions),
* a clean place to attach additional metadata later (e.g. thermodynamic
  quantities sampled along the phase).

#### Constructor

```text
class Phase:
    def __init__(
        self,
        key,
        X: npt.NDArray[np.float64],
        T: npt.NDArray[np.float64],
        dXdT: npt.NDArray[np.float64],
    ) -> None:
        ...
```

* `key`
  Any hashable identifier. In most of the code we use integers (`0, 1, 2, …`)
  in order of discovery, but strings (or tuples) also work.
* `X` : array of shape `(n_T, n_fields)`
  Values of the minimum at each temperature.
* `T` : array, shape `(n_T,)`
  Temperatures corresponding to rows of `X`.
* `dXdT` : array, shape `(n_T, n_fields)`
  Derivative of the minimum w.r.t. temperature. Currently this is mostly used
  for diagnostics, but later blocks can exploit it for thermodynamic
  derivatives.

During initialization:

1. The arrays are **sorted in `T`**, to guard against rare cases where the
   tracer steps “backwards” in temperature.

2. A **B-spline** is built using `scipy.interpolate.splprep`:

   ```text
   k = 3 if T.size > 3 else 1  # cubic if enough points, otherwise linear
   tck, _ = interpolate.splprep(X.T, u=T, s=0.0, k=k)
   self.tck = tck
   ```

   This is the core representation used by `valAt`.

3. Two sets are initialized:

   * `low_trans` : phases connected by (approx.) second-order transitions at
     **lower** temperatures.
   * `high_trans` : phases connected at **higher** temperatures.

   These sets are filled by `addLinkFrom` once other phases are known.

#### `valAt` method

```text
def valAt(self, T, deriv: int = 0) -> npt.NDArray[np.float64]:
    """
    Evaluate x_min(T) or its T-derivatives using the stored spline.
    """
```

* `T` can be scalar or array-like.
* `deriv` chooses the derivative order with respect to T:

  * `deriv=0` → returns `x_min(T)`,
  * for cubic splines, `deriv=1, 2, 3` are also available via `splev`.

Return shape:

* scalar `T` → `(n_fields,)`,
* array `T` of length `n_T` → `(n_T, n_fields)`.

Example:

```text
phase = Phase(key=0, X=res.X, T=res.T, dXdT=res.dXdT)

# Field value at T=100
phi_100 = phase.valAt(100.0)  # shape (n_fields,)

# Values on a grid
T_grid = np.linspace(50.0, 200.0, 100)
phi_grid = phase.valAt(T_grid)  # shape (100, n_fields)
```

Later, when we implement thermodynamic helpers, you can imagine patterns like:

```text
# E.g. evaluate potential along the traced phase
V_along_phase = V(phase.valAt(T_grid), T_grid)
```

#### `addLinkFrom`: second-order transition links

```python
def addLinkFrom(self, other_phase: "Phase") -> None:
    """
    Register a link from other_phase to this phase, and infer second-order
    transitions from the overlap of their T-ranges.
    """
```

The logic is purely based on the **temperature ranges**:

* If this phase lives only at **lower T** than `other_phase`:

  * `min(self.T) >= max(other_phase.T)` → this is a candidate **low-T child**;
  * we record `other_phase.key` in `self.low_trans`.
* If this phase lives only at **higher T**:

  * `max(self.T) <= min(other_phase.T)` → this is a candidate **high-T parent**;
  * we record `other_phase.key` in `self.high_trans`.

The method is symmetric: when adding a link from `other_phase` to `self`, it
also updates the **reverse** sets on `other_phase`. This ensures that the phase
graph remains consistent as we trace multiple minima and discover where they
connect.

This is used later to:

* distinguish **first-order** transitions (tunneling) from
* **second-order**/continuous transitions (where minima merge without a barrier).

#### `__repr__`

The `__repr__` implementation is a debugging helper:

* it prints `key`, a compact preview of `X`, `T`, and `dXdT`,
* formatting is shortened to avoid flooding logs.

It is not meant to be parsed; it is just handy when exploring trajectories in
an interactive session.

---

### `traceMultiMin` – tracing the full phase structure

```text
def traceMultiMin(
    f: Callable[[np.ndarray, float], float],
    d2f_dxdt: Callable[[np.ndarray, float], np.ndarray],
    d2f_dx2: Callable[[np.ndarray, float], np.ndarray],
    points: Sequence[Tuple[np.ndarray, float]],
    tLow: float,
    tHigh: float,
    deltaX_target: float,
    dtstart: float = 1e-3,
    tjump: float = 1e-3,
    forbidCrit: Optional[Callable[[np.ndarray], bool]] = None,
    single_trace_args: Optional[Mapping[str, object]] = None,
    local_min_args: Optional[Mapping[str, object]] = None,
) -> Dict[Hashable, Phase]:
    ...
```

`traceMultiMin` is the **workhorse** for reconstructing the phase structure of a finite-temperature potential. 
Where `traceMinimum` follows a *single* minimum as a function of temperature, `traceMultiMin`:

* Starts from a set of seed minima `(x, T)`,
* Traces them **up and down in temperature** with `traceMinimum`,
* Looks for new minima that appear when a phase disappears,
* Organizes each continuous branch of minima into a `Phase` object,
* Returns a dictionary of all discovered phases.

Conceptually, you can think of it as building a **graph of phases** in the `(field, T)` plane, starting from a few trusted minima and exploring outwards.

#### Inputs and interpretation

* **`f(x, T)`**
  Same function used by `traceMinimum`: a scalar potential (or free energy) as a function of field configuration and temperature.

* **`d2f_dxdt(x, T)` and `d2f_dx2(x, T)`**
  Second derivatives needed by `traceMinimum`:

  * `d2f_dxdt` is the derivative of `∂f/∂x` with respect to `T`.
  * `d2f_dx2` is the Hessian matrix with respect to `x`.

  In practice you will often build these using the generic derivative utilities
  (e.g. `gradientFunction`, `hessianFunction`) so that the **thermal module** does not have to worry about finite differences.

* **`points`** – initial seeds
  A sequence of initial guesses:

  ```text
  points = [
      (x_min_1, T_1),
      (x_min_2, T_2),
      ...
  ]
  ```

  Each `x_min_i` must already be close to a local minimum of `f(·, T_i)`. `traceMultiMin` will refine them with a local minimizer before tracing.

* **Temperature window `tLow`, `tHigh`**
  All phases are traced only within this interval. Any seed outside this window (or pointing “outwards” at the edge) is discarded.

* **`deltaX_target`**
  Target step size in field space (same philosophy as in `traceMinimum`):

  * Sets the tolerance for the internal minimizations,
  * Controls when two minima are considered “the same phase” (redundancy check),
  * Governs how aggressively we search for *new* minima between existing ones.

* **`dtstart` and `tjump` (dimensionless)**
  These are specified **relative** to the total temperature span:

  ```text
  dt_abs   = dtstart * (tHigh - tLow)
  tjump_abs = tjump   * (tHigh - tLow)
  ```

  * `dtstart` → initial step size used by `traceMinimum`.
  * `tjump`   → how far in temperature we jump beyond the last point of a trace when we look for new phases.

  Physically: `tjump` should be **small enough** not to skip narrow windows where new phases appear, but **large enough** to avoid excessive redundant searches.

* **`forbidCrit(x)`** (optional)
  A predicate that returns `True` when a point in field space is “forbidden” (e.g. unphysical, outside a trusted domain, or numerically problematic).
  If either endpoint of a traced branch violates this predicate, the entire `Phase` is discarded and all follow-up seeds spawned from it are removed.

* **`single_trace_args`**
  Extra keyword arguments passed directly to `traceMinimum` (e.g. `dtabsMax`, `dtfracMax`). This is where you tune the **adaptive T-stepper** while keeping the logic of `traceMultiMin` simple.

* **`local_min_args`**
  Extra keyword arguments to `findApproxLocalMin`, such as the number of sample points `n` or the edge trimming fraction `edge`. The `args` entry is **ignored** here and always set internally (we use it to pass the appropriate temperature).

#### Algorithmic flow

For each seed `(x_seed, T_seed)`:

1. **Refine to a true minimum** at `T_seed` using a small-tolerance `fmin` wrapper.
2. Reject if:

   * It lies outside `[tLow, tHigh]` or points outwards at the boundary, or
   * `forbidCrit(x_seed)` is True.
3. **Check redundancy**:

   * Compare this minimum with the minima of existing `Phase` objects at the same temperature,
   * If `‖x_phase(T_seed) − x_seed‖ < 2 * deltaX_target`, we treat it as already covered and optionally update links between phases.
4. If **not redundant**, we:

   * Call `traceMinimum` downward in T (if `T_seed > tLow`),
   * Call `traceMinimum` upward in T (if `T_seed < tHigh`),
   * From each end (`overX`, `overT`) we:

     * Jump by `±tjump_abs` in temperature,
     * Minimize again to get a new seed,
     * Use `findApproxLocalMin` between the last point of the trace and this new seed to see if **intermediate minima** exist.
       Each candidate is minimized and added to the queue of seeds to be explored.
5. Finally, we **join the down/up traces**:

   * Only down → use that branch,
   * Only up → use that branch,
   * Both → concatenate, avoiding duplication at the joint.
6. If the resulting branch has more than one point and passes `forbidCrit` at both ends:

   * Construct a `Phase` object with a new key,
   * Connect it to its “parent” phase via `Phase.addLinkFrom` when appropriate.
7. Loop until the queue of seeds is empty.

The result is a dictionary:

```text
phases: Dict[Hashable, Phase] = {
    0: Phase(...),
    1: Phase(...),
    ...
}
```

Each `Phase` encodes a **continuous branch of minima** over some T-interval, plus connectivity information (`low_trans`, `high_trans`) that will be used later to build thermal histories and identify second-order transitions.

---

### `findApproxLocalMin` – detecting intermediate phases along a line

```text
def findApproxLocalMin(
    f: Callable[..., np.ndarray],
    x1: np.ndarray,
    x2: np.ndarray,
    args: Tuple[object, ...] = (),
    n: int = 100,
    edge: float = 0.05,
) -> np.ndarray:
    ...
```

`findApproxLocalMin` is a **lightweight detector of “missing” phases** between two known minima. It is meant to be cheap and robust, not perfect.

Given two points `x1` and `x2` in field space, it:

1. Constructs a straight line between them

   $$
   x(t) = x_1 + t(x_2 - x_1); \qquad t \in [\text{edge}, 1-\text{edge}],
   $$

2. Samples `f(x(t), *args)` at `n` evenly spaced values of `t`,

3. Identifies *discrete* local minima along this 1D grid (`y[k]` lower than its neighbors),

4. Returns the corresponding positions `x(t_k)`.

These approximate minima are then **polished** by a real minimizer in `traceMultiMin`.

#### Important details

* **Vectorized evaluation**
  The function `f` must accept an array of points with shape `(n_points, ndim)` and return an array of values with shape `(n_points,)`. 
This is consistent with how many of the CosmoTransitions internals are written: you get good performance by avoiding Python loops at this level.

* **`edge` parameter**
  Controls how close to the endpoints the search goes:

  * `edge = 0.0` → search from exactly `x1` to `x2`,
  * `edge = 0.05` (default) → ignore the first and last 5% of the segment,
  * `edge = 0.5` → only probe the midpoint.

  This is useful to avoid re-detecting minima that are already known at the endpoints.

* **Return value**
  An array of shape `(n_min, ndim)`; it may be empty if no interior minima are found.

Physically, you can interpret `findApproxLocalMin` as a **sanity check** when we jump in temperature and see that the endpoint of one branch and the seed of another branch are not the same. It preserves the possibility that there was a *third* phase in between, which would otherwise be skipped.

---

### `_removeRedundantPhase` and `removeRedundantPhases` – cleaning up the phase graph

```text
def _removeRedundantPhase(
    phases: MutableMapping[Hashable, Phase],
    removed_phase: Phase,
    redundant_with_phase: Phase,
) -> None:
    ...
```

```text
def removeRedundantPhases(
    f: Callable[[np.ndarray, float], float],
    phases: MutableMapping[Hashable, Phase],
    xeps: float = 1e-5,
    diftol: float = 1e-2,
) -> None:
    ...
```

Even with careful redundancy checks inside `traceMultiMin`, it is still possible that the same physical phase is represented twice:

* e.g. because you seeded from two different points,
* or because numerical noise leads to slightly different traces that later converge.

`removeRedundantPhases` is a **post-processing cleanup** step that:

1. Scans all pairs of phases,
2. Checks whether their temperature ranges overlap,
3. If they overlap, compares their minima at the ends of the overlap (`T_min`, `T_max`),
4. If they are within `diftol` at **both** ends, treats them as the same phase and merges them.

#### Refining before comparison

Before comparing minima, the function uses a small-tolerance local minimizer:

```text
def fmin(x, t):
    return optimize.fmin(
        f, x, args=(t,), xtol=xeps, ftol=np.inf, disp=False
    )
```

so that:

* if a phase was constructed by spline interpolation at some temperature,
* we still compare **true** minima of `f(·, T)` rather than spline artifacts.

This helps avoid spurious mismatches in the presence of small numerical errors.

#### Merging logic

There are two main cases:

1. **Two phases completely overlap and are indistinguishable**
   Then one is kept, the other is removed by `_removeRedundantPhase`, which:

   * Transfers all `low_trans` and `high_trans` connections from the removed phase to the surviving one,
   * Updates neighbor phases accordingly,
   * Deletes the redundant entry from the `phases` dictionary.

2. **Two phases coincide at both ends of their overlapping interval but cover different T ranges**
   In this case we:

   * Take the lower-T part from the phase that extends further to low temperatures,
   * Take the higher-T part from the phase that extends further to high temperatures,
   * Stitch them into a new `Phase` object with a combined key (e.g. `"1_3"`),
   * Insert this new phase into the dictionary and remove the old two via `_removeRedundantPhase`.

If phases coincide at **only one end** of their overlap, the situation is more subtle (you would need to find the exact splitting point and re-spline). This case is currently **not implemented on purpose**, and you get a clear `NotImplementedError` with an explanatory message. This is safer than silently doing something ambiguous.

#### When to call `removeRedundantPhases`

A typical usage pattern is:

```text
phases = traceMultiMin(...)
removeRedundantPhases(V, phases)
```

before you proceed to:

* building thermal histories,
* finding tunneling paths,
* or computing the full phase transition sequence.

Physically, this ensures that each branch in `phases` really corresponds to a **distinct vacuum / phase** rather than a numerically duplicated copy.

---

### `getStartPhase` – identifying the high-temperature phase

```text
def getStartPhase(
    phases: Mapping[Hashable, Phase],
    V: Optional[Callable[[np.ndarray, float], float]] = None,
) -> Hashable:
    ...
```

`getStartPhase` is a small but conceptually important helper: it answers

> “Which phase should I start from at high temperature?”

This is needed, for example, by `findAllTransitions`, which will later walk through the thermal history starting from the high-T phase and following successive transitions as the Universe cools.

#### Selection rule

1. Among all phases in the dictionary, find those with the **largest** `T_max = phase.T[-1]`.
   These are the phases that survive to the highest temperatures in your scan.

2. If there is a **single** such phase, or if no potential `V` is provided:

   * Return its key directly.

3. If there are **several** phases with the same `T_max`, and `V(x, T)` is provided:

   * Evaluate the potential at their high-T endpoints: `V(phase.X[-1], phase.T[-1])`,
   * Select the phase with the **lowest** potential there (thermodynamically favored at high T).

The returned key is guaranteed to exist in the input mapping.

#### Physical interpretation

In a typical cosmological application:

* `getStartPhase` identifies the phase in which the Universe sits at very high temperature (e.g. the **symmetric phase** with vanishing VEV),
* If multiple candidate phases survive to the same temperature (e.g. due to approximate symmetries), the potential is used to pick the **true** equilibrium one.

Once this starting phase is known, the rest of the module (in later blocks):

* Follows how it transitions to lower-T phases (first-order or second-order),
* Finds nucleation temperatures and actions for tunneling,
* Builds the full sequence of phase transitions.

---

These functions together complete **Block A**’s role:

* `traceMinimum` and `Phase` handle **single** branch tracing and interpolation,
* `traceMultiMin` and `findApproxLocalMin` build the **full phase graph**,
* `removeRedundantPhases` cleans that graph,
* `getStartPhase` picks the **entry point** at high temperature.

In the next blocks, this phase structure will be the backbone for:

* Determining critical temperatures,
* Solving for bounce solutions and actions between phases,
* Extracting thermodynamic and gravitational-wave–relevant quantities from the transition history.


---


## How Block A fits into the bigger picture

Even though Block A knows nothing about nucleation rates, `S(T)`, α, β, or
gravitational waves, it is the **geometric backbone** of the whole pipeline:

1. For each phase, we use `traceMinimum` to get `X(T)` and wrap it in `Phase`.
2. A later block (`traceMultiMin`) will:

   * discover multiple phases,
   * use `Phase.addLinkFrom` and other heuristics to build a phase graph.
3. The tunneling block will:

   * pick two `Phase` objects,
   * at each temperature, find their minima,
   * solve for the instanton and compute `S_E(T)` using `tunneling1D`.
4. Thermodynamic & GW blocks will:

   * use the traced `X(T)` to evaluate `V(φ_min(T), T)` and its derivatives,
   * build `S(T)/T`, α, β/H, etc., and finally
   * predict GW spectra from the transition history.

All of that starts here with a robust and well-understood implementation of
minimum tracing and the `Phase` abstraction.


