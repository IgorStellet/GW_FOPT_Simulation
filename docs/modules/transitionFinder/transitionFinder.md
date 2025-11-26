# Transition finder

## Block A — Following minima and building phases

In this section we focus on the “kinematics” of phases as a function of temperature:

* **what the code does numerically** (so you can read it without fear), and
* **what it means physically** (as if a sharp physicist were staring at a phase diagram).

We follow this order:

1. Physical overview of the module.
2. `traceMinimum`.
3. `Phase`.
4. `traceMultiMin`.
5. `findApproxLocalMin`.
6. `_removeRedundantPhase` and `removeRedundantPhases`.
7. `getStartPhase`.

---

### 1. Physical overview of the module

Context: you have a finite-temperature effective potential $V(\phi, T)$ (written as $f(x,T)$ in the code), possibly with several field components.

* For each fixed (T) there are **local minima** (phases) — some stable, some metastable.
* As (T) changes:

  * those minima move in field space,
  * some disappear (turn into saddles or spinodals),
  * new minima can appear (new phases),
  * in some temperature ranges two (or more) phases coexist and phase transitions become possible.

The `transitionFinder` module bridges between:

* the **microscopic potential** $V(\phi, T)$, and
* the **macroscopic phase history**: which minima exist at each T, how they connect, and which are involved in 1st/2nd-order transitions.

The pieces in Block A are:

* `traceMinimum` + `Phase`: follow **a single minimum** as a function of temperature.
* `traceMultiMin`: reconstruct **all relevant phases** in a given temperature interval and link them.
* `findApproxLocalMin`: detect intermediate minima that may correspond to new phases.
* `removeRedundantPhases`: “deduplicate” multiple copies of the same phase.
* `getStartPhase`: choose the high-temperature phase.

Physically, this gives you something like a **1D phase diagram in T**, but with the full information about the location of the minimum in $\phi$ and how it evolves.

---

### 2. `traceMinimum`: following a minimum as T changes

#### 2.1. The mathematical problem

We want to follow a solution $x_{\min}(T)$ such that

$$
\frac{\partial f}{\partial x}\bigl(x_{\min}(T), T\bigr) = 0
$$

where $x \in \mathbb{R}^{N_\text{fields}}$.

Differentiating this condition with respect to T:

$$
\frac{d}{dT} \left( \frac{\partial f}{\partial x} \right)
= \frac{\partial^2 f}{\partial x^2}\frac{dx_{\min}}{dT}+ \frac{\partial}{\partial T}\left(\frac{\partial f}{\partial x}\right)
  = 0.
$$

Define

* $H = \partial^2 f / \partial x^2$ (the Hessian),
* $b = \partial/\partial T \left(\partial f/\partial x\right)$.

Then we have the implicit equation

$$
H \cdot \frac{dx_{\min}}{dT} = -b.
$$

If the minimum is well defined (Hessian invertible and positive definite), we can write

$$
\frac{dx_{\min}}{dT} = -H^{-1} b.
$$

`traceMinimum` acts as an ODE integrator in T for this equation, but with **periodic corrections** using numerical minimization, 
so that it stays glued to the actual minimum of (f).

---

#### 2.2. Signature and main inputs

```text
def traceMinimum(
    f,
    d2f_dxdt,
    d2f_dx2,
    x0,
    t0,
    tstop,
    dtstart,
    deltaX_target,
    dtabsMax=20.0,
    dtfracMax=0.25,
    dtmin=1e-3,
    deltaX_tol=1.2,
    minratio=1e-2,
) -> TraceMinimumResult:
```

Physical reading:

* `f(x, T)`: the effective potential $V(\phi, T)$.
* `d2f_dxdt(x, T)`: $\partial/\partial T (\partial f / \partial x)$, the right-hand side in the linear system.
* `d2f_dx2(x, T)`: Hessian $H = \partial^2 f / \partial x^2$.
* `x0`, `t0`: a known minimum at some initial temperature (you must start at a minimum).
* `tstop`: final temperature you want to reach.
* `dtstart`: initial step in T (its sign sets the direction: positive → going to higher (T); negative → to lower (T)).
* `deltaX_target`: target displacement in field space per accepted step (the adaptive step controller tries to keep the actual motion “of this order”).

The other arguments are safety knobs for the integrator:

* `dtabsMax`, `dtfracMax`: cap the maximum allowed $|\Delta T|$.
* `dtmin`: if $|\Delta T|$ becomes smaller than this, the routine gives up.
* `deltaX_tol`: relative tolerance to decide whether a step is “good”.
* `minratio`: criterion to say “the Hessian is becoming degenerate/negative → we are near a saddle/instability”.

---

#### 2.3. Main steps in the code

High-level view of the internal logic:

##### 2.3.1. Initial setup

```text
x = np.atleast_1d(np.asarray(x0, dtype=float))
Ndim = x.size

M0 = np.asarray(d2f_dx2(x, t0), dtype=float)
eig0 = linalg.eigvalsh(M0)
...
minratio = float(minratio) * float(
    np.min(np.abs(eig0)) / np.max(np.abs(eig0))
)
```

* Convert `x0` into a 1D array: uniform handling of one or many fields.
* Compute the initial Hessian and its eigenvalues.
* If all eigenvalues are zero, the minimum is singular → error.
* **Rescale `minratio`** according to the local conditioning:

  * if the Hessian is already poorly conditioned, the “almost singular” threshold is relaxed;
  * if it is very well conditioned, you can afford a stricter ratio before declaring trouble.

Physical interpretation: this is a quasi-theoretical check that you are really at a **stable minimum** (all eigenvalues positive and not tiny).

---

##### 2.3.2. Auxiliary function `dxmindt`

```text
def dxmindt(x_now, t_now):
    M = np.asarray(d2f_dx2(x_now, t_now), dtype=float)
    if np.abs(linalg.det(M)) < (1e-3 * np.max(np.abs(M))) ** Ndim:
        return None, False
    b = -np.asarray(d2f_dxdt(x_now, t_now), dtype=float).reshape(Ndim)
    eigs = linalg.eigvalsh(M)
    try:
        dxdt_local = linalg.solve(M, b, overwrite_a=False, overwrite_b=False)
        isneg = (eigs <= 0).any() or (
            np.min(eigs) / np.max(eigs) < minratio
        )
    except linalg.LinAlgError:
        dxdt_local = None
        isneg = False
    return dxdt_local, isneg
```

What it does:

1. Grab the Hessian `M` at `(x_now, T_now)`.
2. Rough singularity check via determinant:

   * if `det(M)` is smaller than a threshold → return `dxdt=None`.
3. Form `b = - d2f_dxdt` as the right-hand side.
4. Solve `M * dxdt_local = b` for `dx/dT`.
5. Compute eigenvalues `eigs` and define

   * `isneg = True` if there is any non-positive eigenvalue or if `min/max` falls below `minratio`, signaling a nearly flat/negative direction.

Physical meaning:

* When `dxdt` exists and all eigenvalues are positive and reasonable, you are in a **well-defined phase**, with positive mass squared in every direction.
* If any eigenvalue becomes zero or negative:

  * a massless or tachyonic mode appears → you are at the **end of that phase** (spinodal) or at a second-order/continuous transition.

---

##### 2.3.3. Local minimization `fmin`

```text
xeps = float(deltaX_target) * 1e-2

def fmin(x_guess, t_now):
    x_guess = np.asarray(x_guess, dtype=float).reshape(Ndim)
    res = optimize.fmin(
        f,
        x_guess,
        args=(t_now,),
        xtol=xeps,
        ftol=np.inf,
        disp=False,
    )
    return np.asarray(res, dtype=float).reshape(Ndim)
```

* Given an initial guess `x_guess` (coming from the ODE integration), this function slides down to the actual local minimum of `f` at that temperature.
* `xtol = xeps`: position tolerance proportional to `deltaX_target`.
* `ftol = np.inf`: the minimizer does not care about the value of `f`, only about convergence in (x).

**Why re-minimize?**

Because integrating only the ODE `dx/dT = -H^{-1} b` accumulates error: you are following an approximate equation. The `fmin` correction keeps the trajectory stuck to the true minimum at each step.

---

##### 2.3.4. Scales and initial state

```text
tscale = abs(float(dtstart))
dtabsMax = float(dtabsMax) * tscale
dtmin = float(dtmin) * tscale
deltaX_tol_abs = float(deltaX_tol) * float(deltaX_target)

t = float(t0)
dt = float(dtstart)
dxdt, negeig = dxmindt(x, t)

X_list = [x.copy()]
T_list = [t]
dXdT_list = [np.zeros_like(x) if dxdt is None else dxdt.copy()]
overX = x.copy()
overT = t
```

* Convert `dtstart` into an absolute scale and define:

  * `dtabsMax` = maximum allowed $|\Delta T|$,
  * `dtmin` = minimum allowed $|\Delta T|$.

* Compute the first `dxdt` at the initial point.

* Initialize the output lists with the starting point.

---

##### 2.3.5. Main loop in T

Skeleton of the update:

```text
while dxdt is not None:
    tnext = t + dt
    x_pred = x + dxdt * dt
    xnext = fmin(x_pred, tnext)
    dxdt_next, negeig = dxmindt(xnext, tnext)

    if dxdt_next is None or negeig:
        dt *= 0.5
        overX, overT = xnext.copy(), float(tnext)
    else:
        err1 = np.linalg.norm(x + dxdt * dt - xnext)
        err2 = np.linalg.norm(xnext - dxdt_next * dt - x)
        xerr = max(err1, err2)

        if xerr < deltaX_tol_abs:
            # accept step
            ...
        else:
            # step too aggressive
            dt *= 0.5
            overX, overT = xnext.copy(), float(tnext)
    ...
```

Step-by-step:

1. **Prediction**:

   * `tnext = t + dt`,
   * `x_pred = x + dxdt * dt`: ODE prediction.

2. **Minimum correction**:

   * `xnext = fmin(x_pred, tnext)` refines to the nearest minimum at the new temperature.

3. **New dx/dT**:

   * `dxdt_next, negeig = dxmindt(xnext, tnext)`.

4. **If there is trouble** (`dxdt_next is None` or `negeig`):

   * reduce the step: `dt *= 0.5`,
   * update `overX, overT` to this “problematic” point (where the minimum disappears or becomes unstable).

5. **If everything is OK**, estimate the error:

   * `err1`: difference between the ODE prediction `x + dxdt * dt` and the refined minimum `xnext`;

   * `err2`: consistency check of going forward and then backward in T using local slopes;

   * `xerr = max(err1, err2)`, compare with `deltaX_tol_abs`.

   * If `xerr` is small → **accept the step**:

     * append `tnext`, `xnext`, `dxdt_next` to the lists,

     * adapt `dt`:

       ```text
       dt *= deltaX_target / (xerr + 1e-100)
       ```

     * update `(x, t, dxdt)` to the new point.

   * If `xerr` is large → reject the step, cut `dt` in half.

6. **Stopping criteria**:

   * If `|dt| < dtmin` → we found a transition (or at least a point where further evolution is impractical) and stop.
   * If we are crossing or reaching `tstop`, force a final step exactly up to `tstop` and then exit.
   * If `|dt|` grows beyond the allowed maximum, clip it back.

At the end, the lists `X_list`, `T_list`, `dXdT_list` are turned into arrays and returned together with `overX`, `overT`.

---

#### 2.4. Physical interpretation of `traceMinimum`

Think of a simple temperature-dependent double-well:

* at very high T, only a minimum at $\phi = 0$ exists;
* as you cool down, broken-symmetry minima at $\phi \neq 0$ appear;
* following one of these minima, you reach a point where the solution ceases to exist (spinodal) or the Hessian develops a negative mode (loss of stability).

`traceMinimum`:

* gives you the curve $x_{\min}(T)$ for **one** of these minima;
* tells you **where it ceases to be stable/metastable**, via `overT`;
* this curve is the basis for

  * building $V(\phi_{\min}(T), T)$,
  * computing thermodynamic quantities along the phase,
  * providing boundary conditions for tunneling (e.g. in `tunneling1D`).

---

### 3. `Phase`: encapsulating a minimum as a physical object

`Phase` takes the output of `traceMinimum` and packages it as an object that holds

* the arrays `X(T)`, `T`, `dXdT`,
* a spline $T \mapsto X(T)$,
* and second-order links to other phases (`low_trans` / `high_trans`).

#### 3.1. Construction

```text
class Phase:
    def __init__(self, key, X, T, dXdT) -> None:
        self.key = key
        T = np.asarray(T, dtype=float).reshape(-1)
        X = np.asarray(X, dtype=float)
        dXdT = np.asarray(dXdT, dtype=float)

        order = np.argsort(T)
        self.T = T[order]
        self.X = X[order]
        self.dXdT = dXdT[order]

        k = 3 if self.T.size > 3 else 1
        tck, _ = interpolate.splprep(self.X.T, u=self.T, s=0.0, k=k)
        self.tck = tck

        self.low_trans: set = set()
        self.high_trans: set = set()
```

Details:

* Everything is sorted by temperature (to fix any small ordering issues from the tracing routine).
* A parametric B-spline is constructed, $T \mapsto X(T)$, via `splprep`:

  * `k=3` (cubic) when enough points are available,
  * otherwise `k=1` (linear).
* `low_trans` and `high_trans` are initialized as empty sets.

---

#### 3.2. `valAt`: inspecting the phase at arbitrary T

```text
def valAt(self, T, deriv: int = 0) -> np.ndarray:
    if deriv < 0:
        raise ValueError("deriv must be non-negative.")
    T_arr = np.asanyarray(T)
    y = interpolate.splev(T_arr, self.tck, der=deriv)
    arr = np.asanyarray(y).T
    return arr
```

* `deriv = 0`: returns $x_{\min}(T)$.
* `deriv = 1`: returns $dx_{\min}/dT$ from the spline (a smooth approximation to the discrete `dXdT`).
* For scalar T, the result has shape `(n_fields,)`; for an array of temperatures, `(n_T, n_fields)`.

Physical use case:

* This allows you to evaluate, for instance, $V(\phi_{\min}(T), T)$ on a fine (T) grid without re-minimizing:

  ```text
  T_grid = np.linspace(50.0, 200.0, 500)
  phi_grid = phase.valAt(T_grid)
  V_grid = V(phi_grid, T_grid)
  ```

---

#### 3.3. `addLinkFrom`: marking second-order connections

```text
def addLinkFrom(self, other_phase: "Phase") -> None:
    if np.min(self.T) >= np.max(other_phase.T):
        self.low_trans.add(other_phase.key)
        other_phase.high_trans.add(self.key)
    if np.max(self.T) <= np.min(other_phase.T):
        self.high_trans.add(other_phase.key)
        other_phase.low_trans.add(self.key)
```

Logic:

* If this phase only exists at **lower** temperatures than `other_phase` (i.e. it starts where the other ends), then:

  * `self.low_trans` includes `other_phase.key`,
  * `other_phase.high_trans` includes `self.key`.

* If this phase only exists at **higher** temperatures, the opposite happens.

This is a criterion purely based on **temperature overlap**: when two phases “touch” in temperature without a window of coexistence with a barrier, they are candidates for a **second-order/continuous transition**.

Physically:

* You build a graph of phases:

  * edges in `low_trans` / `high_trans` represent smooth connections (no barrier),
  * first-order transitions (with tunneling) will be handled elsewhere (e.g. via `tunneling1D`).

---

### 4. `traceMultiMin`: reconstructing all phases

`traceMinimum` follows **one** minimum. `traceMultiMin` orchestrates this into a global algorithm that:

* starts from several seeds,
* traces each phase up and down in temperature,
* when a phase disappears, looks for new minima nearby,
* and builds a dictionary of `Phase` objects.

#### 4.1. Key inputs

```text
def traceMultiMin(
    f,
    d2f_dxdt,
    d2f_dx2,
    points,
    tLow,
    tHigh,
    deltaX_target,
    dtstart=1e-3,
    tjump=1e-3,
    forbidCrit=None,
    single_trace_args=None,
    local_min_args=None,
) -> Dict[Hashable, Phase]:
```

* `points`: list of seeds $(x_\text{seed}, T_\text{seed})$:

  * each should be near a minimum — typically minima at very high and very low temperatures.

* `tLow`, `tHigh`: temperature interval of interest.

* `deltaX_target`: same philosophy as in `traceMinimum`.

* `dtstart`, `tjump`: **fractions** of `(tHigh - tLow)`:

  * `dtstart_abs = dtstart * (tHigh - tLow)` is the initial temperature step for `traceMinimum`,
  * `tjump_abs = tjump * (tHigh - tLow)` is the temperature offset beyond the end of a phase used for probing new minima.

* `forbidCrit(x)`: function that says “this point in field space is forbidden”.

* `single_trace_args`: extra keyword arguments forwarded to `traceMinimum`.

* `local_min_args`: extra keyword arguments for `findApproxLocalMin` (except `args`, which is always the temperature).

---

#### 4.2. High-accuracy local minimization

Internally there is another `fmin`, now with even tighter tolerances:

```text
xeps = deltaX_target * 1e-2

def fmin(x, t):
    x = np.asarray(x, dtype=float)
    xmin = optimize.fmin(
        f,
        x + xeps,
        args=(t,),
        xtol=xeps * 1e-3,
        ftol=np.inf,
        disp=False,
    )
    return np.asarray(xmin, dtype=float)
```

Note the small shift `x + xeps`: this can help escape pathological flat points.

---

#### 4.3. Seed queue (`next_points`)

```text
phases: Dict[Hashable, Phase] = {}
next_points: List[List[Any]] = []

for x_seed, t_seed in points:
    x_seed = np.asarray(x_seed, dtype=float)
    next_points.append([float(t_seed), dtstart_abs, fmin(x_seed, t_seed), None])
```

Each element of `next_points` is:

1. `T_current` – temperature of the seed.
2. `dt_current` – base step to use in `traceMinimum` from that seed.
3. `x_current` – refined minimum at that temperature.
4. `linked_from_key` – which phase (if any) generated this seed (for second-order linking).

---

#### 4.4. Exploration loop

The main `while next_points:` loop does:

1. **Pop a seed** from the queue.

2. Refine again `x1 = fmin(x1, t1)` to ensure we really start at a minimum.

3. Check if:

   * it lies within `[tLow, tHigh]`,
   * it is allowed by `forbidCrit`.

4. **Check redundancy** against phases already found:

   ```text
   for key, phase in phases.items():
       ...
       x_phase = fmin(phase.valAt(t1), t1)
       if np.linalg.norm(x_phase - x1) < 2 * deltaX_target:
           # already covered
   ```

   If some existing `Phase` has almost the same minimum at `t1`, this is not a new phase; the seed can be discarded (or used only to adjust links).

5. If not redundant:

   * print `Tracing phase starting at ...`,
   * set `phase_key = len(phases)` for the new phase,
   * trace that phase **downwards** and **upwards** in T using `traceMinimum`.

---

#### 4.5. Tracing to lower and higher T

**Downwards:**

```text
if t1 > tLow:
    down_trace = traceMinimum(
        f=f,
        d2f_dxdt=d2f_dxdt,
        d2f_dx2=d2f_dx2,
        x0=x1,
        t0=t1,
        tstop=tLow,
        dtstart=-abs(dt1),
        deltaX_target=deltaX_target,
        **single_trace_kwargs,
    )
    X_down, T_down, dXdT_down = down_trace.X, down_trace.T, down_trace.dXdT
    nX_down, nT_down = down_trace.overX, down_trace.overT

    t2 = nT_down - tjump_abs
    dt2 = 0.1 * tjump_abs
    x2 = fmin(nX_down, t2)
    next_points.append([t2, dt2, x2, phase_key])

    if np.linalg.norm(X_down[-1] - x2) > deltaX_target:
        for point in findApproxLocalMin(
            f,
            X_down[-1],
            x2,
            args=(t2,),
            **local_min_kwargs,
        ):
            next_points.append([t2, dt2, fmin(point, t2), phase_key])

    X_down = X_down[::-1]
    T_down = T_down[::-1]
    dXdT_down = dXdT_down[::-1]
```

Physical picture:

* You take the minimum at `T = t1` and follow it down to `T = tLow`.

* `traceMinimum` stops when the phase ceases to exist (Hessian issues, etc.).

* The point `overX, overT` is where the phase “dies”.

* You then step further in temperature (`t2 = overT - tjump_abs`), minimize again, and use this as a new seed.

  * If a new phase has nucleated beyond the end of the old one, this seed will fall into its minimum.

* `findApproxLocalMin` between `X_down[-1]` and `x2` searches for additional minima along that straight segment (potential **hidden intermediate phases**).

**Upwards:**

The upward block is analogous:

```text
if t1 < tHigh:
    up_trace = traceMinimum(
        ...,
        tstop=tHigh,
        dtstart=+abs(dt1),
        ...
    )
    X_up, T_up, dXdT_up = up_trace.X, up_trace.T, up_trace.dXdT
    nX_up, nT_up = up_trace.overX, up_trace.overT

    t2 = nT_up + tjump_abs
    dt2 = 0.1 * tjump_abs
    x2 = fmin(nX_up, t2)
    next_points.append([t2, dt2, x2, phase_key])

    if np.linalg.norm(X_up[-1] - x2) > deltaX_target:
        for point in findApproxLocalMin(...):
            next_points.append([t2, dt2, fmin(point, t2), phase_key])
```

---

#### 4.6. Building a `Phase` from down/up pieces

After tracing down and up:

```text
if X_down is None:      # only traced upwards
    X, T, dXdT = X_up, T_up, dXdT_up
elif X_up is None:      # only traced downwards
    X, T, dXdT = X_down, T_down, dXdT_down
else:
    # join, avoiding duplicating the pivot
    X = np.append(X_down, X_up[1:], axis=0)
    T = np.append(T_down, T_up[1:], axis=0)
    dXdT = np.append(dXdT_down, dXdT_up[1:], axis=0)
```

Then it applies the `forbidCrit` filter to the endpoints; if the phase is allowed and contains more than one point:

```text
new_phase = Phase(phase_key, X, T, dXdT)
if linked_from is not None:
    new_phase.addLinkFrom(phases[linked_from])
phases[phase_key] = new_phase
```

If the phase is forbidden, it and its descendants are treated as dead ends.

At the end of the `while` loop, `phases` is a dictionary `{key → Phase}` covering all relevant phases in `[tLow, tHigh]`, with continuous (second-order) connections encoded through `addLinkFrom`.

Physically: you get a **tree/graph of phases**. From a modest set of seeds, the algorithm explores the potential, discovering all phases connected through minima as temperature changes, and mapping how they appear and disappear.

---

### 5. `findApproxLocalMin`: hunting for intermediate phases along a line

```text
def findApproxLocalMin(
    f,
    x1,
    x2,
    args=(),
    n=100,
    edge=0.05,
) -> np.ndarray:
```

#### Physical idea

When a phase ends at `(overX, overT)` and you step slightly in T and minimize again, you may land in a new minimum far away in field space. Between those two points `x1` and `x2` there may exist a **third minimum** that you would miss by simply re-minimizing.

`findApproxLocalMin` is a cheap detector of “possible hidden phases” along the straight segment from `x1` to `x2`.

#### Algorithm

1. Ensure `x1` and `x2` have the same shape and reinterpret them as 1D vectors: `x1_vec`, `x2_vec`.

2. Build a grid in a parameter $t \in [\text{edge}, 1 - \text{edge}]$:

   ```text
   t_grid = np.linspace(edge, 1.0 - edge, n).reshape(n, 1)
   x_grid = x1_vec + t_grid * (x2_vec - x1_vec)  # (n, ndim)
   ```

   The `edge` parameter trims the endpoints:

   * `edge=0` includes the endpoints,
   * `edge=0.05` ignores the first and last 5% of the line, so you do not simply re-detect the original minima.

3. Evaluate `f` on the grid:

   * Try vectorized evaluation: `y_raw = f(x_grid, *args)`.
   * If the shape does not match or the call fails, fall back to a scalar loop `_evaluate_scalar_grid` which calls `f` point by point.
   * In the end, `y` is a 1D array of length `n`.

4. Look for discrete internal minima:

   ```text
   is_min = (y[2:] > y[1:-1]) & (y[:-2] > y[1:-1])
   minima = x_grid[1:-1][is_min]
   ```

   This is the classic finite-difference criterion: a point is a minimum if it is lower than its immediate neighbors.

5. Return an array with shape `(k, ndim)` containing the approximate positions of these minima. It may be empty.

Refinement is done later by the internal `fmin` of `traceMultiMin`.

---

### 6. `_removeRedundantPhase` and `removeRedundantPhases`: cleaning the phase graph

Despite all heuristics, you may still discover the **same physical phase** more than once (e.g. from different seeds). `removeRedundantPhases` is a post-processing step that:

* scans pairs of phases,
* checks whether they represent the same minimum over their overlapping temperature range,
* if they do, merges or discards one,
* and updates second-order links (`low_trans` / `high_trans`) consistently.

#### 6.1. `removeRedundantPhases`

```text
def removeRedundantPhases(
    f,
    phases,
    xeps=1e-5,
    diftol=1e-2,
) -> None:
```

`phases` is modified in place.

##### 6.1.1. Local minimization

Again a local `fmin` is defined:

```text
def fmin(x, t):
    xmin = optimize.fmin(
        f,
        x,
        args=(t,),
        xtol=xeps,
        ftol=np.inf,
        disp=False,
    )
    return np.asarray(xmin, dtype=float)
```

The idea is to guarantee that when comparing two phases you compare **true minima of (f)**, not just spline values.

##### 6.1.2. Loop until no redundancies remain

```text
has_redundant_phase = True
while has_redundant_phase:
    has_redundant_phase = False
    keys = list(phases.keys())

    for i in keys:
        ...
        for j in keys:
            ...
            phase1 = phases[i]
            phase2 = phases[j]

            tmax = min(phase1.T[-1], phase2.T[-1])
            tmin = max(phase1.T[0], phase2.T[0])
            if tmin > tmax:
                continue
            ...
```

For each pair `phase1`, `phase2`:

* Compute the overlap in temperature:

  * `tmin` = larger of the two Tmin’s,
  * `tmax` = smaller of the two Tmax’s.

* If there is no overlap (`tmin > tmax`), skip.

Then:

* Compare positions at `tmax` and `tmin`:

  * if the boundary matches exactly a stored point, use that,
  * otherwise interpolate via `valAt` and refine with `fmin`.

Define:

```text
dif = np.linalg.norm(x1 - x2)
same_at_tmax = dif < diftol
...
same_at_tmin = dif < diftol
```

If `same_at_tmin` and `same_at_tmax` are both true:

* the two phases are indistinguishable at both ends of their overlapping temperature interval.

---

##### 6.1.3. Merging cases

If they are redundant:

* Set `has_redundant_phase = True` so that the outer loop restarts after modifying `phases`.

* Define

  ```text
  p_low  = phase1 if phase1.T[0] < phase2.T[0] else phase2
  p_high = phase1 if phase1.T[-1] > phase2.T[-1] else phase2
  ```

* Two scenarios:

  **(a) `p_low is p_high`**: they cover essentially the same temperature range.

  * Keep one and discard the other:

    ```text
    p_reject = phase1 if p_low is phase2 else phase2
    _removeRedundantPhase(phases, p_reject, p_low)
    ```

  **(b) `p_low` and `p_high` differ**: one covers lower temperatures, the other higher temperatures.

  * Cut and stitch:

    ```text
    mask_low  = p_low.T  <= tmax
    mask_high = p_high.T >  tmax

    T = np.append(p_low.T[mask_low],  p_high.T[mask_high])
    X = np.append(p_low.X[mask_low],  p_high.X[mask_high], axis=0)
    dXdT = np.append(p_low.dXdT[mask_low], p_high.dXdT[mask_high], axis=0)

    new_key = f"{p_low.key}_{p_high.key}"
    new_phase = Phase(new_key, X, T, dXdT)
    phases[new_key] = new_phase

    _removeRedundantPhase(phases, p_low,  new_phase)
    _removeRedundantPhase(phases, p_high, new_phase)
    ```

  You create a “stitched” phase covering the whole temperature range and replace the two old phases.

If two phases coincide at only one end of the overlap, you hit:

```text
elif same_at_tmin or same_at_tmax:
    raise NotImplementedError(...)
```

This makes it explicit that the ambiguous “touching at one end only” case is not handled automatically (it would require more sophisticated logic to split and stitch correctly).

---

#### 6.2. `_removeRedundantPhase`: reconnecting the graph

```python
def _removeRedundantPhase(phases, removed_phase, redundant_with_phase):
    for key in removed_phase.low_trans:
        if key != redundant_with_phase.key:
            p = phases[key]
            p.high_trans.discard(removed_phase.key)
            redundant_with_phase.addLinkFrom(p)
    for key in removed_phase.high_trans:
        if key != redundant_with_phase.key:
            p = phases[key]
            p.low_trans.discard(removed_phase.key)
            redundant_with_phase.addLinkFrom(p)
    del phases[removed_phase.key]
```

* For each neighbor in `low_trans` and `high_trans`:

  * remove links to the phase that is being deleted,
  * create the corresponding link with the surviving phase `redundant_with_phase`.

* Finally, delete the redundant phase from the dictionary.

Physically: you keep the phase graph coherent — no dangling links pointing to a deleted phase, and second-order connections are preserved via the “merged” phase.

---

### 7. `getStartPhase`: picking the high-temperature phase

```text
def getStartPhase(
    phases: Mapping[Hashable, Phase],
    V: Optional[Callable[[np.ndarray, float], float]] = None,
) -> Hashable:
```

A seemingly simple function, but conceptually important. It answers:

> “In which phase is the Universe at very high temperature?”

#### Logic

1. If `phases` is empty → raise an error.

2. Scan all phases to find those with the **largest** maximum temperature:

   ```text
   start_candidates = []
   Tmax = None
   for key, phase in phases.items():
       phase_Tmax = phase.T[-1]
       if Tmax is None or phase_Tmax > Tmax:
           Tmax = phase_Tmax
           start_candidates = [key]
       elif phase_Tmax == Tmax:
           start_candidates.append(key)
   ```

3. If there is **only one** candidate, or if `V` is `None`:

   * return that candidate.

4. If there are several phases that survive up to the same `Tmax` and a potential `V` is provided:

   * evaluate `V(phase.X[-1], phase.T[-1])` for each candidate,
   * pick the one with the **lowest potential** — thermodynamically preferred.

#### Physical interpretation

* In many models, at high enough T, there is a single “symmetric” phase, e.g. $\phi=0$.
* Sometimes, due to approximate symmetries, several phases may survive to the same maximal scanned temperature; then it is important to check the value of (V) and select the true global minimum.
* This “start phase” is the initial state for any routine that builds the **sequence of transitions** as the Universe cools down.

---

### Putting it all together: integrated picture

The conceptual pipeline for Block A is:

1. **Input**: a potential $V(\phi,T)$ and its derivatives, plus a handful of seed minima.

2. `traceMultiMin`:

   * uses `traceMinimum` to follow each minimum in temperature,
   * when a phase ends, explores nearby points in (T) and field space (`findApproxLocalMin`) to find new phases,
   * builds a dictionary of `Phase` objects with second-order connections (`low_trans` / `high_trans`).

3. `removeRedundantPhases`:

   * ensures each physical phase appears only once in the graph.

4. `getStartPhase`:

   * selects the high-temperature phase from which cosmological evolution starts.

From here, the later blocks (e.g. `tunneling1D` and related routines) will:

* take pairs of phases (false/true vacuum),
* for a given (T), evaluate $\phi_\text{false}(T)$, $\phi_\text{true}(T)$,
* solve the bounce equation and obtain $S_3(T)/T$ or $S_4(T)$,
* build $S(T)/T$ vs. (T), find $T_n$, $T_c$, etc.,
* and ultimately estimate parameters like $\alpha$, $\beta/H$ and the corresponding gravitational-wave spectrum.

In the tests we will use a concrete example potential, for instance

$$
V(\phi, T) = \lambda \phi^4 - ET\phi^3 + D(T^2 - T_0^2)\phi^2
$$

and show explicitly how the functions in Block A reconstruct the phase structure and prepare the ground for tunneling and gravitational-wave calculations.

---

## Block B – Tunneling core: instantons and nucleation temperature

In this block the focus changes:
in Block A we *only* built the **phase diagram as a function of $T$** – which phases exist, where they start/end, and how the minima move in field space.

Block B answers the really crucial physics question:

> Given a metastable phase (false vacuum) and other, more stable phases,
> **when** does the Universe actually make the transition via tunneling?
> And **which** bounce (instanton) dominates this nucleation?

The pipeline is:

1. Read the phase diagram (`Phase` object + `phases` dictionary from Block A).
2. At each temperature $T$, look at which phases are energetically accessible with $V_\text{low}(T) < V_\text{high}(T)$.
3. For each candidate, try to solve the bounce (via `pathDeformation.fullTunneling` or `tunneling1D`).
4. Choose the **bounce with the smallest action** $S(T)$.
5. Search for the $T$ at which the nucleation condition is satisfied (by default $S(T)/T \simeq 140$).

From the numerical point of view, Block B provides:

* a clean backend for the bounce `_solve_bounce`,
* a “scanner in $T$” `_tunnelFromPhaseAtT`,
* helper functions to find the maximum $T_\text{crit}$ where the phase still makes sense,
* and the high-level function `tunnelFromPhase`, which directly returns the instanton and $T_\text{nuc}$.

What follows is a function-by-function description.

---

### 1. `_solve_bounce`: unified backend for the instanton

```text
def _solve_bounce(
    x_high: np.ndarray,
    x_low: np.ndarray,
    V_fixed: Callable[[np.ndarray], float],
    dV_fixed: Callable[[np.ndarray], np.ndarray],
    T: float,
    fullTunneling_params: Optional[Mapping[str, Any]] = None,
) -> Tuple[Optional[Any], float, int]:
    ...
```

**Numerical goal**

Given:

* a metastable minimum (“false vacuum”) at `x_high`,
* a stable minimum at `x_low`,
* a fixed-temperature potential $V_\text{fixed}(x) = V(x, T)$,
* and its gradient $dV_\text{fixed}(x) = \nabla_x V(x, T)$,

this function calls `pathDeformation.fullTunneling` to:

* find the bounce $\phi_b(r)$ that connects $x_\text{high} \to x_\text{low}$,
* extract the Euclidean action $S$,
* classify the type of transition.

All of this is encapsulated in a single place, with standardized error handling.

**Treatment of edge cases**

`pathDeformation.fullTunneling` can raise `tunneling1D.PotentialError` with standard messages:

* `"no barrier"`
  → there is no barrier between the minima
  → **not a first-order transition** (or second-order limit)
  → `trantype = 0`, `action = 0.0`.

* `"stable, not metastable"`
  → the “false vacuum” is actually stable (no metastable well)
  → there is no relevant bounce
  → `trantype = 0`, `action = +∞`.

Any other message is propagated as an unexpected error (this is important: we do **not** hide genuine numerical problems).

**Output**

* `instanton`:

  * the object returned by `fullTunneling` if a bounce is found;
  * `None` in the `"no barrier"` or `"stable, not metastable"` cases.

* `action`:

  * $S$ if `trantype = 1`,
  * `0.0` or `np.inf` in special cases.

* `trantype`:

  * `1` → first-order transition with bounce,
  * `0` → no bounce (second order, stable, etc.).

**Physical interpretation**

`_solve_bounce` is the “engine” that turns a snapshot of the potential at a given temperature $T$ into a physical object:

* radial profile $\phi_b(r)$,
* action $S_3(T)$ (or $S_4$, depending on the backend implementation),
* type of transition (first order or not).

Everything else in Block B is essentially organization around repeated calls to `_solve_bounce` for different temperatures and phase pairs.

---

### 2. `_tunnelFromPhaseAtT`: scan all possible transitions at a given $T$

```text
def _tunnelFromPhaseAtT(
    T,
    phases,
    start_phase,
    V,
    dV,
    phitol,
    overlapAngle,
    nuclCriterion,
    fullTunneling_params,
    verbose,
    outdict,
) -> float:
    ...
```

This function is the **“$S(T)$ wrapper”**: it receives a temperature $T$ and returns

$$
nuclCriterion\bigl( S_\text{min}(T), T \bigr),
$$

where $S_\text{min}(T)$ is the smallest action among all possible transitions from `start_phase` to any other phase with $V_\text{low}(T) < V_\text{high}(T)$.

It also **fills the cache** `outdict[T]` with the best transition found.

#### 2.1. Input and cache

* `T` may come as a scalar or array (because of `optimize.fmin`):

  * the function always converts it to a scalar `T_val`.

* If `T_val` is already in `outdict`, it means we have already computed the best transition at that $T$:

  * it is enough to return `nuclCriterion(outdict[T_val]["action"], T_val)`.

This is important for efficiency: `brentq` and `fmin` will call `_tunnelFromPhaseAtT` many times with the same $T$, and solving the bounce is expensive.

#### 2.2. Finding minima at fixed $T$

It defines a precise 1D minimizer:

```text
def fmin_min(x0):
    xmin = optimize.fmin(V, x0, args=(T_val,), xtol=phitol, ftol=np.inf, disp=False)
    return np.asarray(xmin, dtype=float)
```

* First, it refines the minimum of the initial phase:

  ```text
  x0_guess = start_phase.valAt(T_val)
  x0 = fmin_min(x0_guess)
  V0 = V(x0, T_val)
  ```

* Then it scans all other phases and looks for candidates:

  ```text
  for key, phase in phases.items():
      if key == start_phase.key:
          continue
      # phase must exist at T
      if phase.T[0] > T_val or phase.T[-1] < T_val:
          continue

      x1_guess = phase.valAt(T_val)
      x1 = fmin_min(x1_guess)
      V1 = V(x1, T_val)
      if V1 >= V0:
          continue  # only care if V_low < V_high
      tunnel_list.append({ ... })
  ```

**Physics:**
this step builds the list of possible transitions

$$
\text{high phase} \to \text{low phase}
$$

in which the “low” phase has smaller free energy at temperature $T$.

#### 2.3. `overlapAngle`: pruning strongly aligned targets

If `overlapAngle > 0`, we introduce a geometric criterion:

* For each pair of candidates $(i, j)$:

  * define the vectors

$$\Delta x_i = x_i - x_0,\quad \Delta x_j = x_j - x_0$$

  * compute the angle between them:

$$\cos\theta_{ij}= \frac{ \Delta x_i \cdot \Delta x_j }{ \lvert\Delta x_i\rvert \lvert\Delta x_j\rvert }$$

* If $\theta_{ij}$ is smaller than `overlapAngle` (in degrees), that is,

  ```text
  dotij >= sqrt(xi2 * xj2) * cos_overlap
  ```

  then the two transitions are essentially “in the same direction” in field space.

* We keep only the one closer (in norm) to the false vacuum and discard the other.

**Physics:**
this avoids spending time solving very similar bounces that, in practice, have almost identical actions, keeping only the **shortest** direction in field space (the most promising for having smaller action).

If you want to study **all** directions, just use `overlapAngle=0.0` in the high-level call.

#### 2.4. Solving bounces and choosing the smallest $S$

It defines $T$-fixed wrappers:

```text
def V_fixed(x): return V(x, T_val)
def dV_fixed(x): return dV(x, T_val)
```

Then, for each candidate in `tunnel_list`:

* prints information if `verbose=True`,

* calls `_solve_bounce(...)`,

* fills:

  * `tdict["instanton"]`,
  * `tdict["action"]`,
  * `tdict["trantype"]`,

* updates `lowest_action` and `lowest_tdict` if the action is smaller.

In the end:

* it stores `lowest_tdict` in `outdict[T_val]`,
* it returns `nuclCriterion(lowest_action, T_val)`.

**Physics:**
for each $T$, you construct the envelope

$$
S_\text{min}(T)
= \min_{\text{target phases}}
S_{\text{bounce}, \text{high phase} \to \text{low phase}}(T),
$$

which is what enters the nucleation condition.

---

### 3. `_potentialDiffForPhase`: who is energetically preferred?

```text
def _potentialDiffForPhase(
    T: float,
    start_phase: Phase,
    other_phases: Sequence[Phase],
    V: Callable[[np.ndarray, float], float],
) -> float:
    ...
```

This function is simple but conceptually important. It answers:

> At a given temperature $T$, what is the smallest free-energy difference
> between any other phase and `start_phase`?

More precisely:

* it computes $V_0 = V(x_\text{start}(T), T)$,

* for each other phase $i$:

  * computes $V_i = V(x_i(T), T)$,
  * evaluates $V_i - V_0$,

* and returns the **minimum** over all phases:

$$
\Delta V_\text{min}(T)
= \min_i \bigl[ V_i(T) - V_0(T) \bigr].
$$

**Interpretation**

* If $\Delta V_\text{min}(T) > 0$:
  the potential of `start_phase` is lower than that of any other phase → it is *energetically preferred* (stable) at that $T$.

* If $\Delta V_\text{min}(T) < 0$:
  there exists some other phase with $V < V_0$ → `start_phase` is *energetically disfavored* (unstable) at that $T$.

This sign is what `_maxTCritForPhase` uses to locate critical temperatures where `start_phase` ceases to be preferred.

---

### 4. `_maxTCritForPhase`: maximum critical $T$ for the initial phase

```text
def _maxTCritForPhase(
    phases: Mapping[Hashable, Phase],
    start_phase: Phase,
    V: Callable[[np.ndarray, float], float],
    Ttol: float,
) -> float:
    ...
```

**Physical question**

For the phase `start_phase`, what is the **largest temperature** $T_\text{crit}$ such that there exists another phase degenerate in free energy?

$$
V_\text{start}(T_\text{crit}) \approx V_\text{other}(T_\text{crit}).
$$

This temperature is relevant to delimit the range in which the false vacuum still makes sense as a metastable minimum.

**Numerical steps**

1. It builds the list `other_phases` with all phases different from `start_phase`.

2. It defines the relevant $T$ bounds:

   * `Tmin` = largest initial $T$ among the other phases;
   * `Tmax` = smallest final $T$ among the other phases;
   * then it intersects with $[T_\text{start,min}, T_\text{start,max}]$.

3. It evaluates:

   ```text
   DV_Tmin = _potentialDiffForPhase(Tmin, start_phase, other_phases, V)
   DV_Tmax = _potentialDiffForPhase(Tmax, start_phase, other_phases, V)
   ```

   * If `DV_Tmin >= 0`: at `Tmin` the `start_phase` is already stable (lower than the others) → it returns `Tmin`.
   * If `DV_Tmax <= 0`: at `Tmax` it is already unstable (some other phase has smaller $V$) → it returns `Tmax`.

4. Otherwise, the signs of $\Delta V$ at `Tmin` and `Tmax` are opposite:

   * it uses `optimize.brentq` on `_potentialDiffForPhase` to find the zero:

     $$
     \Delta V_\text{min}(T_\text{crit}) = 0.
     $$

   * it returns this `Tcrit` with tolerance `Ttol`.

**Use inside Block B**

`tunnelFromPhase` uses `_maxTCritForPhase` when the initial search in $T$ does not directly find a nucleation solution:

* it restricts `Tmax` to this `Tmax_crit` to avoid searching for nucleation in a region where the minimum has already ceased to be physically meaningful (the phase has stopped being a local minimum).

---

### 5. `tunnelFromPhase`: high-level interface for $T_n$ and the instanton

```text
def tunnelFromPhase(
    phases,
    start_phase,
    V,
    dV,
    Tmax,
    Ttol=1e-3,
    maxiter=100,
    phitol=1e-8,
    overlapAngle=45.0,
    nuclCriterion=lambda S, T: S/(T + 1e-100) - 140.0,
    verbose=True,
    fullTunneling_params=None,
) -> Optional[Dict[str, Any]]:
    ...
```

This is the function you will call “in practice” when you want to:

* **find $T_n$** (the nucleation temperature) for the transition starting from `start_phase`,
* **obtain the dominant bounce** in the form of the `instanton` object.

#### 5.1. Main inputs

* `phases`: `{key → Phase}` dictionary from Block A.

* `start_phase`: `Phase` object representing the initial metastable phase.

* `V`, `dV`: potential and gradient, $V(x, T)$ and $dV(x, T)$.

* `Tmax`: maximum $T$ allowed for nucleation (you choose the range of interest).

* `Ttol`: tolerance in $T$ for locating the zero of `nuclCriterion`.

* `phitol`: tolerance used in 1D minimizations in field space (to find minima).

* `overlapAngle`: parameter used to prune nearly redundant directions in field space (passed to `_tunnelFromPhaseAtT`).

* `nuclCriterion(S, T)`:

  * default: $S/T - 140$ (a standard choice in cosmological FOPTs),
  * the routine seeks a $T$ such that `nuclCriterion(...) = 0`.

* `fullTunneling_params`: extra kwargs for `pathDeformation.fullTunneling`.

#### 5.2. Range in $T$

First, it defines the effective interval:

* `Tmin = start_phase.T[0]`,
* `T_highest_other` = largest final $T$ among all phases,
* `Tmax_eff = min(Tmax, T_highest_other)`.

It checks whether `Tmax_eff >= Tmin`; otherwise, the search does not make numerical sense.

#### 5.3. Two-step strategy

The function tries, in order:

1. **Direct root-finding with `brentq`**:

   * it defines $f(T) = _tunnelFromPhaseAtT(T, ...)$,

   * it tries to find a zero of $f$ between `Tmin` and `Tmax_eff`:

     ```text
     Tnuc = brentq(f, Tmin, Tmax_eff, ...)
     ```

   * if this works, we are done: `Tnuc` is found directly.

2. If `brentq` fails with the typical error
   `"f(a) and f(b) must have different signs"`:

   * it means that $nuclCriterion(S(T), T)$ has the same sign over the whole interval:

     * either it never crosses zero,
     * or it does cross, but the endpoints do not show that clearly.

   Then the function switches to a more careful mode:

   * it ensures that the endpoints `Tmin` and `Tmax_eff` have been computed (via `_tunnelFromPhaseAtT`) and cached in `outdict`;

   * it checks:

     * whether $nuclCriterion(S(T_\text{max,eff}), T_\text{max,eff}) > 0$ (tunneling too suppressed at high $T$);
     * and whether $nuclCriterion(S(T_\text{min}), T_\text{min}) < 0$ (strong tunneling at low $T$).

   * if such a sign change exists, it restricts the maximum temperature to a `Tmax_crit` obtained from `_maxTCritForPhase` (the region where the phase is still physically meaningful).

   * it then performs a **minimization in $T$** using `optimize.fmin` on `_tunnelFromPhaseAtT`, with a `callback`:

     * as soon as the function enters a region with $nuclCriterion \le 0$, the callback raises a `StopIteration(T)`, containing an estimate of the $T$ where this happens.

   * once this `Tmin_opt` is obtained, it checks whether at that point $nuclCriterion$ is indeed $\le 0$; if it is still $> 0$, the function concludes that there is no efficient nucleation → it returns `None`.

   * if there is a genuine crossing, it performs a second `brentq` in a narrower interval $(T_\text{min,opt}, T_\text{max,crit})$ to better locate the zero.

   * otherwise (for example, if $nuclCriterion$ never becomes $< 0$ even at `Tmin`), it concludes that the transition **does not nucleate** in the given interval → it returns `None`.

If, on the other hand, the nucleation condition is already satisfied at `Tmax_eff` (i.e. $nuclCriterion \le 0$ there), the function simply defines:

* `Tnuc = Tmax_eff`: nucleation “happens immediately” at the upper bound of the range.

#### 5.4. Output

After determining `Tnuc`:

* it ensures that `outdict[Tnuc]` has been filled by `_tunnelFromPhaseAtT`,
* it takes `rdict = outdict[Tnuc]`.

If `rdict["trantype"] > 0` (first-order transition with bounce), it returns the dictionary:

* `Tnuc` : nucleation temperature;
* `low_vev`, `high_vev` : low- and high- T minima;
* `low_phase`, `high_phase` : keys of the phases involved;
* `action` : instanton action;
* `instanton` : object returned by `fullTunneling`;
* `trantype` : `1` for first-order transition.

If `trantype` is `0` (no bounce – second-order case or stable), it returns `None`.

**Physical interpretation**

$T_n$ is the temperature at which, according to the chosen criterion (typically $S(T)/T \simeq 140$), the nucleation probability per Hubble volume becomes relevant.

In terms of cosmological history:

* above $T_n$, the Universe remains trapped in the metastable phase (unless there is a second-order transition);
* close to $T_n$, bubbles of the true phase appear efficiently;
* the returned instanton is the solution that dominates this nucleation.

---

### 6. Closing Block B: integrated view

With Blocks A and B together, the full conceptual flow for a first-order phase transition is:

1. **Build the phase diagram in $T$**:

   * use `traceMultiMin` and `Phase` to find all phases;
   * clean redundancies with `removeRedundantPhases`;
   * choose the high-$T$ phase with `getStartPhase`.

2. **Choose a metastable phase of interest** (`start_phase`).

3. **Run the tunneling core**:

   * for each $T$, `_tunnelFromPhaseAtT` builds all possible bounces starting from `start_phase` → $S_\text{min}(T)$;
   * `_maxTCritForPhase` helps determine up to which temperature it makes sense to demand metastability;
   * `tunnelFromPhase` combines everything, applies the nucleation criterion, and returns $T_n$ and the dominant instanton.

4. **From $T_n$ and the instanton**, other blocks (or user scripts) can:

   * reconstruct $S(T)/T$ as a function of $T$;
   * determine $T_c$ (the critical temperature where the phases become degenerate);
   * extract cosmological parameters such as $\alpha$ and $\beta/H$;
   * compute gravitational-wave spectra.

In the next test files, we will implement concrete examples using the same one-component Landau–Ginzburg potential, to visualize:

* how `tunnelFromPhase` selects the correct target phase,
* how $S(T)/T$ crosses $140$,
* and how this connects to the phase diagram built in Block A.

---

## Block C – Transition history (`secondOrderTrans`, `findAllTransitions`, `findCriticalTemperatures`, `addCritTempsForFullTransitions`)

Block C takes everything built in Blocks A and B

* **phases** traced in temperature (`Phase`, `traceMultiMin`, `getStartPhase`);
* **bounce solutions** and **nucleation temperatures** (`tunnelFromPhase`);

and turns it into something you actually want as a cosmologist:

> A **coherent thermal history** of the Universe:
> which phase is realized at which temperature, when first-order vs second-order transitions happen, and how far the system supercools before tunneling.

Conceptually:

* Block A: *“What are the minima as a function of T?”*
* Block B: *“Given two minima, when and how do we tunnel between them?”*
* **Block C**: *“Chaining everything together, what is the full sequence of transitions as the Universe cools?”*

Below we describe each Block C function (how it works, what it returns, and what it means physically) in the same style as the previous blocks. 

---

### 1. `secondOrderTrans` – encoding second-order transitions

**Role:**
`secondOrderTrans(high_phase, low_phase, Tstr="Tnuc")` is a small helper that builds a **transition dictionary** for cases where the transition is **second order** (no barrier, no bounce).

In the language of Block A:

* Both `high_phase` and `low_phase` are `Phase` objects.
* The **graph structure** (via `low_trans`) tells us that, as T goes down, the system can slide continuously from `high_phase` into `low_phase`.

**Code behavior**

* It constructs a dictionary with keys:

  * `Tstr` (usually `"Tnuc"` or `"Tcrit"`),
  * `high_phase`, `low_phase`,
  * `high_vev`, `low_vev`,
  * `action`, `instanton`, `trantype`.
* For a second-order transition we use:

  * `action = 0.0`,
  * `instanton = None`,
  * `trantype = 2`,
  * `high_vev` and `low_vev` set to the same field value (the first point of `high_phase.X`), reflecting that the VEV is *continuous* across the transition.

Physically, this is a **continuous symmetry restoration/breaking**:

* No barrier → no instanton,
* Correlation length diverges at the critical point,
* Thermal fluctuations, not quantum tunneling, control the dynamics.


---

### 2. `findAllTransitions` – building a full thermal history

**Role:**
`findAllTransitions(phases, V, dV, tunnelFromPhase_args=...)` takes the phase structure (Block A) and the tunneling machinery (Block B), and constructs a **single cooling path**:

$$
\text{Phase}_\text{high} ;\rightarrow; \text{Phase}_1 ;\rightarrow; \text{Phase}_2 ;\rightarrow; \dots
$$

ordered from **high temperature to low temperature**.

**High-level algorithm**

1. **Start at high T**

   * Use `getStartPhase(phases, V)` to identify the dominant high-T phase (usually symmetric, φ ≈ 0).
   * Set `Tmax` to the highest T where this phase is defined.

2. **Iterative step from a given `start_phase`**

   1. Remove `start_phase` from the working map (we won’t tunnel *into* it again).

   2. Call `tunnelFromPhase(phases_work, start_phase, V, dV, Tmax, **tunnelFromPhase_args)`:

      * If a **first-order** transition is found, `tunnelFromPhase` returns a dict with:

        * `Tnuc`, `low_phase`, `high_phase`, `low_vev`, `high_vev`,
        * `instanton`, `action`, `trantype = 1`.
      * If **no** first-order solution exists, it returns `None`.

   3. **If first-order succeeded:**

      * Append the returned dict to the `transitions` list.
      * Set `start_phase = phases_work[low_phase]`.
      * Update `Tmax = Tnuc` (we only look for *later* transitions).

   4. **If first-order failed:**

      * Look at `start_phase.low_trans` (if present) – these are “downstream” phases that can be reached continuously (second order).
      * If at least one of them is still in `phases_work`, pick the first and build a second-order transition via `secondOrderTrans(start_phase, low_phase)`.
      * Append that dict to `transitions`, update `start_phase = low_phase`, and set `Tmax` to the top of the new phase’s T-range.
      * If there are no valid `low_trans` targets, the history stops.

**Outputs**

The function returns a **list of transition dictionaries**, each of which looks like:

* First-order:

  * `{"Tnuc", "low_vev", "high_vev", "low_phase", "high_phase", "action", "instanton", "trantype=1"}`
* Second-order:

  * Same keys but with `action=0.0`, `instanton=None`, `trantype=2`.

They are automatically ordered from **hottest to coldest** along a single, consistent path in phase space.

**Physical interpretation on the toy model**

In the Landau–Ginzburg model used in Blocks A and B:

* At high T: we start in the symmetric phase (φ ≈ 0).
* As T drops:

  * `findAllTransitions` should find a **first-order** transition from the symmetric to the broken phase, with:

    * A `Tnuc` slightly **below** the critical temperature (supercooling),
    * Non-zero `action` and a `trantype=1`.
* If additional broken phases existed (multi-step symmetry breaking), they would appear as further entries, either first-order or second-order, depending on the barrier structure.


---

### 3. `findCriticalTemperatures` – scanning for degeneracies `Tcrit`

**Role:**
`findCriticalTemperatures(phases, V, start_high=False)` decouples the *geometric* information about **phase degeneracies** from the dynamical tunneling story.

For every ordered pair of phases `(phase1 → phase2)` with overlapping temperature ranges, it asks:

> “Is there a temperature `Tcrit` where
> `V(φ₁(Tcrit), Tcrit) = V(φ₂(Tcrit), Tcrit)`?”

If yes, `Tcrit` is a **critical temperature** where the two branches are degenerate in free energy.

**Code behavior**

1. For each pair `(i, j)`:

   * Identify the overlap interval in T:

     $$
     [t_{\min}, t_{\max}] =
     [\max(T_i^{\min}, T_j^{\min}), \min(T_i^{\max}, T_j^{\max})].
     $$

2. If there is **no overlap** (`tmin >= tmax`):

   * Check for a second-order link via `phase2.key in phase1.low_trans`.
   * If such a link exists, construct a second-order transition using
     `secondOrderTrans(phase1, phase2, "Tcrit")`.

3. If there **is** overlap:

   * Define:

     $$
     \Delta V(T) = V(\phi_1(T), T) - V(\phi_2(T), T),
     $$

     with `φ1 = phase1.valAt(T)` and `φ2 = phase2.valAt(T)`.

   * Evaluate `DV(tmin)` and `DV(tmax)`.

   * Require:

     * `DV(tmin) >= 0` and `DV(tmax) <= 0`, so that `phase1` is higher
       at one end and lower at the other → sign change.

   * If the sign change exists, use `optimize.brentq` to find `Tcrit` in `[tmin, tmax]`.

   * Build a transition dict:

     ```text
     {
       "Tcrit": Tcrit,
       "high_vev": phase1.valAt(Tcrit),
       "high_phase": phase1.key,
       "low_vev": phase2.valAt(Tcrit),
       "low_phase": phase2.key,
       "trantype": 1,
     }
     ```

4. After looping over all pairs, sort the list in **decreasing `Tcrit`**.

**Physics: `Tcrit` vs `Tnuc`**

* For a **first-order** transition, `Tcrit` is the temperature where two minima become **degenerate**.

* But nucleation actually happens at a lower `Tnuc` where:

  $$
  \frac{S_3(T)}{T} \approx 140
  $$

  (or whatever criterion you use in `tunnelFromPhase`).

* The difference `Tcrit − Tnuc` is a direct measure of **supercooling**.

For a purely **second-order** transition:

* The minima coalesce at `Tcrit`, there is **no barrier**, and we typically have `Tnuc ≈ Tcrit` and `trantype=2`.

---

### 4. `addCritTempsForFullTransitions` – matching `Tnuc` with `Tcrit`

**Role:**
`addCritTempsForFullTransitions(phases, crit_trans, full_trans)` glues together:

* the **dynamical** information from supercooled transitions (`full_trans`, typically output from `findAllTransitions`), and
* the **static** degeneracy information from `findCriticalTemperatures` (`crit_trans`).

For each nucleation transition with a given `Tnuc`, it tries to find the “corresponding” `Tcrit` along the graph of phases, and attaches that dictionary under the key `"crit_trans"`.

**Why this is useful**

In any realistic model you often want to quote both:

* the **critical temperature** (thermodynamic degeneracy),
* the **nucleation temperature** (kinetic onset of the transition),

for each step in the cosmological history. Knowing both is essential for:

* estimating latent heat and strength of the transition,
* computing gravitational-wave spectra (supercooling directly affects the signal),
* comparing with analytic expectations.

**How the matching works (logic overview)**

1. **Build “parent” chains in the critical graph**

   * For each phase `i` in `phases`, we construct a list `parents_dict[i]`:

     ```text
     parents = [i, ...]
     ```

     by scanning `crit_trans` from low T to high T and, whenever we see a critical transition `high → low` with `low` already in `parents`, we append `high`.

   * Intuitively: these chains encode **ancestry along critical transitions** when moving from low T → high T.

2. **For each nucleation transition `tdict` in `full_trans`:**

   * Extract `low_phase` and `high_phase`.
   * Get ancestry lists:

     * `low_parents = parents_dict[low_phase]`,
     * `high_parents = parents_dict[high_phase]`.
   * Find the set of **common parents** between these chains and carefully prune them:

     * Remove common ancestors from the tail of `low_parents`,
     * Trim the tail of `high_parents` so that they share only the relevant high-T part.
   * This focuses our search on the segment of the phase graph that actually distinguishes the two phases associated with this nucleation event.

3. **Scan `crit_trans` (from low T to high T):**

   * For each candidate `tcdict` in `crit_trans`:

     * Skip if `Tcrit < Tnuc` (we only accept degeneracies above or at the nucleation temperature).

     * Check whether:

       ```text
       tcdict["low_phase"]  in low_parents
       tcdict["high_phase"] in high_parents
       ```

     * The first such match is taken to be the **critical counterpart** of the nucleation transition.

   * Attach this dict as:

     ```text
     tdict["crit_trans"] = tcdict
     ```

   * If no match is found, set:

     ```text
     tdict["crit_trans"] = None
     ```

---

If you want to see the full Block C implementation, look at the `transitionFinder` module in the same file where Blocks A and B live. Block C is the “glue layer” that:

* takes the **phase structure** from Block A,
* the **bounce physics** from Block B,
* and turns them into a **human-readable thermal history** with:

  * which phases appear,
  * at which temperatures they become degenerate,
  * at which temperatures they actually nucleate,
  * and whether each step is first- or second-order.

---
