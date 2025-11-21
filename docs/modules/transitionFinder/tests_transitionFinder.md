# tests Transition Finder

This document summarizes the tutorial-style tests for the `transitionFinder`
module, focusing on the **visual outputs** (plots) and the physical intuition
behind them.

The goal is that, by reading this file and looking at the figures, a user can
understand what the Block A primitives (`traceMinimum`, `Phase`, `traceMultiMin`,
`findApproxLocalMin`, `removeRedundantPhases`, `getStartPhase`) are doing in a
concrete, finite-temperature scalar-field model.

---

## Block A – Landau–Ginzburg toy model and phase tracing

All tests in Block A use the same 1D finite-temperature potential

$$
V(\phi, T) = D (T^2 - T_0^2)\,\phi^2 \;-\; E\,T\,\phi^3 \;+\; \frac{\lambda}{4}\,\phi^4
$$

with $D > 0$, $\lambda > 0$, and a small cubic term $E > 0$.
This is the standard Landau–Ginzburg toy model for a first-order phase
transition:

- At high temperature, there is a unique minimum at $\phi = 0$ (symmetric phase).
- At lower temperature, non-trivial minima at $\phi \neq 0$ appear
  (broken phase).
- In an intermediate range, symmetric and broken phases coexist, separated by a barrier.
- Below a spinodal temperature, the symmetric minimum disappears.

The tests are organized to show how Block A routines reconstruct this structure.

---

### Test 1 – Potential shape and minima at high and low temperature

**Script:** `test_blockA_1_potential_shape_and_minima`

**What it does**

- Evaluates $V(\phi, T)$ on a grid in $\phi$ for two temperatures:
  - $T_\text{high} = 200$ (well above $T_0$),
  - $T_\text{low} = 50$ (well below $T_0$).
- Finds numerically:
  - the minimum near $\phi \simeq 0$ at high T,
  - the broken minimum at low T.
- Checks the curvature $m^2 = d^2V/d\phi^2$ at these points to verify stability.

**Expected plot**

A single figure with two curves:

- Horizontal axis: $\phi$.
- Vertical axis: $V(\phi, T)$.
- Curves:
  - $V(\phi, T_\text{high})$: single well at $\phi \approx 0$.
  - $V(\phi, T_\text{low})$: double-well structure with a deeper minimum at $\phi \neq 0$.

**What to look for**

- At **high T**:
  - the curve has a single minimum at (or extremely close to) $\phi = 0$.
- At **low T**:
  - the origin becomes unstable (local maximum or shallow region),
  - a new **broken minimum** appears at $\phi > 0$.

**Placeholder for figure**

![Test 1 – V(φ, T_high) vs V(φ, T_low)](assets/Lot_A_1.png)


---

### Test 2 – `traceMinimum` on the symmetric phase (descending in T)

**Script:** `test_blockA_2_traceMinimum_symmetric_phase_downwards`

**What it does**

* Starts from the symmetric minimum:

  * initial condition: $\phi = 0$ at (T = 200).
* Uses `traceMinimum` to follow this minimum **downwards in temperature** until it
  becomes unstable.
* Records:

  * the temperature grid `T`,
  * the traced minimum $\phi_{\min}(T)$,
  * the curvature $m^2(T) = d^2V/d\phi^2|*{\phi*{\min}(T)}$.
* Compares the numerically extracted spinodal temperature `res.overT` with the
  analytic spinodal (for the symmetric phase) at $T_\text{spin} = T_0$.

**Expected plots**

1. **Symmetric branch $\phi_{\min}(T)$**

   * Horizontal axis: (T).
   * Vertical axis: $\phi_{\min}(T)$.
   * Markers along the traced points from `traceMinimum`.
   * Vertical dashed line at $T = T_0$ (analytic spinodal).

   Behaviour:

   * $\phi_{\min}(T)$ should stay very close to 0 at all temperatures where
     the symmetric minimum exists.
   * Near $T \approx T_0$, the branch ends (spinodal).

2. **Curvature $m^2(T)$ along the symmetric trace**

   * Horizontal axis: (T).
   * Vertical axis: $m^2(T) = d^2V/d\phi^2|*{\phi*{\min}(T)}$.
   * A horizontal line at $m^2 = 0$ indicating the stability threshold.

   Behaviour:

   * For $T \gg T_0$, $m^2 > 0$: the symmetric minimum is stable.
   * As $T \to T_0$, $m^2 \to 0$.
   * This signals the **spinodal point** where the symmetric phase loses stability.

**Placeholders for figures**

![Test 2 – Symmetric phase: φ_min(T) from traceMinimum](assets/Lot_A_2.png)

![Test 2 – Symmetric phase: m²(T) along the traced branch](assets/Lot_A_3.png)


---

### Test 3 – `traceMinimum` on the broken phase (ascending in T)

**Script:** `test_blockA_3_traceMinimum_broken_phase_upwards`

**What it does**

* First finds a broken minimum at low temperature:

  * (\phi_b(T=50)) via a 1D minimization.
* Uses `traceMinimum` starting from $\phi_b(50)$ at (T = 50), and follows this
  broken minimum **upwards in temperature**.
* Records the branch until the broken phase disappears or becomes unstable.
* Computes the curvature $m^2(T)$ along the broken branch.

**Expected plot**

1. **Broken branch $\phi_{\min}(T)$**

   * Horizontal axis: (T).
   * Vertical axis: $\phi_{\min}(T)$ for the broken phase.
   * A horizontal line at $\phi = 0$.

   Behaviour:

   * At low T, $\phi_{\min}(T)$ is significantly away from 0
     (spontaneous symmetry breaking).
   * As T increases, $|\phi_{\min}(T)|$ decreases, tending towards 0 as you
     approach the region where the symmetric phase dominates.
   * The branch ends near the broken-phase spinodal temperature `res.overT`.

**Placeholder for figure**


![Test 3 – Broken phase: φ_min(T) from traceMinimum](assets/Lot_A_4.png)



---

### Test 5 – `traceMultiMin` and `Phase`: global phase structure

**Script:** `test_blockA_5_traceMultiMin_and_Phase_structure`

**What it does**

* Builds the phase structure in the interval $T \in [50, 200]$ using:

  * seeds at $(\phi = 0, T = 200)$ (symmetric),
  * and $(\phi_b(T=50), T = 50)$ (broken).

* Runs `traceMultiMin` to trace all minima that arise from these seeds, and
  then `removeRedundantPhases` to clean up duplicates.

* Constructs `Phase` objects for each branch and checks that:

  * there is exactly one symmetric-like phase (with $\phi \approx 0$ at high T),
  * at least one broken-like phase.

* Uses `getStartPhase` to identify the high-temperature phase.

* For each `Phase`, compares the spline-based `valAt(T)` with a direct
  minimization of $V(\phi, T)$ at a few T values (sanity check).

**Expected plot**

**Phase structure: $\phi_{\min}(T)$ from `Phase` splines**

* Horizontal axis: (T).
* Vertical axis: $\phi_{\min}(T)$.
* One curve per `Phase` object, plotted as a function of T over its domain.

Behaviour:

* A **symmetric** branch:

  * stays near $\phi \approx 0$ over a wide range of temperatures,
  * is present at the highest T.
* A **broken** branch:

  * exists at lower T with $\phi \neq 0$,
  * bends towards $\phi \to 0$ as T increases,
  * terminates at its own spinodal.

This plot is the closest thing to a “phase diagram in T” in Block A: it shows how
each phase’s vacuum expectation value evolves with temperature.

**Placeholder for figure**


![Test 5 – Phase structure: φ_min(T) for all Phase branches](assets/Lot_A_5.png)


---

### Test 6 – `findApproxLocalMin` on a simple segment

**Script:** `test_blockA_6_findApproxLocalMin_on_simple_segment`

**What it does**

* Focuses on a fixed temperature $T = 150 > T_0$, where the potential has a
  **unique** minimum at $\phi = 0$.

* Considers a straight segment in field space:

  $$
  \phi \in [-3, +3]
  $$

  and calls `findApproxLocalMin` along this segment.

* `findApproxLocalMin` samples the segment, looks for discrete local minima in
  $V(\phi, T)$, and returns approximate positions of minima between the
  segment endpoints.

**Expected plot**

**Potential along the segment and found approximate minima**

* Horizontal axis: $\phi$.
* Vertical axis: $V(\phi, T)$ for fixed (T = 150).
* Curve: $V(\phi, 150)$ over $\phi \in [-3, 3]$.
* Markers: positions returned by `findApproxLocalMin` (approximate minima).

Behaviour:

* The curve should be a single well centred at $\phi = 0$.
* `findApproxLocalMin` should identify minima very close to $\phi = 0$.
* If it finds more than one minimum, they should cluster around the origin,
  reflecting the discrete sampling.

**Placeholder for figure**

![Test 6 – findApproxLocalMin along a segment at T=150](assets/Lot_A_6.png)


---

If you want to see the full test script of this block go to [tests/transitionFInder](/tests/transitionFinder/Lot_A.py)

---
