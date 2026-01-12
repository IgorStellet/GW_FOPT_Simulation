# `gravitational_Waves`

High-level helpers to extract **thermodynamic parameters** relevant for **gravitational-wave (GW)** production from **finite-temperature first-order phase transitions**.

This module is designed to sit *on top* of the core CosmoTransitions workflow:

- `transitionFinder` builds the **thermal phase structure** (minima traced as a function of temperature);
- `tunneling1D` / the unified backend inside `transitionFinder` computes the **bounce** and its action;
- **this module** turns that into the numbers you actually plot in a GW paper:
  - the slope $ d/dT [S_3(T)/T] $ and therefore $\beta/H_*$,
  - the strength parameter $\alpha$,
  - $R_* H_*$ (characteristic bubble size in Hubble units),
  - and the **phenomenological GW spectra** $h^2\Omega(f)$ (sound waves, turbulence, collisions).

It also includes **detector sensitivity fits** (PIS, s-channel) from [arXiv:2002.04615](arXiv:2002.04615) for LISA / DECIGO / BBO.

---

## 1) What problem this module solves

In a thermal first-order transition, the key dynamical input is the **thermal bounce action**

$$
S_3(T) \quad \Rightarrow \quad \frac{S_3(T)}{T},
$$
because the tunneling probability is exponentially sensitive to $S_3/T$.

A common “minimal GW forecast pipeline” is:

1. find a temperature $T_n$ (nucleation) or $T_*$ (production/percolation);
2. evaluate:
   - $ \alpha(T_*) $ from the free-energy difference,
   - $\beta/H_*(T_*) $ from the slope of $S_3/T$,
   - $R_*H_* \sim (8\pi)^{1/3} v_w/(\beta/H_*)$;
3. plug those into GW fitting functions to get
$$
h^2\Omega_{\rm sw}(f),\quad h^2\Omega_{\rm turb}(f),\quad h^2\Omega_{\rm coll}(f),
$$
and compare to detector sensitivities.

This module implements exactly that, with a clean separation between:

- **stateful** computations (you *have phases and a potential* → compute $S_3$, $\alpha$, $\beta/H_*$);
- **stateless** spectra (you *already have $\alpha,\beta/H_*,T_*,g_*,v_w$* → compute spectra).

---

## 2) Public API overview

Exported symbols:

- **Class**
  - `GravitationalWaveCalculator`

- **Stateless spectral helpers**
  - `gw_f_sw_peak`, `gw_omega_sw_h2`
  - `gw_h_star_Hz`, `gw_f_turb_peak`, `gw_omega_turb_h2`
  - `gw_f_coll_peak`, `gw_omega_coll_h2`
  - `gw_omega_total_h2`

- **Detector sensitivity fits** ([arXiv:2002.04615](), PIS, s-channel)
  - `lisa_sensitivity_s_pis`
  - `decigo_sensitivity_s_pis`
  - `bbo_sensitivity_s_pis`

---

## 3) The stateful engine: `GravitationalWaveCalculator`

### 3.1. Conceptual role

`GravitationalWaveCalculator` assumes you already built the phase structure with `transitionFinder`:

- you have a mapping `phases[key] -> Phase`,
- each `Phase` contains splines for the minimum $ \phi_{\min}(T) $,
- and you choose which phase is “high-T/metastable” and which is “low-T/stable”.

Then the class can do two essential jobs:

1. **Compute** $S_3(T)$ by calling the unified tunneling backend (`transitionFinder._solve_bounce`) between the two minima at that temperature.
2. Differentiate $S_3(T)/T$ in $T$ using a robust finite-difference stencil, giving you $ \beta/H_* $.

Once you have $\alpha$ and $\beta/H_*$, the same instance can generate GW spectra (sound waves, turbulence, envelope collisions) using standard fits.

---

### 3.2. Construction

**Signature**
```text
GravitationalWaveCalculator(
    V, dV, dVdT,
    phases,
    high_phase_key, low_phase_key,
    *,
    fullTunneling_params=None,
)
````

#### Parameters

* `V(x, T) -> float`
  Finite-temperature effective potential. Here it is treated as a **Helmholtz free-energy density**.

* `dV(x, T) -> ndarray`
  Field-gradient $\partial V/\partial \phi_i$. Same shape as `x`.

* `dVdT(x, T) -> float`
  Temperature derivative $\partial V/\partial T$ evaluated at the field point `x`.
  This is only required if you want to compute `alpha()`.

* `phases: Mapping[key, Phase]`
  Output from `transitionFinder.traceMultiMin` (or equivalent pipeline).

* `high_phase_key`, `low_phase_key`
  Keys selecting the metastable high-T phase and stable low-T phase.

* `fullTunneling_params: dict`
  Passed through to `transitionFinder._solve_bounce` (e.g. backend options, tolerances).

#### Notes

* The class **pre-computes the overlapping temperature interval** where both phases exist.
* A small internal cache stores computed values of $S_3(T)$ to avoid recomputing at repeated stencil points.

---

### 3.3. Internal: common temperature range

The calculator enforces that both phases are defined at the temperature where you evaluate anything:

* Let `Th = high_phase.T`, `Tl = low_phase.T`.
  The overlap is:
  $$
  T_{\min}=\max(Th[0],Tl[0]),\quad
  T_{\max}=\min(Th[-1],Tl[-1]).
  $$
* Any evaluation outside ([T_{\min},T_{\max}]) raises a `ValueError`.

This is not pedantry: many numerical failures people attribute to “bounce instability”
are just “you asked for tunneling at a temperature where one minimum doesn’t exist.”

---

## 4) Thermodynamic derivatives and (\beta/H_*)

### 4.1. `dS_dT(T, dT)`

Fourth-order central finite-difference estimate of
$$
\frac{d}{dT}\left[\frac{S_3(T)}{T}\right].
$$

It uses the standard 5-point stencil:

$$ \left.\frac{d}{dT}\left[\frac{S_3}{T}\right]\right|_{T}
\approx
\frac{
\frac{S_3(T - 2\Delta T)}{T - 2\Delta T} - 8\frac{S_3(T - \Delta T)}{T - \Delta T} + 8\frac{S_3(T + \Delta T)}{T + \Delta T} - \frac{S_3(T + 2\Delta T)}{T + 2\Delta T}
  }{12\Delta T}.$$

**Why fourth-order?**
Because $S_3/T$ can be steep in the interesting region, and low-order derivatives
will inject fake “features” into $\beta/H_*$ (which you’ll then mistakenly interpret
as physics).

#### Raises

* `ValueError` if `dT <= 0` or if any stencil temperature leaves $[T_{\min},T_{\max}]$
* `RuntimeError` if the bounce action cannot be computed at any stencil point

---

### 4.2. `beta_over_H(Tn, dT, *, H=None)`

Estimate:
$$
\frac{\beta}{H_*}\simeq
T_n\left.\frac{d}{dT}\left[\frac{S_3(T)}{T}\right]\right|_{T=T_n}.
$$

This is the standard approximation under radiation-dominated cooling
(and identifying the dominant time dependence of the nucleation rate with the
temperature dependence of $S_3/T$).

* If you pass `H`, the method additionally returns $\beta = (\beta/H_*)H_*$.

#### Returns

* `float` if `H is None`
* `(beta_over_H, beta)` if `H` is provided

#### Raises

* `ValueError` if `H <= 0`
* plus anything raised by `dS_dT`

---

## 5) Strength parameter (\alpha)

### 5.1. `alpha(T, g_star, *, return_delta_rho=False)`

This implements the common thermodynamic definition:

Treat (V(\phi,T)) as the **free-energy density** (F(\phi,T)).
Then the energy density is
$$
\rho(\phi,T)=F(\phi,T)-T\frac{\partial F(\phi,T)}{\partial T}.
$$
Define the released energy density as the difference between the phases:
$$
\Delta\rho(T)=\rho_{\rm low}(T)-\rho_{\rm high}(T),
$$
and normalize to the radiation energy density
$$
\rho_{\rm rad}(T)=\frac{\pi^2}{30}g_*T^4,\qquad
\alpha(T)=\frac{\Delta\rho(T)}{\rho_{\rm rad}(T)}.
$$

#### Parameters

* `T`: temperature (typically $T_n$ or $T_*$)
* `g_star`: effective relativistic d.o.f. at that temperature
* `return_delta_rho`: if `True`, return `(alpha, delta_rho)`

#### Raises

* `ValueError` if `g_star <= 0` or `T` outside the overlap interval
* `AttributeError` if the instance was constructed without `dVdT`

#### Important implementation note (read this once)

The current implementation computes

```text
rho = V - (1/4) * T * dVdT
```

instead of
$\rho = V - T\partial_T V$.

That **extra factor (1/4)** is not the standard textbook definition.
If this factor is intentional in your convention (e.g. you are using a specific
normalization of the thermal piece), document it consistently across the project.
If it is not intentional, this is exactly the type of “small constant” that can
silently shift (\alpha) by an (\mathcal{O}(1)) factor and mislead you later.
(For the MD file: I’m flagging it as a convention-sensitive point so it doesn’t
get buried.)

---

## 6) Bubble size proxy: (R_*H_*)

### 6.1. `bubble_radius_over_H(beta_over_H, v_w=1.0)`

Defines
$$
R_* \simeq \frac{(8\pi)^{1/3}v_w}{\beta}
\quad\Rightarrow\quad
R_*H_* \simeq \frac{(8\pi)^{1/3}v_w}{\beta/H_*}.
$$

Returns the **dimensionless** $R_*H_*$.

---

## 7) Nucleation rate $\Gamma(T)$

### 7.1. `nucleation_rate(T, *, A_scale=1.0, use_linde_prefactor=True)`

Implements the semiclassical O(3) thermal rate:
$$
\Gamma(T)\simeq A(T)e^{-S_3(T)/T}.
$$

By default it uses the Linde prefactor:
$$
A_{\rm Linde}(T)\approx
\left[\frac{S_3(T)}{2\pi T}\right]^{3/2}T^4.
$$

Alternatively, you can choose a simpler $A(T)\approx A_{\rm scale}T^4$.

#### Returns

* $\Gamma(T)$ in natural units (if (T) is GeV, this is typically GeV(^4))

---

## 8) GW spectra (sound waves, turbulence, collisions)

The model here is intentionally “phenomenology-first”:
you feed in $\alpha, \beta/H_*, T_*, g_*, v_w$, and you get $h^2\Omega(f)$.

### 8.1. Sound waves: `omega_sw_h2(f, ...)`

Uses the common fit:
$$
h^2\Omega_{\rm sw}(f)=
h^2\Omega^{\rm peak}*{\rm sw},S*{\rm sw}(f),
$$
with
$$
h^2\Omega^{\rm peak}*{\rm sw}\propto
\frac{v_w}{\beta/H**}
\left(\frac{\kappa_{\rm sw}\alpha}{1+\alpha}\right)^2
\left(\frac{100}{g_*}\right)^{1/3}
Y_{\rm sup}.
$$

* If `kappa_sw` is not provided, it uses a standard fit $\kappa_{\rm sw}(\alpha)$.
* If `y_sup` is not provided, it estimates a suppression based on a fluid RMS velocity proxy.

### 8.2. Turbulence: `omega_turb_h2(f, ...)`

Implements a standard turbulent fit:
$$
h^2\Omega_{\rm turb}(f)=
h^2\Omega^{\rm peak}*{\rm turb},S*{\rm turb}(f),
$$
with the extra $h_*$ scale (redshifted Hubble frequency at the transition).

`kappa_turb` can be specified directly, or built as
$\kappa_{\rm turb}=\epsilon\kappa_{\rm sw}$.
(Just keep in mind: turbulence modeling is the least controlled piece of this pipeline.)

### 8.3. Bubble collisions (envelope): `omega_coll_h2(f, ...)`

Implements the scalar-field “envelope approximation” component.

Default behavior:

* `kappa_coll = 0.0` unless specified (appropriate for **non-runaway** walls).

This is deliberate: in most thermal transitions, sound waves dominate;
collisions only compete in runaway or very specific regimes.

### 8.4. Total spectrum: `omega_total_h2(f, ...)`

Convenience wrapper returning a dict:

```text
{
  "sw": ...,
  "turb": ...,
  "coll": ...,
  "total": sw + turb + coll,
}
```

You can toggle any component with `include_sw/include_turb/include_coll`.

---

## 9) Stateless spectrum helpers

The `gw_*` functions reproduce the class spectrum methods but do **not** require
a `GravitationalWaveCalculator` instance.

Use these when:

* you already computed $\alpha$ and $\beta/H_*$ elsewhere,
* you are scanning a parameter space and want a fast spectrum call,
* you are validating the implementation by comparing class vs stateless output.

### Exported stateless helpers

* Peaks:

  * `gw_f_sw_peak(beta_over_H, T_star, g_star, v_w)`
  * `gw_h_star_Hz(T_star, g_star)`
  * `gw_f_turb_peak(beta_over_H, T_star, g_star, v_w)`
  * `gw_f_coll_peak(beta_over_H, T_star, g_star, v_w)`

* Spectra:

  * `gw_omega_sw_h2(f, *, alpha, beta_over_H, T_star, g_star, v_w, ...)`
  * `gw_omega_turb_h2(f, *, alpha, beta_over_H, T_star, g_star, v_w, ...)`
  * `gw_omega_coll_h2(f, *, alpha, beta_over_H, T_star, g_star, v_w, ...)`
  * `gw_omega_total_h2(f, *, alpha, beta_over_H, T_star, g_star, v_w, ...)`

---

## 10) Detector sensitivity curves (PIS, s-channel)

### 10.1. What these functions return

These are analytic polynomial-like fits from *arXiv:2002.04615* for the **PIS sensitivity**
expressed as a GW energy density, i.e.
$$
h^2\Omega_{\rm sens,PIS}(f).
$$

They are provided for:

* `lisa_sensitivity_s_pis(f_mHz)`
* `decigo_sensitivity_s_pis(f_mHz)`
* `bbo_sensitivity_s_pis(f_mHz)`

### 10.2. Frequency convention (do not mix this up)

These functions take **frequency in mHz**:

* argument name: `f_mHz`
* internally: $x_s = f/(1,{\rm mHz})$, so numerically $x_s = f_{\rm mHz}$

They return $h^2\Omega_{\rm sens,PIS}(f)$ as a dimensionless array.

### 10.3. Implementation helper

`_pisc_poly_sum(f_mHz, terms, scale)` computes:
$$
{\rm scale}\times \sum_i c_i x^{p_i},
\qquad x=f_{\rm mHz}.
$$

---

## 11) Units and sanity checks (my “save your future self” section)

This module mixes three unit conventions that you must keep consistent:

1. **Field theory / thermodynamics**: natural units
   Temperatures typically in **GeV**, potentials in **GeV(^4)**.

2. **GW spectra**: frequencies intended in **Hz**, output as $h^2\Omega(f)$ dimensionless.

3. **Sensitivity fits**: frequencies explicitly in **mHz** (per the fit convention in the paper).

### Practical rule

* When plotting spectra against sensitivities:

  * either convert your **spectrum frequency grid** to mHz when using the PIS fits,
  * or convert the PIS fit argument back to Hz by `f_mHz = 1e3 * f_Hz`.

### Another implementation note (same spirit as the $\alpha$ comment)

In the peak-frequency helpers the prefactors are labeled as “mHz” in comments,
but the functions are documented as returning “Hz”.
Make sure your plotting layer uses a consistent convention (or explicitly converts).

---

## 12) Minimal usage sketch (end-to-end logic)

This is the conceptual flow (not a full example script):

```text
# 1) Build phases (transitionFinder)
phases = traceMultiMin(...)

# 2) Choose the tunneling pair and build a calculator
gwcalc = GravitationalWaveCalculator(
    V=V, dV=dV, dVdT=dVdT,
    phases=phases,
    high_phase_key="high",
    low_phase_key="low",
    fullTunneling_params={...},
)

# 3) Evaluate thermodynamics at Tn (or your chosen T*)
betaH = gwcalc.beta_over_H(Tn, dT=0.1)
alpha = gwcalc.alpha(Tn, g_star=106.75)
RH    = gwcalc.bubble_radius_over_H(betaH, v_w=0.6)

# 4) Build GW spectrum
f = np.logspace(-5, 1, 2000)  # Hz
spec = gwcalc.omega_total_h2(f, alpha=alpha, beta_over_H=betaH, T_star=Tn, g_star=106.75, v_w=0.6)

# 5) Compare to sensitivity (convert frequency!)
f_mHz = 1e3 * f
lisa = lisa_sensitivity_s_pis(f_mHz)
```

---

## 13) Known limitations (by design)

This module intentionally does *not* try to decide the subtle physics choices for you:

* it does not compute percolation temperature, reheating, or full nucleation history;
* it does not solve hydrodynamics (so $v_w$, $\kappa$, and suppression factors are model-dependent);
* it does not validate whether bubble collisions should be included (default assumes non-runaway walls);
* it provides standard fits meant for quick forecasts and intuition building.

That’s the point: it gives you a clean, inspectable “first-pass GW layer” on top of the
core tunneling / phase-tracing machinery.

---

## 14) Function-by-function index

### Class: `GravitationalWaveCalculator`

* `dS_dT(T, dT)`
* `beta_over_H(Tn, dT, *, H=None)`
* `alpha(T, g_star, *, return_delta_rho=False)`
* `omega_sw_h2(f, *, alpha, beta_over_H, T_star, g_star, v_w=1.0, ...)`
* `omega_turb_h2(f, *, alpha, beta_over_H, T_star, g_star, v_w=1.0, ...)`
* `omega_coll_h2(f, *, alpha, beta_over_H, T_star, g_star, v_w=1.0, ...)`
* `omega_total_h2(f, *, alpha, beta_over_H, T_star, g_star, v_w=1.0, ...)`
* `nucleation_rate(T, *, A_scale=1.0, use_linde_prefactor=True)`
* `bubble_radius_over_H(beta_over_H, v_w=1.0)`

### Stateless helpers

* `gw_f_sw_peak`, `gw_omega_sw_h2`
* `gw_h_star_Hz`, `gw_f_turb_peak`, `gw_omega_turb_h2`
* `gw_f_coll_peak`, `gw_omega_coll_h2`
* `gw_omega_total_h2`

### Sensitivity curves (PIS, s-channel; *arXiv:2002.04615*)

* `lisa_sensitivity_s_pis(f_mHz)`
* `decigo_sensitivity_s_pis(f_mHz)`
* `bbo_sensitivity_s_pis(f_mHz)`

