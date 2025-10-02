# Spline Thermal Integrals (Jb, Jf)

---

## Spline Thermal Integrals (J_f)

### Purpose

Provide a **fast, differentiable surrogate** for the exact fermionic thermal integral

$$J_f(\theta)\equiv J_f\bigl(x^2\bigr)\quad \theta=(m/T)^2\in\mathbb{R}$$

by fitting a **cubic B-spline** to samples of the exact function (J_f(\theta)) on a **non-uniform grid**.
The spline preserves the **legacy API/behavior**:

* Input is **($\theta=x^2$)** (not (x)); ($\theta$) can be **negative** (imaginary (x) branch).
* For ($\theta < \theta_{\min}$): returns the **clamped** value ($J_f(\theta_{\min})$).
* For ($\theta > \theta_{\max}$): returns **0** (and its derivatives 0), matching the physical larg -mass suppression in the legacy code.
* Supports **derivatives w.r.t. ($\theta$)** via `n` (uses `BSpline.derivative(n)`).

**Domain used:**
($\theta_{\min} = -6.82200203,\quad \theta_{\max} = 1.35\times 10^3.$)

> Notes:
> * The spline is built from `Jf_exact2(theta)` (exact quadrature) and cached to disk for reproducibility/speed.
> * Physically meaningful inputs here are **real ($\theta$)**. A complex and real (x) (thus complex ($\theta$)) has no direct meaning in this spline API—use the exact routines if you need analytic continuation details.

---

### `Jf_spline`

#### Signature

```python
Jf_spline(X: float | np.ndarray, n: int = 0) -> float | np.ndarray
```

#### Parameters

* `X` (`float | array_like`): Input **theta** values ($\theta=(m/T)^2$). Scalar-in → scalar-out.
* `n` (`int`, default `0`): Derivative **order w.r.t. (\theta)** (0 for the function value, 1 for first derivative, etc).

#### Returns

* `out` (`float | ndarray`): ($J_f(\theta)$) (or its (n)-th ($\theta$)-derivative) evaluated by the spline.
  Behavior outside the fit domain:

  * If ($\theta < \theta_{\min}$): returns the **constant** value at (\theta_{\min}).
  * If ($\theta > \theta_{\max}$): returns **0.0** (derivatives also 0.0), per legacy behavior.

#### Notes

* Backend: `scipy.interpolate.BSpline` (created via `make_interp_spline(..., k=3)`).
* The grid is **denser** near ($\theta\le 0$) and small positive ($\theta$), and **sparser** on the large-($\theta$) tail.
* Accuracy is typically at the **few × $10^{-6}–10^{-8}$** level relative to the exact integral over the intended domain (depends on the internal grid sizes).

---

### `_ensure_Jf_spline`

#### Signature

```python
_ensure_Jf_spline() -> scipy.interpolate.BSpline
```

#### Purpose

Construct (once) and return the **global** `BSpline` object for (J_f(\theta)).
First tries to **load** a cached spline; if not found, it **builds** the dataset with `Jf_exact2()`, fits a cubic spline, and **saves** the spline parameters to disk.

#### Notes

* The cache file (by default) is `Jf_spline_v1.npz`, containing the **knot vector** `t`, **coefficients** `c`, and degree `k`.
* If the directory is **read-only**, the code silently **skips** saving (still works in-memory).

---

### `_build_Jf_dataset`

#### Signature

```python
_build_Jf_dataset(n_neg: int = 420, n_pos_lin: int = 380, n_pos_log: int = 300) -> tuple[np.ndarray, np.ndarray]
```

#### Purpose

Generate a **non-uniform theta grid** and its exact values ($J_f(\theta)$) used to fit the spline.

* Negative branch: **linear** grid on ($[\theta_{\min}, 0]$) (dense).
* Positive small: **linear** grid on ($[0, 50]$).
* Positive tail: **log** grid on $(50, \theta_{\max}])$.

Returns `(theta, y)` with `y = Jf_exact2(theta)`.

---

### `_load_Jf_cache` / `_save_Jf_cache`

#### Signatures

```python
_load_Jf_cache() -> None | tuple[np.ndarray, np.ndarray, int]
_save_Jf_cache(theta: np.ndarray, coeffs: np.ndarray, t: np.ndarray, k: int) -> None
```

#### Purpose (brief)

* `_load_Jf_cache`: try to **load** a previously saved spline `(t, c, k)`. Returns `None` if not available.
* `_save_Jf_cache`: best-effort **save** of spline parameters to disk so subsequent runs are instantaneous.

---

### Reproducibility & Performance

* The spline is deterministic given the internal grid sizes and the exact integrator tolerances (in `Jf_exact2`).
* First build takes **~seconds** (quadrature over a few hundred points), thereafter **~milliseconds** per call thanks to evaluation of a cubic B-spline (and its derivatives).

---

## Spline Thermal Integrals (J_b)

### Purpose

Provide a **fast, differentiable surrogate** for the exact bosonic thermal integral
($J_b(\theta) \equiv J_b(x^2)$) with ($\theta=(m/T)^2\in\mathbb{R}$), by fitting a **cubic B-spline** to samples of the exact function ($J_b(\theta)$) on a **non-uniform grid**.
The spline preserves the **legacy API/behavior**:

* Input is **($\theta=x^2$)** (not (x)); ($\theta$) may be **negative** (imaginary-mass branch handled via the exact routine during fitting).
* For ($\theta < \theta_{\min}$): return the **clamped** value ($J_b(\theta_{\min})$).
* For ($\theta > \theta_{\max}$): return **0.0** (and derivatives 0.0), matching the legacy tail suppression.
* Support **derivatives w.r.t. (\theta)** via parameter `n` (uses `BSpline.derivative(n)`).

**Domain used (legacy-compatible):**
($\displaystyle \theta_{\min} = -3.72402637,\qquad \theta_{\max} = 1.41\times 10^3.$)

> Physically meaningful inputs here are **real ($\theta$)**. Complex (x) (thus complex ($\theta$)) is not used in this spline API; for analytic-continuation details use the **exact** routines.

---

### `Jb_spline`

#### Signature

```python
Jb_spline(X: float | np.ndarray, n: int = 0) -> float | np.ndarray
```

#### Parameters

* `X` (`float | array_like`): Input **theta** values ($\theta=(m/T)^2$). Scalar-in → scalar-out.
* `n` (`int`, default `0`): Derivative **order w.r.t. (\theta)** (0 for the function value, 1 for first derivative, etc.).

#### Returns

* `out` (`float | ndarray`): ($J_b(\theta)$) (or its (n)-th ($\theta$)-derivative) evaluated by the spline.
  Outside the fitted domain:

  * If ($\theta < \theta_{\min}$): returns the **constant** value at ($\theta_{\min}$).
  * If ($\theta > \theta_{\max}$): returns **0.0** (derivatives also 0.0), per legacy behavior.

#### Notes

* Backend: `scipy.interpolate.BSpline` (created via `make_interp_spline(..., k=3)`).
* The grid used for fitting is **denser** near ($\theta\le 0$) and small positive ($\theta$), and **sparser** on the large-($\theta$) tail.
* Choice of ($\theta_{\min}$) coincides with the **minimum** of ($J_b$), so the clamp at the left boundary plus the vanishing right tail makes the evaluated curve **monotonic increasing** and ($C^1$)-continuous (matching the legacy intent).

---

### Spline construction & caching (same pattern as *Jf_spline*)

The internal helpers follow the **same design** as for `Jf_spline`:

* **Dataset build**: generate a **non-uniform** (\theta) grid
  (linear on ($[\theta_{\min},0]$), linear on ($[0,50]$), logarithmic on ((50,\theta_{\max}])) and compute **ground-truth** values via `Jb_exact2(theta)`.
  *(Function: `_build_Jb_dataset`.)*
* **Spline fit**: build a **cubic interpolating spline** (`make_interp_spline`) with `extrapolate=False`.
  *(Created inside `_ensure_Jb_spline`.)*
* **Caching**: first try to **load** a cached spline `(t, c, k)` from
  `Jb_spline_v1.npz` under `spline_data_path`; if not present, **fit** and then **save** best-effort.
  *(Functions: `_load_Jb_cache`, `_save_Jb_cache`, called by `_ensure_Jb_spline`.)*

**Performance:** first build takes seconds (due to exact quadrature at a few hundred (\theta) nodes); subsequent runs are **milliseconds** (BSpline evaluation).
**Reproducibility:** the cache pins the fitted spline; if the directory is read-only, the code still works (keeps the spline in memory).

---
