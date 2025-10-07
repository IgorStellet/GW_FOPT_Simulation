# CosmoTransitions (modernized)

A modern, test-driven update of **CosmoTransitions** for studying **first-order phase transitions (FOPTs)** and their **gravitational-wave** signatures — now in Python 3.13, with clearer APIs, numerical utilities, examples, and documentation.

> Original project by **Carroll L. Wainwright** (MIT).  
> Modernization by **Igor Almeida da Silva Gouvêa Stellet** (advisor: **Felipe Tovar Falciano**).

---
## Quick links

- 🧭 **Roadmap & Schedule** → [roadmap.md](roadmap.md)  
- 🧩 **Architecture & Module Flow** → [architecture.md](architecture.md)

---

## Index of all functions

---

### Helper Functions
  - Miscellaneous Functions → [miscellaneous_functions.md](modules/helper_functions/Miscellaneous_functions.md)
  - Numerical integration Functions → [Numerical_integration.md](modules/helper_functions/Numerical_integration.md)
  - Numerical Derivatives Functions → [Numerical_derivatives.md](modules/helper_functions/Numerical_derivatives.md)
  - Interpolation Functions → [intepolation_functions.md](modules/helper_functions/interpolation_functions.md)

### Finite T Functions
  - Exact Thermal Integrals → [Exact_Thermal_Integrals.md](modules/finiteT/Exact_Thermal_Integrals.md)
  - Spline Thermal Integrals →  [Spline_Thermal_Integrals.md](modules/finiteT/Spline_Thermal_Integrals.md)
  - Approx Thermal Integrals →  [Approx_Thermal_Integrals.md](modules/finiteT/Approx_Thermal_Integrals.md)
  - Short Hand for all Thermal Integrals → [Short_Hand_Jb&Jf.md](modules/finiteT/Short_Hand_Jb&Jf.md)

---

## Install (dev) & Quick Start

> Requires Python 3.11+ (targeting 3.13). We recommend a fresh virtualenv.

```bash
python -m pip install -U pip
pip install -e .[dev]   # editable install with dev deps (pytest, ruff, black)
```


---
