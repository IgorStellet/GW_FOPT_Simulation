# CosmoTransitions (modernized)

A modern, test-driven update of **CosmoTransitions** for studying **first-order phase transitions (FOPTs)** and their **gravitational-wave** signatures — now in Python 3.13, with clearer APIs, numerical utilities, examples, and documentation.

> Original project by **Carroll L. Wainwright** (MIT).  
> Modernization by **Igor Almeida da Silva Gouvêa Stellet** (advisor: **Felipe Tovar Falciano**).

---
## Quick links

- 🧭 **Roadmap & Schedule** → [roadmap.md](roadmap.md)  
- 🧩 **Architecture & Module Flow** → [architecture.md](architecture.md)

---

## Examples (Core modules only)

If you are only interested in learning how to use the main modules and obtain the results/graphs for the phase transition, here is the place.
Below are links to explanations of only the most important aspects needed to use each main function and get the respective results/plots.

The main explanation of the codes used and all images can be found in the [example](examples) folder.

### Tunneling 1D
  - Single Field Instaton → [example_single_field.md](examples/example_tunneling1D.md)

### Transition Finder
  - TF example → [example_transitionFinder.md](examples/example_transitionFinder.md)

### Generic Potential & Gravitational Waves
  - GP & GW example → [docs/examples/example_paper](examples/example_paper.md)


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

### Tunneling1D Functions
  - Single Field functions: [modules/tunneling1D/single_field](modules/tunneling1D/single_field.md)
  - Tests of SF functions: [modules/tunneling1D/tests_single_field](modules/tunneling1D/tests_single_field.md)

### transitionFinder Functions
  - Transition Finder Complete: [modules/transitionFinder/](modules/transitionFinder/transitionFinder.md)
  - Tests of TF functions: [modules/transitionFinder/tests_](modules/transitionFinder/tests_transitionFinder.md)

### generic_potential Functions
  - Generic Potential Complete: [modules/generic_potential/](modules/generic_potential/generic_potential.md)
  - Example using of GP functions: [docs/examples/example_paper](examples/example_paper.md)

### gravitational_Waves Functions
  - Gravitational Waves Complete: [modules/generic_potential/](modules/gravitational_Waves/gravitational_Waves.md)
  - Example using of GW functions: [docs/examples/example_paper](examples/example_paper.md)





---

## Install (dev) & Quick Start

> Requires Python 3.11+ (targeting 3.13). We recommend a fresh virtualenv.

```bash
python -m pip install -U pip
pip install -e .[dev]   # editable install with dev deps (pytest, ruff, black)
```


---
