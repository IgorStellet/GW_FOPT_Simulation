# GW_FOPT_Simulation

With the emergence and rapid development of gravitational waves (GW's) detectors such as LIGO, LISA, Einstein Telescope [e.g., Virgo, KAGRA, Cosmic Explorer], studies of GW's generation mechanisms and sources have become essential and quite “hot” in the field of physics over the last few years. Cosmology is, of course, part of this effort. One of the mechanisms that has received growing attention is the production of GW's from first-order phase transitions (FOPTs). This mechanism can generate a cosmological stochastic background of GW's which, if detected, can probe the scalar sector and high energy scales still inaccessible to terrestrial colliders, offering insights into the early-universe dynamics, and provide a way to test theories beyond the Standard Model (that make FOPTs).

However, FOPTs face a central challenge: starting from a model’s effective potential, one must find bounce solutions and the thermodynamic parameters that characterize the phase transition in order to finally predict the gravitational wave spectrum, its peak frequencies, and other properties. This requires robust numerical codes to simulate the theory and obtain reliable results, so that we can assess detectability and frequency bands of the GWs.

Today, several codes are used in the literature toward this goal: AnyBubble, BubbleProfiles, FindBounce, and others—mostly written in C++. Notably, there is also **CosmoTransitions**, written in Python.

Because it is simple, open, and implemented in a widely used language, CosmoTransitions remains a common choice in the literature, even though its author, Carroll L. Wainwright, left the field and the code dates back to 2011 (i.e., it is no longer up to date).[Original repo: GitHub:https://github.com/clwainwright/CosmoTransitions]

Recognizing the code’s relevance to the community, and to my Master’s project on FOPTs, I, Igor Almeida da Silva Gouvêa Stellet, together with my advisor, Felipe Tovar Falciano, set out to update it to a modern Python version (3.13), improving its syntax, indentation, and incode explanations across modules. Most importantly, I am adding examples and plots for every function in each module, as a new layer of consistency checks that did not exist in the original version. I am also adding clearer error handling in functions, so users can more easily diagnose where something “breaks.” Thanks to the previous author's work in making an already very good code, I have the possibility to improve it now with a lot of details.

The goal is not only to improve my programming skills, but also my academic and professional development, my GitHub presence, the preparation for my own Master’s research, and, above all that, to share with the community a widely used code in a more modern form that can serve as a foundation for future research and researchers (even though C/C++ codes will often remain more performant).

Therefore, this project aims to deliver substantial improvements to the original code, making it more optimized, modern, and intuitive, with modules, docstrings, and functions better explained and documented, and with example-based tests to build intuition. I am also adding more explicit error signals in each function. Below you will find the overall links and organization of the project.

---

## 🔗 Quick links

- 📚 **Documentation:** see [docs/index.md](docs/index.md), [Roadmap & Schedule](docs/roadmap.md) and [Architecture & Module Flow](docs/architecture.md)  
- 🧩 **New code (modernized):** [`src/CosmoTransitions`](src/CosmoTransitions/)  
- 🧪 **Tests:** [`tests/`](tests/) 
- 📓 **Examples:** [`examples/`](examples/)  
- 🗄️ **Legacy (original layout):** [`legacy/cosmoTransitions`](legacy/cosmoTransitions/)

---

## 🚦 Project status

- **Phase 0** (planning): ✅ done  
- **Phase 1.1** (modifying helper functions): ✅ done  
- **Phase 1.5.1** (tests of helper functions ): ✅ done
- **Phase 1.2** (modifying finiteT): in progress — see [Roadmap](docs/roadmap.md)

---

## ⚙️ Install (dev) & quick start

```bash
python -m pip install -U pip
pip install -e .[dev]   # editable install with dev deps (pytest, ruff, black)
```
---
## 📜 License & citation
- License: **MIT** (see [`LICENSE`](LICENSE)) — original notices preserved.
- If you use this code, please cite **both**:
  1)**The original CosmoTransitions paper** (Wainwright, 2012) and
  2)**This modernization** (software citation below).
**BibTeX**

```bibtex
@software{cosmotransitions_modernized,
  title        = {CosmoTransitions (modernized): Python utilities for cosmological phase transitions and GW forecasts},
  author       = {Igor Almeida da Silva Gouvêa Stellet and Felipe Tovar Falciano},
  year         = {2025},
  version      = {0.1.0},
  url          = {https://github.com/IgorStellet/GW_FOPT_Simulation.git}
}

@article{wainwright2012cosmotransitions,
  title   = {CosmoTransitions: Computing Cosmological Phase Transition Temperatures and Bubble Profiles With Multiple Fields},
  author  = {Wainwright, Carroll L.},
  journal = {Computer Physics Communications},
  volume  = {183},
  number  = {10},
  pages   = {2006--2013},
  year    = {2012},
  doi     = {10.1016/j.cpc.2012.04.004},
  eprint  = {1109.4189},
  archivePrefix = {arXiv}
}
```
---
## 🙏 Acknowledgments

Thanks to **Carroll L. Wainwright** for the original CosmoTransitions (MIT).  
Thanks to **Felipe Tovar Falciano** for advising and guidance.  
Thanks to the community for feedback, issues, and suggestions.

---
