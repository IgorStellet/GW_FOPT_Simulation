# GW_FOPT_Simulation

With the emergence and rapid development of gravitational waves (GW's) detectors such as LIGO, LISA, Einstein Telescope [e.g., Virgo, KAGRA, Cosmic Explorer], studies of GW's generation mechanisms and sources have become essential and quite “hot” in the field of physics over the last few years. Cosmology is, of course, part of this effort. One of the mechanisms that has received growing attention is the production of GW's from first-order phase transitions (FOPTs). This mechanism can generate a cosmological stochastic background of GW's which, if detected, can probe the scalar sector and high energy scales still inaccessible to terrestrial colliders, offering insights into the early-universe dynamics, and provide a way to test theories beyond the Standard Model (that make FOPTs).

However, FOPTs face a central challenge: starting from a model’s effective potential, one must find bounce solutions and the thermodynamic parameters that characterize the phase transition in order to finally predict the gravitational wave spectrum, its peak frequencies, and other properties. This requires robust numerical codes to simulate the theory and obtain reliable results, so that we can assess detectability and frequency bands of the GWs.

Today, several codes are used in the literature toward this goal: AnyBubble, BubbleProfiles, FindBounce, and others—mostly written in C++. Notably, there is also **CosmoTransitions**, written in Python.

Because it is simple, open, and implemented in a widely used language, CosmoTransitions remains a common choice in the literature, even though its author, Carroll L. Wainwright, left the field and the code dates back to 2011 (i.e., it is no longer up to date).[Original repo: GitHub:https://github.com/clwainwright/CosmoTransitions]

Recognizing the code’s relevance to the community, and to my Master’s project on FOPTs, I, Igor Almeida da Silva Gouvêa Stellet, together with my advisor, Felipe Tovar Falciano, set out to update it to a modern Python version (3.13), improving its syntax, indentation, and incode explanations across modules. Most importantly, I am adding examples and plots for every function in each module, as a new layer of consistency checks that did not exist in the original version. I am also adding clearer error handling in functions, so users can more easily diagnose where something “breaks.” Thanks to the previous author's work in making an already very good code, I have the possibility to improve it now with a lot of details.

The goal is not only to improve my programming skills, but also my academic and professional development, my GitHub presence, the preparation for my own Master’s research, and, above all that, to share with the community a widely used code in a more modern form that can serve as a foundation for future research and researchers (even though C/C++ codes will often remain more performant).

Therefore, this project aims to deliver substantial improvements to the original code, making it more optimized, modern, and intuitive, with modules, docstrings, and functions better explained and documented, and with example-based tests to build intuition. I am also adding more explicit error signals in each function. Below you will find the overall plan and organization of the project.

📅 Schedule & Approach: The project’s timeline and flowchart are presented below. The main idea is to split this long task into three primary phases, each lasting up to one month, while running consistency tests throughout development and after finishing each phase. Each phase follows the cycle: Modification → Testing → Fixes → Validation.

The first phase aims to update CosmoTransition's auxiliary modules, which are called by the main modules. The second phase, the codes that find the bounce solution and the thermodynamic parameters (two main modules). Finally, the third and final phase aims to modify the functions that create the generic potential and the plots generated given the initial parameters. Everything will be done for the 1D part for now; the 2D part will remain as before.

Depending on the progress of the project, a fourth phase will be carried out to add new plots and graphs to the code, as well as update the part that calculates multiple fields.

### 📖 Documentation & Recommended Reading (pre-modifications)
Before modifying any module, consult the official documentation and/or the original paper to understand the algorithms:

* **Official Documentation:**[https://clwainwright.net/CosmoTransitions/index.html](https://clwainwright.net/CosmoTransitions/index.html)
* **Original Paper (arXiv):** [arXiv:1109.4189](https://arxiv.org/abs/1109.4189)
* **Computer Physics Communications:** [10.1016/j.cpc.2012.04.004](https://doi.org/10.1016/j.cpc.2012.04.004
