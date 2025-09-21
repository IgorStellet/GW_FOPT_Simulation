📅 Schedule & Approach: The project’s timeline and flowchart are presented below. The main idea is to split this long task into three primary phases, each lasting up to one month, while running consistency tests throughout development and after finishing each phase. Each phase follows the cycle: Modification → Testing → Fixes → Validation.

The first phase aims to update CosmoTransition's auxiliary modules, which are called by the main modules. The second phase, the codes that find the bounce solution and the thermodynamic parameters (two main modules). Finally, the third and final phase aims to modify the functions that create the generic potential and the plots generated given the initial parameters. Everything will be done for the 1D part for now; the 2D part will remain as before.

Depending on the progress of the project, a fourth phase will be carried out to add new plots and graphs to the code, as well as update the part that calculates multiple fields.

### 📖 Documentation & Recommended Reading (pre-modifications)
Before modifying any module, consult the official documentation and/or the original paper to understand the algorithms:

* **Official Documentation:**[https://clwainwright.net/CosmoTransitions/index.html](https://clwainwright.net/CosmoTransitions/index.html)
* **Original Paper (arXiv):** [arXiv:1109.4189](https://arxiv.org/abs/1109.4189)
* **Computer Physics Communications:** [10.1016/j.cpc.2012.04.004](https://doi.org/10.1016/j.cpc.2012.04.004

## Project Timeline

```mermaid
gantt
    title Development Timeline - CosmoTransitions
    dateFormat  YYYY-MM-DD
    axisFormat  %d/%m
    
    section Phase 0: Planning
    Flowchart and Timeline          :done, 2025-08-27, 7d
    Methodology Definition          :done, 2025-09-05, 10d
    
    section Phase 1: Numerical Integration Function Updates
    Modify helper_function.py       :done, 2025-09-08, 10d
    Modify finiteT.py               :active, 2025-09-19, 10d
    
    section Phase 1.5: Testing the Modifications
    Testing all modifications       :2025-09-30, 5d
    Fixes and Adjustments           :2025-09-30, 5d
    
    section Phase 2: Bounce-Solution Functions
    Modify Tunneling1D.py           :2025-10-01, 12d
    Modify transitionsFinder.py     :2025-10-12, 12d
    
    section Phase 2.5: Intermediate Testing
    Bounce-Solution Tests           :2025-10-24, 5d
    Numerical Validation            :2025-10-24, 5d
    
    section Phase 3: Potentials and Function Outputs
    Modify generic_potential.py     :2025-11-03, 18d
    
    section Phase 3.5: Final Tests
    Full Test Suite                 :2025-11-27, 6d
    Documentation                   :2025-11-27, 6d
    
    section Phase 4: Extras (Optional)
    Additional Plots                :2025-12-01, 10d
    Multi-field Solution            :2025-12-10, 10d
    Final Optimizations             :2025-12-10, 10d
```

- [x] **Phase 0**: Planning and first meeting 
  - Create dependency flowchart 
  - Create refactoring schedule 

- [ ] **Phase 1**: Numrical core (auxiliary functions) 
  - Refactor `helper_functions.py` 
  - Refactor `finiteT.py`

- [ ] **Phase 1.5**: Modification tests 
  - Validate isolated functions with simple analytic examples  
  - exercise error paths and validations

- [ ] **Phase 2**: Bounce solution and transition parameters 
  - Refactor `tunneling1D.py`  
  - Improve `transitionFinder.py` (more efficiente search algorithms)  

- [ ] **Phase 2.5**: Intermediate tests  
  - Reproduce example results (`fullTunneling.py`)  
  - Compare critical actions with the legacy version 

- [ ] **Phase 3**: Potential and outputs  
  - Modernize `generic_potential.py` 
  - Update plotting, add energy density and other figures useful for paper/thesis

- [ ] **Phase 3.5**: Final tests
  - Run all examples and validate cosistency 
  - Create notebooks replacing scripts

- [ ] **Phase 4** *(optional)*: Extensions
  - New plot types (e.g., direct GW spectrum, GW energy density vs T, etc.) 
  - Modernize multi-field plotting codes `mult_field_plotting.py` and `path_deformation.py`

```mermaid
graph TD
    Start[Project Start] --> Phase1[Phase 1: Numerical Integration]
    Start --> Phase2[Phase 2: Bounce Solution]
    Start --> Phase3[Phase 3: Potential and Visualization]
    
    Phase1 --> Test1[Consistency Tests]
    Phase2 --> Test2[Consistency Tests]
    Phase3 --> Test3[Consistency Tests]
    
    Test1 --> Adjust1[Adjustments and Fixes]
    Test2 --> Adjust2[Adjustments and Fixes]
    Test3 --> Adjust3[Adjustments and Fixes]
    
    Adjust1 --> FinalValidation[Final Validation]
    Adjust2 --> FinalValidation
    Adjust3 --> FinalValidation
    
    FinalValidation --> Decision{Satisfactory progress?}
    
    Decision -- Yes --> Phase4[Phase 4: New Plots and Figures]
    Decision -- No --> Review[Review and Optimizations]
    
    Phase4 --> ProjectEnd[Project Completed]
    Review --> ProjectEnd

    style Start fill:#e1f5fe
    style Phase1 fill:#e8f5e8
    style Phase2 fill:#fff3e0
    style Phase3 fill:#f3e5f5
    style Phase4 fill:#ffebee
    style ProjectEnd fill:#c8e6c9
```


