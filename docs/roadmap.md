# 📅 Roadmap & Schedule

**Approach**: The project’s timeline and flowchart are presented below. The main idea is to split this long task into four primary phases, each lasting up to ~2 weeks, while running consistency tests throughout development and after finishing each phase. Each phase follows the cycle: Modification → Testing → Fixes → Validation.

- See also: [Architecture & Module Flow](architecture.md)

---
## Phases (overview)
  The **first phase** aims to update CosmoTransition's **auxiliary modules**, which are called by the main modules. The **second phase** and **third phase**, the codes that find the bounce solution and the thermodynamic parameters (**two main modules**). Finally, the **fourth and final phase** aims to modify the functions that create the **generic potential** and the plots generated given the initial parameters. Everything will be done for the 1D part for now; the 2D part will remain as before.

Depending on the progress of the project, a **fifth phase** will be carried out to add new plots and graphs to the code, as well as update the part that calculates multiple fields.

---

## 📖 Documentation & Recommended Reading (pre-modifications)
Before modifying any module, consult the official documentation and/or the original paper to understand the algorithms:

- **Official Documentation:**[https://clwainwright.net/CosmoTransitions/index.html](https://clwainwright.net/CosmoTransitions/index.html)
- **Original Paper (arXiv):** [arXiv:1109.4189](https://arxiv.org/abs/1109.4189)
- **Computer Physics Communications:** [10.1016/j.cpc.2012.04.004](https://doi.org/10.1016/j.cpc.2012.04.004)

---
## Project Timeline


```mermaid
gantt
    %%%%%%%%
    title Development Timeline - CosmoTransitions
    dateFormat  YYYY-MM-DD
    axisFormat  %d/%m
    
    section Phase 0: Planning
    Flowchart and Timeline          :done, 2025-08-27, 7d
    Methodology Definition          :done, 2025-09-05, 10d
    
    section Phase 1: Auxiliar Functions Update
    Modify helper_function.py       :done, 2025-09-08, 12d
    Modify finiteT.py               :done, 2025-09-20, 12d
    
    section Phase 1.5: Testing the Modifications of Auxiliar Functions
    Testing all modifications       :done,2025-10-02, 4d
    Fixes and Adjustments           :done,2025-10-02, 4d
    
    section Phase 2: Tunneling 1D Module
    Modify/tests Tunneling1D.py     :done,2025-10-07, 12d
    Examples Tunneling1D.py         :done,2025-10-20, 3d
    
    section Phase 3: Transtions Finder Module
    Modify/tests transitionFinder.py :done,2025-10-24, 12d      
    Examples transitionFinder        :done,2025-11-06, 3d
    
    section Phase 4: Generic Potential Module
    Modify/tests generic_potential.py  :2025-11-10, 12d
    Examples generic_potential.py      :2025-11-25, 5d
    
    section Phase 5: Extras (Optional)
    Additional Plots                :2025-12-01, 10d
    Multi-field Solution            :2025-12-10, 10d
    Final Optimizations             :2025-12-10, 10d
```


---

## Milestones & Checklist
- [x] **Phase 0**: Planning and first meeting 
  - [x] Create dependency flowchart 
  - [x] Create refactoring schedule 

- [x] **Phase 1**: Auxiliary functions 
  - [x] Refactor `helper_functions.py`
  - [x] Refactor `finiteT.py`

- [x] **Phase 1.5**: Modification tests 
  - [x] Validate isolated functions with simple analytic examples  
  - [x] exercise error paths and validations

- [x] **Phase 2**: Tunneling 1D Core Module 
  - [x] Refactor `tunneling1D.py` 
  - [x] Examples modernized `tunneling1D.py` 

- [x] **Phase 3**: Transtions Finder Core Module 
  - [x] Refactor `transtionFinder.py` 
  - [x] Test modernized `transtionFinder.py`

- [ ] **Phase 4**: Generic Potential Core Module  
  - Refactor `generic_potential.py` 
  - Test modernized `generic_potential.py`
  - Run all old examples and validate cosistency between versions

- [ ] **Phase 5** *(optional)*: Extensions
  - Update plotting, add energy density and other figures useful for paper/thesis and other parameters
  - New plot types (e.g., direct GW spectrum, GW energy density vs T, etc.) 
  - Modernize multi-field plotting codes `mult_field_plotting.py` and `path_deformation.py`
---

```mermaid
graph TD
    Start[Project Start] --> Phase1[Phase 1: Auxiliar Functions]
    Start --> Phase2[Phase 2: Tunneling 1d]
    Start --> Phase3[Phase 3: Transition Finder]    
    Start --> Phase4[Phase 4: Generic Potential]
    
    Phase1 --> Test1[Consistency Tests]
    Phase2 --> Test2[Consistency Tests]
    Phase3 --> Test3[Consistency Tests]
    Phase4 --> Test4[Consistency Tests]
    
    Test1 --> Adjust1[Adjustments, Documentation & Fixes]
    Test2 --> Adjust2[Adjustments, Documentation & Fixes]
    Test3 --> Adjust3[Adjustments, Documentation & Fixes]
    Test4 --> Adjust4[Adjustments, Documentation & Fixes]
    
    Adjust1 --> FinalValidation[Final Validation]
    Adjust2 --> FinalValidation
    Adjust3 --> FinalValidation
    Adjust4 --> FinalValidation
    
    FinalValidation --> Decision{Satisfactory progress?}
    
    Decision -- Yes --> Phase5[Phase 5: New Plots and Figures, 3D+ & More parameters]
    Decision -- No --> Review[Review and Optimizations]
    
    Phase5 --> ProjectEnd[Project Completed]
    Review --> ProjectEnd

    style Start fill:#e1f5fe
    style Phase1 fill:#e8f5e8
    style Phase2 fill:#fff3e0
    style Phase3 fill:#f3e5f5
    style Phase4 fill:#ffebee
    style Phase5 fill:#ffebee
    style ProjectEnd fill:#c8e6c9
```



