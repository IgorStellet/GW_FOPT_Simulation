# Architecture & Module Flow
This page summarizes how modules relate and call each other during typical workflows.

> See also: per-module pages under **Docs → [Modules](modules)**.

---
## Flowchart of the modules

```mermaid
graph TD
    %% ========== MAIN MODULES ==========
    subgraph "MAIN MODULES"
        T1D[Tunneling1D<br/>Bounce solution in 1 field]
        PD[pathDeformation<br/>Bounce solution in multiple fields]
        TF[transitionFinder<br/>Locate Tn and phase structure]
        GP[generic_potential<br/>Model definition and potential plotting]
        
        T1D --> PD
        T1D --> TF
        T1D --> GP
        PD --> TF
        PD --> GP
        TF --> GP
    end

    %% ========== AUXILIARY MODULES ==========
    subgraph "AUXILIARY MODULES"
        HF[helper_functions<br/>Utility functions]
        FT[finiteT<br/>Finite-temperature corrections]
        MFP[multi_field_plotting<br/>Visualization for 3+ fields]
    end

    %% ========== DEPENDENCIES ==========
    AUX --> MAIN
    
    %% ========== STYLES ==========
    style T1D fill:#357a38,color:white
    style PD fill:#d32f2f,color:white
    style TF fill:#357a38,color:white
    style GP fill:#357a38,color:white
    style HF fill:#1565c0,color:white
    style FT fill:#1565c0,color:white
    style MFP fill:#d32f2f,color:white
    
    linkStyle 0 stroke:#1b5e20,stroke-width:2px
    linkStyle 1 stroke:#1b5e20,stroke-width:2px
    linkStyle 2 stroke:#1b5e20,stroke-width:2px
    linkStyle 3 stroke:#1b5e20,stroke-width:2px
    linkStyle 4 stroke:#1b5e20,stroke-width:2px
    linkStyle 5 stroke:#1b5e20,stroke-width:2px
    linkStyle 6 stroke:#0d47a1,stroke-width:3px
```
---

## 📦 Main Modules
| Module                 | Description                                   | Methods & Functionality                                                                            | Links                                                                                     |
| :--------------------- | :-------------------------------------------- | :---------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------- |
| **Tunneling1D**        | Computes bounce (instanton) solution for a single scalar field. | Uses the **overshoot/undershoot** method to solve the Euclidean equation of motion and find the tunneling profile               | [Module page](modules/tunneling1D/tunneling1D.md)                                                     |
| **pathDeformation**    | Computes instantons for multiple scalar fields.  | First finds a 1D solution constrained to an initial path in field space. Then iteratively deforms this path until transverse forces vanish, yielding the correct multi-dimensional solution.                    | Module page - None |
| **transitionFinder** | Computes the phase structure of the potential at finite temperature. |Locates potential minima as a function of temperature, determines critical temperatures (degenerate vacua), and computes the nucleation temperature for phase transitions | [Module page](modules/transitionFinder/transitionFinder.md)                                               |
| **generic\_potential** | Abstract class that defines the physical model of interest.       | The user provides a subclass implementing the specific effective potential $V(\phi, T)$. Also provides methods to **plot the potential** and visualize its phase structure.   | [Module page](modules/generic_potential/generic_potential.md)                                               |


---
## 🔧 Auxiliary Modules
| Module                     | Description                                                                         | Purpose                                               | Links                                   |
| :------------------------- | :---------------------------------------------------------------------------------- | :--------------------------------------------------------------- |:----------------------------------------|
| **helper\_functions**      | Numerical utilities (e.g., numerical integration, interpolation, numerical differentiation). | Called by **all** core modules.                                  | [Module page](modules/helper_functions) |
| **finiteT**                | Finite-temperature effective-potential corrections (boson/fermion thermal pieces).  | Feeds **generic\_potential** and **transition\_finder**.         | [Module page](modules/finiteT)          |
| **multi\_field\_plotting** | Visualization helpers for 3+ field landscapes and paths.                            | Used mainly with **pathDeformation** and **generic\_potential**. | Module page - None                      |
                               
---
