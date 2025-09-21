
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
### 📦 Main Modules
| Module                                                  | Description                                                          | Methods/Functionality                                                                                                                                                                                                     |
| :------------------------------------------------------ | :------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| <span style="color:green">**Tunneling1D**</span>        | Computes the bounce (instanton) solution for a single scalar field.  | Uses the **overshooting/undershooting** method to solve the Euclidean equation of motion and find the tunneling profile.                                                                                                  |
| <span style="color:red">**pathDeformation**</span>      | Computes instantons for multiple scalar fields.                      | First finds a 1D solution constrained to an initial path in field space. Then **iteratively deforms** this path until transverse forces vanish, yielding the correct multi-dimensional solution.                          |
| <span style="color:green">**transitionFinder**</span>   | Computes the phase structure of the potential at finite temperature. | Locates potential minima as a function of temperature, determines **critical temperatures** (degenerate vacua), and computes the **nucleation temperature** for phase transitions.                                        |
| <span style="color:green">**generic\_potential**</span> | Abstract class that defines the physical model of interest.          | The user provides a subclass implementing the specific effective potential $V(\phi, T)$. Also provides methods to **plot the potential** and visualize its phase structure (**I still don’t understand this very well**). |

### 🔧 Auxiliary Modules
| Module                                                    | Description                                                         | Purpose                                                                                                                      |
| :-------------------------------------------------------- | :------------------------------------------------------------------ | :--------------------------------------------------------------------------------------------------------------------------- |
| **helper\_functions**                                     | A set of numerical utility functions.                               | Provides helper operations (e.g., numerical integration, interpolation, numerical differentiation) used by the main modules. |
| **finiteT**                                               | Computes finite-temperature corrections to the effective potential. | Implements the temperature-dependent partition-function terms (bosonic and fermionic loops). Used by `generic_potential`.    |
| <span style="color:red">**multi\_field\_plotting**</span> | Visualization class for potentials with 3+ fields.                  | Tools for producing plots and visualizations of the high-dimensional effective potential.                                    |
