flowchart TD

subgraph CosmoTransitions["📂 cosmoTransitions/"]
    HF["📦 helper_functions.py\n(Funções numéricas auxiliares:\ninterpolação, integração, raízes)"]
    FT["📦 finiteT.py\n(Correções de temperatura finita\npara o potencial)"]
    GP["📦 generic_potential.py\n(Classe base para definir\num modelo de potencial)"]
    MF["📦 multiFieldPlotting.py\n(Ferramentas de plotagem\npara potenciais e perfis)"]
    TF["📦 transitionFinder.py\n(Localiza temperaturas críticas\ne parâmetros da transição)"]
    T1D["📦 tunneling1D.py\n(Solução do bounce em 1 campo)"]
end

subgraph Examples["📂 examples/"]
    FTUN["📦 fullTunneling.py\n(Exemplo completo de cálculo\ndo bounce e transição)"]
    TM1["📦 testeModel1.py\n(Modelo exemplo simples)"]
end

subgraph User["Usuário"]
    U["Define modelo específico\n(herdando generic_potential)"]
end

%% Relações
U --> GP
GP --> FT
GP --> HF
TF --> GP        
TF --> HF
TF --> T1D
T1D --> HF
T1D --> GP
MF --> GP
MF --> T1D
FTUN --> TF
FTUN --> T1D
FTUN --> MF
TM1 --> GP
