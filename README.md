# GW_FOPT_Simulation
Project to simulate the thermodynamic parameters of the phase transition and calculate the spectrum of gravitational waves given an effective potential

A ideia do projeto é atualizar e modificar o **CosmoTransitions** que, por mais que seja um código amplamente utilizado na literatura para transições de fase cosmológicas, está desatualizado por ter sido desenvolvido há bastante tempo. 

Desse modo, o projeto visa realizar melhorias significativas no código, tornando-o mais otimizado, moderno e intuitivo, alinhado-o com os pacotes existentes atuais para python. 

📅 Cronograma e Abordagem: O cronograma e fluxograma do projeto se encontram abaixo. A ideia central é dividir essa tarefa em 3 partes principais, cada uma com duração de 1 mês, realizando testes de consistência ao longo de todo o desenvolvimento, ao finalizar as modificações. Cada fase seguirá o ciclo: Modificação → Testes → Correção → Validação.

A primeira fase visa atualizar os códigos de integração numérica e os módulos independentes, que são chamados pelos módulos principais. A segunda fase, os códigos que encontram a solução de bounce. Por fim, a terceira e última fase visa modificar as funções que criam o potencial genérico e os plots feitos dado os parâmetros iniciais. Tudo será feito para a parte 1D apenas.

Dependendo do andamento do projeto será feito uma quarta fase visando acrescentar novos plots e gráficos ao código, assim como atualizar a parte que calcula o plot 3D.

Há ainda alguns problemas em aberto em relação a como fazer os testes, eles se encontram no final da página.


## Flowchart of the modules

```mermaid
graph TD
    %% ========== MÓDULOS PRINCIPAIS ==========
    subgraph "MÓDULOS PRINCIPAIS"
        T1D[Tunneling1D<br/>Solução de bounce em 1 campo]
        PD[pathDeformation<br/>Solução de bounce em múltiplos campos]
        TF[transitionFinder<br/>Localização de transições Tn e estrutura de Fase]
        GP[generic_potential<br/>Definição de modelos e plot do potencial]
        
        T1D --> PD
        T1D --> TF
        T1D --> GP
        PD --> TF
        PD --> GP
        TF --> GP
    end

    %% ========== MÓDULOS AUXILIARES ==========
    subgraph "MÓDULOS AUXILIARES"
        HF[helper_functions<br/>Funções auxiliares]
        FT[finiteT<br/>Correções de temperatura finita]
        MFP[multi_field_plotting<br/>Visualização para 3 ou mais campos]
    end

    %% ========== DEPENDÊNCIAS ==========
    AUX --> PRINCIPAL
    
    %% ========== ESTILOS ==========
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
### 📦 Módulos Principais

| Módulo | Descrição | Métodos/Funcionalidades |
| :--- | :--- | :--- |
| <span style="color:green">**Tunneling1D**</span> | Calcula a solução de bounce (instantons) para um único campo escalar. | Utiliza o método de **overshooting/undershooting** para resolver a equação de movimento euclidiana e encontrar o perfil do túnel. |
| <span style="color:red">**pathDeformation**</span> | Calcula instantons para múltiplos campos escalares. | Primeiro encontra uma solução 1D restrita a um caminho inicial no espaço de campo. Em seguida, **deforma iterativamente** esse caminho até que as forças perpendiculares a ele se anulem, encontrando a solução multidimensional correta. |
| <span style="color:green">**transitionFinder**</span> | Calcula a estrutura de fase do potencial em temperatura finita. | Localiza os mínimos do potencial em função da temperatura, determina **temperaturas críticas** (onde os vácuos são degenerados) e calcula a **temperatura de nucleação** para transições entre fases. |
| <span style="color:green">**generic_potential**</span> | Classe abstrata que define o modelo físico de interesse. | O usuário fornece a subclasse para implementar o potencial efetivo específico (V(φ,T)). Também fornece métodos para **plotar o potencial** e visualizar sua estrutura de fases (Ainda não entendo tão bem). |

### 🔧 Módulos Auxiliares

| Módulo | Descrição | Função |
| :--- | :--- | :--- |
| **helper_functions** | Conjunto de funções utilitárias numéricas. | Oferece operações auxiliares (ex.: interpolação, derivação numérica) usadas pelos módulos principais. |
| **finiteT** | Calcula correções de temperatura finita ao potencial efetivo. | Implementa os termos da função de partição (loop de bósons e férmions) que dependem de T. É usado por `generic_potential`. |
| <span style="color:red">**multi_field_plotting**</span> | Classe para visualização de potenciais com 3+ campos. | Fornece ferramentas para gerar gráficos e visualizações do potencial efetivo em alta dimensão. |

### 📖 Documentação e Leituras Recomendadas pré modificações

Antes de modificar qualquer módulo, consultar a documentação oficial e/ou o artigo original para entender os algoritmos:
*   **Documentação Oficial:** [https://clwainwright.net/CosmoTransitions/index.html](https://clwainwright.net/CosmoTransitions/index.html)
*   **Artigo Original (arXiv):** [arXiv:1109.4189](https://arxiv.org/abs/1109.4189) 
*   **Paper no Computer Physics Communications:** [10.1016/j.cpc.2012.04.004](https://doi.org/10.1016/j.cpc.2012.04.004)


## Cronograma do Projeto

```mermaid
gantt
    title Cronograma de Desenvolvimento - CosmoTransitions
    dateFormat  YYYY-MM-DD
    axisFormat  %d/%m
    
    section Fase 0: Planejamento
    Fluxograma e Cronograma          :done, 2025-08-27, 7d
    Definição de Metodologias        :active, 2025-09-05, 10d
    
    section Fase 1: Modificação de Funções de Integração Numérica
    Modificação helper_function.py            :2025-09-08, 10d
    Modificação finiteT.py           :2025-09-19, 10d
    
    section Fase 1.5: Testes das Modificações
    Testes Integração Numérica       :2025-09-30, 5d
    Correções e Ajustes              :2025-09-30, 5d
    
    section Fase 2: Modificação das Funções de Soluções Bounce
    Modificação Tunneling1D.py       :2025-10-01, 12d
    Modificação transitionsFinder.py :2025-10-12, 12d
    
    section Fase 2.5: Testes das Modificações
    Testes Soluções Bounce           :2025-10-24, 5d
    Validação Numérica               :2025-10-24, 5d
    
    section Fase 3: Modificação dos Potenciais e Saídas das funções
    Modificação generic_potential.py           :2025-11-03, 18d
    
    section Fase 3.5: Testes Finais
    Testes Completos                 :2025-11-27, 6d
    Documentação                     :2025-11-27, 6d
    
    section Fase 4: Extras (Opcional)
    Plots Adicionais                 :2025-12-01, 10d
    Solução para múltiplos campos    :2025-12-10, 10d
    Otimizações Finais               :2025-12-10, 10d
```

- [x] **Fase 0**: Planejamento e Primeira reunião 
  - Criar fluxograma de dependências  
  - Criar cronograma de refatoração  

- [ ] **Fase 1**: Núcleo numérico (Funções auxiliáres) 
  - Refatorar `helper_functions.py` (usar SciPy para integrais e raízes)  
  - Vetorizar `finiteT.py` (substituir loops por NumPy e atualizar correções)  

- [ ] **Fase 1.5**: Testes de Modificações  
  - Validar funções isoladas com exemplos analíticos simples  
  - Comparar saídas numéricas com versão original  

- [ ] **Fase 2**: Solução do bounce e parâmetros de transição  
  - Refatorar `tunneling1D.py` (usar `scipy.solve_ivp` no solver ODE)  
  - Melhorar `transitionFinder.py` (algoritmos de busca mais eficientes)  

- [ ] **Fase 2.5**: Testes intermediários  
  - Reproduzir resultados dos exemplos (`fullTunneling.py`)  
  - Comparar ações críticas com versão antiga  

- [ ] **Fase 3**: Potencial e saídas  
  - Modernizar `generic_potential.py` (usar `abc.ABC` para interface clara)  
  - Atualizar gráficos plotados, acrescentar densidade de energia e outros úteis para o artigo/tese 

- [ ] **Fase 3.5**: Testes finais  
  - Rodar todos os exemplos e validar consistência  
  - Criar notebooks substituindo scripts  

- [ ] **Fase 4** *(opcional)*: Extensões
  - Novos tipos de plots (ex.: espectro GW direto, densidade de GW no espaço para diferentes T e outros)  
  - Modernizar os códigos que fazem plots para múltiplos campos `mult_field_plotting.py` e `path_deformation.py`

**Problemas ainda em aberto:** Decidir como será testado as modificações, i.e, como iremos comparar o antigo código com o novo que estamos fazendo e termos um teste de consistência. Ideia inicial é:
  - Teste 1: Dentro da própria módulo modificado fazer um teste simples que chamem a função e deem um resultado comparativo de antes e depois do seu output
  - Teste 2: Testar o exemplo de modelo simples, do próprio cosmotransitions
  - Teste 3: Comparar gráficos da forma do potencial antes e depois da modificação e observar as alterações. Possivelmente testar modelos conhecidos como o do próprio artigo do Glauber.

```mermaid
graph TD
    Start[Início do Projeto] --> Phase1[Fase 1: Integração Numérica]
    Start --> Phase2[Fase 2: Solução Bounce]
    Start --> Phase3[Fase 3: Potencial e Visualização]
    
    Phase1 --> Test1[Testes de Consistência]
    Phase2 --> Test2[Testes de Consistência]
    Phase3 --> Test3[Testes de Consistência]
    
    Test1 --> Adjust1[Ajustes e Correções]
    Test2 --> Adjust2[Ajustes e Correções]
    Test3 --> Adjust3[Ajustes e Correções]
    
    Adjust1 --> FinalValidation[Validação Final]
    Adjust2 --> FinalValidation
    Adjust3 --> FinalValidation
    
    FinalValidation --> Decision{Andamento Satisfatório?}
    
    Decision -- Sim --> Phase4[Fase 4: Novos Plots e Gráficos]
    Decision -- Não --> Review[Revisão e Otimizações]
    
    Phase4 --> ProjectEnd[Projeto Concluído]
    Review --> ProjectEnd

    style Start fill:#e1f5fe
    style Phase1 fill:#e8f5e8
    style Phase2 fill:#fff3e0
    style Phase3 fill:#f3e5f5
    style Phase4 fill:#ffebee
    style ProjectEnd fill:#c8e6c9
```

