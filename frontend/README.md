# Frontend - Agente RAG ANEEL (Ceinha)

Este diretório contém a interface de usuário do Agente RAG focado em documentos da ANEEL. A interface foi desenvolvida utilizando **Streamlit**, personalizada com CSS e componentes HTML/Canvas para proporcionar uma experiência imersiva e moderna.

## 🚀 Funcionamento

A aplicação segue o seguinte fluxo de operação:

1.  **Inicialização (Splash Screen):** Ao abrir o app, uma tela de carregamento animada (`loading-screen (2).html`) é exibida enquanto os recursos pesados (banco vetorial e modelos de linguagem) são carregados em cache.
2.  **Interface de Chat:** Após o carregamento, o usuário interage com um chat persistente.
3.  **Processamento RAG:**
    *   **Refinamento:** A pergunta do usuário é enviada para o LLM para ser limpa e otimizada para busca.
    *   **Recuperação:** O sistema busca os trechos mais relevantes no banco de dados vetorial.
    *   **Geração:** O LLM utiliza o contexto recuperado para gerar uma resposta precisa.
4.  **Exibição de Fontes:** As fontes utilizadas para gerar a resposta são exibidas em uma barra lateral, permitindo a verificação da informação.

## 🧠 Teoria e Design

A interface foi projetada com base em três pilares:

*   **Identidade Visual "Neural":** Utiliza simulações de redes neurais via Canvas (`neural-background.html`) para reforçar a natureza tecnológica do projeto.
*   **User Experience (UX):** O uso de uma Splash Screen mitiga a percepção de demora no carregamento de modelos de IA, enquanto o chat focado facilita a interação.
*   **Transparência (Explainability):** A exibição explícita das fontes no sidebar combate alucinações de IA, permitindo que o usuário valide a resposta nos documentos originais da ANEEL.

## 🛠️ Componentes Principais

*   `app.py`: Script principal do Streamlit que orquestra a lógica de UI e integração com o backend RAG.
*   `CEINHA.html`: Avatar e branding da assistente virtual "Ceinha".
*   `neural-background.html`: Fundo animado interativo simulando conexões sinápticas.
*   `loading-screen (2).html`: Interface de splash screen de alta fidelidade visual.

## ❓ Por que usar esta interface?

1.  **Imersão:** Diferente de interfaces genéricas, esta é personalizada para o contexto de IA e energia elétrica.
2.  **Performance:** Utiliza `@st.cache_resource` para garantir que o sistema responda rapidamente após o primeiro carregamento.
3.  **Responsividade:** O design se adapta a diferentes tamanhos de tela, mantendo a legibilidade e estética.
4.  **Facilidade de Uso:** Abstrai toda a complexidade do pipeline RAG em uma interface de chat simples e intuitiva.
