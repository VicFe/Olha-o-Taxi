# Projeto Agente de Táxi Inteligente com Q-Learning (Simplificado)

Este projeto apresenta uma implementação simplificada do algoritmo Q-Learning para resolver o problema clássico do "Táxi" (ambiente Taxi-v3 do Gymnasium/OpenAI Gym). O objetivo principal é ser didático e facilitar a compreensão dos conceitos fundamentais do Aprendizado por Reforço.

## Sobre o Desafio do Táxi

No ambiente Taxi-v3, um agente (o táxi) opera em uma grade 5x5. Existem quatro locais de coleta e entrega de passageiros designados (R, G, Y, B). O objetivo do táxi é:
1.  Navegar até a localização de um passageiro.
2.  Pegar o passageiro.
3.  Navegar até o destino desejado pelo passageiro.
4.  Largar o passageiro no local correto.

O agente recebe recompensas ou penalidades com base em suas ações, incentivando-o a completar a tarefa da forma mais eficiente possível.

## O que é Q-Learning?

Q-Learning é um algoritmo de Aprendizado por Reforço que não requer um modelo do ambiente (model-free). Ele aprende uma política ótima, ou seja, qual a melhor ação a ser tomada em cada estado, através da construção de uma **Q-Table** (Tabela de Valores Q).

*   **Estados (States):** Representam as diferentes situações em que o agente pode se encontrar (posição do táxi, localização do passageiro, destino).
*   **Ações (Actions):** As possíveis movimentações ou operações que o agente pode realizar (mover Norte, Sul, Leste, Oeste, Pegar, Largar).
*   **Recompensas (Rewards):** Feedback numérico que o agente recebe após cada ação. Positivo para ações boas, negativo para ações ruins.
*   **Q-Table:** Uma tabela onde as linhas são os estados e as colunas são as ações. Cada célula `Q(s, a)` armazena o valor esperado da recompensa total futura ao tomar a ação `a` no estado `s` e seguir a política ótima thereafter.

O agente aprende atualizando os valores da Q-Table usando a equação de Bellman, com base na experiência adquirida (tentativa e erro).

## Tecnologias Utilizadas

*   **Python:** Linguagem de programação principal.
*   **Gymnasium (anteriormente OpenAI Gym):** Para o ambiente Taxi-v3 e interações de aprendizado por reforço.
*   **NumPy:** Para manipulação eficiente da Q-Table e cálculos numéricos.
*   **Matplotlib:** Para visualização do progresso do aprendizado (gráfico de recompensas).

## Estrutura do Projeto

*   `q_learning_taxi_limpo.py`: O script Python principal contendo a implementação do agente Q-Learning, treinamento e demonstração.
*   `q_learning_taxi_anotacoes_estudo.md`: Um documento com comentários detalhados sobre o código original (antes da simplificação para apresentação), útil para estudo aprofundado.
*   `explanation.md`: Um arquivo com explicações teóricas sobre Q-Learning, o sistema de recompensas, parâmetros e um roteiro para apresentação do projeto.
*   `recompensa_treinamento_limpo.png`: Imagem do gráfico gerado pelo script, mostrando a evolução da recompensa média durante o treinamento.

## Como Executar o Projeto

1.  **Pré-requisitos:**
    *   Python 3.x instalado.
    *   As seguintes bibliotecas Python: `gymnasium`, `numpy`, `matplotlib`.
    Você pode instalá-las usando pip:
    ```bash
    pip install gymnasium numpy matplotlib
    ```
    Se você estiver usando uma versão mais antiga do Gym, pode precisar instalar `gymnasium[toy_text]` ou `gym` e ajustar a importação no código.

2.  **Executar o Script:**
    Navegue até o diretório onde você salvou o arquivo `q_learning_taxi_limpo.py` e execute-o no terminal:
    ```bash
    python q_learning_taxi_limpo.py
    ```

3.  **O que esperar:**
    *   O script imprimirá informações sobre o ambiente (número de estados e ações).
    *   Iniciará o processo de treinamento, mostrando o progresso a cada X episódios e o valor atual do Epsilon (taxa de exploração).
    *   Ao final do treinamento, salvará um gráfico chamado `recompensa_treinamento_limpo.png` no mesmo diretório, mostrando a evolução da recompensa média.
    *   Em seguida, iniciará uma fase de demonstração, onde você poderá ver o agente treinado em ação por alguns episódios. O estado do ambiente será impresso, e você precisará pressionar Enter para ver cada passo do agente.

## Entendendo os Parâmetros do Q-Learning no Script

*   `alpha` (Taxa de Aprendizado): Controla o quão rápido o agente atualiza seus valores Q com base em novas experiências. (Valor no script: 0.1)
*   `gamma` (Fator de Desconto): Determina a importância das recompensas futuras. Um valor próximo de 1 faz o agente pensar mais no longo prazo. (Valor no script: 0.9)
*   `epsilon` (Taxa de Exploração Inicial): Probabilidade inicial de o agente escolher uma ação aleatória (explorar) em vez da melhor ação conhecida (explotar). (Valor no script: 1.0)
*   `min_epsilon` (Taxa de Exploração Mínima): O valor mínimo que o epsilon pode atingir, garantindo que o agente sempre explore um pouco. (Valor no script: 0.05)
*   `decay_rate` (Taxa de Decaimento do Epsilon): Controla a rapidez com que o epsilon diminui a cada episódio, fazendo o agente explorar menos com o tempo. (Valor no script: 0.01)

## Sistema de Recompensas no Taxi-v3

*   **+20 pontos:** Por entregar o passageiro no destino correto.
*   **-1 ponto:** Por cada passo/movimento realizado (incentiva a eficiência).
*   **-10 pontos:** Por tentar pegar um passageiro no local errado ou tentar largar um passageiro no local errado.

A recompensa de -1 por movimento é crucial para que o agente aprenda a encontrar os caminhos mais curtos, em vez de apenas completar a tarefa de qualquer maneira.

## Contribuições

Este projeto foi desenvolvido com foco didático. Sinta-se à vontade para clonar, modificar e experimentar com os parâmetros para aprofundar seu entendimento sobre Q-Learning!

## Autor Original do Desafio da Faculdade

*   Victor Fernando de Sena Linhares (Matrícula: 01645000) || IAs: DeepSekk e Manus
