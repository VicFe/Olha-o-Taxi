# Projeto Agente de Táxi Inteligente com Q-Learning

## Autor
- Nome: Victor Fernando de Sena Linhares
- Matrícula: 01645000
- Data de Criação do Projeto Base: 02/05/2025
- Data de Desenvolvimento Atualização: 11/05/2025
- IAs usadas: DeepSeek e Manus

## 1. Introdução ao Projeto

Este projeto visa desenvolver um agente inteligente capaz de resolver o problema do "Táxi Virtual" (ambiente Taxi-v3 do OpenAI Gym) utilizando Aprendizado por Reforço, especificamente o algoritmo Q-Learning. O objetivo principal é treinar o agente para que ele aprenda a política ótima de navegação em um ambiente de grade 5x5, de forma a pegar um passageiro em um local específico e levá-lo ao seu destino da maneira mais eficiente possível, maximizando a recompensa acumulada e minimizando penalidades por ações incorretas.

O ambiente Taxi-v3 simula um táxi que opera em uma cidade representada por uma grade. Existem quatro locais de coleta e destino de passageiros (R, G, Y, B). O agente precisa aprender a:
- Navegar pela grade (ações: Norte, Sul, Leste, Oeste).
- Pegar um passageiro no local correto (ação: Pegar).
- Levar o passageiro ao destino correto (ação: Largar).

O agente recebe recompensas positivas por completar a tarefa com sucesso e penalidades por ações ineficientes ou incorretas (como tentar pegar um passageiro onde não há um, ou largar em local errado).

## 2. Tecnologias e Bibliotecas Utilizadas

- **Python 3.11**: Linguagem de programação principal para a implementação do agente.
- **OpenAI Gymnasium (anteriormente Gym)**: Biblioteca utilizada para fornecer o ambiente de simulação do Táxi (Taxi-v3). Ela oferece a interface padrão para ambientes de aprendizado por reforço.
- **NumPy**: Biblioteca fundamental para computação científica em Python, utilizada para a criação e manipulação da Q-table e outros cálculos numéricos.
- **Matplotlib**: Biblioteca para a criação de gráficos estáticos, utilizada para visualizar as métricas de treinamento e avaliação do agente, como a evolução da recompensa média por episódio e o decaimento da taxa de exploração.

## 3. Algoritmos Aplicados e Conceitos de Machine Learning

### 3.1. Q-Learning
O Q-Learning é um algoritmo de aprendizado por reforço model-free (não requer um modelo do ambiente) e off-policy (aprende a política ótima independentemente da política que está seguindo durante o treinamento). Ele busca aprender uma função de valor ação, Q(s, a), que estima a recompensa total esperada ao executar a ação 'a' no estado 's' e seguir a política ótima a partir daí.

**Fórmula de Atualização da Q-Table:**
A Q-table é uma tabela onde as linhas representam os estados (s) e as colunas representam as ações (a). Cada célula Q(s, a) armazena o valor Q para aquele par estado-ação. A atualização é feita iterativamente usando a equação de Bellman:

`Q(s, a) ← Q(s, a) + α * [R + γ * max(Q(s', a')) - Q(s, a)]`

Onde:
- `s`: estado atual.
- `a`: ação tomada no estado `s`.
- `s'`: próximo estado após tomar a ação `a`.
- `R`: recompensa recebida ao transitar de `s` para `s'`.
- `α (alpha)`: Taxa de Aprendizado (Learning Rate). Controla o quanto os novos valores Q substituem os antigos. Um valor entre 0 e 1.
- `γ (gamma)`: Fator de Desconto. Determina a importância das recompensas futuras. Um valor entre 0 e 1. Um valor próximo de 1 dá mais peso a recompensas futuras.
- `max(Q(s', a'))`: O valor Q máximo estimado para todas as ações possíveis no próximo estado `s'`.

### 3.2. Exploração vs. Explotação (Epsilon-Greedy)
Para garantir que o agente explore suficientemente o ambiente e não fique preso em uma política subótima, utilizamos a estratégia epsilon-greedy:
- Com uma probabilidade `ε (epsilon)`, o agente escolhe uma ação aleatória (exploração).
- Com uma probabilidade `1-ε`, o agente escolhe a melhor ação conhecida com base nos valores Q atuais (explotação).

**Ajuste da Taxa de Exploração (Decaimento de Epsilon):**
Inicialmente, `ε` é alto para incentivar a exploração. Conforme o agente aprende, `ε` é gradualmente reduzido (decaimento exponencial) para que o agente passe a explotar mais o conhecimento adquirido. A fórmula de decaimento utilizada foi:

`ε = min_ε + (max_ε - min_ε) * exp(-decay_rate * episódio)`

Onde:
- `min_ε`: Valor mínimo de epsilon.
- `max_ε`: Valor máximo (inicial) de epsilon.
- `decay_rate`: Taxa de decaimento.

### 3.3. Recompensa e Penalidade
O ambiente Taxi-v3 define o sistema de recompensas:
- +20 por largar o passageiro no destino correto.
- -1 por cada passo dado (para incentivar a eficiência).
- -10 por tentar pegar um passageiro ilegalmente ou largar o passageiro em local errado.

O agente aprende a maximizar a recompensa acumulada ao longo de um episódio.

## 4. Estrutura do Projeto e Código-Fonte

O projeto consiste nos seguintes arquivos principais:

- `q_learning_taxi.py`: Contém todo o código Python para a implementação do agente Q-Learning, incluindo:
    - Inicialização do ambiente Taxi-v3.
    - Definição dos hiperparâmetros (alpha, gamma, epsilon, etc.).
    - Inicialização da Q-table.
    - Função para treinamento do agente (`treinar_agente`).
    - Função para avaliação do agente treinado (`avaliar_agente`).
    - Função para visualização dos resultados (`visualizar_resultados`).
    - Bloco principal (`if __name__ == "__main__":`) que orquestra o treinamento, avaliação e visualização.
- `README.md` (este arquivo): Documentação completa do projeto.
- `requirements.txt`: Lista as dependências Python necessárias para executar o projeto.
- `recompensas_treinamento.png`: Gráfico da média móvel das recompensas por episódio durante o treinamento.
- `decaimento_epsilon.png`: Gráfico mostrando o decaimento da taxa de exploração (epsilon) durante o treinamento.
- `recompensas_avaliacao_hist.png`: Histograma da distribuição das recompensas totais por episódio durante a fase de avaliação.
- `q_table.npy`: Arquivo binário contendo a Q-table aprendida pelo agente após o treinamento.

**Cabeçalho dos Arquivos:**
Todos os arquivos de código-fonte incluem um cabeçalho com:
- Nome do arquivo
- Data de criação
- Autor
- Matrícula
- Descrição do que o arquivo implementa e suas funcionalidades.

## 5. Como Executar o Projeto

Para executar o projeto, siga os passos abaixo:

1.  **Clone o repositório (ou baixe os arquivos):**
    Se o projeto estiver em um repositório Git:
    `git clone <URL_DO_REPOSITORIO>`
    `cd <NOME_DA_PASTA_DO_PROJETO>`

    Caso contrário, certifique-se de que todos os arquivos (`q_learning_taxi.py`, `requirements.txt`) estejam no mesmo diretório.

2.  **Crie um ambiente virtual (recomendado):**
    `python -m venv venv`
    `source venv/bin/activate` (Linux/macOS)
    `venv\Scripts\activate` (Windows)

3.  **Instale as dependências:**
    `pip install -r requirements.txt`
    (O arquivo `requirements.txt` deve conter `gymnasium`, `numpy`, `matplotlib`)

4.  **Execute o script principal:**
    `python q_learning_taxi.py`

    O script irá:
    - Inicializar o ambiente e a Q-table.
    - Treinar o agente por um número definido de episódios (25.000 no código atual).
    - Imprimir o progresso do treinamento e a taxa de epsilon.
    - Avaliar o agente treinado em 100 episódios.
    - Imprimir as métricas de avaliação (média de passos, penalidades e recompensa por episódio).
    - Salvar os gráficos de visualização (`recompensas_treinamento.png`, `decaimento_epsilon.png`, `recompensas_avaliacao_hist.png`) e a Q-table (`q_table.npy`) no diretório `/home/ubuntu/project_files/` (ou no diretório especificado na função `visualizar_resultados`).

## 6. Resultados e Análise

Após a execução do script, os seguintes resultados são obtidos e podem ser analisados:

### 6.1. Métricas de Treinamento:
Durante o treinamento, o script imprime a recompensa média (dos últimos 100 episódios) e o valor atual de epsilon em intervalos regulares. Espera-se que a recompensa média aumente ao longo do tempo, indicando que o agente está aprendendo a política correta.

- **Gráfico de Recompensas do Treinamento (`recompensas_treinamento.png`):**
  Este gráfico mostra a média móvel das recompensas acumuladas por episódio. Uma curva ascendente que se estabiliza em um valor positivo alto indica um bom aprendizado. No nosso caso, o agente atinge uma recompensa média positiva estável, mostrando que aprendeu a completar a tarefa eficientemente.

- **Gráfico de Decaimento de Epsilon (`decaimento_epsilon.png`):**
  Este gráfico ilustra como a taxa de exploração (epsilon) diminui exponencialmente ao longo dos episódios de treinamento, começando alta (para exploração) e diminuindo para que o agente explore menos e explore mais o conhecimento adquirido.

### 6.2. Métricas de Avaliação:
Após o treinamento, o agente é avaliado em um conjunto separado de episódios (100 no código atual) sem exploração (epsilon = 0), utilizando apenas a política aprendida (explotação).

As métricas impressas incluem:
- **Média de passos por episódio:** Quantos passos, em média, o agente leva para completar a tarefa. Um valor menor é melhor.
- **Média de penalidades por episódio:** Quantas penalidades, em média, o agente recebe por episódio. Um valor próximo de zero é ideal.
- **Média de recompensa por episódio:** A recompensa total média que o agente consegue obter. Um valor positivo mais alto indica melhor desempenho.

Nos resultados obtidos:
- Média de passos por episódio: ~12.99
- Média de penalidades por episódio: 0.00
- Média de recompensa por episódio: ~8.01

Estes resultados indicam que o agente aprendeu a resolver o problema de forma eficiente, com pouquíssimas ou nenhuma penalidade e uma recompensa positiva consistente.

- **Histograma de Recompensas da Avaliação (`recompensas_avaliacao_hist.png`):**
  Mostra a distribuição das recompensas totais obtidas em cada episódio de avaliação. Idealmente, a maioria dos episódios deve ter recompensas altas e consistentes.

### 6.3. Q-Table (`q_table.npy`):
Este arquivo contém os valores Q aprendidos para cada par estado-ação. Embora difícil de inspecionar diretamente devido ao seu tamanho (500 estados x 6 ações), ela representa o "conhecimento" do agente sobre o ambiente.

## 7. Dificuldades Encontradas e Adaptações

- **Ajuste de Hiperparâmetros:** Encontrar os valores ideais para a taxa de aprendizado (α), fator de desconto (γ), e os parâmetros de decaimento de epsilon (taxa de decaimento, epsilon inicial/mínimo) pode ser um processo iterativo e demorado. Valores inadequados podem levar a um aprendizado lento, instável ou convergência para uma política subótima. Foram testados alguns valores até se chegar a uma configuração que apresentasse bons resultados em um tempo de treinamento razoável.
- **Número de Episódios de Treinamento:** O agente precisa de um número significativo de episódios para explorar o ambiente e convergir para uma boa política. O número de 25.000 episódios foi escolhido para garantir uma boa convergência para o ambiente Taxi-v3.
- **Interpretação dos Estados e Ações do Gym:** É crucial entender corretamente como o Gymnasium representa os estados e ações para interagir com o ambiente e atualizar a Q-table corretamente. O estado no Taxi-v3 é um único inteiro que codifica a posição do táxi, a localização do passageiro e o destino.

## 8. Conclusões e Próximos Passos

O projeto demonstrou com sucesso a aplicação do algoritmo Q-Learning para treinar um agente a resolver o problema do Táxi Virtual. O agente foi capaz de aprender uma política eficiente, resultando em um bom desempenho durante a fase de avaliação, com altas recompensas médias e poucas penalidades.

Os gráficos gerados fornecem uma visualização clara do processo de aprendizado e do comportamento do agente, atendendo aos requisitos do desafio.

**Possíveis Próximos Passos (Sugestões):**
- **Experimentar outros algoritmos:** Comparar o desempenho do Q-Learning com outros algoritmos de Aprendizado por Reforço, como SARSA ou Deep Q-Networks (DQN) para ambientes mais complexos.
- **Otimização de Hiperparâmetros:** Utilizar técnicas mais sistemáticas para otimizar os hiperparâmetros, como Grid Search ou Bayesian Optimization.
- **Análise mais profunda da Q-Table:** Desenvolver ferramentas para visualizar ou analisar a Q-table de forma mais intuitiva, talvez focando em estados específicos ou transições.
- **Generalização para outros ambientes:** Adaptar o agente para resolver outros problemas ou ambientes do Gymnasium.

Este projeto serve como uma excelente introdução prática aos conceitos fundamentais do Aprendizado por Reforço e à implementação do Q-Learning.
