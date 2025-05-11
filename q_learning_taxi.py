# -*- coding: utf-8 -*-
"""
Nome do arquivo: q_learning_taxi_limpo.py
Data de criação: 11/05/2025
Autor: Victor Fernando de Sena Linhares || IAs: DeepSekk e Manus
Matrícula: 01645000

Descrição:
Este script implementa uma versão SIMPLIFICADA do algoritmo Q-Learning
para resolver o problema do Táxi (Taxi-v3) do OpenAI Gym.
Foco em clareza para apresentação.
"""

import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

# --- 1. Configuração Inicial ---
env = gym.make("Taxi-v3", render_mode="ansi")

numero_de_estados = env.observation_space.n
numero_de_acoes = env.action_space.n
print(f"Ambiente: {numero_de_estados} estados, {numero_de_acoes} ações.")

q_table = np.zeros((numero_de_estados, numero_de_acoes))

# Parâmetros do Q-Learning
alpha = 0.1      # Taxa de Aprendizado
gamma = 0.9      # Fator de Desconto
epsilon = 1.0    # Taxa de Exploração Inicial
min_epsilon = 0.05 # Taxa de Exploração Mínima
decay_rate = 0.01 # Taxa de Decaimento de Epsilon

numero_de_episodios_treinamento = 5000
print("\n--- Iniciando Treinamento ---")
recompensas_por_episodio_treinamento = []

# --- 2. Treinamento do Agente --- 
for episodio in range(numero_de_episodios_treinamento):
    estado_tupla = env.reset()
    estado = estado_tupla[0]
    terminado = False
    recompensa_total_episodio = 0

    while not terminado:
        if random.uniform(0, 1) < epsilon:
            acao = env.action_space.sample() # Exploração
        else:
            acao = np.argmax(q_table[estado, :]) # Explotação

        novo_estado_tupla = env.step(acao)
        novo_estado, recompensa, terminado, truncado, info = novo_estado_tupla
        
        if truncado:
            terminado = True

        # Atualização da Q-Table (Fórmula de Bellman)
        valor_antigo_q = q_table[estado, acao]
        melhor_valor_q_proximo_estado = np.max(q_table[novo_estado, :])
        novo_valor_q = valor_antigo_q + alpha * (recompensa + gamma * melhor_valor_q_proximo_estado - valor_antigo_q)
        q_table[estado, acao] = novo_valor_q

        estado = novo_estado
        recompensa_total_episodio += recompensa

    epsilon = max(min_epsilon, epsilon * (1 - decay_rate))
    recompensas_por_episodio_treinamento.append(recompensa_total_episodio)

    if (episodio + 1) % (numero_de_episodios_treinamento // 10) == 0:
        print(f"Episódio {episodio + 1}/{numero_de_episodios_treinamento} | Epsilon: {epsilon:.3f}")

print("Treinamento Concluído!")

# --- 3. Visualização do Aprendizado ---
janela_media_movel = 100
if len(recompensas_por_episodio_treinamento) >= janela_media_movel:
    recompensas_suavizadas = np.convolve(recompensas_por_episodio_treinamento, np.ones(janela_media_movel)/janela_media_movel, mode='valid')
else:
    recompensas_suavizadas = recompensas_por_episodio_treinamento

plt.figure(figsize=(10, 6))
plt.plot(recompensas_suavizadas)
plt.title("Recompensa Média por Episódio (Treinamento Simplificado)")
plt.xlabel("Episódios")
plt.ylabel("Recompensa Média")
plt.grid(True)
plt.savefig("recompensa_treinamento_limpo.png")
print("\nGráfico de recompensa salvo em: recompensa_treinamento_limpo.png")

# --- 4. Teste do Agente Treinado ---
print("\n--- Testando Agente Treinado ---")
numero_episodios_teste = 3

for episodio in range(numero_episodios_teste):
    estado_tupla = env.reset()
    estado = estado_tupla[0]
    terminado = False
    passos_no_episodio = 0
    print(f"\n--- Episódio de Teste {episodio + 1} ---")
    print("Estado Inicial:")
    print(env.render())
    input("Pressione Enter para iniciar...")

    while not terminado and passos_no_episodio < 50:
        acao = np.argmax(q_table[estado, :]) # Apenas explotação
        novo_estado_tupla = env.step(acao)
        novo_estado, recompensa, terminado, truncado, info = novo_estado_tupla
        if truncado:
            terminado = True
        estado = novo_estado
        passos_no_episodio += 1

        print(f"\nPasso {passos_no_episodio} | Ação: {acao}")
        print(env.render())
        print(f"Recompensa: {recompensa}")
        if terminado:
            print("Episódio Concluído!")
        else:
            input("Pressione Enter para próximo passo...")
    
    if not terminado:
        print("Limite de passos atingido.")

print("\n--- Demonstração Concluída ---")

print("\nValores Q para o Estado 0 (exemplo):", q_table[0])
env.close()
print("\nScript finalizado.")

