"""
Projekt:
    Zbudowanie bota(agenta) używając q-learning do gry FrozenLake.
Przygotowanie środowiska:
    Język Python oraz biblioteki gym, numpy.
    - pip install numpy
    - pip install gym
    UWAGA - trzeba zainstalować odpowiednią wersje gym'a - nie najnowszą w tym
    przypadku, niedawno wyszła nowa wersja, która działa trochę inaczej
    i jest z nią trochę problemów.
Działanie aplikacji:
    Bot(agent) gra w naszą grę podaną ilość razy (4000) i mając podane
    parametry takie jak szybkość uczenia się potrzebne dla q-learning
    uzyskuje coraz lepsze wyniki.
    W tej grze wynik(reward) może być albo 0.0 (przegrana) albo 1.0 (wygrana).
    Jeśli średnia dla 100 rozgrywek po danej liczbie zagrań jest większa
    niż 0.75, to możemy uznać, że bot(agent) nauczył się rozwiazywać tę grę.
Autorzy:
    Aleksander Dudek s20155
    Jakub Słomiński  s18552
"""

from typing import List
import gym
import numpy as np
import random

# Ustawiamy konkretny seed aby mieć reprodukowalne wyniki
random.seed(0)
np.random.seed(0)

# Liczba zagrań do nauczenia bota(agenta)
num_episodes = 4000

# Dane dla Q-Learning
discount_factor = 0.8
learning_rate = 0.9

# Interwał z jakim raportujemy wyniki
report_interval = 500

# Format raportowania
report = '100-ep Average: %.2f . Best 100-ep Average: %.2f . Average: %.2f ' \
         '(Episode %d)'


def print_report(rewards: List, episode: int):
    """
    Wyświetl wyniki dla danego zagrania
    - Średni wynik dla 100 ostatnich zagrań
    - Ogólny najlepszy wynik na 100 zagrań
    - Ogólny Średni wynik dla wszystkich zagrań
    """
    print(report % (
        np.mean(rewards[-100:]),
        max([np.mean(rewards[i:i+100]) for i in range(len(rewards) - 100)]),
        np.mean(rewards),
        episode))


def main():
    """
    Funkcja main trenująca bota(agenta) metodą q-learning w grę FrozenLake
    """
    # Tworzymy grę
    env = gym.make('FrozenLake-v1')

    # Wybieramy seed dla reprodukowalnych wyników
    env.seed(100)
    rewards = []

    # Tworzymy Q-Table z zerami
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    # W pętli rozgrywamy rozgrywki
    for episode in range(1, num_episodes + 1):
        # Zerujemy aktulny stan
        current_state = env.reset()
        episode_reward = 0
        while True:
            # Wartość szumu
            noise = np.random.random((1, env.action_space.n)) / (episode**2.)

            # Ustalamy jaka będzie następna akcja
            action = np.argmax(Q[current_state, :] + noise)

            # Wykonujemy 'krok', otrzymujemy stan następnej rozgrywki, wynik
            next_state, reward, done, _ = env.step(action)

            # Ustalamy cel dla Q-Learning
            Qtarget = reward + discount_factor * np.max(Q[next_state, :])

            # Aktualizujemy naszą Q-Table korzystając z algorytmu
            Q[current_state, action] = (
                1-learning_rate) * Q[current_state, action] + learning_rate * Qtarget

            # Zapisujemy wynik aktualnej rozgrywki
            episode_reward += reward

            # Przełączamy się na następną rozgrywkę
            current_state = next_state
            if done:
                # Dodajemy aktualny wynik do wyników
                rewards.append(episode_reward)
                if episode % report_interval == 0:
                    print_report(rewards, episode)
                break
    print_report(rewards, -1)


if __name__ == '__main__':
    main()
# Wyniki
# 100-ep Average: 0.71 . Best 100-ep Average: 0.72 . Average: 0.44 (Episode 500)
# 100-ep Average: 0.58 . Best 100-ep Average: 0.80 . Average: 0.54 (Episode 1000)
# 100-ep Average: 0.75 . Best 100-ep Average: 0.80 . Average: 0.57 (Episode 1500)
# 100-ep Average: 0.72 . Best 100-ep Average: 0.84 . Average: 0.61 (Episode 2000)
# 100-ep Average: 0.74 . Best 100-ep Average: 0.84 . Average: 0.63 (Episode 2500)
# 100-ep Average: 0.73 . Best 100-ep Average: 0.84 . Average: 0.65 (Episode 3000)
# 100-ep Average: 0.76 . Best 100-ep Average: 0.84 . Average: 0.66 (Episode 3500)
# 100-ep Average: 0.73 . Best 100-ep Average: 0.84 . Average: 0.67 (Episode 4000)
# 100-ep Average: 0.73 . Best 100-ep Average: 0.84 . Average: 0.67 (Episode -1)
