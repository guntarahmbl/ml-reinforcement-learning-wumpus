import numpy as np
from rl import WumpusEnv, QLearningAgent, SARSAAgent, greedy_path

env1 = WumpusEnv()
q_agent = QLearningAgent(alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.999)
rewards_q = q_agent.train(env1, episodes=5000, max_steps=100)

env2 = WumpusEnv()
sarsa_agent = SARSAAgent(alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.999)
rewards_s = sarsa_agent.train(env2, episodes=5000, max_steps=100)

import numpy as np

def evaluate_convergence(rewards, name="Agent", window=500, threshold=-100):
    """
    threshold : float
        Ambang rata-rata reward untuk dianggap 'konvergen'.
    """

    print(f"\n=== Evaluasi Konvergensi {name} ===")

    # Rata-rata reward per window episode
    print(f"\nRata-rata reward setiap {window} episode:")
    for i in range(0, len(rewards), window):
        avg = np.mean(rewards[i:i+window])
        print(f"Episode {i+1:5d} -{i+window:5d}: {avg:.2f}")

    # Mencari episode pertama kali reward melampaui threshold
    conv_ep = None
    for i in range(len(rewards)-window):
        avg = np.mean(rewards[i:i+window])
        if avg >= threshold:
            conv_ep = i
            break

    if conv_ep is not None:
        print(f"\n{ name } dianggap mulai konvergen sekitar episode {conv_ep} "
              f"(rata-rata reward ≥ {threshold})")
    else:
        print(f"\n{ name } belum mencapai ambang konvergensi (≥ {threshold})")

    tail = 500 if len(rewards) >= 500 else len(rewards)//2
    std_tail = np.std(rewards[-tail:])
    print(f"Stabilitas akhir: std reward {tail} episode terakhir = {std_tail:.2f}")

evaluate_convergence(rewards_q, name="Q-Learning", window=500, threshold=-100)
evaluate_convergence(rewards_s, name="SARSA", window=500, threshold=-100)


# policy table
def get_policy(Q):
    return np.argmax(Q, axis=1)

policy_q = get_policy(q_agent.Q)
policy_s = get_policy(sarsa_agent.Q)

print("\nPolicy final (Q-Learning):")
print(policy_q)

print("\nPolicy final (SARSA):")
print(policy_s)

# Jalur optimal
path_q, actions_q, rewards_q_path, gold_q, pos_q, done_q = greedy_path(WumpusEnv(), q_agent.Q)
print("\nPath optimal Q-Learning:")
print(path_q)
print(actions_q)

path_s, actions_s, rewards_s_path, gold_s, pos_s, done_s = greedy_path(WumpusEnv(), sarsa_agent.Q)
print("\nPath optimal SARSA:")
print(path_s)
print(actions_s)
