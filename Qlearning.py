import numpy as np
import random

env_states = 5
env_actions = 2
Q_table = np.zeros((env_states, env_actions))
alpha = 0.1
gamma = 0.9
epsilon = 0.2

def get_reward(state, action):
    return random.choice([-1, 0, 1])

def get_next_state(state, action):
    return (state + action) % env_states

for episode in range(100):
    
    state = random.randint(0, env_states - 1)
    
    for step in range(10):
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, env_actions - 1)
            
        else:
            action = np.argmax(Q_table[state])
            
        reward = get_reward(state, action)
        next_state = get_next_state(state, action)
        
        Q_table[state, action] = Q_table[state, action] + alpha * (
            reward + gamma * np.max(Q_table[next_state]) - Q_table[state, action]
        )
        
        state = next_state

print("Final Q-Table:")
print(Q_table)
        