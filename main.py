import numpy as np
import random

class QLearningAgent:
    """
    RL Autonomous Arcade Game Playing Agent
    Trains optimal policies utilizing tabular Q-Learning with decay exploration.
    """
    def __init__(self, state_size, action_size):
        self.q_table = np.zeros((state_size, action_size))
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995

    def choose_action(self, state, action_space):
        if random.uniform(0, 1) < self.epsilon:
            return action_space.sample() # Explore
        return np.argmax(self.q_table[state]) # Exploit

    def update_q_value(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        current = self.q_table[state, action]
        # Bellman Optimality Equation
        self.q_table[state, action] = current + self.learning_rate * (
            reward + self.discount_factor * best_next - current
        )
        self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    agent = QLearningAgent(state_size=10, action_size=4)
    print("Q-Learning autonomous game agent compiled and table initialized.")
