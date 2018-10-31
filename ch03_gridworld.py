import numpy as np
import tabulate
from collections import defaultdict

from gridworld import BaseGridworld, action_to_nwse


# --------------------
# MDP
# --------------------

class Gridworld(BaseGridworld):
    def get_reward(self, state, action, next_state):
        if state == next_state:  # ie going off grid results in return to the same state
            return -1
        if self._is_special(state)[0]:
            return self._is_special(state)[1][1]
        return 0

    def get_state_reward_transition(self, state, action):
        if self._is_special(state)[0]:
            return self._is_special(state)[1]
        else:
            return super().get_state_reward_transition(state, action)

    def _is_special(self, state):
        """
        implement Example 3.5 Gridworld special state transitions A, A', B, B'
        return (True/False, reward, [(next_state, prob)] tuple
        """
        A = (1, 4)
        A_prime = (1, 0)
        B = (3, 4)
        B_prime = (3, 2)

        if state == A:
            return True, (A_prime, 10)
        if state == B:
            return True, (B_prime, 5)
        return False, (None, None)


# --------------------
# Agents and control algorithms
# --------------------

class UniformPolicyAgent:
    def __init__(self, mdp, discount=0.9, eps=1e-2, n_iterations=1000):
        self.mdp = mdp
        self.discount = discount

        # initialize values
        self.values = np.zeros((self.mdp.width, self.mdp.height))
        self.policy = {}

        # Iterative policy evaluation algorithm (Ch 4, p 59)
        for i in range(n_iterations):
            new_values = np.zeros_like(self.values)

            for state in self.mdp.get_states():
                if state in self.mdp.terminal_states:
                    continue

                q_values = {}
                for action in self.mdp.get_possible_actions(state):
                    # uniform action probability
                    action_prob = 1/len(self.mdp.get_possible_actions(state))
                    # compute q_value and update value estimate
                    q_values[action] = self.compute_q_value(state, action)
                    new_values[state] += action_prob * q_values[action]  # Bellman equation (eq. 3.14)

            # if improvement less then eps (after at least 1 iteration), stop iteration
            if np.sum(np.abs(new_values - self.values)) < eps:
                break

            # update values with new_values for the next iteration loop
            self.values = new_values

            # record optimal policy
            self.policy = self.update_policy()

    def compute_q_value(self, state, action):
        # get next state and reward from the transition model
        next_state, reward = self.mdp.get_state_reward_transition(state, action)
        return reward + self.discount * self.values[next_state]

    def update_policy(self):
        policy = {}
        for state in self.mdp.get_states():
            if state in self.mdp.terminal_states:
                continue
            q_values = {}
            for action in self.mdp.get_possible_actions(state):
                q_values[action] = self.compute_q_value(state, action)
            policy[state] = [a for a, v in q_values.items() if round(v, 5) == round(max(q_values.values()), 5)]
        return policy


class OptimalValueAgent:
    def __init__(self, mdp, discount=0.9, eps=1e-2, n_iterations=1000):
        self.mdp = mdp

        # initialize values
        self.values = np.zeros((self.mdp.width, self.mdp.height))  # eq 3.
        self.policy = {}

        # Iterative policy evaluation algorithm (Ch 4, p 59)
        for i in range(n_iterations):
            new_values = np.zeros_like(self.values)

            for state in self.mdp.get_states():
                # if terminal state, nothing to recurse down
                if state in self.mdp.terminal_states:
                    continue

                # if not terminal state, use Bellman eq to recurse value calculation
                q_values = {}
                for action in self.mdp.get_possible_actions(state):
                    # uniform action probability
                    action_prob = 1/len(self.mdp.get_possible_actions(state))
                    # get next state and reward from the transition model
                    next_state, reward = self.mdp.get_state_reward_transition(state, action)
                    # compute q_value and update value estimate
                    q_values[action] = reward + discount * self.values[next_state]

                # record optimal value
                new_values[state] = max(q_values.values())  # Bellman optimality equation (eq. 3.19)

                # record optimal policy
                self.policy[state] = [a for a, v in q_values.items() if v == max(q_values.values())]

            # if improvement less then eps (after at least 1 iteration), stop iteration
            if np.sum(np.abs(new_values - self.values)) < eps:
                break

            # update values with new_values for the next iteration loop
            self.values = new_values


# --------------------
# Figure 3.2: Gridworld example: exceptional reward dynamics (left) and state-value function
# for the equiprobable random policy (right).
# --------------------

def fig_3_2():
    mdp = Gridworld(width=5, height=5)
    agent = UniformPolicyAgent(mdp)

    with open('figures/ch03_fig_3_2.txt', 'w') as f:
        print('Figure 3.2: State-value function (V) for uniform random policy. (V = ∑ action_prob * q_value)', file=f)
        print(tabulate.tabulate(np.flipud(agent.values.T), tablefmt='grid'), file=f)  # transform coords so (0,0) is bottom left


# --------------------
# Figure 3.5: Optimal solutions to the gridworld example.
# --------------------

def fig_3_3():
    mdp = Gridworld(width=5, height=5)
    agent = OptimalValueAgent(mdp)

    f = open('figures/ch03_fig_3_5.txt', 'w')

    print('Figure 3.5: Optimal solutions to the gridworld example. (V = max (q_value))', file=f)
    print(tabulate.tabulate(np.flipud(agent.values.T), tablefmt='grid'), file=f)  # transform coordinates so (0,0) is bottom left

    grid = [['' for x in range(mdp.width)] for y in range(mdp.height)]
    for (x,y), v in agent.policy.items():
        grid[y][x] = [action_to_nwse(v_i) for v_i in v]

    # invert vertical coordinate so (0,0) is bottom left of the displayed grid
    grid = grid[::-1]

    print('Optimal policy:', file=f)
    print(tabulate.tabulate(grid, tablefmt='grid'), file=f)

    f.close()


if __name__ == '__main__':
    fig_3_2()
    fig_3_3()
