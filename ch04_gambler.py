import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class Gambler:
    """ Example 4.3: Gambler's Problem """
    def __init__(self, goal, ph):
        self.goal = goal    # terminal state of capital reached
        self.ph = ph        # probability of heads

    def get_states(self):
        """ states are gambler's capital """
        return np.arange(1, self.goal)

    def get_possible_actions(self, state):
        """ actions are stakes """
        return np.arange(min(state, self.goal - state) + 1)

    def get_reward(self, state, action, next_state):
        if next_state == self.goal:
            return 1
        else:
            return 0

    def get_transitions_states_and_probs(self, state, action):
        return [(state + action, self.ph), (state - action, 1 - self.ph)]


class ValueIterationAgent:
    """ Implementation of Section 4.4 value iteration algorithm

    Note: the output policy (in Figure 4.3) is very sensitive to:
          the eps variable (1e-4) and the q_value precision while finding the argmax (4 significant digits)
    """
    def __init__(self, mdp, n_sweeps, discount=1, eps=1e-4):
        self.mdp = mdp
        self.discount = discount
        self.eps = eps

        # initialize
        self.values = defaultdict(int)
        self.policy = {}
        self.value_sweeps = np.zeros((n_sweeps, self.mdp.get_states().shape[0] + 1))

        for i in range(n_sweeps):

            for state in self.mdp.get_states():
                old_value = self.values[state]
                q_values = {}

                # compute q values
                for action in self.mdp.get_possible_actions(state):
                    q_values[(state, action)] = self.compute_q_value(state, action)

                # update value function in-place with the optimal q value
                self.values[state] = max(q_values.values())

                # update policy if a value function update occured;
                if abs(self.values[state] - old_value) > eps:
                    # for policy choose the min action; that is if more than one actions have max q_values,
                    # then agent minimizes stake at risk from among the new set of actions
                    # alt: python max break ties by data structure index order; so the first q_value where the max
                    # occurs is recorded; thus the action corresponding to the first max(q_val) is the argmax.
                    actions = [a for (s,a), v in q_values.items() if round(v,4) == round(max(q_values.values()),4)]
                    self.policy[state] = min(actions)

            # record value sweeps for plotting
            self.value_sweeps[i] = self.get_values()

    def compute_q_value(self, state, action):
        q_value = 0
        for next_state, prob in self.mdp.get_transitions_states_and_probs(state, action):
            reward = self.mdp.get_reward(state, action, next_state)
            q_value += prob * (reward + self.discount * self.values[next_state])
        return q_value

    def get_values(self):
        values = np.zeros(self.mdp.get_states().shape[0] + 1)
        for k, v in self.values.items():
            if k == self.mdp.goal: continue
            values[k] = v
        return values

    def get_value_sweeps(self):
        return self.value_sweeps

    def get_policy(self):
        return self.policy


# --------------------
# Figure 4.3: The solution to the gambler’s problem for ph = 0.4.
# The upper graph shows the value function found by successive sweeps of value iteration.
# The lower graph shows the final policy.
# --------------------

def fig_4_3():
    mdp = Gambler(goal=100, ph=0.4)
    agent = ValueIterationAgent(mdp, n_sweeps=50)
    value_sweeps = agent.get_value_sweeps()
    policy = agent.get_policy()

    plt.subplot(2,1,1)
    for sweep in [1,2,3,32]:
        plt.plot(value_sweeps[sweep-1], label='sweep {}'.format(sweep), lw=1)
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.legend()

    plt.subplot(2,1,2)
    plt.scatter(policy.keys(), policy.values(), s=1)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')

    plt.tight_layout()


if __name__ == '__main__':
    fig_4_3()
