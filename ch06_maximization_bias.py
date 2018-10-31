import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

from agents import QLearningAgent


class MDP:
    """ Example 6.7 Maximization bias

    State representation: int from [-2, -1, 0, 1] for states left-of-B, B, A, right-of-A
    Start state: A = 0
    Action representation: left = -1; right = +1

    E.g State A=0 can transition to state B=-1 after left action of -1.

    """
    def __init__(self):
        self.start_state = 0  # state A is 0
        self.terminal_states = [-2, 1]  # left action from B -2; or, right action from A +1

    def get_possible_actions(self, state):
        if state == self.start_state: # at state A, can go left -1 or right +1
            return [-1, 1]
        elif state == -1: # at state B -- "many actions all of which cause immediate termination"
            return -np.arange(1,10)
        else:  # at the terminal states
            return [None]

    def get_state_reward_transition(self, state, action):
        """ return reward given action (state is always the starting state A=0 """
        next_state = state + action

        if next_state in [-1,1]:  # after left action frmo A to B (=-1) or right action from A to +1
            reward = 0
        else:  # when left action from B, return drawn from Normal(-0.1, 1)
            reward = np.random.normal(-0.1, 1)

        return next_state, reward

    def reset_state(self):
        return self.start_state

    def is_goal(self, state):
        return state not in [-1, 0]


class QQAgent(QLearningAgent):
    def reset(self):
        super().reset()
        self.q_values = {}
        self.q_values[0] = defaultdict(float)
        self.q_values[1] = defaultdict(float)

    def get_q_value(self, state, action, i=None):
        if i == None: # no index selected so choose the average of the two action-value estimates
            return np.sum([self.q_values[i][(state, action)] for i in self.q_values.keys()])
        else:
            return self.q_values[i][(state, action)]


    def update(self, state, action, reward, next_state):
        """ Double Q learning update to the policy -- eq 6.10 and algorithm box in Section 6.7 """

        # select which q_value to update at random, uniformly
        i = int(np.random.rand() > 0.5)
        j = (i + 1) % len(self.q_values)

        # old q value at current state
        q_i_t0 = self.get_q_value(state, action, i)

        # new q value at next state
        actions = self.mdp.get_possible_actions(next_state)
        best_i_action_idx = np.argmax([self.get_q_value(next_state, a, i) for a in self.mdp.get_possible_actions(next_state)])
        best_i_action = actions[best_i_action_idx]
        q_j_t1 = self.get_q_value(next_state, best_i_action, j)

        # q learning update per eq 6.10
        new_value_i = q_i_t0 + self.alpha * (reward + self.discount * q_j_t1 - q_i_t0)

        # perform update
        self.q_values[i][(state, action)] = new_value_i

        return new_value_i


# --------------------
# Figure 6.7: Comparison of Q-learning and Double Q-learning on a simple episodic MDP (shown inset).
# Q-learning initially learns to take the left action much more often than the right action, and always takes it
# significantly more often than the 5% minimum probability enforced by ε-greedy action selection with ε = 0.1.
# In contrast, Double Q-learning is essentially unaffected by maximization bias. These data are averaged over 10,000 runs.
# The initial action-value estimates were zero. Any ties in ε-greedy action selection were broken randomly.
# --------------------

def fig_6_7():
    mdp = MDP()
    epsilon = 0.1
    alpha = 0.1
    agents = [QLearningAgent(mdp=mdp, epsilon=epsilon, alpha=alpha),
              QQAgent(mdp=mdp, epsilon=epsilon, alpha=alpha)]

    n_runs = 10000
    n_episodes = 300

    actions_taken = np.zeros((len(agents), n_runs, n_episodes))
    rewards_received = np.zeros_like(actions_taken)

    for r in tqdm(range(n_runs)):
        # reset the agents
        for a in agents:
            a.reset()

        for e in range(n_episodes):
            for i, a in enumerate(agents):
                _, actions, rewards = a.run_episode()
                actions_taken[i, r, e] = actions[0]  # only plotting % left action from A (which is the first action in the sequence
                rewards_received[i, r, e] = rewards

    actions_fraction = np.mean(actions_taken - 1, axis=1) / -2

    plt.plot(np.arange(n_episodes), actions_fraction[0], label='Q-learning')
    plt.plot(np.arange(n_episodes), actions_fraction[1], label='Double Q-learning')
    optimal = epsilon/len(mdp.get_possible_actions(mdp.start_state))
    plt.gca().axhline(optimal, linestyle='dashed', lw=0.5, label=r'{:.0%} optimal at $\epsilon\$'.format(optimal), c='black')
    plt.xlim(0,n_episodes)
    plt.ylim(0,1)
    plt.xlabel('Episodes')
    plt.ylabel('% left actions from A')
    plt.legend()


    plt.savefig('figures/ch06_fig_6_7.png')
    plt.close()


if __name__ == '__main__':
    np.random.seed(2)
    fig_6_7()
