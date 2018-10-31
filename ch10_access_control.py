import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from tqdm import tqdm

from agents import BaseAgent


# --------------------
# MDP
# --------------------

class AccessQueueMDP:
    """
    Example 10.2 - An access-control queuing task

    States -- list of (number of free servers, current customer priority from head of queue) representing whether servers are free (0) or busy (1),
                and the index in the reward array (ie the index of priority) for serving the current customer.
    Actions -- int in [0, 1] for reject (0) or accept (1) a customer off the queue.
    Rewards -- int in [1,2,4,8] - what a customer pays (corresponding to their priority).

    """
    def __init__(self, k=10, p=0.06, rewards=[1,2,4,8]):
        self.k = k  # number of servers
        self.p = p  # prob that a busy server becomes available at each time step
        self.rewards = rewards  # the priority is the reward

    def reset_state(self):
        # initialize customer queue and servers state array -- free (0) / busy (1)
        self.queue = deque()
        self.servers = np.zeros(self.k)

        # set up initial state (num free servers, priority of customer)
        self.state = [self.k, np.random.choice(len(self.rewards))]

        self.num_steps = 0
        return self.state

    def get_possible_actions(self, state):
        return [0, 1]

    def step(self, state, action):
        # initialize step
        reward = 0
        self.num_steps += 1

        # 1. Update servers status
        # Each busy server becomes free with probability p on each time step.
        rand = np.random.rand(self.k) >= self.p  # 0-1 vec for which server becomes free; free servers are 0s
        self.servers *= rand  # update the servers; 0s (free) stay 0s; 1s (busy) turn to 0s with prob p (when rand is < p)

        # 2. Update customers queue
        # Customers of four different priorities arrive at the queue
        # the priorities of the customers in the queue are eually randomly distributed
        c = np.random.choice(self.rewards)
        self.queue.insert(0, self.rewards.index(c))

        # 3. Serve queue
        # at each time step, the customer at the head of the queue is either accepted (assigned to a server) or rejected
        # (removed from queue with zero reward)
        c = self.rewards[state[1]]
        # check for free (0) server
        free_servers = np.where(self.servers == 0)[0]
        # if there are free, perform action (accept customer or reject)
        if len(free_servers) != 0:
            # free_servers[0] is the first available server
            self.servers[free_servers[0]] = action  # if action is accept (1) then server is busy (1); if reject (0) then server stays free (0)
            reward = c*action  # if action is accept (1) then reward is c*1, if reject (0) then reward is c*0 = 0

        # 4. Update state
        # pop next customer from queue and load into the state representation
        c = self.queue.pop()
        # update state
        next_state = [len(np.where(self.servers == 0)[0]), c]

        return next_state, reward


# --------------------
# Agent and control algorithm
# --------------------

def run_differential_sarsa_step(mdp, agent):
    """ execute the differential semi-gradient Sarsa algorithm in Sec 10.3 """

    # initialize state S and action A
    state = mdp.reset_state()
    action = agent.get_action(state)

    # loop for each step:
    for i in tqdm(range(int(agent.n_steps))):
        # take action A, observe R, S'
        next_state, reward = mdp.step(state, action)

        # choose A' as a function of q(S', ., w)
        next_action = agent.get_action(next_state)

        # differential sarsa update
        agent.update(state, action, reward, next_state, next_action)

        state = next_state
        action = next_action

class DifferentialOneStepSarsaAgent(BaseAgent):
    def __init__(self, n_steps, beta, run_episode_fn=run_differential_sarsa_step, **kwargs):
        self.n_steps = n_steps  # number of loops/time steps to run
        self.beta = beta  # the step size for updates to the avg reward
        super().__init__(run_episode_fn=run_differential_sarsa_step, **kwargs)

    def reset(self):
        self.w = np.random.uniform(-0.01, 0.01, ((self.mdp.k + 1, len(self.mdp.rewards), len(self.mdp.get_possible_actions(None)))))
        self.avg_reward = 0
        self.num_updates = 0

    def get_q_value(self, state, action):
        return self.w[tuple(state + [action])]

    def update(self, state, action, reward, next_state, next_action):
        delta = reward - self.avg_reward + self.get_q_value(next_state, next_action) - self.get_q_value(state, action)

        # update avg reward
        self.avg_reward += self.beta * delta

        # update weights
        self.w[tuple(state + [action])] += self.alpha * delta

        self.num_updates += 1


    def get_policy(self):
        return np.argmax(self.w, axis=-1).T

    def get_value_fn(self):
        return np.max(self.w, axis=-1).T


# --------------------
# Figure 10.5: The policy and value function found by differential semi-gradient one-step Sarsa on the access-control queuing task after
# 2 million steps. The drop on the right of the graph is probably due to insufficient data; many of these states were never experienced.
# The value learned for R was about 2.31.
# --------------------

def fig_10_5():
    mdp = AccessQueueMDP()
    agent = DifferentialOneStepSarsaAgent(mdp=mdp, n_steps=2e6, alpha=0.01, beta=0.01, epsilon=0.1)

    n_steps = 20

    # run episode -- until we reach max_steps
    agent.run_episode()

    policy = agent.get_policy()
    value_fn = agent.get_value_fn()

    plt.subplot(2,1,1)
    sns.heatmap(policy[:,1:], xticklabels=np.arange(1,11), yticklabels=mdp.rewards, cbar=False, annot=True)
    plt.yticks(rotation=0)
    plt.xlabel('Number of free servers')
    plt.ylabel('Piority')
    plt.title('Policy')

    plt.subplot(2,1,2)
    for i, r in enumerate(mdp.rewards):
        plt.plot(value_fn[i], label='priority {}'.format(r))
    plt.gca().axhline(0, linestyle='--', lw=0.5, c='silver')
    plt.xticks(np.arange(mdp.k + 1))
    plt.yticks(np.linspace(-10, 10, 5))
    plt.xlabel('Number of free servers')
    plt.ylabel('Differential value of best action')
    plt.title('Value function')
    plt.legend()

    plt.tight_layout()
    plt.savefig('figures/ch10_fig_10_5.png')
    plt.close()


if __name__ == '__main__':
    np.random.seed(1)
    fig_10_5()
