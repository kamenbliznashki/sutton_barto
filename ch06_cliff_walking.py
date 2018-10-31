import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from collections import deque
from tqdm import tqdm

from gridworld import BaseGridworld
from agents import BaseAgent, QLearningAgent
from ch06_windy_gridworld import SarsaAgent, run_sarsa_episode, print_values, print_optimal_policy


# --------------------
# MDP
# --------------------

class CliffGridworld(BaseGridworld):
    """ Example 6.6 """
    def __init__(self,
            width=12,
            height=4,
            start_state=(0,0),
            goal_state=(11,0),
            terminal_states=[(x,0) for x in range(1,11)]):
        # note: all coordinates in 0-indexed cartesian x-y; with origin (0,0) in the bottom left
        super().__init__(width, height, start_state, goal_state, terminal_states)

    def get_state_reward_transition(self, state, action):
        # perform action
        next_state = np.array(state) + np.array(action)

        # clip to grid in case action resulted in off-the-grid state
        next_state = self._clip_state_to_grid(next_state)

        # make into tuple of ints
        next_state = int(next_state[0]), int(next_state[1])

        if next_state in self.terminal_states:
            next_state = self.start_state
            reward = -100
        else:
            reward = -1

        return next_state, reward


# --------------------
# Agents and control algorithm
# --------------------

class ExpectedSarsaAgent(QLearningAgent):
    def update(self, state, action, reward, next_state):
        """ Expected Sarsa update to the policy -- eq 6.9 """

        q_t0 = self.get_q_value(state, action)

        # for the epsilon-greedy policy, take best action with prob 1-eps and choose uniformly among the rest with prob eps/num_actions
        next_state_exp_value = (1-self.epsilon)*self.get_value(next_state) + \
                               (self.epsilon/4)*sum(self.get_q_value(next_state, a) for a in self.mdp.get_possible_actions(state))

        # q learning update per eq 6.8
        new_value = q_t0 + self.alpha * (reward + self.discount * next_state_exp_value - q_t0)

        # perform update
        self.q_values[(state, action)] = new_value

        return new_value


# --------------------
# Figure 6.4: The cliff-walking task. The results are from a single run, but smoothed by averaging the reward sums 
# from 10 successive episodes.
# --------------------

def fig_6_4():
    mdp = CliffGridworld()

    sarsa_sum_rewards = []
    qlearning_sum_rewards = []
    rewards_history = deque(maxlen=10)

    n_episodes = 500

    # run sarsa agent
    agent = SarsaAgent(mdp=mdp)
    for i in range(n_episodes):
        states, actions, rewards = agent.run_episode()
        rewards_history.append(rewards)
        sarsa_sum_rewards.append(np.mean(rewards_history))
    print_optimal_policy(mdp, agent)
    print_values(mdp, agent)

    rewards_history.clear()

    # run q learning
    agent = QLearningAgent(mdp=mdp)
    for i in range(n_episodes):
        states, actions, rewards = agent.run_episode()
        rewards_history.append(rewards)
        qlearning_sum_rewards.append(np.mean(rewards_history))

    print_optimal_policy(mdp, agent)
    print_values(mdp, agent)

    # plot results
    plt.plot(np.arange(n_episodes), sarsa_sum_rewards, label='Sarsa')
    plt.plot(np.arange(n_episodes), qlearning_sum_rewards, label='Q-learning')
    plt.ylim(-100, 0)
    plt.xlim(0, 500)
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.legend()

    plt.savefig('figures/ch06_fig_6_4.png')
    plt.close()


# --------------------
# Figure 6.6: Interim and asymptotic performance of TD control methods on the cliff-walking task as of α.
# All algorithms used an ε-greedy policy with ε=0.1. Asymptotic performance is an average over 100,000 episodes
# [here 1000] whereas interim performance is an average over the first 100 episodes.
# These data are averages of over 50,000 [here 500] and 10 runs for the interim and asymptotic cases respectively.
# --------------------

def fig_6_6():
    # initialize
    mdp = CliffGridworld()
    agents = [SarsaAgent(mdp=mdp),
              ExpectedSarsaAgent(mdp=mdp),
              QLearningAgent(mdp=mdp)]
    alphas = np.linspace(0.1, 1, 10)
    plt.figure(figsize=(10,8))

    def run_experiment(mdp, agents, n_runs, n_episodes, alphas=alphas):
        avg_sum_rewards = np.zeros((len(agents), len(alphas)))
        for i, alpha in enumerate(alphas):
            # set new alpha for each agent
            for a in agents:
                a.alpha = alpha

            for r in tqdm(range(n_runs)):
                # rest q_values on each run
                for a in agents:
                    a.reset()
                # reset averages over episodes at each run
                avg_over_episodes = np.zeros(len(agents))

                for e in range(n_episodes):
                    for j, a in enumerate(agents):
                        _, _, rewards = a.run_episode()
                        # update avg over episodes online
                        avg_over_episodes[j] += 1/(e+1) * (rewards - avg_over_episodes[j])

                # update avg over runs online
                avg_sum_rewards[:,i] += 1/(r+1) * (avg_over_episodes - avg_sum_rewards[:,i])
        return avg_sum_rewards

    # run interim
    print('Running interim')
    avg_sum_rewards = run_experiment(mdp, agents, n_runs=500, n_episodes=100)

    # plot results
    plt.plot(alphas, avg_sum_rewards[0], label='Sarsa (interim)', linestyle='dotted', c='blue', lw=0.5, marker='v', markerfacecolor='none')
    plt.plot(alphas, avg_sum_rewards[1], label='Expected Sarsa (interim)', linestyle='dotted', c='red', lw=0.5, marker='x')
    plt.plot(alphas, avg_sum_rewards[2], label='Q-learning (interim)', linestyle='dotted', c='black', lw=0.5, marker='s', markerfacecolor='none')


    # run asymptotic
    print('Running asymptotic')
    avg_sum_rewards = run_experiment(mdp, agents, n_runs=10, n_episodes=1000)

    # plot results
    plt.plot(alphas, avg_sum_rewards[0], label='Sarsa (asymptotic)', c='blue', lw=0.5, marker='v', markerfacecolor='none')
    plt.plot(alphas, avg_sum_rewards[1], label='Expected Sarsa (asymptotic)', c='red', lw=0.5, marker='x')
    plt.plot(alphas, avg_sum_rewards[2], label='Q-learning (asymptotic)', c='black', lw=0.5, marker='s', markerfacecolor='none')


    # format plot
    plt.xlabel(r'$\alpha$')
    plt.xlim(0.1, 1)
    plt.xticks(np.linspace(0.1,1,10))
    plt.ylim(-150,0)
    plt.ylabel('Sum of rewards per episode')
    plt.legend()

    plt.savefig('figures/ch06_fig_6_6.png')
    plt.close()

if __name__ == '__main__':
    np.random.seed(1)
    fig_6_4()
    fig_6_6()
