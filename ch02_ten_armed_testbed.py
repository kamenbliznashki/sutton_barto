import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def save_fig(name, ch = 'ch02'):
    plt.savefig('figures/{}_{}.png'.format(ch, name))
    plt.close()


# --------------------
# Bandit problem and functions
# --------------------

class BaseBandit:
    """
    Base class implementation of Section 2.3: The 10-armed Testbed
    """
    def __init__(
            self,
            k_arm=10,     # number of arms
            eps=0,        # explore with prob eps; exploit with prob 1-eps
            initial_q=0,  # initial action-value estimates
            true_q_mean=0 # true reward mean
            ):
        self.k_arm = k_arm
        self.possible_actions = np.arange(self.k_arm)
        self.eps = eps
        self.initial_q = initial_q
        self.true_q_mean = true_q_mean

        self.reset()

    def reset(self):
        # p21: for each bandit the action values q_*(a) were selected
        # from a normal distribution with 0 mean and variance 1
        self.q_true = np.random.randn(self.k_arm) + self.true_q_mean

        # initialize 'A simple bandit algorithm' p24
        self.q_estimate = np.zeros(self.k_arm) + self.initial_q
        self.action_count = np.zeros(self.k_arm)

        # record how often the optimal action is selected
        self.optimal_action_freq = 0

    def act(self):
        # explore with prob eps; exploit with prob 1-eps
        if np.random.rand() < self.eps: # explore
            action = np.random.choice(self.possible_actions)
        else:
            action = np.argmax(self.q_estimate)
        return action

    def reward(self, action_idx):
        # p21: the actual reward is selected from a normal distribution with mean q_*(A_t) and variance 1
        return np.random.randn() + self.q_true[action_idx]

    def update_q(self, action, reward):
        # simple average update (eq. 2.3)
        self.q_estimate[action] += 1/self.action_count[action] * (reward - self.q_estimate[action])

    def step(self):
        # single loop in 'A simple bandit algorithm' p24
        action = self.act()
        reward = self.reward(action)
        self.action_count[action] += 1
        self.update_q(action, reward)

        # online update of average for optimal action frequency (eq 2.3)
        if action == np.argmax(self.q_true):  # action == best possible action
            self.optimal_action_freq += 1/np.sum(self.action_count) * (1 - self.optimal_action_freq)

        return action, reward


class ExponentialAverageBandit(BaseBandit):
    """
    Section 2.5 Tracking a Nonstationary Problem
    """
    def __init__(
            self,
            step_size=0.1,  # exponential weighted average param
            **kwargs
            ):
        super().__init__(**kwargs)
        self.step_size = step_size

    def update_q(self, action, reward):
        # exponential average update (eq. 2.5)
        self.q_estimate[action] += self.step_size * (reward - self.q_estimate[action])


class UCBBandit(BaseBandit):
    """
    Implements Section 2.7 Upper Confidence Bound Action Selection and eq. (2.8)
    """
    def __init__(
            self,
            c=2,  # controls degree of exploration
            **kwargs
            ):
        super().__init__(**kwargs)
        self.c = c

    def act(self):
        if np.random.rand() < self.eps: # explore
            action = np.random.choice(self.possible_actions)
        else:  # exploit (eq. 2.8)
            t = np.sum(self.action_count) + 1
            q = self.q_estimate + self.c * np.sqrt(np.log(t) / (self.action_count + 1e-6))
            action = np.argmax(q)
        return action


class GradientBandit(BaseBandit):
    """
    Implementation of Section 2.8 Gradient Bandit Algorithm
    """
    def __init__(
            self,
            baseline=True,  # use average returns as a baseline for gradient calculation
            step_size=0.1,  # exponential weighted avg param
            **kwargs
            ):
        super().__init__(**kwargs)
        self.baseline = baseline
        self.step_size = step_size
        self.average_reward = 0

    def act(self):
        e = np.exp(self.q_estimate)
        self.softmax = e / np.sum(e)
        return np.random.choice(self.possible_actions, p=self.softmax)

    def update_q(self, action, reward):
        # avg rewards serve as a baseline; if reward > baseline then prob(action) is increased
        # first do online update of average reward (2.3)
        # (note n number of steps == sum of action counts since at each step only one action is chosen
        self.average_reward += 1/np.sum(self.action_count) * (reward - self.average_reward)
        baseline = self.average_reward if self.baseline else 0
        # gradient update (eq 2.10):
        mask = np.zeros_like(self.softmax)
        mask[action] = 1
        self.q_estimate += self.step_size * (reward - baseline) * (mask - self.softmax)


# --------------------
# Evaluate a list of bandit problems
# --------------------

def run_bandits(bandits, n_runs, n_steps):
    """ simulates a list of bandit running each for n_teps and then averaging over n_runs """

    rewards = np.zeros((len(bandits), n_runs, n_steps))
    optimal_action_freqs = np.zeros_like(rewards)

    for b, bandit in enumerate(bandits):
        for run in tqdm(range(n_runs)):
            # runs are independed; so reset bandit
            bandit.reset()
            for step in range(n_steps):
                # step bandit (act -> reward)
                action, reward = bandit.step()
                # record reward averages and optimal action frequence
                rewards[b, run, step] = reward
                if action == np.argmax(bandit.q_true):
                    optimal_action_freqs[b, run, step] = 1

    # average across the n_runs
    avg_rewards = rewards.mean(axis=1)
    avg_optimal_action_freqs = optimal_action_freqs.mean(axis=1)

    return avg_rewards, avg_optimal_action_freqs


# --------------------
# Figure 2.1: An example bandit problem from the 10-armed testbed.
# The true value q⇤(a) of each of the ten actions was selected according to a normal distribution
# with mean zero and unit variance, and then the actual rewards were selected according to a 
# mean q⇤(a) unit variance normal distribution, as suggested by these gray distributions.
# --------------------

def fig_2_1():
    plt.violinplot(np.random.randn(200,10) + np.random.randn(10), showmeans=True)
    plt.xticks(np.arange(1,11), np.arange(1,11))
    plt.xlabel('Action')
    plt.ylabel('Reward distribution')
    save_fig('fig_2_1')


# --------------------
# Figure 2.2: Average performance of eps-greedy action-value methods on the 10-armed testbed.
# These data are averages over 2000 runs with different bandit problems.
# All methods used sample averages as their action-value estimates.
# --------------------

def fig_2_2(runs=2000, steps=1000, epsilons=[0, 0.01, 0.1]):
    bandits = [BaseBandit(eps=eps) for eps in epsilons]
    avg_rewards, avg_optimal_action_freqs = run_bandits(bandits, runs, steps)

    # plot results
    plt.subplot(2,1,1)
    for eps, rewards in zip(epsilons, avg_rewards):
        plt.plot(rewards, label=r'$\epsilon$ = {}'.format(eps), lw=1)
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()

    plt.subplot(2,1,2)
    for eps, optimal_action_freq in zip(epsilons, avg_optimal_action_freqs):
        plt.plot(optimal_action_freq, label=r'$\epsilon$ = {}'.format(eps), lw=1)
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()

    plt.tight_layout()
    save_fig('fig_2_2')


# --------------------
# Figure 2.3: The effect of optimistic initial action-value estimates on the 10-armed testbed.
# Both methods used a constant step-size parameter, step_size = 0.1.
# --------------------

def fig_2_3(runs=2000, steps=1000, epsilons=[0, 0.1], initial_qs=[5, 0]):
    bandits = []
    for eps, initial_q in zip(epsilons, initial_qs):
        bandits.append(ExponentialAverageBandit(eps=eps, initial_q=initial_q, step_size=0.1))

    _, avg_optimal_action_freqs = run_bandits(bandits, runs, steps)

    # plot results
    for i, (eps, initial_q) in enumerate(zip(epsilons, initial_qs)):
        plt.plot(avg_optimal_action_freqs[i], label=r'Q1 = {}, $\epsilon$ = {}'.format(initial_q, eps))
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()

    save_fig('fig_2_3')


# --------------------
# Figure 2.4: Average performance of UCB action selection on the 10-armed testbed.
# As shown, UCB generally performs better than epsilon-greedy action selection,
# except in the first k steps, when it selects randomly among the as-yet-untried actions.
# --------------------

def fig_2_4(runs=2000, steps=1000):
    bandits = [UCBBandit(eps=0, c=2), BaseBandit(eps=0.1)]

    _, avg_optimal_action_freqs = run_bandits(bandits, runs, steps)

    # plot results
    plt.plot(avg_optimal_action_freqs[0], label='UCB c = {}'.format(bandits[0].c))
    plt.plot(avg_optimal_action_freqs[1], label=r'$\epsilon$-greedy $\epsilon$ = {}'.format(bandits[1].eps))
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()

    save_fig('fig_2_4')


# --------------------
# Figure 2.5: Average performance of the gradient bandit algorithm with and without a reward baseline
# on the 10-armed testbed when the q⇤(a) are chosen to be near +4 rather than near zero.
# --------------------

def fig_2_5(runs=2000, steps=1000):
    bandits = [
            GradientBandit(step_size=0.1, true_q_mean=4, baseline=True),
            GradientBandit(step_size=0.4, true_q_mean=4, baseline=True),
            GradientBandit(step_size=0.1, true_q_mean=4, baseline=False),
            GradientBandit(step_size=0.4, true_q_mean=4, baseline=False)]

    _, avg_optimal_action_freqs = run_bandits(bandits, runs, steps)

    # plot results
    for i, bandit in enumerate(bandits):
        plt.plot(avg_optimal_action_freqs[i],
                label='step_size = {}, baseline = {}'.format(bandit.step_size, bandit.baseline))
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()

    save_fig('fig_2_5')



if __name__ == '__main__':
    fig_2_1()
    fig_2_2()
    fig_2_3()
    fig_2_4()
    fig_2_5()
