import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from ch06_random_walk import RandomWalkMDR


class RandomWalk(RandomWalkMDR):
    def __init__(self, n_states=21):
        self.all_states = np.arange(n_states)
        self.start_state = max(self.all_states)//2
        self.reset_state()

    def get_reward(self, state):
        if state == self.all_states[0]:
            return -1
        elif state == self.all_states[-1]:
            return 1
        else:
            return 0


def estimate_v(mdr, n_episodes, n, alpha, gamma=1):
    """ Estimate the value function using n-step TD method.
    This maintains a running estimate of the value function for each episode in range(n_episodes)
    """

    # Initialize records for episode values (v) and values over episodes
    v = np.zeros(len(mdr.get_states()))
    v_over_episodes = np.empty((n_episodes+1, len(mdr.get_states())))
    v_over_episodes[0] = v.copy()

    # Implements Algorithm in Section 7.1 -- n-step TD for estimating v_pi
    for episode in range(1, n_episodes+1):
        # initialize and store S0, T, t
        state = mdr.reset_state()
        T = float('inf')
        t = 0  # time step inside of episode

        # loop for each step of episode, t = 0, 1, 2, ...
        while True:
            # if we haven't reached the terminal state, take an action
            if t < T:
                state, step_reward = mdr.step()
                if mdr.is_terminal(state):
                    T = t + 1

            # update state estimate at time tau
            tau = t - n + 1
            if tau >= 0:
                G = sum(gamma**(i - tau) * mdr.rewards_received[i] for i in range(tau, min(tau + n, T)))
                if tau + n < T:
                    state_tpn =  mdr.states_visited[tau+n]  # state at time step tau + na
                    G += gamma**n * v[state_tpn]
                state_tau = mdr.states_visited[tau]  # state at time step tau
                v[state_tau] += alpha * (G - v[state_tau])

            # episode step
            t += 1
            if tau == T - 1:
                break

        # at the end of each episode, add value estimate for current episode to the aggregates
        v_over_episodes[episode] = v.copy()

    # return average over the episodes for only the non-terminal states
    return v_over_episodes[:,1:-1]


# --------------------
# Figure 7.2: Performance of n-step TD methods as a function of α, for various values of n,
# on a 19-state random walk task (Example 7.1).
# --------------------


def fig_7_2():
    mdr = RandomWalk()
    true_values = np.linspace(-1, 1, 21)[1:-1]

    n_runs = 10
    n_episodes = 10
    ns = 2**np.arange(10)
    alphas = np.hstack((np.linspace(0, 0.1, 10), np.linspace(0.15, 1, 10)))

    rms_error = np.zeros((n_runs, len(ns), len(alphas)))

    for rep in tqdm(range(n_runs)):
        for i, n in enumerate(ns):
            for j, alpha in enumerate(alphas):
                v = estimate_v(mdr, n_episodes, n, alpha)
                # The performance measure for each parameter setting, shown on the vertical axis, is
                # the square-root of the average squared error between the predicitons at the end of the episode
                # for the 19 states and their true values, then averaged over the first 10 episodes and 100 repretitions of 
                # the whole experiement.
                rms_error[rep, i, j] += np.mean(np.sqrt(np.mean((v - true_values)**2, axis=1)))

    rms_error = np.mean(rms_error, axis=0)  # avg over runs

    for i, n in enumerate(ns):
        plt.plot(alphas, rms_error[i], label='n={}'.format(n), lw=1)
    plt.xlabel(r'$\alpha$')
    plt.xlim(plt.gca().get_xlim()[0], max(alphas))
    plt.ylim(0.25, 0.55)
    plt.ylabel('Average RMS error over {} states and first {} episodes'.format(len(mdr.all_states)-2, n_episodes))
    plt.legend()

    plt.savefig('figures/ch07_fig_7_2.png')
    plt.close()


if __name__ == '__main__':
    np.random.seed(1)
    fig_7_2()
