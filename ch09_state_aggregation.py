import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from ch09_random_walk import RandomWalk

#--------------------
# Policy evaluation algorithms
# --------------------

def state_aggregation_estimate_v_mc(mdp, n_episodes, n_state_bins, alpha):
    """ Estimate the value function using Gradient MC Algorithm and state aggregation - Sec 9.1 """

    # algorithm parameters
    state_bin_size = mdp.num_states // n_state_bins  # state aggregation

    # initialize value-function weights
    w = np.zeros(n_state_bins)

    for episode in range(n_episodes):
        # initialize state
        state = mdp.reset_state()

        # generate an episode S0, A0, R1, S1, A1, ...
        while not mdp.is_terminal(state):
            state, reward = mdp.step()

        # perform mc update at the end of the episode once G_t is known
        G = np.sum(mdp.rewards_received)
        for state in mdp.states_visited:
            # skip if terminal state; not recording weights for the terminal states
            if mdp.is_terminal(state):
                continue

            # place the state in the state aggregation bins, ie modulo the size of the bins
            state = (state - 1) // state_bin_size

            # update weights for this state group:
            # the value of a state is estimated as its group’s component, and when the state is updated,
            # that component alone is updated. State aggregation is a special case of SGD (9.7) in which the gradient,
            # ∇v(St,wt), is 1 for St’s group’s component and 0 for the other components.
            w[state] += alpha * (G - w[state])

        if episode % 100 == 0:
            print('w after episode {}: {}'.format(episode, np.round(w,3)), end='\r')

    return w



def state_aggregation_estimate_v_td(mdp, n, n_episodes, n_state_bins, alpha, quiet=False, gamma=1):
    """ Estimate the value function using Semi-gradient n-step TD Algorithm and state aggregation - Sec 9.1 / 7.2 """

    # algorithm parameters
    state_bin_size = mdp.num_states // n_state_bins  # state aggregation

    # initialize value-function weights
    w = np.zeros(n_state_bins)

    for episode in range(n_episodes):
        # initialize state
        state = mdp.reset_state()
        T = float('inf')
        t = 0 # time step inside of episode

        # loop for each step of episode t=0,1,2,...
        while True:
            # if we haven't reached a terminal state, take an action
            if t < T:
                state, reward = mdp.step()
                if mdp.is_terminal(state):
                    T = t + 1

            # update the state estimate at time tau
            tau = t - n + 1
            if tau >= 0:  # start updating once we've made the n-steps between current state and update target
                G = sum(gamma**(i - tau) * mdp.rewards_received[i] for i in range(tau, min(tau + n, T)))
                # if the n'th step is not at terminal state (tau + n = T), then store the aggregate reward G for the n-step path
                if tau + n < T:  # update the state at timestep tau + n
                    state_tpn = mdp.states_visited[tau + n]
                    state_tpn = (state_tpn - 1) // state_bin_size  # look up the relevant state aggregation bin to map to w
                    G += gamma**n * w[state_tpn]

                # perform the TD(n) update -- update the state at visited at timestep tau with the reward G for the n-steps ahead
                state_tau = mdp.states_visited[tau]
                state_tau = (state_tau - 1) // state_bin_size  # look up the relevant state aggregation bin to map to w
                w[state_tau] += alpha * (G - w[state_tau])

            # step episode
            t += 1
            if tau == T - 1:
                break

        if not quiet and episode % 100 == 0:
            print('w after episode {}: {}'.format(episode, np.round(w,3)), end='\r')

    return w


# --------------------
# Plotting helpers
# --------------------

def expand_aggregate_weights(w, num_states, state_bin_size):
    x = np.arange(1, num_states+1) / state_bin_size
    x = np.floor(x)
    x *= state_bin_size

    w = np.ones(int(state_bin_size)) * w.reshape(-1,1)
    w = w.flatten()

    return w, x


# --------------------
# Figure 9.1: Function approximation by state aggregation on the 1000-state random walk task, 
# using the gradient Monte Carlo algorithm (page 163).
# --------------------

def fig_9_1():
    mdp = RandomWalk(num_states=1000, left_window=100, right_window=100)

    # experiment parameters
    n_episodes = 100000
    alpha = 2e-5
    n_state_bins = 10  # bins for state aggregation
    state_bin_size = mdp.num_states / n_state_bins

    # 1. plot the state distribution and true value function

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.set_ylim(0, 0.015)
    ax2.set_yticks([0, 0.0015, 0.014])
    ax1.set_xlim(0, mdp.num_states)
    ax1.set_xticks([1, mdp.num_states])
    ax1.set_ylim(-1, 1)
    ax1.set_yticks([-1, 0, 1])
    ax1.set_xlabel('State')
    ax1.set_ylabel('Value scale')
    ax2.set_ylabel('Distribution scale')

    # calculate the steady state transition matrix
    steady_state_T = mdp.T**50
    # the stationary distribution for the start state (excluding the 2 terminal states on the left and right
    state_dist = steady_state_T[mdp.start_state, 1:-1].reshape(-1,1)
    # the true value function at steady state
    true_v = ((mdp.true_T**100) @ mdp.rewards)[:,1:-1].reshape(-1,1)

    # plot the state dist and true value fn
    ax1.plot(true_v, label=r'True value $v_{\pi}$', lw=1)
    ax2.plot(state_dist, label=r'State distribution $\mu$', c='lightgray', lw=1)
    ax2.fill_between(np.arange(state_dist.shape[0]), np.zeros(state_dist.shape[1]), np.asarray(state_dist).flatten(), color='lightgray', alpha=0.5)


    # 2. plot the approximation
    w = state_aggregation_estimate_v_mc(mdp, n_episodes, n_state_bins, alpha)

    x = np.arange(1, mdp.num_states+1) / state_bin_size
    x = np.floor(x)
    x *= state_bin_size

    y = np.ones(int(state_bin_size))
    y = y * w.reshape(-1,1)
    y = y.flatten()

    ax1.plot(x, y, label='Approximate MC value')

    ax1.legend(loc='upper left')
    ax2.legend(loc='center right')
    plt.tight_layout()

    plt.savefig('figures/ch09_fig_9_1.png')
    plt.close()



# --------------------
# Figure 9.2: Bootstrapping with state aggregation on the 1000-state random walk task.
# Left  [here (a)]: Asymptotic values of semi-gradient TD are worse than the asymptotic Monte Carlo values in Figure 9.1.
# Right [here (b)]: Performance of n- step methods with state-aggregation are strikingly similar to those with 
#                   tabular representations (cf. Figure 7.2).
# --------------------

def fig_9_2a():
    """ Asymptotic value function for TD(0) """
    mdp = RandomWalk(num_states=1000, left_window=100, right_window=100)

    # experiment parameters
    n_episodes = 100000
    alpha = 2e-3
    n_state_bins = 10  # bins for state aggregation
    state_bin_size = mdp.num_states / n_state_bins

    ax = plt.subplot()

    # 1. plot the state distribution and true value function

    ax.set_xlim(0, mdp.num_states)
    ax.set_xticks([1, mdp.num_states])
    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel('State')
    ax.set_ylabel('Value scale')

    # the true value function at steady state
    true_v = ((mdp.true_T**100) @ mdp.rewards)[:,1:-1].reshape(-1,1)
    ax.plot(true_v, label=r'True value $v_{\pi}$', lw=1)

    # 2. plot the approximation
    w = state_aggregation_estimate_v_td(mdp, 1, n_episodes, n_state_bins, alpha)
    w, x = expand_aggregate_weights(w, mdp.num_states, state_bin_size)
    ax.plot(x, w, label='Approximate TD(0) value')

    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('figures/ch09_fig_9_2a.png')
    plt.close()


def fig_9_2b():
    """ n-step TD method """
    mdp = RandomWalk(num_states=1000, left_window=50, right_window=50)

    # experiment parameters
    n_episodes = 10
    ns = 2**np.arange(10)
    alpha_ranges = np.stack([np.linspace(0, 1 - (n>4)*np.log2(n) / 10, 20) for n in ns], axis=0)
    n_state_bins = 20  # bins for state aggregation
    state_bin_size = mdp.num_states / n_state_bins

    # true value
    true_v = np.asarray(((mdp.true_T**100) @ mdp.rewards)[:,1:-1]).flatten()
    true_v = np.convolve(true_v, np.ones(int(state_bin_size))/state_bin_size, 'valid')[::int(state_bin_size)]

    # run experiment
    for n, alphas in zip(ns, alpha_ranges):
        # init rms_error
        rms_error = np.zeros(len(alphas))

        for i, alpha in enumerate(alphas):
            print('Running experiement: n={}, alpha={}'.format(n, alpha), end='\r')
            w = state_aggregation_estimate_v_td(mdp, n, n_episodes, n_state_bins, alpha, quiet=True)

            # calculate rms error
            rms_error[i] = np.sqrt(np.mean((w - true_v)**2))

        # plot rms error for this n across alphas
        rms_error = np.convolve(np.pad(rms_error, 3, mode='edge'), np.ones(4)/4, 'valid')
        plt.plot(alphas, rms_error[:len(alphas)], label='n={}'.format(n))

    plt.xlabel(r'$\alpha$')
    plt.ylabel('Average RMS error over {} states and first {} episodes'.format(mdp.num_states, n_episodes))
    plt.xlim(0, 1)
    plt.ylim(0.1, 0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig('figures/ch09_fig_9_2b.png')
    plt.close()



if __name__ == '__main__':
    np.random.seed(2)
    np.set_printoptions(linewidth=150)
    fig_9_1()
    fig_9_2a()
    fig_9_2b()

