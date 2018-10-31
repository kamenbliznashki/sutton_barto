import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from ch07_random_walk import RandomWalk


# --------------------
# Prediction algorithms
# --------------------

def offline_lambda_return(mdp, n_episodes, lam, alpha, gamma=1):
    """ Offline lambda return algorithm per Sec 12.1
    Assume linear function approx ie state value is w[state] and follow the Monte Carlo update in Ch 6 - end of episode sweep.
    """

    w = np.zeros(len(mdp.get_states()))

    w_over_episodes = np.empty((n_episodes, len(w)))

    for i in range(n_episodes):
        state = mdp.reset_state()

        # run mdp until termination
        while not mdp.is_terminal(state):
            state, _ = mdp.step()

        # make whole sequence of off-line updates according to semi-grad rule (eq 12.4)
        for t in range(len(mdp.states_visited)):
            # 1. Compute G_t_t+n (eq 12.1) and G_t (eq 3.8) to get G_t_lambda (eq 12.3)
            # grab sequence of rewards from this state onward
            rewards = np.array(mdp.rewards_received)[t:]
            # grab the terminal state values
            # note: paragraph before eq 12.3 -- after a terminal state, all subsequent n-step return are equal to G_t,
            #       that is, we separate the post termination terms from the main sum into eq 12.3
            #       here, terminal values are computed for all terms except for the last, where a [0] is added.
            terminal_state_values = np.array([w[state] for state in mdp.states_visited[t+1:]] + [0])
            # apply discounting to the rewards and terminal state values
            rewards *= gamma ** np.arange(len(rewards))
            terminal_state_values *= gamma ** np.arange(1, len(terminal_state_values)+1)  # discounting is offset by 1 for the terminal states
            # complete eq 12.1; note again the last element of terminal_state_values is 0 (it is G_t and not G_t_t+n -- cf Sec 7.1)
            G_ts = np.cumsum(rewards) + terminal_state_values

            # 2. Compute the lambda multipliers
            lambdas = lam ** np.arange(len(rewards))
            lambdas[:-1] *= (1-lam)  # all but the last term (the G_t) are multiplied by (1-lam)
            assert np.allclose(np.sum(lambdas), 1, atol=1e-10), 'Lambda weights not summing to 1.'

            # 3. Compute G_t_lambda (complete eq 12.3)
            G_t_lambda = np.sum(G_ts * lambdas)

            # 4. update at state t per eq 12.4
            state = mdp.states_visited[t]
            w[state] += alpha * (G_t_lambda - w[state])

        # update averages record
        w_over_episodes[i] = w.copy()

    return w_over_episodes[:, 1:-1]  # return the non-terminal states


def td_lambda(mdp, n_episodes, lam, alpha, gamma=1):
    """ Semi-grad TD(lambda) per Sec 12.2 """

    w = np.zeros(len(mdp.get_states()))

    w_over_episodes = np.empty((n_episodes, len(w)))

    for i in range(n_episodes):
        # init S
        state = mdp.reset_state()

        # init eligibility trace
        z = np.zeros_like(w)

        # run mdp until termination
        while not mdp.is_terminal(state):
            next_state, reward = mdp.step()

            # update z -- accumulating trace
            z *= gamma * lam
            z[state] += 1

            # compute the TD error
            delta = reward + gamma * w[next_state] - w[state]

            # update w
            w += alpha * delta * z

            # update state
            state = next_state

        # update averages record
        w_over_episodes[i] = w.copy()

    return w_over_episodes[:, 1:-1]  # return the non-terminal states


def true_online_td_lambda(mdp, n_episodes, lam, alpha, gamma=1):
    """ True online TD(lambda) per Sec 12.5 """

    w = np.zeros(len(mdp.get_states()))

    w_over_episodes = np.empty((n_episodes, len(w)))

    for i in range(n_episodes):
        # init state
        state = mdp.reset_state()

        # init eligibility trace
        z = np.zeros_like(w)

        # initialize value tracker
        v_old = 0

        # run mdp until termination
        while not mdp.is_terminal(state):
            next_state, reward = mdp.step()

            # store value of state and next state
            v = w[state]
            v_next = w[next_state]

            # compute the TD error
            delta = reward + gamma * v_next - v

            # update z -- dutch trace
            z_old = z[state]
            z *= gamma * lam
            z[state] += 1 - alpha * gamma * lam * z_old

            # update w
            w += alpha * (delta + v - v_old) * z
            w[state] -= alpha * (v - v_old)

            # make a step
            v_old = v_next
            state = next_state

        # update averages record
        w_over_episodes[i] = w.copy()

    return w_over_episodes[:, 1:-1]  # return the non-terminal states


# --------------------
# Plotting functions
# --------------------

def plot_prediction_fn(prediction_fn, filename, title):
    mdp = RandomWalk(n_states=21)
    true_values = np.linspace(-1, 1, len(mdp.all_states))[1:-1]

    # experiment parameters
    n_runs = 10
    n_episodes = 10
    lambdas = [0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
    alphas = np.hstack((np.linspace(0, 0.1, 5), np.linspace(0.15, 1, 10)))

    rms_error = np.zeros((n_runs, len(lambdas), len(alphas)))

    for run in tqdm(range(n_runs)):
        for i, lam in enumerate(lambdas):
            for j, alpha in enumerate(alphas):
                w = prediction_fn(mdp, n_episodes, lam, alpha)
                rms_error[run, i, j] += np.sqrt(np.mean((w - true_values)**2))

    rms_error = np.mean(rms_error, axis=0)  # avg over runs

    for i, lam in enumerate(lambdas):
        plt.plot(alphas, rms_error[i], label=r'$\lambda$={}'.format(lam), lw=1)
    plt.xlabel(r'$\alpha$')
    plt.xlim(plt.gca().get_xlim()[0], max(alphas))
    plt.ylim(0.25, 0.55)
    plt.ylabel('Average RMS error at the end of the episode\nover the first {} episodes'.format(n_episodes))
    plt.title(r''+title)
    plt.legend()

    plt.savefig(filename)
    plt.close()


# --------------------
# Figure 12.3: 19-state Random walk results (Example 7.1):
# Performance of the offline λ-return algorithm alongside that of the n-step TD methods.
# In both case, intermediate values of the bootstrapping parameter (λ or n) performed best.
# The results with the off-line λ-return algorithm are slightly better at the best values of α and λ, and at high α.
# --------------------

def fig_12_3():
    plot_prediction_fn(offline_lambda_return, 'figures/ch12_fig_12_3.png', 'Off-line $\lambda$-return algorithm')


# --------------------
# Figure 12.6: 19-state Random walk results (Example 7.1): Performance of TD(λ) alongside that of the off-line λ-return algorithm.
# The two algorithms performed virtually identically at low (less than optimal) α values, but TD(λ) was worse at high α values.
# --------------------

def fig_12_6():
    plot_prediction_fn(td_lambda, 'figures/ch12_fig_12_6.png', 'TD($\lambda$)')


# --------------------
# Figure 12.8: 19-state Random walk results (Example 7.1): Performance of online and off-line λ-return algorithms.
# The performance measure here is the VE at the end of the episode, which should be the best case for the off-line algorithm.
# Nevertheless, the on-line algorithm performs subtlely better. For comparison, the λ=0 line is the same for both methods.
# --------------------

def fig_12_8():
    plot_prediction_fn(true_online_td_lambda, 'figures/ch12_fig_12_8.png', 'True online TD($\lambda$)')


if __name__ == '__main__':
    np.random.seed(1)
    fig_12_3()
    fig_12_6()
    fig_12_8()

