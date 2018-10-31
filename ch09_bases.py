import numpy as np
import matplotlib.pyplot as plt
import functools
from tqdm import tqdm

from ch09_random_walk import RandomWalk


# --------------------
# Figure 9.3: One-dimensional Fourier cosine-basis features xi, i = 1,2,3,,
# for approximating functions over the interval [0, 1]. After Konidaris et al. (2011).
# --------------------

def fig_9_3():
    xx = np.linspace(0,1,50)
    x_i = lambda i, s: np.cos(i*np.pi*s)

    fig, axs = plt.subplots(1,5, figsize=(14,4))

    for i, ax in enumerate(axs.flatten()):
        ax.plot(xx, x_i(i, xx))
        ax.set_title('i={}'.format(i))
        ax.set_xticks([0,1])
        ax.set_yticks([-1,1])

    plt.tight_layout()
    plt.savefig('figures/ch09_fig_9_3.png')
    plt.close()


# --------------------
# Figure 9.4: A selection of six two-dimensional Fourier cosine features, each labeled by the vector ci that defines it
# (s1 is the horizontal axis, and ci is shown with the index i omitted). After Konidaris et al. (2011).
# --------------------

def fig_9_4():
    x = np.linspace(0,1,50)
    xx, yy = np.meshgrid(x, x)
    # note: need to flipud(yy) because meshgrid return yy increasing going down the vertical axis (numpy array coords -
    # column 1 is [0, ... 1].T) whereas in the plot y increases going up the vertical axis (standard cartesian coords)
    s = np.dstack((xx, np.flipud(yy)))

    cs = np.array([[0,1], [1,0], [1,1], [0,5], [2,5], [5,2]])

    x_i = lambda s, c: np.cos(np.pi * np.dot(s, c))

    fig, axs = plt.subplots(2,3)
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(x_i(s, cs[i]), cmap=plt.get_cmap('gray'))
        ax.set_title('c = {}'.format(cs[i]))
        ax.set_xticks(ax.get_xlim())
        ax.set_yticks(ax.get_ylim())
        ax.set_xticklabels([0,1])
        ax.set_yticklabels([0,1])

    plt.tight_layout()
    plt.savefig('figures/ch09_fig_9_4.png')
    plt.close()


# --------------------
# Figure 9.5: Fourier basis vs polynomials on the 1000-state random walk.
# Shown are learning curves for the gradient Monte Carlo method with Fourier and polynomial bases of order 5, 10, and 20.
# The step-size parameters were roughly optimized for each case: α = 0.0001 for the polynomial basis and
# α = 0.00005 for the Fourier basis. The performance measure (y-axis) is the root mean squared value error (9.1).
# --------------------

#--------------------
# Policy evaluation algorithms
# --------------------

def estimate_v_mc(mdp, basis_fn, n_basis_dims, n_episodes, alpha):
    """ Estimate a linear value function using Gradient MC Algorithm - Sec 9.4.
    The value function is linear in the weights and onto a supplied basis function (e.g. polynomial, Fourier) """

    # initialize value-function weights
    w = np.zeros(n_basis_dims)
    w_history = []

    for episode in tqdm(range(n_episodes)):
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

            # update weights
            w += alpha * (G - np.dot(w, basis_fn(state))) * basis_fn(state)

        # record the weights for plotting
        w_history.append(w.copy())

    return w, np.vstack(w_history)


def fig_9_5():
    mdp = RandomWalk()
    # true value function of the random walk markov chain
    true_v = np.asarray(((mdp.true_T**100) @ mdp.rewards)[:,1:-1]).flatten()

    # basis functions
    # args:
    #   -- state - int; representation of the current state of the random walk mdp or an array of all states;
    #   -- num_states - int; total number of state to normalize the state input into [0,1]
    #   -- n -- int; the number of basis functions to include [0, ... n-1]
    # returns:
    #   -- float or np array of floats; feature values of the basis functions evaluated at the state.
    fourier_basis_fn = lambda state, num_states, n: np.array([np.cos(i * np.pi * state/num_states) for i in range(n)])
    poly_basis_fn = lambda state, num_states, n: (np.array([state/num_states]).reshape(1,-1)**np.arange(n).reshape(-1,1)).squeeze()  # out: (n_order, num_states)

    # experiment parameters
    n_runs = 5
    n_episodes = 5000
    alphas = [5e-5, 1e-4]
    n_basis_order = [5+1, 10+1, 20+1]  # 1 is added to include the constant function
    basis_fns = {'Fourier basis': fourier_basis_fn,
                 'Polynomial basis': poly_basis_fn}

    for (basis_fn_name, basis_fn), alpha in zip(basis_fns.items(), alphas):
        for n_order in n_basis_order:
            # freeze the number of basis functions and number of states
            basis_fn = functools.partial(basis_fn, n=n_order, num_states=mdp.num_states)

            print('Running experiment -- basis: {}, n={} (n_runs={})'.format(basis_fn_name, n_order-1, n_runs))
            rms_error = np.zeros((n_runs, n_episodes))
            for i in range(n_runs):
                # run simulation
                w, w_history = estimate_v_mc(mdp, basis_fn, n_order, n_episodes, alpha)

                # calculate value and rms
                states = np.arange(1, mdp.num_states + 1)
                v = np.dot(w_history, basis_fn(states))  # in dims: (episodes, len(w))  (n_order, num_states)
                                                         # out dims: (episodes, num_states)
                # record rms
                rms_error[i] = np.sqrt(np.mean((v - true_v)**2, axis=1))

            # plot avg rms error
            rms_error = np.mean(rms_error, axis=0)
            plt.plot(rms_error, label='{} n={}'.format(basis_fn_name, n_order-1))
            plt.savefig('figures/ch09_fig_9_5.png')

    plt.xlabel('Episodes')
    plt.ylabel('RMS error averaged over {} runs'.format(n_runs))

    plt.legend()
    plt.savefig('figures/ch09_fig_9_5.png')
    plt.close()


if __name__ == '__main__':
    fig_9_3()
    fig_9_4()
    fig_9_5()
