import numpy as np
import matplotlib.pyplot as plt
import functools
from tqdm import tqdm

from ch09_random_walk import RandomWalk


# --------------------
# Coarse coding
# --------------------

class SquareWave:
    """ Square wave function over the [0,1] interval which is 1 on [0.3, 0.7) and 0 otherwise """
    def __init__(self):
        self.true_fn = lambda x: ((0.3 <= x) & (x < 0.7))
        self.reset_steps()

    def sample(self):
        """ sample the function uniformly on [0,1); return the position/state s and function value v """
        s = np.random.rand()
        v = self.true_fn(s)

        self.steps += 1

        return s, float(v)

    def reset_steps(self):
        self.steps = 0


class CoarseCodingFeatures:
    """ Example 9.3 Coarsenes of Coarse Coding
    Implements 1-dim coarse coding given a base interval (over which to perform coarse coding), a receptive field size, and
    target coverage density (overlap of the coding surfaces). In the 1-d case this constructs intervals of the target
    receptive field size uniformly over the base interval.
    """
    def __init__(self, receptive_field_size, base_interval=[0,1], density=50, alpha=0.2):
        self.receptive_field_size = receptive_field_size  # the interval in 1-d
        self.alpha = alpha

        # build the receptive fields targeting density ~ 50;
        # that is - density / receptive field size = number of intervals needed under uniform sampling
        # intervals are build by sampling the midpoint from [0,1] uniformly then
        # subtracting and adding the half interval
        self.num_intervals = int(density / receptive_field_size)
        mid_points = np.random.uniform(min(base_interval), max(base_interval), size=self.num_intervals)
        self.intervals = np.stack((mid_points - receptive_field_size/2, mid_points + receptive_field_size/2), axis=1)

        print('Setting up features -- interval size = {}, target density = {}'.format(receptive_field_size, density))
        density = [len(np.where((i>=self.intervals[:,0]) & (i<self.intervals[:,1]))[0]) \
                    for i in np.arange(0.01, 1.01, receptive_field_size)]
        print(' .. Number of intervals created: {}; Average density: {:.0f}'.format(len(self.intervals), np.mean(density)))

        # initialize weights
        self.w = np.zeros(self.num_intervals)

    def get_active_features(self, state):
        """ return a 0-1 vector of size num_intervals with 1's for the active features and 0's for the inactive """
        return ((state >= self.intervals[:,0]) & (state < self.intervals[:,1])).astype(np.int)

    def value(self, state):
        """ return a vector of size len(w) with values only for the active features """
        out = self.get_active_features(state)
        return np.dot(out, self.w)

    def update_weights(self, state, target_value):
        n = np.sum(self.get_active_features(state))  # alpha factor
        self.w += (self.alpha/(n+1)) * (target_value - self.value(state)) * self.get_active_features(state)

    def __call__(self, x):
        """ evaluate over all points in np array x """
        x = x.reshape(-1,1)
        active_features = self.get_active_features(x)
        return np.dot(active_features, self.w)


# --------------------
# Tile coding
# --------------------

def estimate_v_mc(mdp, n_episodes, num_tilings, tile_width, alpha, true_v):
    """ Estimate the value function using Gradient MC Algorithm and tile coding - Sec 9.5.4 """

    # algorithm parameters
    # active tiles fn -- returns the active tiles indexed over tiling and tile per tiling;
    # args
    #   -- state - int; representation of the current state of the random walk mdp
    #   -- tile_width - int; size of the individual tile (partition) within each tiling of the state space
    #   -- offset - int; step of number of states over which the multiple tilings are offet from each other
    # returns
    #   -- tuple of the row index over the tilings and column index over the tiles per tiling, of the active tiles given the state
    #
    # example:
    # tile width = 200
    # offet = 4
    #
    # state = 9
    # row index is state % tile_width = 9 > np.arange(0,200,4) -- returns a binary activations vector for the activated tilings
    # column index is state // tile width = 0 -- return the column index of the tile for each tiling (all are equally spaced
    # the returned vectors are:
    #   row idx -- [1,1,1,0,...0]
    #   col idx -- [0, ... 0]
    #
    # state = 209; would return:
    #   row idx -- [1,1,1,0,...0]
    #   col idx -- [1, ... 1]
    #
    # note: the tilings are indexed [0, num_tiling] inclusive, and the tiles [0, num_mdp_states/tile_width] inclusive, 
    #       so the weight function includes the +1's
    active_tiles = lambda state, tile_width, offset: ((state % tile_width > np.arange(0, tile_width, offset)).astype(np.int),\
                                                     (state // tile_width)*np.ones(int(tile_width/offset), dtype=np.int))
    active_tiles = functools.partial(active_tiles, tile_width=tile_width, offset=tile_width//num_tilings)

    # initialize value-function weights as a matrix of shape (num tilings, num tile per tiling)
    w = np.zeros((num_tilings+1, int(mdp.num_states/tile_width)+1))
#    w = np.random.uniform(0, 0.02, size=(num_tilings+1, int(mdp.num_states/tile_width)+1))  # note -- result is quite different with this init

    # precompute the `encoding` of every state onto the tilings+tiles grid; be used to compute the value fn estimate
    states_to_features_idxs = [active_tiles(s) for s in mdp.all_states[1:-1]]
    # record the RMS error of the estimated value fn from the true v for plotting
    rms_errors = []

    for i in tqdm(range(n_episodes)):
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

            # update weights for the active features
            active_features = active_tiles(state)
            w[active_features] += (alpha/num_tilings) * (G - np.sum(w[active_features]))

        # record the rms error for plotting
        v = np.array([np.sum(w[idx]) for idx in states_to_features_idxs])
        rms_errors.append(np.sqrt(np.mean((v - true_v)**2)))

    return w, rms_errors


# --------------------
# Figure 9.8: Example of feature width’s strong effect on initial generalization (first row)
# and weak effect on asymptotic accuracy (last row).
# --------------------

def fig_9_8():
    target_fn = SquareWave()

    fig, axs = plt.subplots(6,3, figsize=(14,10))
    x = np.linspace(0,1,100)
    for ax in axs.flatten():
        ax.axis('off')

    # experiment params
    receptive_field_sizes = [1/12, 1/6, 2/5]
    n_steps = [10*(4**i) for i in range(6)]

    # run experiment
    for i, receptive_field_size in enumerate(receptive_field_sizes):
        # setup
        np.random.seed(1)
        target_fn.reset_steps()
        feature_fn = CoarseCodingFeatures(receptive_field_size)

        while target_fn.steps < max(n_steps):
            # sample the target and get the sampled state/position and the target value there
            s, v = target_fn.sample()

            # update the weights vector
            feature_fn.update_weights(s, v)

            # if a plotting step, calculate the value fn and plot
            if target_fn.steps in n_steps:
                step_idx = n_steps.index(target_fn.steps)
                axs[step_idx, i].plot(x, feature_fn(x), lw=1, label='Approximation')

    # plot the desired function on the top axis
    for ax in axs[0]:
        ax.plot(x, target_fn.true_fn(x), color='silver', alpha=0.5, label='Desired function', lw=1)
        ax.legend(loc='upper right', fontsize=8)

    # plot and annotate the feature intervals on the bottom axis
    for i, (ax, feature_type) in enumerate(zip(axs[-1], ['Narrow', 'Medium', 'Broad'])):
        ax.annotate('{} features'.format(feature_type), xy=(0.3, -0.2), xycoords='axes fraction')
        ax.plot(np.linspace(0.25, 0.25 + receptive_field_sizes[i], 2), -0.1*np.ones(2), c='black')
    plt.annotate('feature\nwidth', xy=(9/10, 1/10), xycoords='figure fraction')

    # annotate the examples numbers
    plt.annotate('# examples', xy=(1/30,9/10), xycoords='figure fraction')
    for n_step, ax in zip(n_steps, axs[:,0]):
        ax.annotate(n_step, xy=(-2/5,1/2), xycoords='axes fraction')


    plt.savefig('figures/ch09_fig_9_8.png')
    plt.close()


# --------------------
# Figure 9.10: Why we use coarse coding. Shown are learning curves on the 1000-state random walk example
# for the gradient Monte Carlo algorithm with a single tiling and with multiple tilings.
# The space of 1000 states was treated as a single continuous dimension, covered with tiles each 200 states wide.
# The multiple tilings were offset from each other by 4 states. The step-size parameter was set so that the initial
# learning rate in the two cases was the same, α = 0.0001 for the single tiling and α = 0.0001/50 for the 50 tilings.
# --------------------

def fig_9_10():
    mdp = RandomWalk()
    # true value function of the random walk markov chain
    true_v = np.asarray(((mdp.true_T**100) @ mdp.rewards)[:,1:-1]).flatten()

    # experiment parameters
    n_runs = 1
    n_episodes = 5000
    n_tilings_range = [1, 50]
    tile_width = 200
    alpha = 1e-4

    rms_error = np.zeros((len(n_tilings_range), n_runs, n_episodes))

    for i, n_tilings in enumerate(n_tilings_range):
        print('Running experiment -- n_tilings: {} (n_runs={})'.format(n_tilings, n_runs))
        for j in range(n_runs):
            # run simulation
            w, rms_error_this_run = estimate_v_mc(mdp, n_episodes, n_tilings, tile_width, alpha, true_v)

            # record rms error
            rms_error[i, j] = rms_error_this_run

    # plot avg rms error
    rms_error = np.mean(rms_error, axis=1)
    plt.plot(rms_error[0], label='State aggregation ({} tiling)'.format(n_tilings_range[0]))
    plt.plot(rms_error[1], label='Tile coding ({} tilings)'.format(n_tilings_range[1]))

    plt.xlabel('Episodes')
    plt.ylabel('RMS error averaged over {} runs'.format(n_runs))

    plt.legend()
    plt.savefig('figures/ch09_fig_9_10.png')
    plt.close()



if __name__ == '__main__':
    np.random.seed(3)
    fig_9_8()
    fig_9_10()


