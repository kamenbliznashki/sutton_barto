import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class RandomWalkMDR:
    """ Defines the Markov reward process in Example 6.2

    States are [0, 1, 2, 3, 4, 5, 6] = [Terminal state, A, B, C, D, E, Terminal State]
    Actions are [-1, 1] for left and right steps
    Returns are 0 everywhere except for landing at the right terminal state (state 6)

    """
    def __init__(self):
        self.all_states = np.arange(7)
        self.start_state = 3 # all episodes start at the center state C (here 3)
        self.reset_state()

    def reset_state(self):
        self.state = self.start_state
        self.states_visited = [self.state]
        self.rewards_received = []
        return self.state

    def get_states(self):
        return self.all_states

    def get_reward(self, state):
        # return +1 when an episode terminates on the right
        return int(state == self.all_states[-1])

    def step(self):
        action = [-1, 1][np.random.rand() >= 0.5]  # go left or right with equal probability
        next_state = self.state + action
        reward = self.get_reward(next_state)
        self.rewards_received.append(reward)

        if not self.is_terminal(next_state):
            self.state = next_state
            self.states_visited.append(next_state)

        return next_state, reward

    def is_terminal(self, state):
        # the two ends of the random walk path are the terminal states
        return (state == self.all_states[0]) or (state == self.all_states[-1])


def estimate_v(mdr, n_episodes, method, alpha=0.1):
    """ Estimate the value function using TD(0) or MC.
    This maintains a running estimate of the value function for each episode in range(n_episodes)
    """

    # initialize state values to 0.5 per Example 6.2 except v(terminal) = 0
    v = 0.5*np.ones(len(mdr.get_states()))
    v[0] = v[-1] = 0
    # initialize aggregate tracker over all episodes
    v_over_episodes = np.empty((n_episodes+1, len(v)))
    v_over_episodes[0] = v.copy()

    # Implements Algorithm in Section 6.1 -- Tabular TD(0) for estimating v_pi
    for episode in range(1, n_episodes+1):
        # initialize S
        state = mdr.reset_state()
        episode_reward = 0
        # loop until state is terminal
        while not mdr.is_terminal(state):
            next_state, step_reward = mdr.step()
            episode_reward += step_reward
            # perform td updates after every step -- eq. 6.2
            if method == 'td':
                v[state] += alpha * (step_reward + v[next_state] - v[state])
            state = next_state

        # perform mc updates at the end of the episode (when reward (G_t) is known) -- eq 6.1
        if method == 'mc':
            for state in mdr.states_visited:  # record the episode returns for each state visited
                v[state] += alpha * (episode_reward - v[state])

        # at the end of each episode, add value estimate for current episode to the aggregates
        v_over_episodes[episode] = v.copy()

    # return only the non-terminal states
    return v_over_episodes[:,1:-1]


def batch_estimate_v(mdr, n_episodes, method, alpha=1e-4):
    """ batch update algorithm in Section 6.3 """

    v = 0.5*np.ones_like(mdr.get_states())
    v[0] = v[-1] = 0
    # initialize aggregate tracker over all episodes
    v_over_episodes = np.empty((n_episodes+1, len(v)))
    v_over_episodes[0] = v.copy()

    states_visited = []
    rewards_received = []

    for episode in range(1, n_episodes+1):
        # run an episode through the mdp
        state = mdr.reset_state()
        while not mdr.is_terminal(state):
            state, _ = mdr.step()
        # record the episode states and rewards
        states_visited.append(mdr.states_visited)
        rewards_received.append(mdr.rewards_received)

        # Batch Update:
        # update to the value function for the current and all previous episodes until convergence
        while True:
            batch_update = np.zeros_like(v)
            # loop through all experience to-date
            for states, rewards in zip(states_visited, rewards_received):
                for i, (state, reward) in enumerate(zip(states, rewards)):
                    if method == 'td':
                        # terminal states are not recorded in the trace so return the zero terminal state
                        # if we've reached the end of the trace
                        next_state = 0 if i==len(states)-1 else states[i+1]
                        batch_update[state] += alpha * (reward + v[next_state] - v[state])
                    if method == 'mc':
                        batch_update[state] += alpha * (sum(rewards) - v[state])
            if np.sum(np.abs(batch_update)) < 1e-3:
                break
            v += batch_update

        # at the end of each episode, add value estimate for current episode to the aggregates
        v_over_episodes[episode] = v.copy()

    # return only the non-terminal states
    return v_over_episodes[:,1:-1]



def example_6_2():

    mdr = RandomWalkMDR()
    true_values = np.arange(1,6) / 6

    fig, axs = plt.subplots(1,2, figsize=(12,5))
    x = np.arange(1,6)  # the states to plot

    # --------------------
    # Example 6.2 left graph: The left graph above shows the values learned after various numbers of episodes
    # on a single run of TD(0). The estimates after 100 episodes are about as close as they ever come to the true values —
    # with a constant step-size parameter (α = 0.1 in this example), the values fluctuate indefinitely in response to
    # the outcomes of the most recent episodes.
    # --------------------

    estimated_v = estimate_v(mdr, n_episodes=100, method='td')
    for episode in [0, 1, 10, 100]:
        axs[0].plot(x, estimated_v[episode], marker='o', markersize=4, label='{} episodes'.format(episode))  # exclude the terminal states from est_v

    axs[0].plot(x, true_values, label='True values', marker='o', markersize=4)
    axs[0].set_title('Estimated value')
    axs[0].set_xlabel('State')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(['A', 'B', 'C', 'D', 'E'])
    axs[0].legend(loc='lower right')


    # --------------------
    # Example 6.2 right graph: The right graph shows learning curves for the two methods for various values of α.
    # The performance measure shown is the root mean-squared (RMS) error between the value function learned and the true value
    # function, averaged over the five states, then averaged over 100 runs. In all cases the approximate value function was
    # initialized to the intermediate value V (s) = 0.5, for all s.
    # The TD method was consistently better than the MC method on this task.
    # --------------------

    mc_alphas = [0.01, 0.02, 0.03, 0.04]
    td_alphas = [0.05, 0.1, 0.15]
    n_runs = 100
    n_episodes = 100

    mc_rms_error = np.zeros((len(mc_alphas), n_episodes+1))
    td_rms_error = np.zeros((len(td_alphas), n_episodes+1))

    for r in tqdm(range(n_runs)):
        # run mc
        for a, alpha in enumerate(mc_alphas):
            v = estimate_v(mdr, n_episodes, 'mc', alpha)
            # calculate rms
            mc_rms_error[a] += np.sqrt(np.mean((v - true_values)**2, axis=1))
        # run td
        for a, alpha in enumerate(td_alphas):
            v = estimate_v(mdr, n_episodes, 'td', alpha)
            # calculate rms
            td_rms_error[a] += np.sqrt(np.mean((v - true_values)**2, axis=1))

    mc_rms_error /= n_runs
    td_rms_error /= n_runs

    for i, a in enumerate(mc_alphas):
        axs[1].plot(np.arange(n_episodes+1), mc_rms_error[i], linestyle='dashed', label=r'MC, $\alpha$ = {}'.format(a))
    for i, a in enumerate(td_alphas):
        axs[1].plot(np.arange(n_episodes+1), td_rms_error[i], label=r'TD(0), $\alpha$ = {}'.format(a))

    axs[1].set_xlabel('Walks / Episodes')
    axs[1].set_title('Empirical RMS error, averaged over states')
    axs[1].legend(loc='upper right')

    plt.savefig('figures/ch06_ex_6_2.png')
    plt.close()


def example_6_3():

    mdr = RandomWalkMDR()
    true_values = np.arange(1,6) / 6

    n_runs = 100
    n_episodes = 100
    rms_error = np.zeros((2, n_episodes+1))

    for r in tqdm(range(n_runs)):
        # run mc
        v = batch_estimate_v(mdr, n_episodes, 'mc')
        rms_error[0] += np.sqrt(np.mean((v - true_values)**2, axis=1))

        # run td
        v = batch_estimate_v(mdr, n_episodes, 'td')
        rms_error[1] += np.sqrt(np.mean((v - true_values)**2, axis=1))

    rms_error /= n_runs

    plt.plot(np.arange(n_episodes+1), rms_error[0], label='MC')
    plt.plot(np.arange(n_episodes+1), rms_error[1], label='TD')
    plt.xlim(0,100)
    plt.ylim(0,0.25)
    plt.xlabel('Walks / Episodes')
    plt.ylabel('RMS error, avg over states')
    plt.title('Batch Training')
    plt.legend()

    plt.savefig('figures/ch06_fig_6_2.png')
    plt.close()


if __name__ == '__main__':
    np.random.seed(1)
    example_6_2()
    example_6_3()

