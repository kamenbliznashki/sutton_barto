import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm

from agents import BaseAgent
from tiles3 import IHT, tiles


# --------------------
# MDP
# --------------------

class MountainCar:
    """ Defines the Mountain Car MDP in Example 10.1.

    States representaion -- tuple of (x, xdot) for position and velocity; this is a continuous state representation, which
                            the agent converts to discrete binary features via grid tilings.
    Actions representation -- int in [-1,0,1] corresponding to full throttle reverse, zero throttle, full throttle forward.
    Rewards representation -- int; -1 on all time steps until the car moves past its goal position, which ends the episode.

    """
    def __init__(self, x_bound=[-1.2, 0.5], xdot_bound=[-0.07, 0.07]):
        self.x_min, self.x_max = x_bound
        self.xdot_min, self.xdot_max = xdot_bound

        self.reset_state()

    def reset_state(self):
        x = np.random.uniform(low=-0.6, high=-0.4)
        xdot = 0
        self.state = (x, xdot)
        return self.state

    def get_reward(self, state, action, next_state):
        next_x, next_xdot = next_state
        if next_x >= self.x_max:
            return 0
        else:
            return -1

    def get_possible_actions(self, state):
        return [-1, 0, 1]

    def get_state_reward_transition(self, state, action):
        x, xdot = state

        next_xdot = xdot + 0.001*action - 0.0025*np.cos(3*x)
        next_xdot = np.clip(next_xdot, self.xdot_min, self.xdot_max)

        next_x = x + next_xdot
        next_x = np.clip(next_x, self.x_min, self.x_max)

        if next_x == self.x_min:
            next_xdot = 0

        next_state = next_x, next_xdot
        reward = self.get_reward(state, action, next_state)

        return next_state, reward

    def is_goal(self, state):
        return np.round(state[0], 6) >= np.round(self.x_max, 6)


# --------------------
# Agent and control algorithm
# --------------------

def run_nstep_sarsa_episode(mdp, agent):
    """ execute the n-step semi-gradient Sarsa algorithm -- Sec 10.2 """

    # algorithm parameters
    T = float('inf')
    t = 0
    n = agent.n
    discount = agent.discount

    # initialize and store S0
    state = mdp.reset_state()
    states_history = [state] + [None]*n  # all store and access operations (S_t, A_t, R_t) can take their index mod n+1
    rewards_history = [None]*(n+1)        # all store and access operations (S_t, A_t, R_t) can take their index mod n+1

    # select and store an action A0
    action = agent.get_action(state)
    actions_history = [action] + [None]*n # all store and access operations (S_t, A_t, R_t) can take their index mod n+1

    # loop for each step of episode, t = 0, 1, 2, ...
    while True:
        # if we haven't reached the terminal state, take an action
        if t < T:
            # take action A, observe and store R_t+1, S_t+1
            state, reward = mdp.get_state_reward_transition(state, action)

            states_history[(t+1) % (n+1)] = state
            rewards_history[(t+1) % (n+1)] = reward

            if mdp.is_goal(state):
                T = t + 1
            else:
                # select and store next action A_t+1
                action = agent.get_action(state)
                actions_history[(t+1) % (n+1)] = action

        # update state estimate at time tau
        tau = t - n + 1
        if tau >= 0:  # start updating once we've made the n-step between current state and update target
            G = sum(discount**(i - tau) * rewards_history[(i+1) % (n+1)] for i in range(tau, min(tau + n, T)))

            # if not at the terminal state, then increment the aggregate reward G with the state-value of future rewards discounted
            if tau + n < T:
                state_tpn = states_history[(tau+n) % (n+1)]  # state at time step tau + n
                action_tpn = actions_history[(tau+n) % (n+1)]
                G += discount**n * agent.get_q_value(state_tpn, action_tpn)

            # update q values for time step tau
            state_tau = states_history[tau % (n+1)]
            action_tau = actions_history[tau % (n+1)]
            # perform update
            agent.update(state_tau, action_tau, G)

        # episode step
        t += 1
        if tau == T - 1:
            break

    return t


class NStepSarsaAgent(BaseAgent):
    def __init__(self, mdp, n, run_episode_fn=run_nstep_sarsa_episode, num_tilings=8, num_tiles=8, max_size=4096, snapshot_timesteps=[], **kwargs):
        self.n = n
        self.snapshot_timesteps = snapshot_timesteps

        # continuous states of the mdp are converted into discrete by using tiling (Sec 9.5.4)
        # tilings form a grid of num_tilings, each containing num_tiles
        self.max_size = max_size
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles

        super().__init__(mdp=mdp, run_episode_fn=run_episode_fn, **kwargs)


    def reset(self):
        # set up tiles to extract discrete feature represetation of the continuous mdp state rep and discrete action rep
        self.iht = IHT(self.max_size)
        self.state_scale_factor = [self.num_tiles / abs(self.mdp.x_max - self.mdp.x_min),
                                   self.num_tiles / abs(self.mdp.xdot_max - self.mdp.xdot_min)]

        # setup a tiles fn which returns a list of the active tiles given the state, action pair
        self.tiles = lambda state, action: tiles(self.iht,
                                                 self.num_tilings,
                                                 [state[0]*self.state_scale_factor[0], state[1]*self.state_scale_factor[1]],
                                                 [action])
        self.w = np.zeros(self.max_size)
        self.num_updates = 0
        self.w_history = []

    def get_q_value(self, state, action):
        active_tiles = self.tiles(state, action)
        return np.sum(self.w[active_tiles])

    def update(self, state, action, rewards):
        active_tiles = self.tiles(state, action)
        self.w[active_tiles] += self.alpha * (rewards - np.sum(self.w[active_tiles]))

        self.num_updates += 1

        # plotting functionality
        self.snapshot_weights()

    # plotting functionality
    def snapshot_weights(self):
        if self.num_updates in self.snapshot_timesteps:
            self.w_history.append(self.w.copy())


# --------------------
# Figure 10.1: The Mountain Car task (upper left panel) and the cost-to-go function (− maxa qˆ(s, a, w)) learned during one run.
# --------------------

def get_value_fn_array(agent, x, y):
    """ Populates the value function (= - max_a q(s,a,w) per figure 10.1) at the state (x, xdot) coordinates in coords """
    v = np.zeros((len(x), len(y)))

    for x_coord in range(len(x)):
        for y_coord in range(len(y)):
            v[x_coord, y_coord] = - agent.get_value((x[x_coord], y[y_coord]))

    return v

def plot_value_fn(ax, xx, yy, zz):
    ax.plot_surface(xx, yy, zz, color='white', edgecolors='gray', linewidth=0.25, shade=False)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zlim([0, np.round(np.max(zz),0)])
    ax.set_zticks([0, np.round(np.max(zz),0)])

def fig_10_1():
    mdp = MountainCar()
    agent = NStepSarsaAgent(mdp, n=1, snapshot_timesteps=[428], epsilon=0, alpha=0.3/8)

    episodes_to_plot = [12, 104, 1000, 9000]

    fig = plt.figure(figsize=(12,8))
    res = 50
    x = np.linspace(mdp.x_min, mdp.x_max, res)
    y = np.linspace(mdp.xdot_min, mdp.xdot_max, res)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    for i in tqdm(range(max(episodes_to_plot))):
        agent.run_episode()

        # plot 'step 428' of the first episode
        if i == 0:
            ax = fig.add_subplot(2, 3, 2, projection='3d')
            plot_value_fn(ax, xx, yy, get_value_fn_array(agent, x, y))
            ax.set_title('Step {}'.format(agent.snapshot_timesteps[0]))
            ax.set_xticks([mdp.x_min, mdp.x_max])
            ax.set_yticks([mdp.xdot_min, mdp.xdot_max])

        # plot
        if i+1 in episodes_to_plot:
            ax = fig.add_subplot(2, 3, episodes_to_plot.index(i+1)+3, projection='3d')
            plot_value_fn(ax, xx, yy, get_value_fn_array(agent, x, y))
            ax.set_title('Episode {}'.format(i+1))
            plt.savefig('figures/ch10_fig_10_1.png')

    plt.savefig('figures/ch10_fig_10_1.png')
    plt.close()


# --------------------
# Figure 10.2: Mountain Car learning curves for the semi-gradient Sarsa method with tile-coding function approximation
# and ε-greedy action selection.
# --------------------

def fig_10_2():
    mdp = MountainCar()
    agent = NStepSarsaAgent(mdp, n=1, epsilon=0.1)

    # experiment params
    n_runs = 25
    n_episodes = 500
    alphas = [0.1/agent.num_tilings, 0.2/agent.num_tilings, 0.5/agent.num_tilings]

    # init records
    avg_steps_per_episode = np.zeros((len(alphas), n_runs, n_episodes))

    # run experiment
    for i, alpha in enumerate(alphas):
        # set up agent at this alpha
        agent.alpha = alpha
        agent.reset()

        print('Running experiment {} of {}; n_runs={}'.format(i+1, len(alphas), n_runs))
        for j in range(n_runs):
            # reset weights for each run
            agent.reset()

            for k in tqdm(range(n_episodes)):
                # run episode and record steps
                avg_steps_per_episode[i,j,k] = agent.run_episode()

    avg_steps_per_episode = np.mean(avg_steps_per_episode, axis=1)

    # plot results
    for i, alpha in enumerate(alphas):
        plt.plot(avg_steps_per_episode[i], label=r'$\alpha$ = {}/{}'.format(alpha*agent.num_tilings, agent.num_tilings))
    plt.ylim(100, 1000)
    plt.gca().set_yscale('log')
    plt.xlim(0, n_episodes)
    plt.xlabel('Episode')
    plt.ylabel('Mountain car\n steps per episode\n log scale\n (average over {} runs)'.format(n_runs))

    plt.legend()
    plt.tight_layout()

    plt.savefig('figures/ch10_fig_10_2.png')
    plt.close()


# --------------------
# Figure 10.3: Performance of one-step vs 8-step semi-gradient Sarsa on the Mountain Car task.
# Good step sizes were used: α=0.5/8 for n=1 and α=0.3/8 for n=8.
# --------------------

def fig_10_3():
    mdp = MountainCar()

    n_runs = 25
    n_episodes = 500
    ns = [1, 8]
    alphas = [0.5/8, 0.3/8]


    for i, (n, alpha) in enumerate(zip(ns, alphas)):
        print('Running experiment {} of {}; n={}, n_runs={}'.format(i+1, len(alphas), n, n_runs))

        agent = NStepSarsaAgent(mdp, n=n, alpha=alpha, epsilon=0)

        avg_steps_per_episode = np.zeros(n_episodes)

        for _ in range(n_runs):
            agent.reset()

            for j in tqdm(range(n_episodes)):
                avg_steps_per_episode[j] += agent.run_episode()

        # average
        avg_steps_per_episode /= n_runs

        # plot
        plt.plot(avg_steps_per_episode, label='n={}'.format(n))

    plt.xticks([0, n_episodes])
    plt.ylim(100, 1000)
    plt.gca().set_yscale('log')
    plt.xlabel('Episode')
    plt.ylabel('Mountain car\n steps per episode\n log scale\n (average over {} runs)'.format(n_runs))

    plt.legend()
    plt.tight_layout()

    plt.savefig('figures/ch10_fig_10_3.png')
    plt.close()


# --------------------
# Figure 10.4: Effect of the α and n on early performance of n-step semi-gradient Sarsa and tile-coding function approximation
# on the Mountain Car task. As usual, an intermediate level of bootstrapping (n = 4) performed best. These results are for 
# selected α values, on a log scale, and then connected by straight lines. The standard errors ranged from 0.5 (less than the line width) for n = 1 to about 4 for n = 16, so the main effects are all statistically significant.
# --------------------

def fig_10_4():
    mdp = MountainCar()

    n_runs = 5
    n_episodes = 50
    ns = 2**np.arange(5)
    alpha_range = np.stack([np.linspace(0.2, 1.75 - np.log2(n) / 4, 10) for n in ns], axis=0)
    alpha_range /= 8  # number of tilings

    for n, alphas in zip(ns, alpha_range):
        avg_steps_per_episode = np.zeros(len(alphas))

        for i, alpha in enumerate(alphas):
            print('Running experiment at n={}, alpha={}, n_runs={}'.format(n, alpha, n_runs))

            agent = NStepSarsaAgent(mdp, n=n, alpha=alpha, epsilon=0)

            for _ in range(n_runs):
                agent.reset()

                for _ in tqdm(range(n_episodes)):
                    avg_steps_per_episode[i] += agent.run_episode()

        # average episodes and runs across the alphas
        avg_steps_per_episode /= n_runs * n_episodes

        # plot
        plt.plot(alphas*agent.num_tilings, avg_steps_per_episode, label='n={}'.format(n))
        plt.savefig('figures/ch10_fig_10_4.png')

    plt.xlim(0, 1.75)
    plt.xticks(np.arange(0, 2, 0.5))
    plt.ylim(220, 300)
    plt.xlabel(r'$\alpha$ x number of tilings ({})'.format(agent.num_tilings))
    plt.ylabel('Mountain car\n steps per episode\n log scale\n (average over {} runs)'.format(n_runs))

    plt.legend()
    plt.tight_layout()

    plt.savefig('figures/ch10_fig_10_4.png')
    plt.close()




if __name__ == '__main__':
    np.random.seed(3)
    fig_10_1()
    fig_10_2()
    fig_10_3()
    fig_10_4()
