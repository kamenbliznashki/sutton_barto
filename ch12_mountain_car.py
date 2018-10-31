import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from agents import BaseAgent
from ch10_mountain_car import MountainCar
from tiles3 import IHT, tiles


# --------------------
# Control algorithms
# --------------------

def run_sarsa_lambda_episode(mdp, agent):
    """ Execute the sarsa(lambda) algorithm of Sec 12.7 """

    # initialize state S
    state = mdp.reset_state()

    # initialize action A under the agent policy
    action = agent.get_action(state)

    # initialize eligibility trace -- type is determined by the update_trace function called during the episode
    z = np.zeros_like(agent.w)

    # record steps
    steps = 0

    # loop for each step of episode:
    while True:
        # take action A and observe R, S'
        next_state, reward = mdp.get_state_reward_transition(state, action)
        agent.total_rewards += reward

        # update delta
        delta = reward

        # update delta and z for the active feature
        delta -= agent.get_q_value(state, action)
        z = agent.update_trace(agent, z, state, action)

        # if S' is terminal update weight and go to next episode
        if mdp.is_goal(next_state):
            agent.update(delta, z)
            break

        # choose A' under the agent policy given next_state
        next_action = agent.get_action(next_state)

        # update delta for the active tiles
        delta += agent.discount * agent.get_q_value(next_state, next_action)

        # update weights
        agent.update(delta, z)
        # decay trace
        z *= agent.discount * agent.lam
        # episode step
        state = next_state
        action = next_action
        steps += 1

        # limit the number of steps for plotting
        # at very high lambdas and very high alphas, number of steps per episode blows up (cf fig 12.10);
        if steps >= agent.max_steps:
            break

    return steps


def run_true_online_sarsa_lambda_episode(mdp, agent):
    """ Execute true online sarsa lambda algorithm of Sec 12.7 """

    # initialize S
    state = mdp.reset_state()

    # chhose A ~ agent policy
    action = agent.get_action(state)

    # grab the active features
    active_tiles = agent.tiles(state, action)

    # initialize eligibility trace
    z = np.zeros_like(agent.w)

    # initialize value tracker
    q_old = 0

    # record steps
    steps = 0

    # loop for each step of episode:
    while not mdp.is_goal(state):
        # take action A, observe R, S'
        next_state, reward = mdp.get_state_reward_transition(state, action)
        agent.total_rewards += reward

        # choose A' ~ agent policy
        next_action = agent.get_action(next_state)

        # store q value of state-action and next_state-next_action
        q = agent.get_q_value(None, None, active_tiles)
        q_next = agent.get_q_value(next_state, next_action)

        # compute the TD error
        delta = reward + agent.discount * q_next - q

        # decay trace
        z_dot_x = np.sum(z[active_tiles])
        z *= agent.discount * agent.lam
        z[active_tiles] += 1 - agent.alpha * agent.discount * agent.lam * z_dot_x

        # update weights
        delta_trace = delta + q - q_old
        delta_active_tiles = q - q_old
        agent.update(delta_trace, delta_active_tiles, z, active_tiles)

        # make a step
        q_old = q_next
        active_tiles = agent.tiles(next_state, next_action)
        state = next_state
        action = next_action
        steps += 1

        # limit the number of steps for plotting
        # at very high lambdas and very high alphas, number of steps per episode blows up (cf fig 12.10);
        if steps >= agent.max_steps:
            break



# --------------------
# Tracing functions updates -- how the eligibility trace function is updated during an episode
# --------------------

def update_replacing_traces_fn(agent, z, state, action):# active_tiles, **kwargs):
    active_tiles = agent.tiles(state, action)
    z[active_tiles] = 1
    return z

def update_accumulating_traces_fn(agent, z, state, action):# active_tiles, **kwargs):
    active_tiles = agent.tiles(state, action)
    z[active_tiles] += 1
    return z

def update_replacing_clearing_traces_fn(agent, z, state, action):#active_tiles, inactive_tiles):
    # cf sec 12.7, figure 12.11 --
    # replacing trace in which, on each time step, the trace for the state and actions not selected were set to 0
    active_tiles = agent.tiles(state, action)
    inactive_tiles = [agent.tiles(state, a) for a in agent.mdp.get_possible_actions(state) if a != action]
    for t in inactive_tiles:
        z[t] = 0
    z[active_tiles] = 1
    return z

def update_dutch_trace_fn(agent, z, state, action):
    active_tiles = agent.tiles(state, action)
    z_dot_x = np.sum(z[active_tiles])
    z[active_tiles] += 1 - agent.alpha * agent.discount * agent.lam * z_dot_x
    return z


# --------------------
# Agents
# --------------------

class TileCodingAgent(BaseAgent):
    """ Increment of the BaseAgent class to use tile coding and linear function approximation. 
    cf http://incompleteideas.net/tiles/tiles3.html for tiles use.
    """
    def __init__(self, num_tilings=8, num_tiles=8, max_size=4096, **kwargs):
        # continuous states of the mdp are converted into discrete by using tiling (Sec 9.5.4)
        # tilings form a grid of num_tilings, each containing num_tiles
        self.max_size = max_size
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles

        super().__init__(**kwargs)


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

        # setup weight vector for linear function approximation
        self.w = np.zeros(self.max_size)
        self.total_rewards = 0

    def get_q_value(self, state, action, active_tiles=None):
        if not active_tiles:
            active_tiles = self.tiles(state, action)
        return np.sum(self.w[active_tiles])


class SarsaLambdaAgent(TileCodingAgent):
    def __init__(self, lam, update_trace_fn, run_episode_fn=run_sarsa_lambda_episode, **kwargs):
        self.lam = lam  # lambda
        self.update_trace = update_trace_fn
        self.max_w = None  # whether the weight vector should be clipped to prevent exploding grads
        self.max_steps = float('inf')  # whether to cut agent off a specific number of step per episode
        super().__init__(run_episode_fn=run_sarsa_lambda_episode, **kwargs)

    def update(self, delta, z):
        self.w += self.alpha * delta * z

        # clip weights for stability (otherwise accumulating traces blows up
        if self.max_w:
            self.w = np.clip(self.w, -self.max_w, +self.max_w)


class TrueOnlineSarsaLambdaAgent(TileCodingAgent):
    def __init__(self, lam, run_episode_fn=run_true_online_sarsa_lambda_episode, **kwargs):
        self.lam = lam  # lambda
        self.max_w = None  # whether the weight vector should be clipped to prevent exploding grads
        self.max_steps = float('inf')  # whether to cut agent off a specific number of step per episode
        super().__init__(run_episode_fn=run_true_online_sarsa_lambda_episode, **kwargs)

    def update(self, delta_trace, delta_active_tiles, z, active_tiles):
        self.w += self.alpha * delta_trace * z
        self.w[active_tiles] -= self.alpha * delta_active_tiles

        # clip weights (otherwise accumulating traces blows up) -- this is only added here for symmetry
        # with the SarsaLambdaAgent so results can be compared
        if self.max_w:
            self.w = np.clip(self.w, -self.max_w, +self.max_w)

# --------------------
# Figure 12.10: Early performance on the Mountain Car task of Sarsa(λ) with replacing traces and n-step Sarsa
# (copied from Figure 10.4) as a function of the step size, α.
# --------------------

def fig_12_10():
    mdp = MountainCar()

    # experiment params
    n_runs = 10
    n_episodes = 50
    lambdas = [0, 0.68, 0.84, 0.92, 0.96, 0.98, 0.99]
    alphas = np.linspace(0.2, 1.75, 10) / 8  # number of tilings

    for lam in tqdm(lambdas, desc='experiments'):
        avg_steps_per_episode = np.zeros(len(alphas))

        with tqdm(total=len(alphas)*n_runs*n_episodes, leave=False) as pbar:

            for i, alpha in enumerate(alphas):
                pbar.set_description('lambda {}; alpha {}; runs/episodes'.format(lam, alpha))

                # run experiment
                agent = SarsaLambdaAgent(mdp=mdp, update_trace_fn=update_replacing_traces_fn, lam=lam, alpha=alpha, epsilon=0)
                for _ in range(n_runs):
                    agent.reset()
                    for _ in range(n_episodes):
                        avg_steps_per_episode[i] += agent.run_episode()
                        pbar.update(1)

        # average episodes and runs across the alphas
        avg_steps_per_episode /= n_runs * n_episodes

        # plot
        plt.plot(alphas*agent.num_tilings, avg_steps_per_episode, label=r'$\lambda$={}'.format(lam))
        plt.savefig('figures/ch12_fig_12_10.png')

    plt.xlim(0, 1.75)
    plt.xticks(np.arange(0, 2, 0.5))
    plt.ylim(170, 300)
    plt.xlabel(r'$\alpha$ x number of tilings ({})'.format(agent.num_tilings))
    plt.ylabel('Mountain car\n steps per episode\n (average over first {} episodes and {} runs)'.format(n_episodes, n_runs))
    plt.title(r'Sarsa($\lambda$) with replacing traces')

    plt.legend()
    plt.tight_layout()

    plt.savefig('figures/ch12_fig_12_10.png')
    plt.close()


# --------------------
# Figure 12.11: Summary comparison of Sarsa(λ) algorithms on the Mountain Car task. True Online Sarsa(λ) performed better
# than regular Sarsa(λ) with both accumulating and replacing traces. Also included is a version of Sarsa(λ) with replacing traces
# in which, on each time step, the traces for the state and the actions not selected were set to zero.
# --------------------

def fig_12_11():
    # experiment params
    n_runs = 25
    n_episodes = 20
    alphas = np.arange(0.2, 2.2, 0.2) / 8  # number of tilings
    lam = 0.80
    epsilon = 0

    mdp = MountainCar()
    agents = {'True online Sarsa(lambda)': TrueOnlineSarsaLambdaAgent(mdp=mdp, lam=lam, epsilon=epsilon),
              'Sarsa(lambda) w/ replacing traces': SarsaLambdaAgent(mdp=mdp, update_trace_fn=update_replacing_traces_fn, lam=lam, epsilon=epsilon),
              'Sarsa(lambda) w/ replacing and clearning traces': SarsaLambdaAgent(mdp=mdp, update_trace_fn=update_replacing_clearing_traces_fn, lam=lam, epsilon=epsilon),
              'Sarsa(lambda) w/ accumulating traces': SarsaLambdaAgent(mdp=mdp, update_trace_fn=update_accumulating_traces_fn, lam=lam, epsilon=epsilon)}

    # stability parameters
    max_steps = 1000  # break after this many steps per episode
    max_w = 50  # clip weights; otherwise accumulating traces overflows
    for a in agents.values():
        a.max_steps = max_steps
        a.max_w = max_w

    for agent_name, agent in tqdm(agents.items(), desc='agents'):

        with tqdm(total=len(alphas) * n_runs * n_episodes, leave=False) as pbar:

            # keep records for plotting
            avg_reward_per_episode = np.zeros(len(alphas))

            for i, alpha in enumerate(alphas):
                # update agent alpha
                agent.alpha = alpha

                # keep progress
                pbar.set_description('agent={}, alpha={:.3f}; runs + episodes'.format(agent_name, alpha))

                # run experiment

                for _ in range(n_runs):
                    agent.reset()

                    for _ in range(n_episodes):
                        agent.run_episode()
                        pbar.update()
                        pbar.set_postfix(w_max = np.max(agent.w))

                    # record avg per episode at the end of this run
                    avg_reward_per_episode[i] += agent.total_rewards / n_episodes
                # avg over runs
                avg_reward_per_episode[i] /= n_runs

        # plot this agent
        plt.plot(alphas*agent.num_tilings, avg_reward_per_episode, label=r'' + agent_name.replace('lambda', '$\lambda$'))
        plt.savefig('figures/ch12_fig_12_11.png')


    plt.ylim(-550, -150)
    plt.xlabel(r'$\alpha$ x number of tilings ({})'.format(agent.num_tilings))
    plt.ylabel('Mountain car\n Rewards per episode\n (average over first {} episodes and {} runs)'.format(n_episodes, n_runs))

    plt.legend()
    plt.tight_layout()

    plt.savefig('figures/ch12_fig_12_11.png')
    plt.close()




if __name__ == '__main__':
    np.random.seed(3)
    fig_12_10()
    fig_12_11()
