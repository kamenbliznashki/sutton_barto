import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from agents import BaseAgent, QLearningAgent


# --------------------
# MDP
# --------------------

class MDP:
    """ Branching factor experiment MDP as described in Sec 8.6 """

    def __init__(self, b, num_states=1000, num_actions=2):
        self.b = b  # branching factor
        self.num_states = num_states
        self.num_actions = num_actions

        self.start_state = 0#np.random.randint(num_states - 1)
        self.goal_state = num_states - 1

        # initialize the model
        self.transition_model = {}

        # set up the environment:
        # 1. fix transitions for each state: for each state, two action spossible, each resulting in a
        #       one of b next_states, all equally likely, with a different random selection of b states for each state-action pair
        for s in range(self.num_states-1): # skip the goal_state / terminal_state
            for a in range(self.num_actions):
                next_states = np.random.choice(a=range(self.num_states-1), size=self.b, replace=False)  # sample from all states excl the goal
                next_states = np.append(next_states, self.goal_state)

                transition_probs = np.ones(len(next_states))
                transition_probs[-1] = 0.1  # on all transitions there is a 0.1 probability of transition to the terminal state
                transition_probs[:-1] = 0.9/(len(next_states)-1)

                self.transition_model[(s, a)] = list(zip(next_states, transition_probs))

        # 2. fix rewards for each transition -- map the state-action pair to a reward for each b + 1 (branching + terminal state)
        # get_rewards args are: state, action, next_state
        self.rewards = np.random.normal(loc=0, scale=0.5, size=(self.num_states, self.num_actions, self.b+1))
        self.rewards[..., -1] = 0  # reward for the terminal value is same for all states

    def reset(self):
        self.__init__(self.b, self.num_states, self.num_actions)

    def reset_state(self):
        self.state = self.start_state
        return self.state

    def get_states(self):
        return list(range(self.num_states))

    def get_possible_actions(self, state):
        return list(range(self.num_actions))

    def get_reward(self, state, action, next_state):
        # find 'which' branch the next_state is wrt the available transitions from (state, action) in the transition model
        # and map return to that branch
        next_branch = [s for s, p in self.transition_model[(state, action)]]
        return self.rewards[state, action, next_branch.index(next_state)]

    def get_transition_states_and_probs(self, state, action):
        return list(self.transition_model[(state, action)])

    def get_state_reward_transition(self, state, action):
        # sample the transition model
        rand = np.random.rand()
        cum_prob = 0

        for next_state, prob in self.transition_model[(state, action)]:
            cum_prob += prob
            if rand < cum_prob:
                return next_state, self.get_reward(state, action, next_state)
            if cum_prob > 1:
                raise 'Invalid probability'

    def is_goal(self, state):
        return state == self.goal_state


# --------------------
# Agents and control algorithms
# --------------------

class ExpectedUpdateQLearningAgent(QLearningAgent):
    """
    One-step expected tabular updates agent per Sec 8.6
    Per Sec 8.6 -- On-policy case simulates episodes, all starting at the same place, updating each state-action pair
    that occured under the current e-greedy policy -- ie run q-learning with tabular updates.
    """
    def __init__(self, sampling_freq, **kwargs):
        super().__init__(**kwargs)
        self.sampling_freq = sampling_freq
        self.start_state_values = []

    def update(self, state, action, reward, next_state):
        """ implement eq 8.1 for the expected update """

        # store the start state value for plotting;
        if self.num_updates % self.sampling_freq == 0:
            self.start_state_values.append(self.get_value(self.mdp.start_state))

        # perform update
        new_value = 0

        # calculate expected update prob-weighted by the possible transitions
        for next_state, prob in self.mdp.get_transition_states_and_probs(state, action):
            reward = self.mdp.get_reward(state, action, next_state)
            new_value += prob * (reward + self.discount * self.get_value(next_state))

        # update the q_value
        self.q_values[(state, action)] = new_value

        self.num_updates += 1

        return new_value


def run_uniform_update_episode(mdp, agent):

    for state, action in mdp.transition_model.keys():

        # perform update
        new_value = 0

        for next_state, prob in mdp.get_transition_states_and_probs(state, action):
            reward = mdp.get_reward(state, action, next_state)
            new_value += prob * (reward + agent.discount * agent.get_value(next_state))

        # update the q_value in place
        agent.q_values[(state, action)] = new_value

        agent.num_updates += 1

        # store the start state value for plotting
        if agent.num_updates % agent.sampling_freq == 0:
            agent.start_state_values.append(agent.get_value(mdp.start_state))


class UniformUpdateAgent(BaseAgent):
    """ Uniform sampling agent per Sec 8.6:
    Per Sec 8.6 -- Cycle through all state-action pairs, updating each in place.
    """
    def __init__(self, sampling_freq, run_episode_fn=run_uniform_update_episode, **kwargs):
        super().__init__(run_episode_fn=run_uniform_update_episode, **kwargs)
        self.sampling_freq = sampling_freq
        self.start_state_values = []


# --------------------
# Figure 8.9: Relative efficiency of updates dis- tributed uniformly across the state space versus focused
# on simulated on-policy trajectories, each starting in the same state. Results are for ran- domly generated
# tasks of two sizes and various branching factors, b.
# --------------------

def run_experiment(mdp, agent, n_sample_tasks, n_updates):
    start_state_values = []

    for i in tqdm(range(n_sample_tasks)):
        # reset mdp (randomize state transitions) and agent (q_values and num_updates)
        mdp.reset()
        agent.reset()

        # run agent for the number of updates (updates are recorded at the sampling frequency)
        while agent.num_updates < n_updates:
            agent.run_episode()

        # record only at the sampling frequency
        start_state_values.append(agent.start_state_values[:int(n_updates / agent.sampling_freq)])

    start_state_values = np.vstack(start_state_values)

    return np.mean(start_state_values, axis=0)

def fig_8_9():

    n_sample_tasks = 50
    n_states = np.array([1000, 10000])
    n_updates = np.array([20000 + 1000, 200000 + 1000])  # add some to leave room for the convolution
    b_range = [1, 3, 10]

    # setup sampling freq per num_updates; agent saves a start state value modulo every sampling freq
    # in the uniform case, we're looping over all states-action pairs; so the start state value is only updated
    # when its transition states are, which is 2*b times per cycle since there are 2 possible actions;
    # e.g. 1000 states, 2 actions = 2000 updates for a full cycle / 10 branching factor = 200 updates
    #      10k states, 2 actions = 20k updates / 10 branching = 2k
    #      that is 200 (updates on avg to see change in the children of the start state, thus for change in the start state value.
    sampling_freq = np.array([400, 4000])
    n_samples = ((n_updates - 1000) / sampling_freq).astype(np.int)

    fig, axs = plt.subplots(2,1, figsize=(8,12))
    for i, ax in enumerate(axs.flatten()):
        ax.set_ylim(-0.05, 4.5)
        ax.set_yticks(np.arange(5))
        ax.set_xlim(-0.05, n_samples[i])
        ax.set_xticks(np.linspace(0, n_samples[i], 5))
        ax.set_xticklabels(np.linspace(0, n_samples[i] * sampling_freq[i], 5).astype(np.int))
        ax.set_xlabel('Computation time, in expected updates')
        ax.set_ylabel('Value of start state under greedy policy')


    # Run simulation
    for i, s in enumerate(n_states):
        n_avg = 5  # n-running average to smooth the result

        for b in b_range:
            # only plot 10k states at branching factor 1
            if i > 0 and b > 1:
                continue

            # setup agents and mdp
            mdp = MDP(b=b, num_states=n_states[i])
            q = ExpectedUpdateQLearningAgent(mdp=mdp, epsilon=0.1, discount=1, sampling_freq=sampling_freq[i])
            u = UniformUpdateAgent(mdp=mdp, discount=1, sampling_freq=sampling_freq[i])

            # run experiment and plot
            print('Running experiment: {} states; {} branches'.format(n_states[i], b))
            start_state_values = run_experiment(mdp, q, n_sample_tasks, n_updates[i])
#            start_state_values = np.convolve(start_state_values, np.ones(n_avg), 'full')/n_avg
            axs[i].plot(np.arange(n_samples[i]), start_state_values[:n_samples[i]], label='on-policy, b={}'.format(mdp.b))

            start_state_values = run_experiment(mdp, u, n_sample_tasks, n_updates[i])
            start_state_values = np.convolve(start_state_values, np.ones(n_avg), 'full')/n_avg
            axs[i].plot(np.arange(n_samples[i]), start_state_values[:n_samples[i]], label='uniform, b={}'.format(mdp.b))

            axs[i].legend()

            plt.savefig('figures/ch08_fig_8_9.png')

    plt.tight_layout()
    plt.savefig('figures/ch08_fig_8_9.png')
    plt.close()



if __name__ == '__main__':
    np.random.seed(2)
    fig_8_9()

