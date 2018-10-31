import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from tqdm import tqdm

from gridworld import BaseGridworld, action_to_nwse, print_grid
from agents import QLearningAgent


# --------------------
# MDP
# --------------------

class Gridworld(BaseGridworld):
    def __init__(self,
            width=9,
            height=6,
            start_state=(0,3),
            goal_state=(8,5),
            blocked_states=[(2,2), (2,3), (2,4), (5,1), (7,3), (7,4), (7,5)]):
        super().__init__(width, height, start_state, goal_state, blocked_states=blocked_states)
        self.time_step = 0

    def step(self):
        self.time_step += 1

    def get_reward(self, state, action, next_state):
        if self.is_goal(next_state):
            return 1
        else:
            return 0

class BlockingMaze(Gridworld):
    def __init__(self, blocked_states_1, blocked_states_2, change_blocked_time_step,
                 width=9, height=6, start_state=(3,0), goal_state=(8,5)):
        super().__init__(width, height, start_state, goal_state)
        self.blocked_states = blocked_states_1
        self.blocked_states_1 = blocked_states_1
        self.blocked_states_2 = blocked_states_2
        self.change_blocked_time_step = change_blocked_time_step
        self.time_step = 0

    def step(self):
        self.time_step += 1
        if self.time_step == self.change_blocked_time_step:
            self.blocked_states = self.blocked_states_2


# --------------------
# Agent and control algorithm
# --------------------

class DynaQAgent(QLearningAgent):
    """ Tabular Dyna-Q algorithm per Section 8.2 """
    def __init__(self, n_planning_steps, **kwargs):
        super().__init__(**kwargs)
        self.n_planning_steps = n_planning_steps

    def reset(self):
        super().reset()
        self.model = {}

    def sample_model(self):
        # sample state
        past_states = [k[0] for k in self.model.keys()]
        sampled_state = past_states[np.random.choice(len(past_states))]
        # sample action, previously taken from the sampled state
        past_actions = [k[1] for k in self.model.keys() if k[0] == sampled_state]
        sampled_action = past_actions[np.random.choice(len(past_actions))]
        # model assumes deterministic environment so no need to sample from the (R,S') pair under model(S,A)
        reward, next_state = self.model[(sampled_state, sampled_action)][1]
        return sampled_state, sampled_action, reward, next_state

    def update(self, state, action, reward, next_state):
        """ Execute the Q-learning off-policy algorithm in Section 6.5 with Dyna-Q model update/planning in Section 8.2 """

        # perform q-learning update (Section 8.2 - Tabular Dyna-Q algorithm line (d))
        super().update(state, action, reward, next_state)  # note this is stepping the num_updates counter

        # update model (Sec 8.2 - Dyna-Q line (e))
        # model assumes deterministic environment
        self.model[(state, action)] = self.num_updates, (reward, next_state)

        # perform planning (Sec 8.2 - Dyna-Q line(f))
        # Loop repeat n times for the n_planning_steps
        for i in range(self.n_planning_steps):
            # sample randomly previously observed state (S) and sample randomly action previously taken at S
            super().update(*self.sample_model())  # update q_values with the planning sample

        self.mdp.step()      # keep track of mdp number of update steps to change the mdp dynamically per example 8.2 blocking maze


class DynaQPlusAgent(DynaQAgent):
    """ Dyna-Q+ algorithm per Section 8.3 and footnote 1 """
    def __init__(self, k, **kwargs):
        super().__init__(**kwargs)
        self.k = k      # scale multiplier tying reward and timesteps: reward + k*sqrt(time delta)

    def sample_model(self):
        # Sec 8.3 + footnote 1:agent changed in the following ways:
        # 1. actions that have never been tried before from a state are allowed to be considered
        # 2. initial model for such actions is that they lead back to the same state with a reward of zero
        # 3. 'bonus reward' for long-untried actions -- planning updates done with new_reward = reward + k * sqrt(time delta)

        # sample a state
        past_states = [k[0] for k in self.model.keys()]
        sampled_state = past_states[np.random.choice(len(past_states))]

        # sample action from all possible action
        # 1. actions that have never been tried before from a state are allowed to be considered
        possible_actions = self.mdp.get_possible_actions(sampled_state)
        past_actions = [k[1] for k in self.model.keys() if k[0] == sampled_state]
        sampled_action = possible_actions[np.random.choice(len(possible_actions))]
        if sampled_action not in past_actions:
            # 2. initial model for such actions is that they lead back to the same state with a reward of zero
            reward = 0
            next_state = sampled_state
            # since this state-action has never been tried, add it to the model
            self.model[(sampled_state, sampled_action)] = self.num_updates, (reward, next_state)
        else:
            # model assumes deterministic environment so no need to sample from the (R,S') pair under model(S,A)
            t_last_update, (reward, next_state) = self.model[(sampled_state, sampled_action)]
            # 3. 'bonus reward' for long-untried actions -- planning updates done with new_reward = reward + k * sqrt(time delta)
            reward += self.k * np.sqrt(self.num_updates - t_last_update)
        return sampled_state, sampled_action, reward, next_state


# --------------------
# Figure 8.3: A simple maze (inset) and the average learning curves for Dyna-Q agents varying in their
# number of planning steps (n) per real step. The task is to travel from S to G as quickly as possible.
# --------------------

def fig_8_3():
    mdp = Gridworld()
    print_grid(mdp)

    n_runs = 30
    n_episodes = 50
    planning_steps = [0, 5, 50]

    agents = [DynaQAgent(mdp=mdp, n_planning_steps=n, alpha=0.1, epsilon=0.1, discount=0.95) for n in planning_steps]

    steps_per_episode = np.zeros((len(agents), n_runs, n_episodes))

    for i, a in enumerate(agents):
        for j in tqdm(range(n_runs)):
            np.random.seed(29)  #29  #47
            a.reset()
            for k in range(n_episodes):
                states, actions, rewards = a.run_episode()
                steps_per_episode[i, j, k] = len(states)

    steps_per_episode = np.mean(steps_per_episode, axis=1)

    for i, a in enumerate(agents):
        plt.plot(np.arange(1, n_episodes), steps_per_episode[i, 1:], label='{} planning steps'.format(a.n_planning_steps))
    plt.xlabel('Episodes')
    plt.ylabel('Steps per episode')
    plt.legend(loc='upper right')

    plt.savefig('figures/ch08_fig_8_3.png')
    plt.close()


# --------------------
# Figure 8.4: Policies found by planning and nonplanning Dyna-Q agents halfway through the second episode.
# The arrows indicate the greedy action in each state; if no arrow is shown for a state, then all of its action values were equal.
# The black square indicates the location of the agent.
# --------------------

def print_policy_delta(mdp, agent, agent_state, f=None):
    # display on a grid
    grid = print_grid(mdp)

    # the mdp keeps numpy indexing so have to flip grid back
    grid = grid[::-1]
    for state in mdp.get_states():  # note higher y is lower in the list, so will need to invert to match the grid coordinates
        x, y = state
        # show the best action for this state
        actions = mdp.get_possible_actions(state)
        q_values = [agent.get_q_value(state, a) for a in actions]
        if np.allclose(q_values, 1e-11):#all([q == max(q_values) for q in q_values]):
            marker = grid[y][x]  # all q values are the same so show blank
        else:
            marker = action_to_nwse(actions[np.argmax(q_values)])  # show the best action
        # update grid with marker
        grid[y][x] = marker

    grid = grid[::-1]

    x, y = agent_state
    grid[y][x] = 'X'

    print(tabulate(grid, tablefmt='grid'), file=f)
    return grid


def fig_8_4():
    mdp = Gridworld()
    agents = [DynaQAgent(mdp=mdp, n_planning_steps=0, alpha=0.1, epsilon=0.1, discount=0.95),
              DynaQAgent(mdp=mdp, n_planning_steps=50, alpha=0.1, epsilon=0.1, discount=0.95)]

    f = open('figures/ch08_fig_8_4.txt', 'w')
    print('Policies found by planning and nonplanning Dyna-Q agents:', file=f)

    for a in agents:
        print('\nWith{} planning (n={})'.format('out' if a.n_planning_steps == 0 else '', a.n_planning_steps), file=f)
        # run two episodes
        np.random.seed(1)
        for i in range(2):
            states, _, _= a.run_episode()
            print('After episode {}'.format(i+1), file=f)
            print_policy_delta(mdp, a, states[len(states)//2], f)

    f.close()


# --------------------
# Figure 8.5: Average performance of Dyna agents on a blocking task. The left environment was used for the first 1000 steps,
# the right environment for the rest. Dyna-Q+ is Dyna-Q with an exploration bonus that encourages exploration.
# --------------------

def run_experiment(mdp, agent, n_runs, n_timesteps):
    cum_rewards = np.zeros((n_runs, n_timesteps))

    for j in tqdm(range(n_runs)):
        # reset environment
        mdp.blocked_states = mdp.blocked_states_1
        mdp.time_step = 0

        # reset agent
        agent.reset()

        # reset counters
        episode_rewards = np.array([0])
        step = 0

        while mdp.time_step < n_timesteps:
            _, _, rewards = agent.run_episode()
            # record cumulative returns by tiling the episode reward across the time steps of the episode
            episode_rewards = np.append(episode_rewards, episode_rewards[-1] + np.tile(rewards, mdp.time_step - step))
#            print('Number of steps this episode: {}'.format(mdp.time_step - step), end='\r')
            step = mdp.time_step
        cum_rewards[j] = episode_rewards[1:n_timesteps+1]


    return np.mean(cum_rewards, axis=0)

def find_hyperparams(n_samples):

    def print_best_rewards(cum_reward_at_hypers):
        # print best rewards
        cum_reward_at_hypers = sorted(cum_reward_at_hypers, key=lambda x: -x[0])
        print('Hyperparams at best cumulative rewards:')
        for l in cum_reward_at_hypers[:5]:
            print(l)
            print()

    mdp = BlockingMaze(
            blocked_states_1=[(x, 2) for x in range(8)],
            blocked_states_2=[(x, 2) for x in range(1,9)],
            change_blocked_time_step = 1000)

    cum_reward_at_hypers = []

    n_runs = 3
    n_timesteps = 3000

    for s in range(n_samples):
        print('Running sample {} of {}'.format(s+1, n_samples))

        # sample hyperparams
        np.random.seed() # set below so will persist here after the first loop and should be reset
        k = np.random.uniform(1e-1, 1e-6)
        n_planning_steps = 5#np.random.randint(5,25)
        alpha = np.random.rand()
        epsilon = np.random.rand()

        # reset agent
        agent = DynaQPlusAgent(mdp=mdp, k=k, n_planning_steps=n_planning_steps, alpha=alpha, epsilon=epsilon, discount=0.95)

        # reset tracker
        cum_rewards = np.zeros(n_timesteps)

        # run experiment
        np.random.seed(2)
        cum_rewards = run_experiment(mdp, agent, n_runs, n_timesteps)

        cum_reward_at_hypers.append([np.max(cum_rewards), k, n_planning_steps, alpha, epsilon])

        plt.plot(np.arange(n_timesteps), cum_rewards, label='Dyna-Q+')
        plt.axvline(mdp.change_blocked_time_step, linestyle='dotted', lw=0.5)
        plt.xlabel('Time steps')
        plt.ylabel('Cumulative reward')
        plt.title('k={:.7f}; n_planning={}, alpha={:.4f}, eps={:.4f}'.format(
            k, n_planning_steps, alpha, epsilon), fontsize=8)
        plt.legend()

        plt.savefig('figures/ch08_fig_8_5_{}.png'.format(s))
        plt.close()

        if s % 10 == 0:
            print('Best rewards thus far: ')
            print_best_rewards(cum_reward_at_hypers)

    print('Best rewards at: ')
    print_best_rewards(cum_reward_at_hypers)


def fig_8_5():
    mdp = BlockingMaze(
            blocked_states_1=[(x, 2) for x in range(8)],
            blocked_states_2=[(x, 2) for x in range(1,9)],
            change_blocked_time_step = 1000
            )

    agents = [DynaQAgent(mdp=mdp, n_planning_steps=10, alpha=0.9, epsilon=0.5, discount=0.95),
              DynaQPlusAgent(mdp=mdp, k=1e-4, n_planning_steps=10, alpha=0.9, epsilon=0.5, discount=0.95)]  # 1e-2, 22, .27, 0.15

    n_runs = 5
    n_timesteps = 3000

    cum_rewards = np.zeros((len(agents), n_timesteps))

    for i, agent in enumerate(agents):
        np.random.seed(2)
        print('Running agent {} of {}'.format(i+1, len(agents)))
        cum_rewards[i] = run_experiment(mdp, agent, n_runs, n_timesteps)

    plt.plot(np.arange(n_timesteps), cum_rewards[0], label='Dyna-Q')
    plt.plot(np.arange(n_timesteps), cum_rewards[1], label='Dyna-Q+')
    plt.axvline(mdp.change_blocked_time_step, linestyle='dotted', lw=0.5)
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative reward')
    plt.legend()

    plt.savefig('figures/ch08_fig_8_5.png')
    plt.close()


# --------------------
# Figure 8.6: Average performance of Dyna agents on a shortcut task. The left environment was used for the first 3000 steps,
# the right environment for the rest.
# --------------------

def fig_8_6():
    mdp = BlockingMaze(
            blocked_states_1=[(x, 2) for x in range(1,9)],
            blocked_states_2=[(x, 2) for x in range(1,8)],
            change_blocked_time_step = 3000)

    agents = [DynaQAgent(mdp=mdp, n_planning_steps=10, alpha=0.8, epsilon=0.5, discount=0.95),
              DynaQPlusAgent(mdp=mdp, k=1e-4, n_planning_steps=10, alpha=0.8, epsilon=0.5, discount=0.95)]

    n_runs = 5
    n_timesteps = 6000  # time steps per run

    cum_rewards = np.zeros((len(agents), n_timesteps))

    for i, agent in enumerate(agents):
        np.random.seed(2)
        print('Running agent {} of {}'.format(i+1, len(agents)))
        cum_rewards[i] = run_experiment(mdp, agent, n_runs, n_timesteps)

    plt.plot(np.arange(n_timesteps), cum_rewards[0], label='Dyna-Q')
    plt.plot(np.arange(n_timesteps), cum_rewards[1], label='Dyna-Q+')
    plt.axvline(mdp.change_blocked_time_step, linestyle='dotted', lw=0.5)
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative reward')
    plt.legend()

    plt.savefig('figures/ch08_fig_8_6.png')
    plt.close()


if __name__ == '__main__':
    fig_8_3()
    fig_8_4()
    fig_8_5()
    fig_8_6()
