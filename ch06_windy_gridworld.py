import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tabulate import tabulate
from tqdm import tqdm
from collections import deque


from gridworld import BaseGridworld
from agents import BaseAgent

# --------------------
# MDP
# --------------------

class WindyGridworld(BaseGridworld):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # wind directions
        y_wind = np.zeros(self.width)  # wind in y-axis direction, along the x-axis coorinates
        y_wind[3:-1] += 1
        y_wind[6:8] +=1
        x_wind = np.zeros(self.height)  # wind in x-axis direction, along the y-axis coordinates
        self.wind_directions = {(x,y): (x_wind[y], y_wind[x]) for x in range(self.width) for y in range(self.height)}

    def get_state_reward_transition(self, state, action):
        # perform action
        next_state = np.array(state) + np.array(action)
        # clip to grid in case action resulted in off-the-grid state
        next_state = self._clip_state_to_grid(next_state)
        # offset additionally by the direction of the wind
        next_state = self._offset_state_by_wind(next_state)
        # clip to grid in case wind resulted in off-the-grid state
        next_state = self._clip_state_to_grid(next_state)

        # make into tuple of ints
        next_state = int(next_state[0]), int(next_state[1])

        if next_state == self.goal_state:
            reward = 0
        else:
            reward = -1

        return next_state, reward

    def _offset_state_by_wind(self, state):
        x, y = state
        x_, y_ = self.wind_directions[state]
        return (x + x_, y + y_)


class KingsMovesGridworld(WindyGridworld):
    # ex 6.9 -- include 8 actions along diagonals + stay action
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_possible_actions(self, state):
        actions = [(x,y) for x in range(-1,2) for y in range(-1,2)]
        return actions


# --------------------
# Agents and control algorithm
# --------------------

def run_sarsa_episode(mdp, agent):
    """ Execute the Sarse on-policy algorithm in Section 6.4 """
    # record episode path and actions
    states_visited = []
    actions_performed = []
    episode_rewards = 0

    # initialize S
    state = mdp.reset_state()
    states_visited.append(state)

    # choose A from S using policy derived from Q
    action = agent.get_action(state)
    actions_performed.append(action)

    # loop for each step
    while not mdp.is_goal(state):
        # take action A, observe R, S'
        next_state, reward = mdp.get_state_reward_transition(state, action)

        # choose A' from S' using policy derived from Q
        next_action = agent.get_action(next_state)

        # sarsa update
        agent.update(state, action, reward, next_state, next_action)
        state = next_state
        action = next_action

        # record path
        states_visited.append(state)
        actions_performed.append(action)
        episode_rewards += reward

    return states_visited, actions_performed, episode_rewards


class SarsaAgent(BaseAgent):
    """ Section 6.4 Sarsa: on-policy TD control """
    def __init__(self, **kwargs):
        super().__init__(run_episode_fn=run_sarsa_episode, **kwargs)

    def update(self, state, action, reward, next_state, next_action):
        """ Sarsa update to the policy -- eq 6.7 """

        q_t0 = self.get_q_value(state, action)
        q_t1 = self.get_q_value(next_state, next_action)

        # td0 update per eq 6.7
        new_value = q_t0 + self.alpha * (reward + self.discount * q_t1 - q_t0)

        # perform update
        self.q_values[(state, action)] = new_value

        return new_value


class GoRightAgent(SarsaAgent):
    # only used to test correctness of the implementation
    def __init__(self, mdp):
        super().__init__(mdp)

    def get_action(self, state):
        return (1,0)


# --------------------
# Helper functions
# --------------------

def print_optimal_policy(mdp, agent, f=None):
    # compute optimal policy
    agent.epsilon = 0
    states_visited, _, _ = agent.run_episode()

    # display on a grid
    grid = [[' ' for x in range(mdp.width)] for y in range(mdp.height)]
    for (x,y) in states_visited:  # note higher y is lower in the list, so will need to invert to match the grid coordinates
        grid[y][x] = 'o'

    # mark the start state and terminal state
    x, y = mdp.start_state
    grid[y][x] = 'S'
    x, y = mdp.goal_state
    grid[y][x] = 'G'
    for (x,y) in mdp.terminal_states:
        grid[y][x] = 'T'

    # invert vertical coordinate so (0,0) is bottom-left of the grid
    grid = grid[::-1]

    print(tabulate(grid, tablefmt='grid'), file=f)
    return grid


def print_values(mdp, agent, f=None):
    grid = [[' ' for x in range(mdp.width)] for y in range(mdp.height)]
    for (x,y) in mdp.get_states():  # note higher y is lower in the list, so will need to invert to match the grid coordinates
        grid[y][x] = agent.get_value((x,y))

    # invert vertical coordinate so (0,0) is bottom-left of the grid
    grid = grid[::-1]

    print(tabulate(grid, tablefmt='grid', floatfmt='.2f'), file=f)
    return grid


# --------------------
# Figure 6.3: Results of Sarsa applied to a gridworld (shown inset [here printed] in which movement is altered by the
# location-dependent, upward 'wind'. A trajectory under the optimal policy is also shown.
# --------------------

def fig_6_3():
    mdp = WindyGridworld(width=10, height=7, start_state=(0,3), goal_state=(7,3))
    agent = SarsaAgent(mdp=mdp, epsilon=0.1/4)  # e-soft policy is eps / number of actions (sec 5.4)

    time_steps = []  # record time steps per episode for plotting

    n_episodes = 170
    for i in tqdm(range(n_episodes)):
        states, actions, rewards = agent.run_episode()
        time_steps += [len(states)]

    # display policy and q values
    with open('figures/ch06_fig_6_3_policy.txt', 'w') as f:
        print_optimal_policy(mdp, agent, f)
    print('Average episode length %i' %int(np.mean(time_steps[-5:])))

    time_steps = np.cumsum(time_steps)
    plt.plot(time_steps, np.arange(n_episodes))
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')

    plt.savefig('figures/ch06_fig_6_3.png')
    plt.close()


def ex_6_9():
    mdp = KingsMovesGridworld(width=10, height=7, start_state=(0,3), goal_state=(7,3))
    agent = SarsaAgent(mdp=mdp, epsilon=0.1/4)  # e-soft policy is eps / number of actions (sec 5.4)

    time_steps = []  # record time steps per episode for plotting

    n_episodes = 170
    for i in tqdm(range(n_episodes)):
        states, actions, rewards = agent.run_episode()
        time_steps += [len(states)]

    print("Optimal policy path for King's Moves Gridworld:")
    print_optimal_policy(mdp, agent)
    print('Average episode length %i' %int(np.mean(time_steps[-5:])))

    time_steps = np.cumsum(time_steps)
    plt.plot(time_steps, np.arange(n_episodes))
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.show()


if __name__ == '__main__':
    np.random.seed(1)
    fig_6_3()
    ex_6_9()
