import numpy as np
from tabulate import tabulate

from agents import BaseAgent
from ch07_gridworld import Gridworld
from ch12_mountain_car import run_sarsa_lambda_episode


# --------------------
# Agent
# --------------------

class SarsaLambdaAgent(BaseAgent):
    def __init__(self, lam, update_trace_fn, run_episode_fn=run_sarsa_lambda_episode, **kwargs):
        self.lam = lam  # lambda
        self.update_trace = update_trace_fn
        self.possible_actions = kwargs.get('mdp').get_possible_actions(None)
        self.max_steps = float('inf')  # whether to cut agent off a specific number of step per episode
        super().__init__(run_episode_fn=run_sarsa_lambda_episode, **kwargs)

    def reset(self):
        # initialize q_value storage as tabular (ie linear weights) updates
        self.w = np.zeros((self.mdp.width, self.mdp.height, len(self.possible_actions)))
        self.total_rewards = 0

    def get_q_value(self, state, action):
        x, y = state
        action_idx = self.possible_actions.index(action)
        return self.w[x, y, action_idx]

    def update(self, delta, z):
        self.w += self.alpha * delta * z


def update_replacing_traces_fn(agent, z, state, action):
    z[state] = 1
    return z


# --------------------
# Example 12.1: Traces in Gridworld The use of eligibility traces can substantially increase the efficiency of control algorithms
# over one-step methods and even over n-step methods. The reason for this is illustrated by the gridworld example below.
# --------------------

def print_action_value_delta(mdp, agent, f=None):
    # display on a grid
    grid = [[' ' for x in range(mdp.width)] for y in range(mdp.height)]
    for (x,y) in mdp.get_states():  # note higher y is lower in the list, so will need to invert to match the grid coordinates
        v = agent.get_value((x,y))
        grid[y][x] = '{:.3f}'.format(v) if v != 0 else '   '

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

def example_12_1():
    mdp = Gridworld(width=10, height=8, start_state=(1,3), goal_state=(6,3))
    agent = SarsaLambdaAgent(mdp=mdp, update_trace_fn=update_replacing_traces_fn, lam=0.9)

    # run agent
    np.random.seed(371)  # 371, 102, 26 shows a simple/reasonable unoptimized path
    agent.run_episode()

    # record results
    with open('figures/ch12_ex_12_1.txt', 'w') as f:
        with open('figures/ch07_fig_7_4.txt', 'r') as f_old:
            for line in f_old.readlines():
                f.write(line)
        print('\nAction values increased by Sarsa(lambda) with lambda={}:'.format(agent.lam), file=f)
        print_action_value_delta(mdp, agent, f)


if __name__ == '__main__':
    example_12_1()

