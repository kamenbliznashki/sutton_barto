import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

from gridworld import BaseGridworld
from agents import BaseAgent
from ch06_windy_gridworld import SarsaAgent, print_optimal_policy, print_values


# --------------------
# MDP
# --------------------

class Gridworld(BaseGridworld):
    def get_reward(self, state, action, next_state):
        if next_state == self.goal_state:
            reward = 1
        else:
            reward = 0
        return reward


# --------------------
# Agents and control algorithm
# --------------------

def run_nstep_sarsa_episode(mdp, agent):
    """ Execute the Sarse on-policy algorithm Section 7.2 """
    # record episode path and actions
    states_visited = []
    actions_performed = []
    rewards_received = []

    # initialize S
    state = mdp.reset_state()
    states_visited.append(state)

    # choose A from S using policy derived from Q
    action = agent.get_action(state)
    actions_performed.append(action)

    T = float('inf')
    t = 0
    n = agent.n
    gamma = agent.discount
    # loop for each step of episode, t = 0, 1, 2, ...
    while True:
        # if we haven't reached the terminal state, take an action
        if t < T:
            # take action A, observe R, S'
            next_state, reward = mdp.get_state_reward_transition(state, action)
            rewards_received.append(reward)
            if mdp.is_goal(next_state):
                T = t + 1
            else:
                states_visited.append(next_state)
                # choose A' from S' using policy derived from Q
                next_action = agent.get_action(next_state)
                actions_performed.append(next_action)

        # update state estimate at time tau
        tau = t - n + 1
        if tau >= 0:
            G = sum(gamma**(i - tau) * rewards_received[i] for i in range(tau, min(tau + n, T)))
            if tau + n < T:
                state_tpn =  states_visited[tau+n]  # state at time step tau + n
                action_tpn = actions_performed[tau+n]
                G += gamma**n * agent.get_q_value(state_tpn, action_tpn)
            state_tau = states_visited[tau]  # state at time step tau
            action_tau = actions_performed[tau]

            # perform update for the tau timestep
            agent.update(state_tau, action_tau, G)


        # update episode and records
        if t < T:  # we took an action above
            state = next_state
            if not mdp.is_goal(next_state):
                action = next_action

        # episode step
        t += 1
        if tau == T - 1:
            break

    return states_visited, actions_performed, rewards_received


class NStepSarsaAgent(BaseAgent):
    def __init__(self, n, **kwargs):
        super().__init__(run_episode_fn=run_nstep_sarsa_episode, **kwargs)
        self.n = n

    def update(self, state, action, returns):
        """ n-step Sarsa update to the policy -- Section 7.2 """

        self.q_values[(state, action)] += self.alpha * (returns - self.q_values[(state, action)])

        return self.q_values[(state, action)]


# --------------------
# Helper functions
# --------------------

def print_action_value_delta(mdp, agent, f=None):
    # display on a grid
    grid = [[' ' for x in range(mdp.width)] for y in range(mdp.height)]
    for (x,y) in mdp.get_states():  # note higher y is lower in the list, so will need to invert to match the grid coordinates
        v = agent.get_value((x,y))
        grid[y][x] = '{:.1f}'.format(v) if v != 0 else '  '

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


# --------------------
# Figure 7.4: Gridworld example of the speedup of policy learning due to the use of n-step methods.
# The first panel shows the path taken by an agent in a single episode, ending at a location of high reward, marked by the G.
# In this example the values were all initially 0, and all rewards were zero except for a positive reward at G. The arrows in
# the other two panels show which action values were strengthened as a result of this path by one-step and n-step Sarsa methods.
# The one-step method strengthens only the last action of the sequence of actions that led to the high reward, whereas the
# n-step method strengthens the last n actions of the sequence, so that much more is learned from the one episode.
# --------------------

def fig_7_4():
    mdp = Gridworld(width=10, height=8, start_state=(1,3), goal_state=(6,3))
    agents = [NStepSarsaAgent(mdp=mdp, n=1),
              NStepSarsaAgent(mdp=mdp, n=10)]
    random_seed = 371  # shows a simple/reasonable unoptimized path (26, 102, 371)

    with open('figures/ch07_fig_7_4.txt', 'w') as f:
        print('Path taken:\n', file=f)
        np.random.seed(random_seed)
        print_optimal_policy(mdp, agents[0], f)

        for a in agents:
            np.random.seed(random_seed)
            print('\nAction values increased by {}-step Sarsa:'.format(a.n), file=f)
            a.reset()
            _, _, _ = a.run_episode()
            print_action_value_delta(mdp, a, f)


if __name__ == '__main__':
    fig_7_4()




