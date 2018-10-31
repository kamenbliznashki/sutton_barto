import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import heapq

from gridworld import BaseGridworld, print_grid, print_path
from agents import BaseAgent, QLearningAgent

from ch08_dyna_maze import Gridworld, DynaQAgent


# --------------------
# MDP
# --------------------

class ResGridworld(Gridworld):
    def __init__(self, res, **kwargs):
        # initialize with original arguments; then update based on resolution and manually reinitialize
        super().__init__(**kwargs)

        # compute the relative positions
        orig_w = self.width
        orig_h = self.height
        orig_start_state = self.start_state
        rel_start_state = (orig_start_state[0]/orig_w, orig_start_state[1]/orig_h)

        orig_blocked_states = self.blocked_states
        # reparametrize blocked states from (x,y) coords to (rel_x, rel_y, w, h) where xy are relative to width and height
        rel_blocked_states = [(x/orig_w, y/orig_h, 1, 1) for (x,y) in orig_blocked_states]

        # compute positions at the new resolution
        self.width = res[0] * orig_w
        self.height = res[1] * orig_h
        self.start_state = (int(self.width*rel_start_state[0]), int(self.height*rel_start_state[1]))
        self.goal_state = (self.width - 1, self.height - 1)

        # new blocked states at the (x, y, w, h) parametrization need to be broken down into (x,y) pairs
        blocked_states = [(int(self.width*rel_x), int(self.height*rel_y), res[0]*w, res[1]*h) \
                for (rel_x, rel_y, w, h) in rel_blocked_states]
        self.blocked_states = []
        for (x, y, w, h) in blocked_states:
            for i in range(w):
                for j in range(h):
                    self.blocked_states.append((x+i, y+j))


# --------------------
# Agent and control algorithms
# --------------------

class PrioritizedSweepingAgent(QLearningAgent):
    """ Prioritized sweeping algorithm per Section 8.4

    Proposed upates to q values are kept in a priority queue as a list with python's heapq module.
    Heapq as a min-heap returns the min entry; min comparison for a python tuple compares each element in turn to break ties.

    Here the entry in the PQ is (-abs(proposed_update), -np.sign(proposed_update), (state, action)):
        -- heap index -abs(value) returns the min of the negative absolute updates i.e. the maximum update in abolute value;
        -- heap index -np.sign(value) returns the inverted sign of the actual update; thus heapq breaks ties for the first index
            by returning the min of -sign
            i.e. the entry with with + sign since rewards in this experiment are +1 for goal and 0 otherwise -- that is we'd like
            to prioritize large positive updates to the q values

    E.g.
        proposed_update_1 = +1
        heappush((-abs(1), -sign(1), ...)) pushes entry = (-1, -1, ...)
        proposed_update_2 = -1
        heappush((-abs(-1), -sign(-1), ...)) pushes entry = (-1, 1, ...)
        min heap property returns (-1, -1, ...) first which corresponds to the update proposal +1

    """
    def __init__(self, n_planning_steps, theta, **kwargs):
        super().__init__(**kwargs)
        self.n_planning_steps = n_planning_steps
        self.theta = theta  # the minimum magnitude of q_value update to be performed

    def reset(self):
        super().reset()
        self.model = {}
        self.pq = PriorityQueue()
        self.predecessors = defaultdict(set)

    def _update_predecessors(self, state, action, next_state):
        # add predecessors as a set of (state, action) tuples
        self.predecessors[next_state].add((state, action))

    def update(self, state, action, reward, next_state):
        """ Execute the Q-learning off-policy algorithm in Section 6.5 with
        Prioritized Sweeping model update/planning in Section 8.4 """

        # update model (Sec 8.4 - line (d))
        # model assumes deterministic environment
        self.model[(state, action)] = (reward, next_state)
        # keep track of predecessors for the pq loop below
        self._update_predecessors(state, action, next_state)

        # compute q value proposed update and update priority queue (Sec 8.4 - line (e-f))
        proposed_update = reward + self.discount * self.get_value(next_state) - self.get_q_value(state, action)
        if abs(proposed_update) > self.theta:
            self.pq.push((state,action), -abs(proposed_update))

        # loop over n_planning steps while pq is not empty (Sec 8.4 - line(g)
        for i in range(self.n_planning_steps):
            if self.pq.is_empty():
                break

            # pop best update from queue and transition from model
            state, action = self.pq.pop()
            reward, next_state = self.model[(state, action)]

            # update q values for this state-action pair
            super().update(state, action, reward, next_state)

            # loop for all S', A' predicted to lead to the above state
            for s, a in self.predecessors[state]:
                # get predicted reward from the predecessor leading to `state`
                r, _ = self.model[(s, a)]
                # calculate the proposed update to (s,a)
                proposed_update = r + self.discount * self.get_value(state) - self.get_q_value(s, a)
                # add to priority queue if greater than min threshold
                if abs(proposed_update) > self.theta:
                    self.pq.push((s,a), -abs(proposed_update))


# --------------------
# Helper classes and functions
# --------------------

class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.key_index = {}  # key to index mapping
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        _, _, item = heapq.heappop(self.heap)
        return item

    def is_empty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        for idx, (p, c, i) in enumerate(self.heap):
            if i == item:
                # item already in, so has either lower or higher priority
                # if already in with smaller priority, don't do anything
                if p <= priority:
                    break
                # if already in with larger priority, update the priority and restore min-heap property
                del self.heap[idx]
                self.heap.append((priority, c, i))
                heapq.heapify(self.heap)
                break
            else:
                # item is not in, so just add to priority queue
                self.push(item, priority)


def dijkstra(mdp):
    # init priority queue with problem start state
    init_state = mdp.reset_state()
    pq = PriorityQueue()
    pq.push(init_state, 0)

    # tracker {child: parent, cost}
    path = {init_state: (None, 0)}

    while not pq.is_empty():
        # visit a node
        state  = pq.pop()
        in_cost = path[state][1]

        if mdp.is_goal(state):
            break

        # construct the successors
        actions = mdp.get_possible_actions(state)
        successors = []
        for action in actions:
            next_state, reward = mdp.get_state_reward_transition(state, action)
            # prevent loops
            if next_state == state:
                continue
            successors.append(next_state)

        # relax the successors
        for next_state in successors:
            path_cost = in_cost + 1
            # if never seen record the path with cost
            if next_state not in path:
                path[next_state] = state, path_cost
                pq.push(next_state, path_cost)
            # if visited but the path was longer, then update the pq and the path tracker
            if path_cost < path[next_state][1]:
                pq.update(next_state, path_cost)
                path[next_state] = state, path_cost

    # recontruct shortest path
    # `state` var currently refers to the goal state after the loop above exits
    states_visited = [state]
    while state is not init_state:
        state, _ = path[state]  # grab the parent path[child] = parent
        states_visited.insert(0, state)

    return states_visited


def print_optimal_policy(mdp, agent, f=None):
    # run the agent greedy
    orig_epsilon = agent.epsilon
    agent.epsilon = 0  # set eps to 0 so no exploration and measure length of path

    states_visited, _, _ = agent.run_episode()
    agent.epsilon = orig_epsilon

    print_path(mdp, states_visited, f)
    return len(states_visited)


def run_experiment(mdp, agent, n_runs, tol=0.25):
    shortest_path = dijkstra(mdp)
    shortest_path_len = len(shortest_path)
#    print('Shortest path:')
#    print_path(mdp, shortest_path)
    print('Target shortest path of {}'.format(shortest_path_len))

    num_updates = np.zeros(n_runs)

    for i in range(n_runs):
        # reset agent
        agent.reset()

        while True:
            agent.run_episode()

            path_len = print_optimal_policy(mdp, agent, open(os.devnull, 'w'))
#            print('Current optimal policy path length ', path_len)
            num_updates[i] = agent.num_updates

            if (path_len - shortest_path_len)/path_len < tol:
                break

        print('Number of updates to shortest path: {}'.format(num_updates), end='\r')
    print()

    return np.mean(num_updates)

def test_dijkstra():
    mdp = ResGridworld((2,4))
    print_grid(mdp)
    optimal_path = dijkstra(mdp)
    print_path(mdp, optimal_path)

def test_run_experiment():
    mdp = ResGridworld((1,1))
    print_grid(mdp)
    agent = PrioritizedSweepingAgent(mdp=mdp, n_planning_steps=5, theta=1e-5, alpha=0.6, epsilon=0.15, discount=0.95)

    avg_num_updates = run_experiment(mdp, agent, 5)
    print(avg_num_updates)


# --------------------
# Example 8.4 Prioritized Sweeping on Mazes
# Prioritized sweeping has been found to dramatically increase the speed at which optimal solutions are found in maze tasks,
# often by a factor of 5 to 10. A typical example is shown below.a
# These data are for a sequence of maze tasks of exactly the same structure as the one shown in Figure 8.3,
# except that they vary in the grid resolution. Prioritized sweeping maintained a decisive advantage over unprioritized Dyna-Q.
# Both systems made at most n = 5 updates per environmental interaction.
# --------------------

def ex_8_4():
    # experiment parameters
    n_runs = 5
    grid_resolutions = [(2**x, 2**y) for x in range(3) for y in range(x-1, x+1)][1:]

    # display and plotting labels
    dummy_mdp = ResGridworld((1,1))  # set to initialize the gridworld_size and agents lists and will be overwriten during the experiment
    gridworld_sizes = [(dummy_mdp.width * dummy_mdp.height - len(dummy_mdp.blocked_states))*x*y for x, y in grid_resolutions]
    print('Running grid sizes: ', gridworld_sizes)

    # initialize records
    avg_num_updates_to_best_path = np.zeros((2, len(grid_resolutions)))

    # run experiments
    for j, res in enumerate(grid_resolutions):
        mdp = ResGridworld(res)
        agents = [PrioritizedSweepingAgent(mdp=mdp, n_planning_steps=5, theta=1e-5, alpha=0.6, epsilon=0.15, discount=0.95),
                  DynaQAgent(mdp=mdp, n_planning_steps=5, alpha=0.6, epsilon=0.15, discount=0.95)]
        for i, agent in enumerate(agents):
            print('Running resolution {} (free positions: {}) ...'.format(res, mdp.width*mdp.height - len(mdp.blocked_states)))
            avg_num_updates_to_best_path[i, j] = run_experiment(mdp, agent, n_runs)

    print(avg_num_updates_to_best_path)

    plt.plot(np.arange(len(gridworld_sizes)), avg_num_updates_to_best_path[0], label='Prioritized sweeping')
    plt.plot(np.arange(len(gridworld_sizes)), avg_num_updates_to_best_path[1], label='Dyna-Q')
    plt.xlabel('Gridworld size (# states)')
    plt.xticks(np.arange(len(gridworld_sizes)))
    plt.gca().set_xticklabels(gridworld_sizes)
    plt.ylabel('Updates until optimal solution')
    plt.gca().set_yscale('log')
    plt.ylim(10, 10**6)
    plt.yticks([pow(10,i) for i in range(1,7)])
    plt.legend()

    plt.savefig('figures/ch08_ex_8_4.png')
    plt.close()


if __name__ == '__main__':
    np.random.seed(6)
    ex_8_4()
