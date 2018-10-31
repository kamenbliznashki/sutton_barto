import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

class MDP:
    """ Defines the MDP in Figure 5.4.

    States are 0 (=S) and 1 (=T)
    Actions are 0 (right; to state S=0) and 1 (left; to state T=1)
    Returns are 0 (for action 0 to state 0) and 1 (for action 1 to state 1)

    Thus the possible paths from s given action are:
        for left action:
            S->A->T would be 0+1=1 with prob 0.1
            S->A->S would be 0+0=0 with prob 0.9
        for right action:
            S->A->T would be 0+1=1 with prob 1

    """
    def __init__(self):
        self.total_steps = 0
        self.states_visited = defaultdict(list)
        self.reset_state()

    def reset_state(self):
        self.state = 0
        return self.state

    def get_states(self):
        return [0, 1]

    def get_possible_actions(self):
        return [0, 1]

    def step(self, action):
        """ transition function:
                right action causes a deterministic transition
                left action transitions, with prob 0.9, back to s or, with prob 0.1, on to termination
        """

        if action == 1:  # right action
            next_state = 1
            reward = 1
        else:  # left action
            # with prob 0.1 transition left otherwise return
            rand = np.random.rand()
            if rand <= 0.1:
                next_state = 1
                reward = 1
            else:
                next_state = 0
                reward = 0

        self.state = next_state
        self.total_steps += 1

        # Ch 5, eq 5.5: define the set of all time steps in which state s is visited
        self.states_visited[next_state].append(self.total_steps)

        return next_state, reward



def estimate_v(mdp, n_episodes):
    """ estimate the value function of the mdp for each possible state"""

    states = mdp.get_states()
    actions = mdp.get_possible_actions()
    v = np.zeros((int(n_episodes), len(states)))


    # 1. calculate the importance sampling ratio for a single time step (eq. 5.3)
    #    note: action probabilities are independent of the state so p(a_k|s_k) = p(a_k)
    p_target = np.array([1, 0])          # target policy selects left-right actions with prob 1 and 0
    p_behavior = np.array([0.5, 0.5])   # behavior policy selects left-right actions uniformly
    rho = p_target/p_behavior

    # initialize numerator of eq 5.5 for each state
    _v = np.zeros(len(mdp.get_states()))

    for i in tqdm(range(int(n_episodes))):
        rewards_this_episode = defaultdict(list)
        actions_this_episode = []

        # loop episode
        state = mdp.reset_state()
        while state == 0:  # game ends with transition to terminal state 1
            # get action from behavior policy
            action = actions[np.random.rand() >=0.5]
#            action = np.random.choice(actions, p=p_behavior)  # this is much slower to run
            # transition the mdp with the action selected
            state, reward = mdp.step(action)
            # record
            rewards_this_episode[state].append(reward)
            actions_this_episode.append(action)

        rewards_this_episode = np.array([sum(v) for v in rewards_this_episode.values()])  # undiscounted sum

        # get rho per eq 5.3 the product of importance sampling ratio over the actions|states taken this episode
        rho_t = rho[actions_this_episode]  # ratio at each time step in this episode (probs here are independent of the state)
        rho_this_episode = np.prod(rho_t)  # product across all the time steps in this episode

        # eq 5.5 for this episode:
        # numerator
        _v += rho_this_episode * rewards_this_episode
        # denominator is # times the state was visited
        _d = np.array([len(v) for v in mdp.states_visited.values()])

        # record result
        v[i] = _v / _d

    return v



# --------------------
# Figure 5.4: Ordinary importance sampling produces surprisingly unstable estimates on the one-state MDP
# shown inset (Example 5.5). The correct estimate here is 1 (gamma = 1), and, even though this is the
# expected value of a sample return (after importance sampling), the variance of the samples is infinite,
# and the estimates do not converge to this value. These results are for o↵-policy first-visit MC.
# --------------------

def fig_5_4(n_runs=10):
    for _ in range(n_runs):
        mdp = MDP()
        # running estimate of the value function
        # columns are the states of the mdp and rows the number of episodes
        v = estimate_v(mdp, 1e6)
        plt.semilogx(range(len(v)), v[:,1])  # plot estimate of state 1

    plt.xlabel('Episodes (log scale)', size=8)
    plt.ylabel(r'Monte-Carlo estimate of $v_{{\pi}}(s)$ with ordinary importance sampling ({} runs)'.format(n_runs),
               rotation=0, wrap=True, verticalalignment='center', labelpad=40, fontsize=8)
    plt.gcf().subplots_adjust(left=0.2)

    plt.savefig('figures/ch05_fig_5_4.png')
    plt.close()



if __name__ == '__main__':
    np.random.seed(2)
    fig_5_4()
