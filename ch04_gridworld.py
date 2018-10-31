import numpy as np
import tabulate

from gridworld import BaseGridworld, action_to_nwse
from ch03_gridworld import UniformPolicyAgent


class Gridworld(BaseGridworld):
    def get_reward(self, state, action, next_state):
        if state in self.terminal_states:
            return 0
        else:
            return -1


# --------------------
# Figure 4.1: Convergence of iterative policy evaluation on a small gridworld.
# The left column is the sequence of approximations of the state-value function for the random policy
# (all actions equally likely). The right column is the sequence of greedy policies corresponding to
# the value function estimates (arrows are shown for all actions achieving the maximum, and the numbers
# shown are rounded to two significant digits). The last policy is guaranteed only to be an improvement
# over the random policy, but in this case it, and all policies after the third iteration, are optimal.
# --------------------

def fig_4_1():
    mdp = Gridworld(width=4, height=4, terminal_states=[(0,3), (3,0)])

    f = open('figures/ch04_fig_4_1.txt', 'w')

    print('Figure 4.1: Convergence of iterative policy evaluation.', file=f)

    for n_iter in [0, 1, 2, 3, 10, 1000]:
        agent = UniformPolicyAgent(mdp=mdp, discount=1, n_iterations=n_iter)
        print('#'*30, file=f)
        print('##', ' '*10, 'k = {}'.format(n_iter), file=f)

        print('V(k) for the random policy:', file=f)
        print(tabulate.tabulate(np.flipud(agent.values.T), tablefmt='grid'), file=f)  # transform so (0,0) is bottom-left

        grid = [['' for x in range(mdp.width)] for y in range(mdp.height)]
        for (x,y), v in agent.policy.items():
            grid[y][x] = [action_to_nwse(v_i) for v_i in v]
        # invert vertical coordinate so (0,0) is bottom left of the displayed grid
        grid = grid[::-1]

        print('Greedy policy wrt v(k):', file=f)
        print(tabulate.tabulate(grid, tablefmt='grid'), file=f)

    f.close()


if __name__ == '__main__':
    fig_4_1()

