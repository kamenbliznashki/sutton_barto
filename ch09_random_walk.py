import numpy as np

# --------------------
# MDP
# --------------------

class RandomWalk:
    """ The 1000-state random walk task for Ch 9;

    States are represented as int in [1-1000] with 0 and 1001 the terminal states.
    Actions are selected from the left and right window of a state based on uniform probability.
    Rewards are -1 on the left side (state 0) and +1 on the right side (state 1001), otherwise 0.
    """
    def __init__(self, num_states=1000, left_window=100, right_window=100):
        self.num_states = num_states
        self.left_window = left_window
        self.right_window = right_window
        self.all_states = np.arange(num_states + 2)  # num_states non-terminal states + 2 terminal states

        # starting state
        self.start_state = (num_states + 2)//2

        # Transition matrix (episodic)
        # State transitions are from the current state to one of the 100 neighboring states to its left,
        # or to one of the 100 neighboring states to its right, all with equal probability.
        # Termination occurs in the left and right ends; after termination, transition is to the start state
        #
        # T_ij is the probability of transition from state i to state j
        T = np.zeros((num_states, num_states + 2))
        i, j = np.indices(T.shape)

        # populate the columns for right and left transition with uniform probability
        u = 1 / (left_window + right_window)  # transition prob
        for k in range(2, 2 + right_window):
            T[i==j-k] = u
        for k in range(left_window):
            T[i-k==j] = u

        # If the current state is near an edge, then there may be fewer than 100 neighbors on that side of it.
        # In this case, all the probability that would have gone into those missing neighbors goes into the
        # probability of terminating on that side (thus, state 1 has a 0.5 chance of terminating on the left,
        # and state 950 has a 0.25 chance of terminating on the right).
        row_sums = np.sum(T, axis=1)
        for k in range(T.shape[0]):
            if T[k,0] != 0:
                T[k,0] += 1 - row_sums[k]
            if T[k,-1] !=0:
                T[k,-1] += 1 - row_sums[k]

        # Add first and last rows for the left and right terminal states;
        # terminal states end the episode and return the process to the start state
        T = np.vstack((np.zeros(T.shape[1]), T, np.zeros(T.shape[1])))
        true_T = T.copy()  # store true returns separately
        T[0, self.start_state] = 1  # in the episodic transitions, terminal states return to the starting state
        T[-1, self.start_state] = 1

        # true returns are maintained separately -- terminal states transition to themselves ie not episodic
        true_T[0,0] = 1
        true_T[-1,-1] = 1

        assert T.shape == (num_states + 2, num_states + 2)  # shape is num_states + 2 terminal states
        assert np.allclose(T.sum(axis=1), np.ones(T.shape[0]))  # each row is a probability distribution

        # the final episodic transition matrix and true returns matrix
        self.T = np.matrix(T)
        self.true_T = np.matrix(true_T)
        del T
        del true_T

        # set up rewards
        self.rewards = np.zeros(num_states + 2)
        self.rewards[0] = -1
        self.rewards[-1] = 1

        self.reset_state()

    def reset_state(self):
        self.state = self.start_state
        self.states_visited = [self.state]
        self.rewards_received = []
        return self.state

    def is_terminal(self, state):
        # the process terminates at the left or right ends
        return (state == self.all_states[0]) or (state == self.all_states[-1])

    def step(self):
        # sample the next_state uniformly
        rand = np.random.rand()
        cum_sum = 0
        # grab the non-zero entries of the transition matrix at the current state - this is the transition distribution
        probs = self.T[self.state, np.where(self.T[self.state]>0)[1]]
        next_state_idxs = np.flatnonzero(self.T[self.state])  # the idx of the possible transitions
        for i, p in enumerate(np.asarray(probs).flatten()):
            cum_sum += p
            if rand < cum_sum:
                next_state = next_state_idxs[i]
                break
            if np.round(cum_sum, 10) > 1:
                raise 'Invalid probability'

        reward = self.rewards[next_state]
        self.rewards_received.append(reward)

        # check if terminal
        if not self.is_terminal(next_state):
            self.state = next_state
            self.states_visited.append(next_state)

        return next_state, reward


