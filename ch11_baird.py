import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# --------------------
# MDP
# --------------------

class BairdMDP:
    """
    Baird's counterexample of off-policy divergence -- Section 11.2

    States -- 7 states; represented as int in [0, 6]
    Actions -- 2 actions; dashed action takes the system to one of the six upper states with equal prob;
                          solid action takes the system to the seventh state.
                          (each action here is a transition vector shape (num states,) with the transition probability out of that state,
                            since it is the same out of every state (given the action), we only need a vector and not a matrix)
    Rewards -- 0 on all transitions.
    """
    def __init__(self, n_states=7):
        self.n_states = n_states

        # 2 actions -- dashed and solid represented as transition matrixces; then A @ state = next_state
        self.dashed = np.zeros(n_states)
        self.dashed[:-1] = 1/6  # takes the system to one of the six upper states with equal prob

        self.solid = np.zeros(n_states)
        self.solid[-1] = 1  # takes the system to the seventh state

        self.reset_state()

    def reset_state(self):
        # the starting distribution is uniform (same for all non-terminal states)
        self.state = np.random.randint(low=0, high=self.n_states)
        self.num_steps = 0
        return self.state

    def get_possible_actions(self, state):
        return [self.dashed, self.solid]

    def get_state(self):
        return self.state

    def get_reward(self, state, action, next_state):
        return 0

    def step(self, state, action):
        next_state = np.random.choice(self.n_states, p=action)
        reward = self.get_reward(state, action, next_state)

        # store
        self.state = next_state
        self.num_steps += 1
        return next_state, reward


# --------------------
# Agents and policy evaluation algorithm
# --------------------

class BaseAgent:
    """ Base class for the agents/algorithms in Ch 11.
    Builds the feature representation and target/behavioral policies for the Baird counterexample.
    """
    def __init__(self, mdp, w_dim=8, p_behavior=np.array([6/7, 1/7]), p_target=np.array([0,1]), alpha=0.01, gamma=0.99):
        self.mdp = mdp
        self.n_states = mdp.n_states
        self.w_dim = w_dim
        self.p_behavior = p_behavior
        self.p_target = p_target
        self.alpha = alpha
        self.gamma = gamma

        # build feature representation -- shape (state dim, weight dim)
        # each row corresponds to the state and represents the active features
        self.x = np.zeros((self.n_states, self.w_dim))
        self.x[:-1,:-2] += 2 * np.eye(self.n_states - 1)
        self.x[:,-1] = 1
        self.x[-1,-1] = 2
        self.x[-1,-2] = 1

        self.reset()

    def reset(self):
        # initialize weight vector -- per fig 11.7 initial values
        self.w = np.ones(self.w_dim)
        self.w[-2] = 10

        # track number of updates
        self.num_updates = 0

    def compute_state_value(self, state):
        return self.x[state] @ self.w


class TDAgent(BaseAgent):
    """ semi-gradient off-policy TD(0) agent -- Sec 11.1, eq. 11.2
    This follows the semi-gradient TD(0) algorithm in Sec 9.3 except for adding importance sampling to the weight update
    """
    def __init__(self, mdp, **kwargs):
        super().__init__(mdp, **kwargs)

    def run_episode(self):
        # initialize state
        state = self.mdp.get_state()

        # loop for each step of episode:

        # choose A ~ behavaioral policy
        actions = self.mdp.get_possible_actions(state)
        action_idx = np.random.choice(len(actions), p=self.p_behavior)
        action = actions[action_idx]

        # take action A, observe R, S'
        next_state, reward = self.mdp.step(state, action)

        # update weights
        delta = reward + self.gamma * self.compute_state_value(next_state) - self.compute_state_value(state)  # eq 11.3
        rho = (self.p_target / self.p_behavior)[action_idx]
        self.update(delta, rho, state, next_state)

        # step
        self.num_updates += 1

        return self.w

    def update(self, delta, rho, state, next_state):
        grad_v = self.x[state]

        # perform update per eq 11.2
        self.w += self.alpha * rho * delta * grad_v


class TDCAgent(TDAgent):
    """ gradient corrected TD(0) algorithm -- Sec 11.7
    Gradient is taken wrt PBE, the projected bellman error
    """
    def __init__(self, beta, **kwargs):
        self.beta = beta
        super().__init__(**kwargs)

    def reset(self):
        # add the secondary parameter vector v to the initialization
        super().reset()
        self.v = np.zeros_like(self.w)

    def update(self, delta, rho, state, next_state):
        x_t = self.x[state]         # x at t
        x_tp1 = self.x[next_state]  # x at t+1

        # perform update per section 11.7

        # update v parameter (11.8)
        self.v += self.beta * rho * (delta - self.v @ x_t) * x_t

        # update w parameter (continuing from 11.29 to the 'slightly better algorithm')
        self.w += self.alpha * rho * (delta * x_t - self.gamma * x_tp1 * (x_t @ self.v))
        # alternative update is eq 11.29 + 2; it has much more noise than above
        #self.w += self.alpha * rho * (x_t - self.gamma * x_tp1 * (x_t @ self.v))


class DPAgent(BaseAgent):
    """ semi-gradient expectation-based DP agent -- Sec 11.1, eq. 11.9.
    Expectation is taken of the update target in a DP style - looping over all action and next states under the target policy
    (since we are only interested in off-policy learning of the target policy here)
    """
    def __init__(self, mdp, **kwargs):
        super().__init__(mdp, **kwargs)

    def run_episode(self):
        # init the new weights vector
        new_w = np.zeros_like(self.w)

        # enumerate the update target expectation under eq 4.4, the Bellman eq
        for state in range(self.mdp.n_states):
            # init delta of eq 11.3
            delta = 0

            # compute expected delta over the state distribution under the policy and transition probabilities given the action
            for action_idx, action in enumerate(self.mdp.get_possible_actions(state)):
                # Note: we can use rho at the end or just run DP on the target policy (ie in expectation rho = p_target)
                # build up eq 4.4:
                p_action = self.p_target[action_idx]

                if p_action == 0:  # save calcs since DP adds up p_action * p_transition
                    continue

                for next_state, p_transition in enumerate(action):
                    if p_transition == 0:  # save calcs since DP adds up p_action * p_transition
                        continue

                    # eq 4.4 cont'd
                    reward = self.mdp.get_reward(state, action, next_state)
                    v_next_state = self.compute_state_value(next_state)

                    # the expected td error update -- DP on the Bellman eq, summing over p_action and p_transition in the for loops
                    delta += p_action * p_transition * (reward + self.gamma * v_next_state)

            delta -= self.compute_state_value(state)  # DO expectation-based target in eq 11.9

            # store new weight for this state
            # note: the DP expecatation-based target (delta) given the current state is then averaged over all the 'current' states
            #       that is, divided by number of current states
            grad_v = self.x[state]
            new_w += self.alpha / self.mdp.n_states * delta * grad_v  # semi-gradient update per eq 11.9

        # update weights synchronously
        self.w += new_w

        self.num_updates += 1

        return self.w


class ExpectedTDCAgent(BaseAgent):
    """ gradient corrected expected TD(0) algorithm -- Sec 11.7
    Gradient is taken wrt PBE, the projected bellman error.
    Expectation is taken wrt to the components of PBE (cf sec 11.7) in a DP style; updates are made asynchronously per fig 11.6
    """
    def __init__(self, mdp, beta, **kwargs):
        self.beta = beta
        super().__init__(mdp, **kwargs)

    def reset(self):
        # add the secondary parameter vector v to the initialization
        super().reset()
        self.v = np.zeros_like(self.w)

    def run_episode(self):
        # enumerate the update target expectation under eq 4.4, the Bellman eq
        for state in range(self.mdp.n_states):
            delta = 0
            x_tp1 = 0

            # compute expected delta over the action distributions and transition probabilities (DP)
            for action_idx, action in enumerate(self.mdp.get_possible_actions(state)):
                # Note: we can use rho at the end or just run DP on the target policy (ie in expectation rho = p_target)
                # build up eq 4.4:
                p_action = self.p_target[action_idx]

                if p_action == 0:  # save calcs since DP adds up p_action * p_transition
                    continue

                for next_state, p_transition in enumerate(action):
                    if p_transition == 0:  # save calcs since DP adds up p_action * p_transition
                        continue

                    # eq 4.4 cont'd
                    reward = self.mdp.get_reward(state, action, next_state)
                    v_next_state = self.compute_state_value(next_state)

                    # the expected td error update -- DP on the Bellman eq, summing over p_action and p_transition in the for loops
                    delta += p_action * p_transition * (reward + self.gamma * v_next_state)

                    # the expected next step gradient (first term of the grad_PBE eq)
                    x_tp1 += p_action * p_transition * self.x[next_state]

            delta -= self.compute_state_value(state)  # the DP expectation-based target given current state

            # update primary (value function weights) and secondary (v) parameters asynchronously (per fig 11.6 description)
            # note: the DP expecatation-based target (delta) given the current state is then averaged over all the 'current' states
            #       that is, divided by number of current states (as in eq 11.9)
            # update v -- eq 11.28
            x_t = self.x[state]
            self.v += self.beta / self.mdp.n_states * (delta - np.dot(self.v, x_t)) * x_t
            # update w -- eq 11.29 continued
            self.w += self.alpha / self.mdp.n_states * (delta * x_t - self.gamma * x_tp1 * np.dot(x_t, self.v))

        self.num_updates += 1

        return self.w


class ExpectedEmphaticTDAgent(BaseAgent):
    """
    Emphatic TD algorithm per Sec 11.8 where the parameters are obtained by iteratively computing their expectation.

    Note the M is update differs slightly from book in incorporating a lambda parameter per
    Sutton, True Online Emphatic TD(λ): Quick Reference and Implementation Guide, https://arxiv.org/pdf/1507.07147.pdf
    where M = lam * I + (1-lam) * F_t and F_t = rho * gamma * F_t-1 + I
    """
    def __init__(self, mdp, lam=0.95, **kwargs):
        self.lam = lam  # 
        super().__init__(mdp, **kwargs)

    def reset(self):
        super().reset()
        self.M = 0

    def run_episode(self):
        # iteratively compute the expectation of w -- ie at every step? every episode?
        new_w = np.zeros_like(self.w)
        new_M = 0

        for state in range(self.mdp.n_states):
            delta = 0

            # compute the expectation of delta
            actions = self.mdp.get_possible_actions(state)
            for action_idx, action in enumerate(actions):
                # Note: we can use rho at the end or just run DP on the target policy (ie in expectation rho = p_target)
                # build up eq 4.4
                p_action = self.p_target[action_idx]

                if p_action == 0:  # save calcs since DP adds up p_action * p_transition
                    continue

                for next_state, p_transition in enumerate(action):
                    if p_transition == 0:  # save calcs since DP adds up p_action * p_transition
                        continue

                    # eq 4.4 cont'd
                    reward = self.mdp.get_reward(state, action, next_state)
                    v_next_state = self.compute_state_value(next_state)

                    # the expected td error update -- DP on the Bellman eq, summing over p_action and p_transition in the for loops
                    delta += p_action * p_transition * (reward + self.gamma * v_next_state)

            delta -= self.compute_state_value(state)  # the DP expectation-based target given current state

            # compute rho given this state
            rho = ((self.p_target @ actions) / (self.p_behavior @ actions))[state]

            # update the expected M and weight vectors
            # note: the DP expecatation-based target (delta) given the current state is then averaged over all the 'current' states
            #       that is, divided by number of current states (as in eq 11.9)
            grad_v = self.x[state]
            M = self.gamma * rho * self.M + 1   # note: the emphatic update is gamme_t * rho_t-1 * M_t-1 + I_t (cf Sutton 2015) 
                                                # previous M is the current state of self.M, before the current episode update is performed
                                                # previous rho in expectation = current rho
            # store update to M and weights
            new_w += self.alpha / self.mdp.n_states * M * delta * grad_v  # compute expectation of w over all the states; ie avg the deltas
            new_M += M / self.mdp.n_states  # compute the expectation of M over all the states, ie avg


        # perform the updates in expectation to M and weights vector
        self.M = self.lam * new_M + (1 - self.lam) * 1   # cf Sutton paper section 2, eq 3; there M is a convex combination of I and F, before
                                                         # before used to update the weight vector
        self.w += new_w

        self.num_updates += 1

        return self.w


# --------------------
# Figure 11.2: Demonstration of instability on Baird’s counterexample. Shown are the evolution of the components of the parameter vector w
# of the two semi-gradient algorithms. The step size was α = 0.01, and the initial weights were w = (1,1,1,1,1,1,10,1)⊤.
# --------------------

def fig_11_2():
    mdp = BairdMDP()
    agents = [TDAgent(mdp), DPAgent(mdp)]

    # experiment params
    n_steps = 1000

    fig, axs = plt.subplots(1, 2, figsize=(10,6), sharex=True, sharey=True)

    for agent, ax in zip(agents, axs.flatten()):
        # run agent
        w_history = np.zeros((agent.w_dim, n_steps))
        for i in tqdm(range(n_steps)):
            w_history[:,i] = agent.run_episode()

        # plot weights components
        for i in range(len(w_history)):
            ax.plot(w_history[i], label=r'$w_{}$'.format(i+1))
        ax.set_xticks([0, n_steps])
        ax.set_yticks([1, 10, 100, 200, 300])
        ax.legend()

    axs[0].set_title('Semi-gradient off-policy TD')
    axs[0].set_xlabel('Steps')
    axs[1].set_title('Semi-gradient DP')
    axs[1].set_xlabel('Sweeps')

    plt.tight_layout()
    plt.savefig('figures/ch10_fig_11_2.png')
    plt.close()


# --------------------
# Figure 11.6: The behavior of the TDC algorithm on Baird’s counterexample.
# On the left is shown a typical sin- gle run, and on the right is shown the expected behavior of this algorithm
# if the updates are done synchronously (analogous to (11.9), except for the two TDC parameter vectors).
# The step sizes were α = 0.005 and β = 0.05.
# --------------------

def compute_ve(mdp, agent, w_history):
    """ Compute the mean square error for an agent given a history of weight vectors (eq 11.11) """
    # VE = sum_s[ mu(s) * (v_w - v_pi])**2 ] per eq 9.1;
    # note: v_pi = 0 (the true value function) since all rewards are 0
    #       mu represents the on-policy distribution.
    # below the on-policy distribution is the dot product of the action probs with the transition probs under those action
    # each action from mdp.get_possible_action() returns the transition probs for that action
    mu = np.dot(agent.p_behavior, mdp.get_possible_actions(None))  # the on-policy distribution
    VE = mu @ (agent.x @ w_history)**2  # eq 9.2 / 11.11 summed over all states
    return np.sqrt(VE)

def compute_pbe(mdp, agent, w_history):
    """ Compute the Projected Bellman Error for an agent given a history of weight vectors (eq 11.11) """
    # PBE = norm(P @ delta_bar_w)**2 per eq 11.14 / sec 11.7
    # delta_bar_w = B_pi(v_w) - v_w  per eq 11.20 where B_pi is the Bellman operator and v_w is the projected value function v_w = X @ w
    # note the bellman operator eq is implified below since:
    #   1. rewards here are 0, so the bellman operator is just a discounted, prob weighted v_w
    #   2. target policy selects solid action with prob 1
    #   3. solid action transition probs select state 7 with prob 1
    #   thus the bellman operator reduces to a scalar of gamma * v(s'=7)

    # prelim
    actions = mdp.get_possible_actions(None)
    mu = np.dot(agent.p_behavior, actions)  # the on-policy distribution

    # calculate delta_bar
    v_w = agent.x @ w_history
    B_pi = np.dot(agent.p_target, actions) # Bellman operator at the target distribution
    delta_bar = agent.gamma * B_pi @ v_w - v_w
    # calculate the projection matrix
    X = agent.x
    D = np.diag(mu)
    P = X @ np.linalg.inv(X.T @ D @ X) @ X.T @ D
    # project delta_bar
    delta_bar_proj = P @ delta_bar
    # calculate the norm of the projected delta_bar under the probability metric of the function space (11.11)
    PBE = mu @ delta_bar_proj**2
    return np.sqrt(PBE)


def fig_11_6():
    mdp = BairdMDP()
    agents = [TDCAgent(mdp=mdp, alpha=0.005, beta=0.05), ExpectedTDCAgent(mdp=mdp, alpha=0.005, beta=0.05)]

    fig = plt.figure(figsize=(8,6))

    # experiment params
    n_steps = 1000

    fig, axs = plt.subplots(1, 2, figsize=(10,6), sharex=True, sharey=True)

    for agent, ax in zip(agents, axs.flatten()):
        # reset mdp and records
        mdp.reset_state()
        w_history = np.zeros((agent.w_dim, n_steps))

        # run agent
        for i in tqdm(range(n_steps)):
            w_history[:,i] = agent.run_episode()

        # plot weights components
        for i in range(len(w_history)):
            ax.plot(w_history[i], label=r'$w_{}$'.format(i+1))

        # plot VE
        ax.plot(compute_ve(mdp, agent, w_history), label=r'$\sqrt{\overline{VE}}$')

        # plot PBE
        ax.plot(compute_pbe(mdp, agent, w_history), label=r'$\sqrt{\overline{PBE}}$')

        ax.axhline(0, linestyle='--', lw=0.5, c='black', alpha=0.4)
        ax.set_xlabel('Steps')
        ax.set_xticks([0, n_steps])
        ax.set_ylim(-3, 10)
        ax.set_yticks([-3, 0, 2, 5, 10])
        ax.legend()

    axs[0].set_title('TDC')
    axs[0].set_xlabel('Steps')
    axs[1].set_title('Expected TDC')
    axs[1].set_xlabel('Sweeps')


    plt.tight_layout()
    plt.savefig('figures/ch10_fig_11_6.png')
    plt.close()


# --------------------
# Figure 11.7: The behavior of the one-step emphatic-TD algorithm in expectation on Baird’s counterexample.
# The step size was α = 0.03.
# --------------------

def fig_11_7():
    mdp = BairdMDP()
    agent = ExpectedEmphaticTDAgent(mdp, alpha=0.03)

    # experiment params
    n_steps = 1000
    w_history = np.zeros((agent.w_dim, n_steps))

    # run agent
    for i in tqdm(range(n_steps)):
        w_history[:,i] = agent.run_episode()

    # plot weights components
    for i in range(len(w_history)):
        plt.plot(w_history[i], label=r'$w_{}$'.format(i+1))

    # plot VE
    plt.plot(compute_ve(mdp, agent, w_history), label=r'$\sqrt{\overline{VE}}$')

    plt.gca().axhline(0, linestyle='--', lw=0.5, c='black', alpha=0.4)
    plt.title('Emphatic-TD')
    plt.xlabel('Sweeps')
    plt.xticks([0, n_steps])
    plt.ylim(-7, 14)
    plt.yticks([-5, 0, 2, 5, 10])

    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.savefig('figures/ch10_fig_11_7.png')
    plt.close()


if __name__ == '__main__':
    np.random.seed(1)
    fig_11_2()
    fig_11_6()
    fig_11_7()
