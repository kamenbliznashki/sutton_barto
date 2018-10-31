import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# --------------------
# MDP
# --------------------

class CorridorGridworld:
    """ Defines the short corridor Gridworld MDP in Example 13.1

    State representation -- int in [0, 3] where 3 is the terminal state.
    Action representation -- np array of the feature vector x(s, right) = [1, 0]; x(s, left) = [0, 1] for all s
    """
    def __init__(self, length=4, start_state=0, goal_state=3):
        self.states = np.arange(length)
        self.start_state = start_state
        self.goal_state = goal_state

        # transition probabilities for each state given an left action [0,1] and a right action [1,0]
        # e.g.  s_0 @ right_action = 1; s_0 @ left_action = 0
        #       s_1 @ right_action = 0; s_1 @ left_action = 2
        #       s_2 @ right_action = 3; s_2 @ left_action = 1
        self.T = np.array([[1,0], [0,2], [3,1], [3,3]])

    def get_possible_actions(self, state):
        # actions are x(s, left) = [0, 1].T and x(s, right) = [1,0].T for all s
        # return left, right action as column vectors in a matrix of available actions
        return np.array([[0, 1],
                         [1, 0]])

    def get_reward(self, *args):
        return -1

    def get_state_reward_transition(self, state, action):
        next_state = self.T[state] @ action.flatten()
        reward = self.get_reward()
        return next_state, reward

    def is_goal(self, state):
        return state == self.goal_state

    def reset_state(self):
        self.state = self.start_state
        return self.state


# --------------------
# REINFORCE algorithm
# --------------------

def reinforce(mdp, agent, n_episodes, max_steps=100):
    """ Execute the REINFORCE monte-carlo policy gradient episodic algorithm -- Sec 13.3 """

    # record all rewards
    total_rewards = np.zeros(n_episodes)

    with tqdm(total=n_episodes, leave=False) as pbar:#, postfix=['policy', dict(policy=[])]) as pbar:
        for i in range(n_episodes):
            state = mdp.reset_state()

            # initialize trackers
            states_history = [state]
            actions_history = []
            rewards_history = []

            # generate an episode S0, A0, R1, ... following π(.|., theta)
            while not mdp.is_goal(state):
                # select action under the policy
                action, action_idx = agent.get_action(state)

                # execute action and observe next_state and reward
                state, reward = mdp.get_state_reward_transition(state, action)

                # update records
                actions_history.append((action, action_idx))
                states_history.append(state)
                rewards_history.append(reward)
                total_rewards[i] += reward

                # break if episode goes for too long
                if len(states_history) > max_steps:
                    break

            # loop for each step of the episode
            for t in range(len(states_history[:-1])):  # the final recorded state is the goal state, so exclude that for the grad updates
                state = states_history[t]
                action, action_idx = actions_history[t]

                # compute return from step t
                # future returns from step t onward, discounted to step 5
                G = agent.gamma**np.arange(len(rewards_history[t:])) * rewards_history[t:]
                # discount from step t to start of the episode
                G = agent.gamma**t * np.sum(G)

                # compute gradient -- eq 13.7
                all_features = agent.extract_features(state, mdp.get_possible_actions(state))
                policy = agent.get_policy(state)
                # 1. current feature -- x(s,a)
                current_feature = all_features[:, action_idx]
                # 3. compute gradient
                gradient = current_feature - np.dot(all_features, policy)

                # make policy parameter update
                agent.update_policy(G, gradient)

                pbar.set_postfix(policy=np.round(policy, 3))
            pbar.update()

    return total_rewards



# --------------------
# Agents
# --------------------

class LinearPolicyAgent:
    """ Agent that follows a differentiable parametrized policy -- exponential softmax policy (13.2) with linear action preferences (13.3). """

    def __init__(self, mdp, features_dim, alpha, gamma=1, min_prob_thresh=0.01):
        self.mdp = mdp
        self.alpha = alpha
        self.gamma = gamma

        # threshold of min probability under the policy,
        # the lowest the policy is allowed to go, else the gradient pushes it to become deterministic
        # specifically: a policy gradient update at this treshold should not push the policy into deterministic
        #               if max_steps per episode = 100; then total reward for the episode = -100 (gamm=1); then:
        #               under REINFORCE with alpha = 2**-12, the max gradient step of alpha * delta * grad = 2**-12 * -100 * 1 = 0.0244
        #               under REINFORCE + baseline with alpha = 2**-9, the max gradient step is: 2**-9 * -100 * 1 (assume w=0) = 0.195
        #
        self.min_prob_thresh = min_prob_thresh

        # setup feature vector extractor; in example 13.1 feature vectors are x(s,right) = [1,0]; x(s,left) = [0,1] for all states
        self.features_dim = features_dim  # dim of the feature vectors
        self.extract_features = lambda state, action: action

        self.reset()

    def reset(self):
        # init policy parameter under linear action preferences
        # specifically init at a e-greedy left point of example 13.1 chart to see policy optimization
        # cf example 13.1 -- e = 0.1 and choosing left with prob 0.1/2 result in v0 = -82 (value of state 0)
        #                    want a policy that starts at left prob = 0.1/2 = np.exp(theta @ x) ie theta @ x = np.log(0.1/2)
        #                    then the probs for left-right actions are [0.95 0.05]
        self.theta = np.array([np.log(0.1/2), np.log(1-0.1/2)])
        assert np.allclose(softmax(self.theta @ np.array([[0, 1], [1,0]])), np.array([0.95, 0.05]))

        self.num_updates = 0

    def get_policy(self, state):
        # get all actions
        actions = self.mdp.get_possible_actions(state)
        features = self.extract_features(state, actions)
        # compute action preferences -- eq 13.3
        h = features @ self.theta
        # compute softmax policy -- eq 13.2
        policy = softmax(h)
        return policy

    def get_action(self, state):
        actions = self.mdp.get_possible_actions(state)
        policy = self.get_policy(state)
        action_idx = np.random.choice(len(actions), p=policy)
        return actions[:, action_idx], action_idx

    def update_policy(self, discounted_delta, eligibility_vector):
        self.theta += self.alpha * discounted_delta * eligibility_vector
        self._check_theta()

    def _check_theta(self):
        # prevent theta from becoming deterministic
        if np.min(self.get_policy(None)) < self.min_prob_thresh:   # when min prob falls below threshold, push it back to treshold
            self.theta[np.argmax(self.theta)] = np.log(self.min_prob_thresh)      # set argmax theta to the low log prob
            self.theta[np.argmin(self.theta)] = np.log(1 - self.min_prob_thresh)


class BaselineLinearPolicyAgent(LinearPolicyAgent):
    """ Policy agent based on REINFORCE with baseline per Sec 13.4 """
    def __init__(self, beta, **kwargs):
        self.beta = beta
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self.w = 0

    def update_policy(self, discounted_delta, eligibility_vector):
        self.w += self.beta * (discounted_delta - self.w)
        self.theta += self.alpha * (discounted_delta - self.w) * eligibility_vector


def softmax(x):
    e_x = np.exp(x + np.max(x) + 1e-6)
    return e_x / np.sum(e_x)


# --------------------
# Example 13.1 Short corridor with switched actions
# --------------------

def example_13_1():
    """
    Example 13.1 - short corridor with switched actions

    Compute the value of state 0 under policy pi, that is v_pi(s_0).
    Use Bellman equation to solve for v(s0) analytically
        v(s0) = ∑π(action|state) * ∑p(s',r|s,a)*(r + v(s'))  assuming no discounting ie gamma=1;
        transitions and rewards p(s',r|s,a) are deterministic given the state and action as described in the example.
        v0 = v(s0) = (2*p - 4) / (p * (1-p)) where p = π(action=right|s)
        argmax(v0) is at dv0/dp = 0, that is at 2 ± sqrt(2); p is a probability so argmax(v0) is at p = 2 - sqrt(2)
    """

    # true value of state 0 given by the Bellman eq, where p is the prob of right action
    v0 = lambda p: (2*p - 4) / (p * (1 -p))
    argmax_v0 = 2 - np.sqrt(2)

    # evaluate v0 at p
    p = np.linspace(0.005, 0.995, 50)

    # annotate max v0 and e-greedy left-right policy values
    annot_pts = np.array([[argmax_v0, v0(argmax_v0)],
                          [0.1/2, v0(0.1/2)],
                          [1-0.1/2, v0(1-0.1/2)]])
    # plot results
    plt.plot(p, v0(p))
    plt.scatter(annot_pts[:,0], annot_pts[:,1])
    plt.annotate('Optimal\nstochastic\npolicy', annot_pts[0] + (0,-13), ha='center')
    plt.annotate(r'$\epsilon$-greedy left', annot_pts[1] + 0.03)
    plt.annotate(r'$\epsilon$-greedy right', annot_pts[2] - 0.03, ha='right')

    plt.xlabel('Probability of right action')
    plt.ylabel(r'J($\theta$) = $v_{\pi_{\theta}}$(S)')
    plt.ylim(-100, -10)
    plt.yticks(list(np.linspace(-100,-20,5).astype(np.int)) + [np.round(v0(argmax_v0),1)])

    plt.tight_layout()
    plt.savefig('figures/ch13_ex_13_1.png')
    plt.close()


# --------------------
# Figure 13.1: REINFORCE on the short-corridor gridworld (Example 13.1).
# With a good step size, the total reward per episode approaches the optimal value of the start state.
# --------------------

def fig_13_1():
    mdp = CorridorGridworld()

    # true optimum
    v0 = lambda p: (2*p - 4) / (p * (1 -p))
    argmax_v0 = 2 - np.sqrt(2)


    # experiment params
    n_runs = 50
    n_episodes = 1000
    alphas = [2**-12, 2**-13, 2**-14]

    avg_reward_per_episode = np.zeros((len(alphas), n_runs, n_episodes))

    for i, alpha in enumerate(alphas):
        agent = LinearPolicyAgent(mdp, len(mdp.get_possible_actions(None)[0]), alpha)

        for j in tqdm(range(n_runs), desc='n_runs'):
            agent.reset()
            avg_reward_per_episode[i, j] = reinforce(mdp, agent, n_episodes)

    avg_reward_per_episode = np.mean(avg_reward_per_episode, axis=1)

    # plot results
    for i, alpha in enumerate(alphas):
        plt.plot(np.arange(n_episodes), avg_reward_per_episode[i], label=r'$\alpha=2^{{{:.0f}}}$'.format(np.log2(alpha)))
    plt.gca().axhline(v0(argmax_v0), color='silver', linestyle='dashed', lw=0.5, label=r'$v_* (s_0)$')
    plt.xlabel('Episode')
    plt.ylabel(r'Total reward per episode $G_{0}$')

    plt.legend()
    plt.tight_layout()

    plt.savefig('figures/ch13_fig_13_1.png')
    plt.close()


# --------------------
# Figure 13.2: Adding a baseline to REINFORCE can make it learn much faster, as illustrated here on the short-corridor gridworld
# (Example 13.1). The step size used here for plain REINFORCE is that at which it performs best (to the nearest power of two;
# see Figure 13.1). Each line is an average over 100 independent runs.
# --------------------

def fig_13_2():
    mdp = CorridorGridworld()

    # true optimum
    v0 = lambda p: (2*p - 4) / (p * (1 -p))
    argmax_v0 = 2 - np.sqrt(2)


    # experiment params
    n_runs = 50
    n_episodes = 1000
    agents = {'REINFORCE with baseline': BaselineLinearPolicyAgent(mdp=mdp,
                                                                   features_dim=len(mdp.get_possible_actions(None)[0]),
                                                                   alpha=2**-9,
                                                                   beta=2**-6),
              'REINFORCE': LinearPolicyAgent(mdp=mdp,
                                             features_dim=len(mdp.get_possible_actions(None)[0]),
                                             alpha=2**-13)}


    avg_reward_per_episode = np.zeros(n_episodes)

    for agent_name, agent in agents.items():
        for j in tqdm(range(n_runs), desc='n_runs'):
            agent.reset()
            avg_reward_per_episode += reinforce(mdp, agent, n_episodes)

        # avg over runs
        avg_reward_per_episode /= n_runs

        # plot
        label = agent_name + r' $\alpha = 2^{{{:.0f}}}$'.format(np.log2(agent.alpha))
        if agent.__dict__.get('beta', None):
            label += r' $\beta = 2^{{{:.0f}}}$'.format(np.log2(agent.beta))
        plt.plot(np.arange(n_episodes), avg_reward_per_episode, label=label)

    plt.gca().axhline(v0(argmax_v0), color='silver', linestyle='dashed', lw=0.5, label=r'$v_* (s_0)$')
    plt.xlabel('Episode')
    plt.ylabel(r'Total reward per episode $G_{0}$')

    plt.legend()
    plt.tight_layout()

    plt.savefig('figures/ch13_fig_13_2.png')
    plt.close()



if __name__ == '__main__':
    np.random.seed(5)
    example_13_1()
    fig_13_1()
    fig_13_2()

