import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
from collections import defaultdict
import itertools
from copy import deepcopy
from tqdm import tqdm


class BlackjackEnvironment:
    """ Class representing the rules of the game and the dynamics.

    Representations:

    state = np array of shape (# players, 14) where
            row 0 = dealer; row 1:n = players
            col 0 = dealer shown card
            col 1:13 = one hot vector for the hand held by the agent

            note: the one-hot representation is against card values from the self.card_deck [1, 2, ..., 10]
                  where an ace is 1 and face cards are 10

    actions are: 1 = stick; 2 = hit

    reward = np array of shape (# players,) where
            pos 0 = dealer, pos 1 = player, ...

            note: reward is sent at the end of each loop and separately at the end of the episode.
            note: reward is sent to agents as a scalar.

    """
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.card_deck = np.clip(np.arange(1,14), 0, 10)
        self.reset()

    def get_state_value(self, state):
        value = state[:,1:].dot(self.card_deck)  # exclude pos 0 for the dealer shown card
        for i, s in enumerate(state):
            # check for a 'usable' ace, i.e. one that can be counted as 11 (at most 1 usable)
            if s[1] != 0:  # aces are in pos 1
                # if aces, check if can be counted as 11 (it is already counted as 1 so increment another 10)
                if value[i] + 10 <= 21:
                    value[i] += 10
        return value

    def get_start_state(self):
        # sample 2 cards for each of the agents (per example 5.1)
        state = []
        for _ in range(self.n_agents):
            state.append(self.sample_cards(2))

        # show one of the dealer's cards at random
        # dealer is state pos 0; the shown card is in pos 0 for the dealer
        dealer = state[0]
        shown = np.random.choice(np.where(dealer != 0)[0])
        dealer[0] = shown

        # set the initial state and make all agents active
        self.state = np.vstack(state)
        self.active_agents = np.ones(self.n_agents)
        return self.state

    def get_current_state(self):
        return self.state

    def get_possible_actions(self, agent):
        return [1, 2]  # stick or hit

    def do_action(self, action, agent):
        """ execute an action for a specific agent """
        state = self.get_current_state()
        self.state = self.get_random_next_state(state, action, agent)

        # update active agents
        # if agent sticks (action 1) or busts (state val > 21) or wins (state val = 21), he is no longer active
        if action == 1 or self.get_state_value(self.state)[agent] > 21:
            self.active_agents[agent] = 0

        # update reward based on new state and active agents
        reward = self.get_reward(self.state, action, agent)

        return self.state, reward

    def get_random_next_state(self, state, action, agent):
        # check if legal action
        if not self.is_agent_active(agent):
            raise 'Agent not active. Cannot perform action {}'.format(action[np.where(action != 0)])

        # if action is hit draw a card from the deck at random for this agent
        if np.sum(action) == 2:
            sample = self.sample_cards(1)
            state[agent] += sample
        # else the state remains unchanged

        return state

    def get_reward(self, state, action, agent):
        """ calculate reward across all agents but return only for agent in the called function """
        reward = np.zeros(self.n_agents)

        # distribute final rewards after either everyone is bust or no active agents remain (they all stick)
        if self.is_terminal(state) or np.all(self.active_agents == 0):
            state_value = self.get_state_value(state)
            d = state_value[0]   # dealer is always index 0

            if d == 21:
                # everyone that doesn't match the dealer loses
                reward[np.where(state_value != d)] = -1
                # everyone that matches the dealer draws
                reward[np.where(state_value == d)] = 0
                # dealer wins only if just the dealer has 21, else draws
                if len(np.where(state_value == d)[0]) == 1:
                    reward[0] = 1
            if d > 21:
                # everyone that isn't bust wins
                reward[np.where(state_value <= 21)] = 1
                # everyone that is bust loses
                reward[np.where(state_value > 21)] = -1
                # dealer loses
                reward[0] = -1
            if d < 21:
                # everyone that has strictly more than the dealer wins (unless they bust, handled below)
                reward[np.where(state_value > d)] = 1
                # everyone that is bust loses
                reward[np.where(state_value > 21)] = -1
                # everyone that has strictly less then the dealer loses
                reward[np.where(state_value < d)] = -1
                # dealer wins only if no one else has already won else ties or loses
                if len(np.where(reward == 1)[0]) > 0:  # someone else won
                    reward[0] = -1
                elif len(np.where(reward == 0)[0]) == 1:  # only dealer hasn't been update; so wins (state where dealer loses and someone else wins is cleared above)
                    reward[0] = 1
                # remaining players == dealer stay at reward 0

            # record episode reward to distribute to all agents
            self.episode_reward = reward

        return reward[agent]

    def get_final_reward(self, agent):
        """ return the end-of-episode reward to the specific agent
        (players have to wait for dealer to play to determine episode reward """
        return self.episode_reward[agent]

    def sample_cards(self, size):
        """ cards drawn with replacement; game rule is 6 decks x 52 cards; so max numbers
        a certain card can be drawn is 6*4, which is very low prob so no need to check for this and resample """
        sample = np.zeros(1 + len(self.card_deck))  # leave pos 0 for dealer shown card
        for _ in range(size):
            drawn = np.random.choice(self.card_deck)
            sample[drawn] += 1
        return sample

    def reset(self):
        self.get_start_state()

    def is_terminal(self, state):
        # if no active agents remain (triggered when agent sticks or is busted)
        return np.all(self.get_state_value(state) >= 21)

    def is_agent_active(self, agent):
        return self.active_agents[agent] == 1


class MCAgent:
    """ Base class for all agents in the game (dealer and player).

    Each agent receives the full state representation from the environment
    (ie the one-hot matrix of cards for each agent holds).

    Internal state representation is kept using state_idx in the compressed representation
    used for plotting (Figure 5.1 and 5.2) of (usable_ace, player_sum, dealer_showing).

    Classes inheriting this class specify the particular value/q-value/policy update algorithms.
    """
    def __init__(self, agent_idx, state_value_fn):
        self.agent_idx = agent_idx  # the position in the state and action arrays that this agent occupies
        self.get_state_value = lambda state: state_value_fn(state)[agent_idx]  # calculate state value for this agent

        # Figure 5.1
        # for monte carlo agents, at the end of each episode record the value function
        # indexed by (usable ace, dealer card shown, player sum)
        # returning a list of [accumulated value, episodes_played] for this index
        self.values = defaultdict(int)

    def update(self, state_idx, action, reward):
        """ implements the value/q-value update algorithm specific to the agent

            First-visit monte carlo algorithm:
                -- states and rewards are collected along the episode
                -- at the end of the episode after final reward is collected,
                    state values and visit counts are update for each of the
                    states visited during the episode.
        """
        # add new state to episode values
        self.episode_values[state_idx] = 0  # the value is updated below for every state visited

        # increment reward for all states previously visited (including current) in this episode
        for k, v in self.episode_values.items():
            self.episode_values[k] += reward

        # after observing the final state (populated in the stop_episode function),
        # update overall value function at the end of the whole episode
        if self.final_state is not None:
            for k, v in self.episode_values.items():
                value = self.values.get(k, [0,0])  # default [value, count] = [0, 1] to avoid divisiion by 0 when calculating avg
                value[0] += v  # increment value
                value[1] += 1  # increment visited count
                self.values[k] = value

    def get_action(self, state):
        """ called by the environment when the agent is in turn to play """
        raise NotImplementedError

    def observe_transition(self, state, action, next_state, delta_reward):
        """ called by the environment to inform agent that a transition has been observed;
            convert the external state representation to internal and run update with the
            specific value algorithm
        """
        self.last_state = state
        self.last_action = action
        state_idx = self.get_state_idx(state, action)
        self.update(state_idx, action, delta_reward)

    def get_state_idx(self, state, action):
        """ compute the tracking index for a given state: (usable ace, player sum, dealer showing) """
        # player sum
        player_sum = self.get_state_value(state)
        # dealer showing
        dealer_showing = state[0][0]  # state row 0, col 0 for the dealer and shown card
        # usable ace
        player_sum_without_aces = state.copy()
        player_sum_without_aces[:,1] = 0
        player_sum_without_aces = self.get_state_value(player_sum_without_aces)
        usable_ace = abs(player_sum_without_aces - player_sum) >= 10

        return (int(usable_ace), int(player_sum), int(dealer_showing)), action

    def start_episode(self):
        """ called by the environment when new episode is starting """
        self.episode_values = defaultdict(int)
        self.start_state = None
        self.final_state = None

    def stop_episode(self, state, final_reward):
        """ called by the environment when episode is done """

        self.final_state = state  # this triggers the update function to aggregate values over the episode

        # with final state set, observe the final reward
        self.observe_transition(state, self.last_action, state, final_reward)


class FixedPolicyValueEstimationMCAgent(MCAgent):
    def __init__(self, decision_rule, **kwargs):
        super().__init__(**kwargs)
        self.decision_rule = decision_rule  # policy threshold for action 'hit' or 'stick'

    def get_action(self, state):
        """ returns an action given the fixed policy in Example 5.1 """
        if self.get_state_value(state) >= self.decision_rule:
            return 1  # stick
        else:
            return 2  # hit

    def get_values(self):
        # aggregate values across actions
        agg_values = defaultdict(int)
        for (state_idx, action), (v, c) in self.values.items():
            val, count = agg_values.get(state_idx, [0, 0])
            new_val = val + v
            new_count = count + c
            agg_values[state_idx] = [new_val, new_count]
        return agg_values


class ExploringStartsMCAgent(MCAgent):
    def __init__(self, actions_fn, **kwargs):
        super().__init__(**kwargs)
        self.get_possible_actions = lambda : actions_fn(self.agent_idx)  # grab legal actions from the environment

    def get_action(self, state):
        """ exploring starts algorithm: choose (S_0 and A_0 such that all pairs have prob > 0)
        and then follow a optimal policy given current q_values """

        legal_actions = self.get_possible_actions()

        # check if episode has just started; if just started, pick random action (exploring starts)
        if self.start_state is None:
            action = np.random.choice(legal_actions)
            self.start_state = state
        else:
            # else pick best_action given q_values (follow best available policy)
            state_idx = self.get_state_idx(state, None)
            action = self.compute_action_from_q_values(state_idx)

        return action


    def compute_action_from_q_values(self, state_idx):
        actions = self.get_possible_actions()

        best_action = None
        best_q_value = float('-inf')
        for action in actions:
            # update state_idx with action
            state_idx = state_idx[0], action
            q_value = self.get_q_value(state_idx)
            if q_value > best_q_value:
                best_action = action
                best_q_value = q_value

        return best_action

    def compute_value_from_q_values(self, state_idx):
        best_action = self.compute_action_from_q_values(state_idx)
        if best_action is None:
            return 0
        else:
            # update state_idxs with best_action
            state_idx = state_idx[0], best_action
            return self.get_q_value(state_idx)

    def get_q_value(self, state_idx):
        value, count = self.values.get(state_idx, [0, 0])
        if count == 0:  # avoid division by 0
            count += 1
        return value / count

    def get_policy(self, state_idx):
        return self.compute_action_from_q_values(state_idx)

    def get_value(self, state_idx):
        return self.compute_value_from_q_values(state_idx)

# --------------------
# Run an episode of the game
# --------------------

def run_episode(agents, environment):
    environment.reset()
    for agent in agents:
        agent.start_episode()

    # arange agents so dealer goes last
    sorted_agents = sorted(agents, key=lambda a: a.agent_idx, reverse=True)

    for agent in sorted_agents:
        # play each agent until they become inactive (until they bust or stick)
        while environment.is_agent_active(agent.agent_idx):
            # 1. get current state
            state = environment.get_current_state().copy()

            # 2. get agent action
            action = agent.get_action(state)

            # 3. execute action
            next_state, reward = environment.do_action(action, agent.agent_idx)

            # 4. update learner
            agent.observe_transition(state, action, next_state, reward)

        if environment.is_terminal(state) or np.all(environment.active_agents == 0):
            break

    # 5. record returns
    for agent in agents:
        last_state = environment.get_current_state()
        episode_reward = environment.get_final_reward(agent.agent_idx)
        agent.stop_episode(state, episode_reward)


# --------------------
# Figure 5.1: Approximate state-value functions for the blackjack policy that sticks only on 20 or 21,
# computed by Monte Carlo policy evaluation.
# --------------------

def make_array_from_dict(values_dict):
    """ convert a values dict with index (usable_ace, player_sum, dealer_showing) to numpy array 
    e.g.: in resulting array; index (0, :, :) represents the data for no usable ace """

    values = np.zeros((2, 22, 11))  # usable ace x player_sum x dealer showing
    for idx, v in values_dict.items():
        # skip values under 12 or busts over 21 since not plotting
        if idx[1] > 21:
            continue
        if idx[1] < 12:
            continue
        values[idx] = v[0] / v[1]  # calculate average reward
    values = values[:, 12:22, 1:] # cut off the zero index which is unpopulated
    return values

def fig_5_1(n_episodes=500000):
    # initiailize environment and agents
    env = BlackjackEnvironment(n_agents=2)
    agents = [FixedPolicyValueEstimationMCAgent(agent_idx=0,
                                                state_value_fn=env.get_state_value,
                                                decision_rule=17),  # dealer
              FixedPolicyValueEstimationMCAgent(agent_idx=1,
                                                state_value_fn=env.get_state_value,
                                                decision_rule=20)]  # player

    # simulate games and extract value functions
    for i in tqdm(range(n_episodes)):
        run_episode(agents, env)
        if i == n_episodes/5:
            n_episodes_interm = i
            values_short_run = deepcopy(agents[1].get_values())
    values_long_run = agents[1].get_values()

    # convert value functions to np arrays for plotting
    values = [make_array_from_dict(values_short_run), make_array_from_dict(values_long_run)]

    # plot
    fig = plt.figure()
    x = np.arange(1,11)  # dealer showing [A, 2, ... 10]
    y = np.arange(10)    # player sum [12, ... 21]
    xx, yy = np.meshgrid(x,y)
    for i in range(4):
        # plot
        ax = fig.add_subplot(2,2,i+1, projection='3d')
        run = i % 2             # column 1 (i=0,2) is values[0]
        usable_ace = int(i<2)   # row 1 (i=0,1) is values[j][1]
        ax.plot_wireframe(xx, yy, values[run][usable_ace])

        # clear labels for all the subplots
        ax.set_xticks([1, 10])
        ax.set_yticks([0, 9])
        ax.set_zlim(-1,1)
        ax.set_zticks([-1, 1])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    # add axes label to just the last axis
    ax.set_xlabel('Dealer showing', fontsize=8)
    ax.set_xticklabels([1, 10])
    ax.set_ylabel('Player sum', fontsize=8)
    ax.set_yticklabels([12, 21])
    ax.set_zticks([-1, 1])
    ax.set_zticklabels([-1, 1])

    # add titles
    axes = fig.get_axes()
    axes[0].set_title('After {} episodes'.format(n_episodes_interm))
    axes[1].set_title('After {} episodes'.format(n_episodes))
    axes[0].annotate('No\nusable\nace', xy=(1/12, 1/4), xycoords='figure fraction', horizontalalignment='center')
    axes[0].annotate('\nUsable \nace', xy=(1/12, 3/4), xycoords='figure fraction', horizontalalignment='center')

    plt.savefig('figures/ch05_fig_5_1.png')
    plt.close()


# --------------------
# Figure 5.2: The optimal policy and state1-9value function for blackjack, found by Monte Carlo ES.
# The state-value function shown was computed from the action-value function found by Monte Carlo ES.
# --------------------

def fig_5_2(n_episodes=1000000):

    # initiailize environment and agents
    env = BlackjackEnvironment(n_agents=2)
    agents = [FixedPolicyValueEstimationMCAgent(agent_idx=0,
                                                state_value_fn=env.get_state_value,
                                                decision_rule=17),  # dealer
              ExploringStartsMCAgent(agent_idx=1,
                                     state_value_fn=env.get_state_value,
                                     actions_fn=env.get_possible_actions)]  # player

    # simulate games and extract value functions
    for i in tqdm(range(n_episodes)):
        run_episode(agents, env)


    # 1. extract policy
    policy = np.zeros((2, 22, 11))  # shape (usable ace, player_sum, dealer_showing)
    value = np.zeros_like(policy)

    # iterate arrays and populate from the agent
    idxs = [(a, y, x) for a in np.arange(2)
                      for y in np.arange(12,22)
                      for x in np.arange(1,11)]  # generate index (usable ace, player sum, dealer_showing)
    for idx in idxs:
        state_idx = (idx, None)
        policy[idx] = agents[1].get_policy(state_idx)
        value[idx] = agents[1].get_value(state_idx)
    value = value[:,12:,1:]
    policy = policy[:,12:,1:]
    policy_labels = np.copy(policy).astype(dtype='<U1')
    policy_labels[policy==2] = 'H'
    policy_labels[policy==1] = 'S'


    a = np.array([0, 1])    # usable ace
    x = np.arange(1,11)     # dealer showing [A, 2, ... 10]
    y = np.arange(10)       # player sum [12, ... 21]
    xx, yy = np.meshgrid(x, y)

    fig = plt.figure(figsize=(12,8))

    for i in range(2):
        # subplots 2 and 4 show value
        ax = fig.add_subplot(2,2,2*(i+1), projection='3d')
        ax.plot_wireframe(xx, yy, value[(i+1) % 2])
        ax.set_xticks([1,10])
        ax.set_yticks([0,9])
        ax.set_yticklabels([12,21])
        ax.set_zlim(-1,1)
        ax.set_zticks([-1,1])

        # subplots 1 and 3 show policy
        ax = fig.add_subplot(2,2, 1 + 2*i)
        sns.heatmap(np.flipud(policy[(i+1)%2]),
                    xticklabels=list(range(1,11)),
                    yticklabels=list(range(21,11,-1)),
                    square=True,
                    cbar=False,
                    fmt='s',
                    annot=np.flipud(policy_labels[(i+1)%2]),
                    ax=ax)
        ax.yaxis.tick_right()
        ax.yaxis.set_tick_params(rotation=0)
        ax.yaxis.set_label_position('right')

    # add titles
    axes = fig.get_axes()
    axes[2].set_xlabel('Dealer showing')
    axes[2].set_ylabel('Player sum')
    axes[3].set_xlabel('Dealer showing')
    axes[3].set_ylabel('Player sum')

    axes[0].annotate(r'$\pi_{{*}}$', size=20, xy=(0.3, .95), xycoords='figure fraction', horizontalalignment='center')
    axes[0].annotate(r'$v_{{*}}$', size=20, xy=(0.7, .95), xycoords='figure fraction', horizontalalignment='center')
    axes[0].annotate('\nUsable \nace', size=14, xy=(1/12, 3/4), xycoords='figure fraction', horizontalalignment='center')
    axes[0].annotate('No\nusable\nace', size=14, xy=(1/12, 1/4), xycoords='figure fraction', horizontalalignment='center')

    plt.savefig('figures/ch05_fig_5_2.png')
    plt.close()


# --------------------
# Figure 5.3: Weighted importance sampling produces lower error estimates of the value of a single 
# blackjack state from off-policy episodes.
# --------------------

class FixedStartBlackjackEnvironment(BlackjackEnvironment):
    """ Modification of the environment to ensure starting state is per Example 5.4:

        Starting state: usable_ace, player_sum, dealer_showing = 1, 13, 2
            (that is the player holds an ace and a deuce or equavalently three aces)

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_start_state(self):
        player_states = np.zeros((2, 1 + len(self.card_deck)))
        player_states[0][1] = 3         # three aces in position 1
        player_states[1][[1,2]] = 1     # one ace in position 1 and one 2 in position 2

        dealer_state = np.zeros(1 + len(self.card_deck))
        dealer_state[2] = 1  # holding one 2 in position 2
        dealer_state[0] = 2  # showing card 2
        dealer_state += self.sample_cards(1)

        # randomly choose among the player start states
        player_state = player_states[int(np.random.rand() > .5)]

        # set the initial state and make all agents active
        self.state = np.vstack((dealer_state, player_state))
        self.active_agents = np.ones(self.n_agents)
        return self.state


class RandomPolicyAgent(FixedPolicyValueEstimationMCAgent):
    def __init__(self, possible_actions, **kwargs):
        super().__init__(decision_rule=None, **kwargs)
        self.actions = possible_actions

    def get_action(self, state):
        return np.random.choice(self.actions)


def fig_5_3(n_runs=100, n_episodes=10000):
    # initiailize environment
    env = FixedStartBlackjackEnvironment(n_agents=2)
    # initialize the value estimate for ordinary and weighted importance sampling
    v = np.zeros((2, n_episodes, n_runs))

    # precompute the importance sampling ratio rho as a dict indexed by (state_idx, action)
    # 1. get probs under target and behavior policies
    # 2. enumerate possible state representations
    # 3. compute rho for each action|state

    # 1. get the probs under the target and behavior policies
    def p_target(state_idx):
        action = state_idx[1]
        target_action = 1 if state_idx[0][1] >= 20 else 2  # stick if player sum >= 20 (action 1) else hit (action 2)
        return int(target_action == action)  # target prob is deterministic; so acts with prob = 1 if correct action; else prob = 0
    actions = env.get_possible_actions(1)  # actions available to the player
    p_behavior = 1/len(actions)  # uniform

    # 2. enumerate possible state representations [(1, 13, 2), ... (1, 21, 2)] for usable ace, player_sum, dealer_showing
    states_list = [(a, x, 2) for a in range(2) for x in range(13, 22)]

    # 3. derive action probability under the target policy and the behavior policy at each state index
    rho = defaultdict(int)  # return 0 for states not already enumerated eg when player sum is > 21
    for s in itertools.product(states_list, actions):  # state_idxs represented as ((usable_ace, player_sum, dealer_showing), action)
        rho[s] = p_target(s) / p_behavior

    # simulate games and extract value function estimate
    # compute eq 5.5 and 5.6 for ordinary and weighted importance sampling
    for i in tqdm(range(n_runs)):
        # reinitialize agents for every run
        agents = [FixedPolicyValueEstimationMCAgent(agent_idx=0,
                                                    state_value_fn=env.get_state_value,
                                                    decision_rule=17),  # dealer
                  RandomPolicyAgent(agent_idx=1,
                                    state_value_fn=env.get_state_value,
                                    possible_actions=env.get_possible_actions(1))]  # player
        a = agents[1]  # get the player agent for later reference
        # reinitialize value and rho trackers for every run
        _v = 0          # numerator for eq 5.5 -- cumulative weighted episode returns
        _rho = 1e-5     # denominator for eq 5.6 -- cumulative weights

        for j in range(n_episodes):
            # for each episode
            # 1. run episode
            # 2. get episode return
            # 3. get episode rho as product for each of the action|state visited in the episode
            # 4. update running weighted return estimate and cumulative rho
            # 5. record value estimate for this episode

            # 1. run episode
            run_episode(agents, env)

            # 2. get episode return
            episode_return = list(a.episode_values.values())[-1]  # get the final return (total episode return ) from the last state recorded

            # 3. get episode rho as product for each of the action|state visited in the episode
            episode_rho = np.array([rho[k] for k in a.episode_values.keys() if k[0][1] <= 21])
            episode_rho = np.prod(episode_rho)  # eq 5.3 -- prob of the full episode trajectory

            # 4. update running weighted return estimate and cumulative rho
            _v += episode_return * episode_rho  # numerator
            _rho += episode_rho                 # denominator for weighted importance

            # 5. record value estimate for this episode
            v[0][j, i] = _v / (j + 1)   # ordinary importance sampling (eq 5.5)
                                        # denominator is over # times the state we're estimated is encounter 
                                        # which occurs as the initial state for each episode,
                                        # hence denominator is trivially the number of the episodes
            v[1][j, i] = _v / _rho      # weighted importance sampling (eq. 5.6)

    # plot results
    # calculate mean square error (average over the runs)
    true_value = -0.27726
    mse = ((v - true_value)**2).mean(axis=-1)

    plt.plot(mse[0], label='Ordinary importance sampling')
    plt.plot(mse[1], label='Weighted importance sampling')
    plt.xscale('log')
    plt.xlabel('Episodes (log scale)')
    plt.ylim(-0.15,4)
    plt.yticks(np.arange(5))
    plt.ylabel('Mean square error (avg over {} runs)'.format(n_runs))
    plt.legend()

    plt.savefig('figures/ch05_fig_5_3.png')
    plt.close()


if __name__ == '__main__':
    fig_5_1()
    fig_5_2()
    fig_5_3()
