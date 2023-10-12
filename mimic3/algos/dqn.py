from gymnasium import Env, Space
import torch.nn as nn
from django.utils.module_loading import import_string
import numpy as np
import random
from model import MDP
import torch
import torch.nn.functional as F
import copy
import networkx as nx


def set_seed(seed: int = 64) -> None:
    """
    In a single run, torch.rand() can produce different results, but across multiple runs the results are the same.
    Same for numpy.
    source: https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class MimicSpace(Space):

    def __init__(self, mdp: MDP, time_period):
        """MDP object which will be used as the base of the experiment.
        Action space signifes a transition to another state, so specifing a node_id can represent both a
        state and an action (transition to the next state).

        Args:
            mdp (MDP): MDP objects used for creating the state dynamics.'graph' attribute will store all
            the information about states and actions.
        """
        self.time_period = time_period
        self.mdp = mdp.make_models().create_states_base(time_period=time_period)

    def create_actions(self):
        self.mdp.create_actions_and_transition_probabilities()

    def sample(self):
        """Select a random action.
        """
        actions = list(self.mdp.graph.edges)
        random_action = random.choice(actions)[1]
        return random_action

    def contains(self, x: int):
        """Check if a state /action is part of the MDP graph state space.

        Args:
            x (int): integer state representing the graph node.
        """
        states = self.mdp.graph.nodes
        return x in states


class MimicEnv(Env):

    def __init__(self, time_period, n_actions_per_state=None):
        mdp = MDP('PNEUMONIA', n_actions_per_state=n_actions_per_state)
        self.time_period = time_period
        self.state_space = MimicSpace(mdp, time_period)
        self.action_space = self.state_space
        self.observation_space = self.state_space
        self.visited_states = set()

    def step(self, action):
        """
        #NOTE With existing implementation, action from current state will always lead to a predefined next_state.
        Returns next_state, reward, terminated, truncated information.
        Args:
            action (_type_): id of the next state (edge connecting to the next node).
        """

        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid."
        assert self.state is not None, "Call reset before using step method."

        self.visited_states.add(self.state['label'])
        next_state_id = None
        for edge in self.observation_space.mdp.graph.edges:
            if edge[0] == self.state['label'] and edge[1] == action:
                next_state_id = edge[1]
                break
        if not next_state_id:
            state_out_edges = list(
                self.state_space.mdp.graph.out_edges(self.state['label']))
            next_state_id = random.choice(state_out_edges)[1]

        next_state = self.observation_space.mdp.graph.nodes[next_state_id]
        if next_state['label'] in self.visited_states:
            # dynamic reward
            reward = next_state['reward']-50
        else:
            reward = next_state['reward']+50
        terminated = next_state.get('goal', False)
        truncated = None
        self.state = next_state
        return next_state, reward, terminated, truncated

    def reset(self):
        """Each state can be as a start state, so we sample randomly.
        """
        state = self.state = self.start_state
        return state


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, h_layer_dim):
        super(QNetwork, self).__init__()
        self.x = nn.Linear(state_dim, h_layer_dim)
        self.y_layer = nn.Linear(h_layer_dim, action_dim)

    def forward(self, state):
        # 1 hidden layer
        xh = F.relu(self.x(state))
        state_action_values = self.y_layer(xh)
        return state_action_values


class Agent(object):
    def __init__(self, state_dim, env, optimizer=torch.optim.Adam, lr=0.01, discount_factor=0.99):
        self.time_period = env.state_space.time_period
        action_dim = len(env.action_space.mdp.graph.nodes)
        self.qnet = QNetwork(state_dim, action_dim, 16)
        self.qnet_target = copy.deepcopy(self.qnet)
        self.optimizer = import_string(optimizer)(
            self.qnet.parameters(), lr=lr)
        self.discount_factor = discount_factor
        self.tau = 0.95
        self.loss_function = nn.MSELoss()
        self.env = env
        self.replay_buffer = []

    def epsilon_greedy_action(self, state, epsilon):
        if np.random.uniform(0, 10) < epsilon:
            # choose random action
            action = self.env.action_space.sample()
            return action
        else:
            output = self.qnet(state['features']).detach().numpy()
            # choose greedy action
            action = np.argmax(output)+1
            return action

    def soft_target_update(self, network, target_network, tau):
        for net_params, target_net_params in zip(network.parameters(), target_network.parameters()):
            target_net_params.data.copy_(
                net_params.data * tau + target_net_params.data * (1 - tau))

    def update_QNetwork(self, state, next_state, action, reward, terminals):

        # torch.gather: from a matrix of values, for each row select the value corresponding to the action index
        # you calculate the value of the state for the action providing the best value
        qsa = torch.gather(self.qnet(state), dim=1, index=action.long())
        qsa_next_actions = self.qnet_target(next_state)
        # you calculate the value of the next_state for the action providing the best value
        qsa_next_action, _ = torch.max(qsa_next_actions, dim=1, keepdim=True)
        not_terminals = 1 - terminals
        qsa_next_target = reward + not_terminals * \
            self.discount_factor * qsa_next_action
        # the network learns to minimize the difference between state's and next_state's best values.
        # the network should be optimized in a way that current state's and next_state's actions are
        # equivalently good, meaning they have close numerical outputs.
        q_network_loss = self.loss_function(qsa, qsa_next_target.detach())
        loss = q_network_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update(self, update_rate):
        for _ in range(update_rate):
            states, next_states, actions, rewards, terminals = self.sample_from_buffer(
                self.replay_buffer, size=8)
            self.update_QNetwork(states, next_states,
                                 actions, rewards, terminals)
            self.soft_target_update(self.qnet, self.qnet_target, self.tau)

    def sample_from_buffer(self, buffer, size):
        minibatch = random.choices(buffer, k=size)
        states, next_states, actions, rewards, terminals = (zip(*minibatch))
        state_features = []
        next_state_features = []
        for state, next_state in zip(states, next_states):
            state_features.append(state['features'])
            next_state_features.append(next_state['features'])

        states = torch.stack(state_features)
        next_states = torch.stack(next_state_features)
        actions = torch.Tensor(actions).view(size, -1)
        rewards = torch.Tensor(rewards).view(size, -1)
        terminals = torch.Tensor(terminals).view(size, -1)
        return states, next_states, actions, rewards, terminals

    def train(self, time_period, max_time_step, epsilon_greedy, update_rate):

        number_of_episodes = 300
        episode_rewards = []
        policy_graph = nx.DiGraph(time_period=time_period)
        for episode in range(1, number_of_episodes+1):
            # print('episode:', episode)
            reward_sum = 0
            state = self.env.reset()
            for _ in range(max_time_step):
                action = self.epsilon_greedy_action(state, epsilon_greedy)
                next_state, reward, terminal, _ = self.env.step(action)
                if episode == number_of_episodes:
                    # print(
                    #     f"state:{state['label']},next_state:{next_state['label']}")
                    policy_graph.add_edge(state['label'], next_state['label'])

                reward_sum += reward
                self.replay_buffer.append(
                    [state, next_state, action-1, reward, terminal])
                state = next_state
                if terminal:
                    break
            # print('sum_of_rewards_for_episode:', reward_sum)
            episode_rewards.append(reward_sum)
            self.update(update_rate)
            self.env.visited_states = set()

        policy_graph.graph['solution'] = terminal
        return episode_rewards, self.qnet, policy_graph
