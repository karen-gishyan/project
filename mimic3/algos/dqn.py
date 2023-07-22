from gymnasium import Env, Space
import torch.nn as nn
import numpy as np
import random
from model import MDP
import torch
import torch.nn.functional as F


mdp=MDP('SEPSIS')

class MimicSpace(Space):

    def __init__(self, mdp: MDP):
        """MDP object which will be used as the base of the experiment.
        Action space signifes a transition to another state, so specifing a node_id can represent both a
        state and an action (transition to the next state).

        Args:
            mdp (MDP): MDP objects used for creating the state dynamics.'graph' attribute will store all
            the information about states and actions.
        """
        self.mdp = mdp.make_models().create_states(time_period=1
        ).create_actions_and_transition_probabilities()

    def sample(self):
        """Select a random action.
        """
        actions=list(self.mdp.graph.edges)
        random_action = random.choice(actions)[1]
        return random_action

    def contains(self, x: int):
        """Check if a state /action is part of the MDP graph state space.

        Args:
            x (int): integer state representing the graph node.
        """
        states=self.mdp.graph.nodes
        return x in states


class MimicEnv(Env):

    state_space=MimicSpace(mdp)
    action_space = state_space
    observation_space = state_space

    def step(self, action):
        """
        #NOTE With existing implementation, action from current state will always lead to a predefined next_state.
        Returns next_state, reward, terminated, truncated information.
        Args:
            action (_type_): id of the next state (edge connecting to the next node).
        """
        #TODO issue here
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid."
        assert self.state is not None, "Call reset before using step method."


        #TODO action may not exist for a given state, such as when it points to same state
        next_state_id=None
        for edge in self.observation_space.mdp.graph.edges:
            if edge[0]==self.state_label and edge[1]==action:
                next_state_id=edge[1]
                break
        if not next_state_id:
            #TODO this may not be right
            next_state_id=self.state_label

        next_state=self.observation_space.mdp.graph.nodes[next_state_id]
        reward=next_state['reward']
        terminated=next_state.get('goal',False)
        truncated=None
        next_state_label=next_state['label']
        next_state=next_state['features']
        self.state=next_state
        self.state_label=next_state_label

        return next_state, reward, terminated, truncated

    def reset(self):
        """Each state can be as a start state, so we sample randomly.
        """
        state=random.choice(self.observation_space.mdp.graph.nodes)
        self.state_label=state['label']
        state=self.state=state['features']
        return state


class QNetwork(nn.Module):
    def __init__(self, state_dim , action_dim, h_layer_dim):
        super(QNetwork, self).__init__()
        self.x = nn.Linear(state_dim, h_layer_dim)
        self.h_layer = nn.Linear(h_layer_dim, h_layer_dim)
        self.y_layer = nn.Linear(h_layer_dim, action_dim)
        print(self.x)

    def forward(self, state):
        xh = F.relu(self.x(state))
        hh = F.relu(self.h_layer(xh))
        state_action_values = self.y_layer(hh)
        return state_action_values

class Agent(object):
    def __init__(self, state_dim, action_dim,env):
        self.qnet = QNetwork(state_dim, action_dim, 32)
        self.optimizer = torch.optim.Adam( self.qnet.parameters(), lr=0.001)
        self.discount_factor = 0.99
        self.loss_function = nn.MSELoss()
        self.env=env
        self.replay_buffer = []

    def epsilon_greedy_action(self, state, epsilon):
        if np.random.uniform(0, 10) < epsilon:
            # choose random action
            return self.env.action_space.sample()
        else:
            output = self.qnet(state).detach().numpy()
            # choose greedy action
            return np.argmax(output)

    def update_QNetwork(self, state, next_state, action, reward, terminals):
        #TODO there is index 16, which is out of bounds here
        qsa = torch.gather(self.qnet(state), dim=1, index=action.long())
        qsa_next_actions = self.qnet(next_state)
        qsa_next_action,_ = torch.max(qsa_next_actions, dim=1, keepdim=True)
        not_terminals = 1 - terminals
        qsa_next_target = reward + not_terminals * self.discount_factor * qsa_next_action
        q_network_loss = self.loss_function(qsa, qsa_next_target.detach())
        self.optimizer.zero_grad()
        q_network_loss.backward()
        self.optimizer.step()

    def update(self, update_rate):
        for _ in range(update_rate):
            states, next_states, actions, rewards, terminals = sample_from_buffer(self.replay_buffer,size=128)
            self.update_QNetwork(states, next_states, actions, rewards, terminals)


def sample_from_buffer(buffer,size):
    minibatch=random.choices(buffer,k=size)
    states, next_states, actions, rewards, terminals=(zip(*minibatch))
    states=torch.stack(states)
    next_states=torch.stack(next_states)
    actions=torch.Tensor(actions).view(size,-1)
    rewards=torch.Tensor(rewards).view(size,-1)
    terminals=torch.Tensor(terminals).view(size,-1)
    return states, next_states,actions, rewards, terminals


def train():
    env=MimicEnv()
    agent = Agent(state_dim=10, action_dim=15,env=env)
    number_of_episodes = 200
    max_time_steps = 200

    for episode in range(number_of_episodes):
        reward_sum = 0
        state = env.reset()
        for _ in range(max_time_steps):
            #TODO issue here
            action = agent.epsilon_greedy_action(state, 0.2)
            next_state, reward, terminal, _ = env.step(action)

            reward_sum += reward
            agent.replay_buffer.append([state,next_state,action,reward,terminal])
            state = next_state
            if terminal:
                print('episode:', episode, 'sum_of_rewards_for_episode:', reward_sum)
                break
        agent.update(40)

train()
