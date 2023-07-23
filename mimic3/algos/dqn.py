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
            if edge[0]==self.state['label'] and edge[1]==action:
                next_state_id=edge[1]
                break
        if not next_state_id:
            #NOTE when the network predicts action which leads to the same state (e.g. 1->1)
            # we randomly select a next state based on existing outgoing edges of the state
            state_out_edges=list(self.state_space.mdp.graph.out_edges(self.state['label']))
            next_state_id=random.choice(state_out_edges)[1]

        next_state=self.observation_space.mdp.graph.nodes[next_state_id]
        reward=next_state['reward']
        terminated=next_state.get('goal',False)
        truncated=None
        self.state=next_state
        return next_state, reward, terminated, truncated

    def reset(self):
        """Each state can be as a start state, so we sample randomly.
        """

        state_id=random.choice(list(self.observation_space.mdp.graph.nodes))
        state=self.state=self.observation_space.mdp.graph.nodes[state_id]
        return state


class QNetwork(nn.Module):
    def __init__(self, state_dim , action_dim, h_layer_dim):
        super(QNetwork, self).__init__()
        self.x = nn.Linear(state_dim, h_layer_dim)
        # self.h_layer = nn.Linear(h_layer_dim, h_layer_dim)
        self.y_layer = nn.Linear(h_layer_dim, action_dim)
        print(self.x)

    def forward(self, state):
        xh = F.relu(self.x(state))
        # hh = F.relu(self.h_layer(xh))
        state_action_values = self.y_layer(xh)
        return state_action_values

class Agent(object):
    def __init__(self, state_dim, action_dim,env):
        self.qnet = QNetwork(state_dim, action_dim, 16)
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
            output = self.qnet(state['features']).detach().numpy()
            # choose greedy action
            return np.argmax(output)+1

    def update_QNetwork(self, state, next_state, action, reward, terminals):
        #TODO there is index 16, which is out of bounds here because one extra state was added

        # torch.gather: from a matrix of values, for each row select the value corresponding to the action index
        # you calculate the value of the state for the action providing the best value
        qsa = torch.gather(self.qnet(state), dim=1, index=action.long())
        qsa_next_actions = self.qnet(next_state)
        # you calculate the value of the next_state for the action providing the best value
        qsa_next_action,_ = torch.max(qsa_next_actions, dim=1, keepdim=True)
        not_terminals = 1 - terminals
        qsa_next_target = reward + not_terminals * self.discount_factor * qsa_next_action
        # the network learns to minimize the difference between state's and next_state's best values.
        q_network_loss = self.loss_function(qsa, qsa_next_target.detach())
        self.optimizer.zero_grad()
        q_network_loss.backward()
        self.optimizer.step()

    def update(self, update_rate):
        for _ in range(update_rate):
            states, next_states, actions, rewards, terminals = sample_from_buffer(self.replay_buffer,size=64)
            self.update_QNetwork(states, next_states, actions, rewards, terminals)


def sample_from_buffer(buffer,size):
    minibatch=random.choices(buffer,k=size)
    states, next_states, actions, rewards, terminals=(zip(*minibatch))
    state_features=[]
    next_state_features=[]
    for state,next_state in zip(states,next_states):
        state_features.append(state['features'])
        next_state_features.append(next_state['features'])

    states=torch.stack(state_features)
    next_states=torch.stack(next_state_features)
    actions=torch.Tensor(actions).view(size,-1)
    rewards=torch.Tensor(rewards).view(size,-1)
    terminals=torch.Tensor(terminals).view(size,-1)
    return states, next_states,actions, rewards, terminals

def train():
    env=MimicEnv()
    # I think action dim 16 is correct, including staying in the same state
    agent = Agent(state_dim=10, action_dim=16,env=env)
    number_of_episodes = 10
    max_time_steps = 50

    for episode in range(number_of_episodes):
        print('episode:', episode)
        reward_sum = 0
        state = env.reset()
        for _ in range(max_time_steps):
            #NOTE somehow right after the first iteration, the network predicts index of the action
            #which leads to a self loop (it does not know the id of the current state).
            # env.step() handles this self loop case.
            #NOTE generally there can be many repeating transition scenarios 1->4, 4->1 or 1->6, 6->5,5->7,7->1
            # and repetition will continue because the network weights do not change when adding to the buffer
            # (epsilon can break this loop, or if repetition is a self loop for one of the states (1->1),
            # it will be solved by the note described above).
            # if repition is not a self loop, the repeating transition may continue without being broken.
            #NOTE currently when many state explorations are observed, it is due to the fact that a transition
            # has been a self loop, and a state has been selected from actual edge transitions
            action = agent.epsilon_greedy_action(state, 0.2)
            next_state, reward, terminal, _ = env.step(action)
            print(f"state:{state['label']},next_state:{next_state['label']}")
            reward_sum += reward
            agent.replay_buffer.append([state,next_state,action-1,reward,terminal])
            state = next_state
            if terminal:
                # when the reward sum is 0, it means once a bad state has been visited before getting to a
                # terminal state.
                print('sum_of_rewards_for_episode:', reward_sum)
                break
        agent.update(50)

#TODO why does 1->1 after the first iteration?
#TODO why are such patterns so common, 1->4,4->1?
#TODO try to reduce repition scenarios
#TODO there should be more postive or more negative rewards, so as we check if the network is learning from
# episode to episode.
train()

