import os
import sys
import json
import networkx as nx
from networkx.algorithms.shortest_paths import single_source_dijkstra_path_length
import numpy as np
from numpy.linalg import norm
from sklearn.linear_model import LogisticRegression


path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(path)
dir_ = os.path.dirname(__file__)
os.chdir(dir_)

from cluster.model import DistanceModel
from helpers import configure_logger
logger=configure_logger(default=False,path=os.path.dirname(__file__))

class MDP:
    def __init__(self, diagnosis,n_actions_per_state=None,initial=False):
        self.graph = nx.DiGraph()
        self.diagnosis = diagnosis
        self.n_actions_per_state=n_actions_per_state
        if initial:
            #TODO check where initial is used
            # first MDP
            self.graph['initial']=True

    def make_models(self):
        models = []
        for t in [1, 2, 3]:
            models.append(DistanceModel(diagnosis=self.diagnosis, timestep=t))

        self.model1, self.model2, self.model3 = models
        self.model1.average_feature_time_series()
        self.model2.average_feature_time_series()
        self.model3.average_feature_time_series()
        return self

    def predict_discharge_based_on_final_features(self):
        X_stage3=self.model3.feature_tensors.numpy()
        y=self.model3.output.numpy()
        logit_model=LogisticRegression().fit(X_stage3,y)
        #prediction stage
        X_stage1=self.model1.feature_tensors.numpy()
        X_stage2=self.model2.feature_tensors.numpy()
        X=np.vstack((X_stage1,X_stage2,X_stage3))
        # probability of non-successful discharge outcome
        probabilities=logit_model.predict_proba(X)[:,0].tolist()

        with open('logit_probabilities.json','w') as file:
            json.dump(probabilities,file)
        return self


    def get_global_target(self):
        good_indices = (self.model3.output == 1).nonzero().flatten()
        global_target = self.model3.feature_tensors.index_select(
            0, good_indices).mean(dim=0)
        return global_target

    def create_states_base(self,time_period):
        """Iterate over tensors, each row is a state (node), assign features to nodes.
        Reward is 0 and this method should be used for dqn state space.
            """

        if time_period==1:
            data=self.model1.feature_tensors
        elif time_period==2:
            data=self.model2.feature_tensors
        else:
            data=self.model3.feature_tensors

        self.global_target = self.get_global_target()
        self.cosine_sim_scores = {}
        for i, train_x in enumerate(data):
            self.graph.add_node(i+1, label=i+1,features=train_x, value=0, reward=0)
            score = self.calculate_cosine_similarity(
                train_x, self.global_target)
            self.cosine_sim_scores.update({i+1: score})

        ordered_scores = list(
            sorted(self.cosine_sim_scores.items(), key=lambda i: i[1]))

        goal_node_id = list(ordered_scores[-1])[0]
        self.goal_state = self.graph.nodes[goal_node_id]
        print(f"Goal State: {self.goal_state['label']}")
        self.graph.nodes[goal_node_id]['goal'] = True
        self.graph.nodes[goal_node_id]['reward'] = 100

        bad_node_id = list(ordered_scores[0])[0]
        self.bad_state = self.graph.nodes[bad_node_id]
        print(f"Bad State: {self.bad_state['label']}")
        self.graph.nodes[bad_node_id]['bad'] = True
        self.graph.nodes[bad_node_id]['reward'] = -100
        return self


    def create_states(self,time_period):
        """Iterate over tensors, each row is a state (node), assign features to nodes.Reward is based on
        probabilities"""

        with open('logit_probabilities.json') as file:
            probabilities=json.load(file)
        individual_shape=int(len(probabilities)/3)

        if time_period==1:
            data=self.model1.feature_tensors
            probabilities=probabilities[:individual_shape]
        elif time_period==2:
            data=self.model2.feature_tensors
            probabilities=probabilities[individual_shape:individual_shape*2]
        else:
            data=self.model3.feature_tensors
            probabilities=probabilities[individual_shape*2:]

        max_value=max(probabilities)
        max_index=probabilities.index(max_value)
        min_value=min(probabilities)
        min_index=probabilities.index(min_value)
        rewards=[50-r*100 for r in probabilities]

        self.global_target = self.get_global_target()
        self.cosine_sim_scores = {}
        for i, train_x in enumerate(data):
            self.graph.add_node(i+1, label=i+1,features=train_x, value=0, reward=rewards[i])

        goal_node_id = min_index
        self.goal_state = self.graph.nodes[goal_node_id]
        print(f"Goal State: {self.goal_state['label']}")
        self.graph.nodes[goal_node_id]['goal'] = True

        bad_node_id = max_index
        self.bad_state = self.graph.nodes[bad_node_id]
        print(f"Bad State: {self.bad_state['label']}")
        self.graph.nodes[bad_node_id]['bad'] = True
        return self

    def calculate_cosine_similarity(self, tensor_a, tensor_b):
        return np.dot(tensor_a, tensor_b)/(norm(tensor_a)*norm(tensor_b))

    def create_actions_and_transition_probabilities(self):
        n_actions=len(self.graph.nodes)-1
        if self.n_actions_per_state:
            assert isinstance(self.n_actions_per_state,int) and self.n_actions_per_state<=n_actions,\
            f"{self.n_actions_per_state} should be less than or equal to {n_actions}."
        else:
            self.n_actions_per_state=n_actions

        for i, state_i in self.graph.nodes(data=True):
            similarities = {}
            for j, state_j in self.graph.nodes(data=True):
                if i == j:
                    continue
                score = self.calculate_cosine_similarity(
                    state_i['features'], state_j['features'])
                # j is correct
                similarities.update({j: score})

            top_actions = list(sorted(similarities.items(), key=lambda dict_: dict_[1]))[-self.n_actions_per_state:]
            scores_only = [i[1] if i[1]>=0  else 0 for i in top_actions]
            for t in top_actions:
                self.graph.add_edge(i, t[0], sim_score=1-t[1], probability=t[1]/sum(scores_only))

        # nx.draw(self.graph, with_labels=True)
        return self


    def one_step_look_ahead(self,state,discount_factor=1):
        """Return list of action_values from the current state."""
        out_edges=self.graph.out_edges(state,data=True)
        action_values=[]
        for edge in out_edges:
            pr=edge[2]['probability']
            next_state=self.graph.nodes[edge[1]]
            value=pr*(next_state['reward']+discount_factor*next_state['value'])
            action_values.append(value)
        return action_values


    def value_iteration(self,theta=0.001):
        """Perform value iteration, returning optimal policy."""
        #TODO converting to list may not be needed here in a few other similar places
        #NOTE number of actions need to be more than 1 for the algorithm to stop
        states=list(self.graph.nodes)
        while True:
            delta=0
            for state in states:
                if self.graph.nodes[state].get('goal'):
                    continue
                current_value=self.graph.nodes[state]['value']
                action_values=self.one_step_look_ahead(state=state)
                max_action_value=max(action_values)
                self.graph.nodes[state]['value']=max_action_value
                delta=max(delta,abs(current_value-max_action_value))
            if delta<=theta:
                break

        policy=[]
        self.policy_states=[]
        for state in states:
            logger.info(f"State:{state},Value:{self.graph.nodes[state]['value']}")
            action_values=self.one_step_look_ahead(state=state)
            best_action_id=np.argmax(action_values)
            best_action_state=list(self.graph.out_edges(state))[best_action_id][1]
            policy.append(f"state: {state}->{best_action_state}.")
            self.policy_states.append((state,best_action_state))
        logger.info(f"Policy\n {policy}")
        return self

    def dijkstra(self,start_state):
        """Computes shortes path length between a node and all other reachable nodes."""
        states=list(self.graph.nodes)
        visited=set()

        distances={state:np.inf for state in states}
        distances[start_state]=0

        while len(visited)!=len(states):
            sorted_=dict(sorted(distances.items(),key=lambda dict_:dict_[1]))
            for k,_ in sorted_.items():
                if k not in visited:
                    state=k
                    break

            visited.add(state)
            out_edges=self.graph.out_edges(state,data=True)
            for edge in out_edges:
                back_cost=distances[state]
                front_cost=edge[2]['sim_score']
                total=back_cost+front_cost
                next_state=edge[1]
                if total<distances[next_state]:
                    distances[next_state]=total
                    self.graph.nodes[next_state]['distance']=total
                    self.graph.nodes[next_state]['parent']=state
        return distances

    def evaluate(self):
        correct_policies=[]
        for t in self.policy_states:
            id=t[1]
            pred=0 if self.graph.nodes[id]['reward']<0 else 1
            # if an added node from the previous layer with type str, assign 1, as it is a good state
            target=self.model3.output[id-1].item() if type(id)==int else 1
            correct_policies.append(pred==target)
        print(sum(correct_policies)/len(correct_policies)*100)

#TODO play special importance to cycles
class StageMDP:
    def __init__(self):
        self.mdp_t1=MDP(diagnosis="PNEUMONIA",n_actions_per_state=3).make_models().create_states(time_period=1)
        self.mdp_t2=MDP(diagnosis="PNEUMONIA",n_actions_per_state=3).make_models().create_states(time_period=2)
        self.mdp_t3=MDP(diagnosis="PNEUMONIA",n_actions_per_state=3).make_models().create_states(time_period=3)
        self.mdp_list=[self.mdp_t1,self.mdp_t2,self.mdp_t3]

    #TODO these  nodes need to be explicitly connected, instead of one extra node being added to the other graph
    def connect_graphs(self):
        """Add goal state of i periods graph to i+1.
        """
        for i in range(len(self.mdp_list)-1):
            goal_state=self.mdp_list[i].goal_state
            label=f"t{i+1}_{goal_state['label']}"
            self.mdp_list[i+1].graph.add_node(label,label=label,features=goal_state['features'],
                                              value=goal_state['value'],reward=0,start=True)
        return self


    def __call__(self):
        #TODO check start logic
        self.connect_graphs()
        self.mdp_t1.create_actions_and_transition_probabilities().value_iteration().evaluate()
        #TODO period t2 and t3 are not learning
        self.mdp_t2.create_actions_and_transition_probabilities().value_iteration().evaluate()
        self.mdp_t3.create_actions_and_transition_probabilities().value_iteration().evaluate()

if __name__=="__main__":
    StageMDP()()
