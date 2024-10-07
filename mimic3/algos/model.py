import os
import sys
import json
import networkx as nx
from networkx.algorithms.shortest_paths import single_source_dijkstra_path_length
import numpy as np
from numpy.linalg import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict



path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(path)
dir_ = os.path.dirname(__file__)
os.chdir(dir_)

from cluster.model import DistanceModel
from helpers import configure_logger
logger=configure_logger(default=False,path=os.path.dirname(__file__))

class MDP:
    def __init__(self, diagnosis,n_actions_per_state=None):
        self.graph = nx.DiGraph()
        self.diagnosis = diagnosis
        self.n_actions_per_state=n_actions_per_state


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
        self.time_period=time_period
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
        self.goal_node_id = goal_node_id
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


    def create_states_for_classification(self,time_period):
        self.time_period=time_period
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

        if time_period!=3:
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
            self.graph.nodes[goal_node_id]['reward'] = -100

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
        #NOTE number of actions need to be more than 1 for the algorithm to work in reasonable computation time
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
        self.policy_graph=nx.DiGraph()
        for state in states:
            logger.info(f"State:{state},Value:{self.graph.nodes[state]['value']}")
            action_values=self.one_step_look_ahead(state=state)
            best_action_id=np.argmax(action_values)
            best_action_state=list(self.graph.out_edges(state))[best_action_id][1]
            policy.append(f"state: {state}->{best_action_state}.")
            self.policy_graph.add_edge(f"{state}_t{self.time_period}",f"{best_action_state}_t{self.time_period}")
        logger.info(f"Policy\n {policy}")
        return self.policy_graph


#When the model is learning, policy_outcome is mostly 1 (it learns to lead to the target class). It is a
# a good RL model, but a bad classification model, as there are no zero outcomes.
class StageMDP:
    def __init__(self):
        # more actions means more the chance of positive policy_outcome
        self.mdp_t1=MDP(diagnosis="SEPSIS",n_actions_per_state=3).make_models().create_states_base(time_period=1)
        self.mdp_t2=MDP(diagnosis="SEPSIS",n_actions_per_state=3).make_models().create_states_base(time_period=2)
        self.mdp_t3=MDP(diagnosis="SEPSIS",n_actions_per_state=3).make_models().create_states_base(time_period=3)
        self.mdp_list=[self.mdp_t1,self.mdp_t2,self.mdp_t3]

    def connect_graphs(self):
        """Add goal state of i periods graph to i+1.
        """
        for i in range(len(self.mdp_list)-1):
            goal_state=self.mdp_list[i].goal_state
            label=f"{goal_state['label']}_t{i+1}"
            self.mdp_list[i+1].graph.add_node(label,label=label,features=goal_state['features'],
                                              value=goal_state['value'],reward=0,start=True)
        return self

    def create_multilayer_graph(self):
        """Separate multilayer graph for modeling and testing, path existence checking."""

        def get_start_node(mdp):
            for node, attr in mdp.graph.nodes(data=True):
                if attr.get("start"):
                    return node

        def get_goal_node(mdp):
            return mdp.goal_node_id

        def get_shortest_path(source):
            """Source node can vary. It should be of this format (1,"layer1")"""
            path = nx.shortest_path(M, source=source, target=(goal_G3,"layer3"), weight=None, method='dijkstra')
            # Print the path
            print(path,"\n")

        # no predefined start node for the first layer
        _= get_start_node(self.mdp_t1)
        goal_G1 = get_goal_node(self.mdp_t1)

        start_G2= get_start_node(self.mdp_t2)
        goal_G2 = get_goal_node(self.mdp_t2)

        start_G3= get_start_node(self.mdp_t3)
        goal_G3 = get_goal_node(self.mdp_t3)

        M = nx.DiGraph()
        for node in self.mdp_t1.graph.nodes:
            M.add_node((node,"layer1"))
        for edge in self.mdp_t1.graph.edges:
            M.add_edge((edge[0],"layer1"),(edge[1],"layer1"))

        for node in self.mdp_t2.graph.nodes:
            M.add_node((node,"layer2"))
        for edge in self.mdp_t2.graph.edges:
            M.add_edge((edge[0],"layer2"),(edge[1],"layer2"))

        for node in self.mdp_t3.graph.nodes:
            M.add_node((node,"layer3"))
        for edge in self.mdp_t3.graph.edges:
            M.add_edge((edge[0],"layer3"),(edge[1],"layer3"))

        M.add_edge((goal_G1, 'layer1'), (start_G2, 'layer2'))
        M.add_edge((goal_G2, 'layer2'), (start_G3, 'layer3'))

        # interlayer edges
        print("Interlayer edges")
        for edge in M.edges:
            if edge[0][1]!=edge[1][1]:
                print(edge)

        # shortest paths from different layer 1 start edges
        for node in M.nodes:
            if node[1]=="layer1":
                try:
                    get_shortest_path(node)
                except nx.exception.NetworkXNoPath as e:
                    print(str(e))

    def evaluate(self):
        """With current evaluation, if the algorithm reaches a full prescription, the outcomes is 1,
        if a self loop happens, the outcome is 0.
        """
        mappings=[]
        for i, _ in enumerate(self.mdp_t1.model1.feature_tensors,1):
            policy_mapping=defaultdict(list)
            # always start from timestep 1
            state=f"{i}_t1"
            current_state=state
            while True:
                out_edge=self.combined_policy_graph.edges(current_state)
                next_state=list(out_edge)[0][1]
                if next_state in policy_mapping[state]:
                    print(f'{i}th:Exiting because of a loop.')
                    policy_mapping['policy_outcome']=0
                    break
                policy_mapping[state].append(next_state)
                state_,time_period=next_state.split("_")
                mdp=getattr(self,f"mdp_{time_period}")
                if mdp.graph.nodes[int(state_)].get('goal'):
                    if mdp.time_period==3:
                        #solution
                        policy_mapping['policy_outcome']=1
                        break
                    #NOTE  if you reach a goal node, get to the second stage
                    # alternative could be not to allow no outgoing edges from the goal state
                    current_state=f"{next_state}_t{int(mdp.time_period)+1}"
                    policy_mapping[state].append(current_state)
                    continue
                current_state=next_state
            policy_mapping['actual_final_outcome']=self.mdp_t3.model3.output[i-1].item()
            mappings.append(policy_mapping)


        policy_outcomes=np.array([state['policy_outcome'] for state in mappings])
        actual_outcomes=np.array(self.mdp_t3.model3.output)
        evaluation=precision_recall_fscore_support(policy_outcomes,actual_outcomes,average='binary')
        print(evaluation)
        mappings.append(evaluation)

        with open("json_files/discrete_policies.json",'w') as file:
            json.dump(mappings,file,indent=4)

    def __call__(self):
        self.connect_graphs()
        obj1=self.mdp_t1.create_actions_and_transition_probabilities()
        obj2=self.mdp_t2.create_actions_and_transition_probabilities()
        obj3=self.mdp_t3.create_actions_and_transition_probabilities()
        self.create_multilayer_graph()

        t1_policy_graph=obj1.value_iteration()
        t2_policy_graph=obj2.value_iteration()
        t3_policy_graph=obj3.value_iteration()
        self.combined_policy_graph=nx.compose_all([t1_policy_graph,t2_policy_graph,t3_policy_graph])
        self.evaluate()


if __name__=="__main__":
    #TODO argparser arguments for experimenting with results with different number of actions as inputs
    StageMDP()()
