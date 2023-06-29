"""
1st experiment:
    DAG construction:
    1.1 multiple shortest paths per DAG without weights (default=1)
    1.1 one shortest path per DAG with weights.
    1.2 Astar seach with heuristics.
2nd experiment:
    Graph construction: (not a DAG, directed cycles are allowed), experimental
    2.1 multiple shortest paths per graph without weights, one path with weights,
    no Astar search
    Tree construction (undirected cycles are not allowed)
    2.2 one path for the tree, 'weight' does not affect number of paths, no Astar search

"""
import re
import random
import string
import os
import sys
import torch
import networkx as nx
from networkx.algorithms.shortest_paths import has_path,shortest_path
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from scipy.stats import kstest
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from utils import hierarchy_pos, topo_pos, Costs_and_Thresholds
from collections import deque
from math import sqrt

path=os.path.dirname(os.path.dirname(__file__))
sys.path.append(path)

dir_=os.path.dirname(__file__)
os.chdir(dir_)

from cluster.model import DistanceModel
from helpers import configure_logger

path=os.path.dirname(__file__)
logger=configure_logger(default=False,path=path)


class Graph:
    """
    Class for performing DAG (graph with no directed cycles) and tree (graph with no
    undirected cycles) constructions for optimal medical treatments.
    make_graphs(), make_graphs_stage_independent() perform the main functionalities.
    Other methods include tree, graph visualizations, shortest path calculations.

    """
    def __init__(self,diagnosis):
        self.diagnosis=diagnosis

    def make_models(self):
        """
        Using DistanceModel, bring stage features to the format that can be used
        for make_graphs() and create_relationships().
        """
        models=[]
        for t in [1,2,3]:
            models.append(DistanceModel(diagnosis=self.diagnosis,timestep=t))

        self.model1,self.model2,self.model3=models
        self.model1.average_feature_time_series().train_test()
        self.model2.average_feature_time_series()
        self.model3.average_feature_time_series()

    def make_graphs(self,n_childs=5):
        """
        Constructs DAGs (can be checked with is_directed_acyclic_graph(G)).
        Formal description:
        1. Select training instances from timestep 3, where the output is 1 (discharge to Home).
        2. Select features corresponding to indices of step 1.
        3. For each testing instance, calculate the top 5 most similar features from stage 1
        based on an RMSE score. Create a directed graph, create nodes with these 5 instances,
        and weighted edges from the starting instance to these nodes.
            3.1 Repeat the same procedure described in 3, this time selecting the top 5 most
            similar features from stage 2 for each of the nodes generated in 3.
            3.2 Repeat the same procedure desctibed in 3, this time selecting the top 5 most
            similar features from stage 3 for each of the nodes generated in 3.1.
        4. Visualize the first graph for each diagnosis.
        5. Calculate A star shortest path for each graph.
        6. Return the list of directed graphs (for each testing instance).

        Args:
            n_childs (int, optional): Number of child nodes each parent can have, default is 5.

        Returns:
            _type_: List[nx.DiGraph]
        """
        self.counter={}
        self.make_models()
        # third timestep should have only the good indices, only one of them is goal currently
        good_indices=(self.model3.output==1).nonzero().flatten()
        #NOTE "DIABETIC KETOACIDOSIS", "SEPSIS" "CONGESTIVE HEART FAILURE" have only 1 non -1 t3 node
        # "ALTERED MENTAL STATUS" has 0, so end node is chosen from t2 nodes.
        self.model3.feature_tensors_limited=self.model3.feature_tensors.index_select(0,good_indices)
        # for empty values having 0 or -1 makes no difference as long as it is the
        # same for all train and test instances

        self.djikstra_paths_list,self.astar_paths_list=[],[]
        #NOTE test_dat_graphs is not used currenty
        test_data_graphs=[]
        visualize=True
        for i,test_x in enumerate(self.model1.test_data):
            # if no values for a given test instance, continue
            if torch.all(test_x==-1):
                continue

            self.graph=nx.DiGraph()
            similarity_scores=[]
            for j,train_x in enumerate(self.model1.train_data):
                #score is 0 when two vectors are exactly the same (e.g.all values for each vector is -1).
                if torch.all(train_x==-1):
                    # fixed penalty of value of 1024
                    score=2**len(train_x)
                else:
                    #+1 for the lowest cost to be 1 instead of 0
                    #NOTE: may need to change to rmse
                    score=1+sqrt(mean_squared_error(test_x,train_x))
                similarity_scores.append((f"start:{i}",f"t1:{j}",score))

            stage1_top_closest=list(sorted(similarity_scores,key=lambda i:i[2])[:n_childs])
            self.graph.add_weighted_edges_from(stage1_top_closest)

            for tuple_ in stage1_top_closest:
                similarity_scores=[]
                int_node=int(re.findall("\d+",tuple_[1])[1])
                node_features=self.model1.feature_tensors[int_node]
                for j, train_x in enumerate(self.model2.feature_tensors):
                    if torch.all(train_x==-1):
                        score=2**len(train_x)
                    else:
                        score=1+sqrt(mean_squared_error(node_features,train_x))
                    similarity_scores.append((f"t1:{int_node}",f"t2:{j}",score))

                stage2_top_closest=list(sorted(similarity_scores,key=lambda i:i[2])[:n_childs])
                self.graph.add_weighted_edges_from(stage2_top_closest)
                for tuple_ in stage2_top_closest:
                    similarity_scores=[]
                    int_node=int(re.findall("\d+",tuple_[1])[1])
                    node_features=self.model2.feature_tensors[int_node]
                    for j, train_x in enumerate(self.model3.feature_tensors_limited):
                        if torch.all(train_x==-1):
                            score=2**len(train_x)
                            #NOTE we continue here but not above, as there is a chance
                            # that this 3rd timestep node is later selected as a final node
                            continue
                        else:
                            score=1+sqrt(mean_squared_error(node_features,train_x))
                        similarity_scores.append((f"t2:{int_node}",f"t3:{j}",score))

                    stage3_top_closest=list(sorted(similarity_scores,key=lambda i:i[2])[:n_childs])
                    self.graph.add_weighted_edges_from(stage3_top_closest)

            test_data_graphs.append(self.graph)
            if visualize:
                self.visualize_tree(self.graph)
                visualize=False

            djikstra_paths,astar_paths=self.compute_path(self.graph,method='ordinary')
            #NOTE currently no multiple shortest, so we take the only path for appending
            self.djikstra_paths_list.append(djikstra_paths[0])
            self.astar_paths_list.append(astar_paths)

        return self

    def create_relationships(self,cost_callable,n_childs=3,allow_cycles=False,non_deterministic=False,incremental_improvement=False):
        """
        For each testing instance with admission features, the goal is
        to construct a DAG which will store the path of the optimal treatment.
        If allow_cycles is False, each DAG is a tree because there are no undirected cycles
        (can be checked with nx.is_tree(G)).
        A treatment is optimal if it results in features which are withing certain
        distance away from the target features.
        The graph construction starts from the start node with initial features,
        which is added to the frontier que (que storing the nodes to be explored).
        We recursively call create_relationships(), and each call explores a single
        child node to see whether the node's features satsify one of the base
        case criterias. If they do satisfy, the graph is returned, else the algorithm
        continues (continuation means adding child nodes to the que based on similarity
        to be further explored). Each edge represents a treatment, and the node's features
        are the result of the treatment.

        Base cases:
            1. (No solution case): The frontier que is empty as there are no more child nodes
            to be explored. This case is reached frequently when we do not allow cycles, meaning
            each new node wil result in a cycle, thus is not added to the que. Here the graph
            is returned, and we fail to output a path where the end node has desired features.
            2. (No solution case): If the DAG reaches a depth of 5 (five treatments) and still
            no nodes with target features, the algorithm is terminated adn the graph is returned.
            3. (Acceptable solution case): After the graph reaches a depth of three, we start to check
            if the node's features are more similar to target features than the start node's features.
            This means the acceptable threshold similarity is not yet reached, but we have been able to
            move the patient to a better state. If yes, then the graph is returned.
            4. (Good solution case): The node's features are within the acceptable threshold, meaning
            we have found a good solution and the graph is returned.

        Formal Description:
            1. Try to take out a node from the frontier que in a FIFO manner.
                1.1 If no node, the base-case 1 is reached, termination.
            2. Add the node to the list of explored nodes.
            3. Obtain node's features, calculate diff=RMSE(node-features,target-features).
            4. If the node is at a depth of three or more.
               4.1 Check if diff <=RMSE(start-node features,target-features).
               If yes, base-case 3 is reached, termination.
            5. If the node is at a depth of five or more, based-case 2 is reached, termination.
            6. If diff <=threshold_distance, base-case 4 is reached, termination.
            7. From the data available for three stages, for each of the n most similar child
               nodes do:
                7.1 If cycles are allowed:
                    7.1.1 If incremental improvement is True, make sure each node's features
                    are better than parent's features. If not, continue.
                    7.1.2 Else, add the child node to the que to be explored.
                7.2 If cycles are not allowed:
                    Remove an edge if it results in an undirected cycle, continue.
                    Calculate node's features based on probability deviation.
                    7.2.1 If incremental improvement is True, make sure each node's features
                    are better than parent's features. If not. continue.
                    7.2.2 Else, relabel the child node and add to the que to be explored.
            8. Perform recursion.

        Args:
            n_childs (int, optional): Number of child nodes each parent can have, default is 5.
            allow_cycles (bool, optional): Whether undirected cycles are allowed or not, default is False.
            incremental_improvement (bool, optional): Whether each child node should have features better than the parent,
                                                      default is False.

        Returns:
            _type_: List[nx.DiGraph]
        """

        try:
            node=self.frontier_que.popleft()
            self.visited_nodes[node['label']]=True
        except IndexError:
            # reason: all nodes in the frontier_que have no children (nothing else to explore),
            # because each new child will result in a cycle formation.
            print(self.diagnosis)
            print('Explored without finding a solution.')
            return self.graph
        features=node['features']
        diff=1+cost_callable(features,self.target_features)

        # max depth is three
        try:
            # len of 2 means a depth of 1, for that reason we subtract
            tree_depth=len(shortest_path(self.graph,self.start_node['label'],node['label']))-1
            #NOTE: tree depth base case is mostly reached when we allow cycles.
            # without cycles, other base conditions are met much sooner.
            #NOTE: decrease depth to relax meeting the base-case criteria.
            if tree_depth>=3:
                # this means maximum 7 treatments
                # return without finding the goal
                # the last node;s features should at least be better than the start'nodes,
                # before returning the graph
                if diff<1+cost_callable(self.start_node['features'],self.target_features):
                    print(self.diagnosis)
                    print("Reached Maximum exploration depth with acceptable features.")
                    self.graph.graph['intermediary_goal_node']=node
                    return self.graph
            if tree_depth>=5:
                print(self.diagnosis)
                print('Explored without finding a solution.')
                return self.graph
        except Exception as e:
            print(e)
            return self.graph

        if diff<=self.threshold_value:
            #NOTE: if start node satisfies the threshhold, we return the node without start node.
            # This is similar to real-world scenarios. If target features are met, no more
            # exploration.
            self.graph.graph['goal_node']=node
            print(self.diagnosis)
            print('Found a goal state with desired features.')

            return self.graph

        similarity_scores=[]
        #stage 1
        for j,train_x in enumerate(self.model1.train_data):
            if node['label']==f"t1:{j}":
                # no self loops
                continue
            if torch.all(train_x==-1):
                # fixed penalty of value of 1024
                score=2**len(train_x)
            else:
                #+1 for the lowest cost to be 1 instead of 0
                score=1+cost_callable(features,train_x)
            similarity_scores.append((f"{node['label']}",f"t1:{j}",score))

        #stage2
        for j,train_x in enumerate(self.model2.feature_tensors):
            if node['label']==f"t2:{j}":
                continue
            if torch.all(train_x==-1):
                # fixed penalty of value of 1024
                score=2**len(train_x)
            else:
                #+1 for the lowest cost to be 1 instead of 0
                score=1+cost_callable(features,train_x)
            similarity_scores.append((f"{node['label']}",f"t2:{j}",score))

        #stage3
        for j,train_x in enumerate(self.model3.feature_tensors):
            if node['label']==f"t3:{j}":
                continue
            if torch.all(train_x==-1):
                # fixed penalty of value of 1024
                score=2**len(train_x)
            else:
                #+1 for the lowest cost to be 1 instead of 0
                score=1+cost_callable(features,train_x)

        top_closest=list(sorted(similarity_scores,key=lambda i:i[2])[:n_childs])
        for tuple_ in top_closest:
            t,int_node=map(lambda i:int(i),re.findall("\d+",tuple_[1]))
            if t==1:
                node_features=self.model1.feature_tensors[int_node]
            elif t==2:
                node_features=self.model2.feature_tensors[int_node]
            else:
                node_features=self.model3.feature_tensors[int_node]

            # if a node has been marked as a cycle node, do not explore again
            #NOTE self.cycle_nodes will only be updated during tree construction,
            # does not affect graph.
            if not self.cycle_nodes.get(tuple_[1],False):
                #without adding the actual edge, cycle cannot be checked
                self.graph.add_edge(tuple_[0],tuple_[1],weight=tuple_[2])
            else:
                continue

            if allow_cycles:
                if non_deterministic:
                    modify=np.random.choice([True,False],p=[0.7,0.3])
                    if modify:
                        probability_of_effectiveness=round(np.random.uniform(0.8,0.9),2)
                        change_percentage=1-probability_of_effectiveness
                        node_features=torch.Tensor(list(map(lambda i:i* \
                            np.random.uniform(1-change_percentage,1+change_percentage),node_features)))

                # if incremental_improvement:
                #     if not 1+cost_callable(node_features,self.target_features)<=diff:
                #         #backtrack
                #         self.graph.remove_node(tuple_[1])
                #         continue

                self.graph.nodes[tuple_[1]]['features']=node_features

                if non_deterministic and modify:
                    unique_id=''.join(random.choices(string.ascii_letters + string.digits, k=3))
                    self.graph.nodes[tuple_[1]]['label']=f"{tuple_[1]}:{probability_of_effectiveness}:{unique_id}"
                    rename={tuple_[1]:f"{tuple_[1]}:{probability_of_effectiveness}:{unique_id}"}
                    nx.relabel_nodes(self.graph,rename,copy=False)

                    #chances are extremely rare but still needed
                    if not self.visited_nodes.get(f"{tuple_[1]}:{probability_of_effectiveness}:{unique_id}",False):
                        self.visited_nodes[f"{tuple_[1]}:{probability_of_effectiveness}:{unique_id}"]=True
                        self.frontier_que.append(self.graph.nodes[f"{tuple_[1]}:{probability_of_effectiveness}:{unique_id}"])

                else:
                    self.graph.nodes[tuple_[1]]['label']=tuple_[1]
                    # add the node to the frontier only if it has not been previously visited
                    # otherwise the same node may be added to the queue more than once
                    # NOTE the final results do not change, but the logic with multiple addition is incorrect
                    if not self.visited_nodes.get(tuple_[1],False):
                        self.visited_nodes[tuple_[1]]=True
                        self.frontier_que.append(self.graph.nodes[tuple_[1]])
            else:
                try:
                    #NOTE orientation='ignore' is a stricter condition,
                    # it can spot both directed and undirected cycles,
                    # while orientation=None cannot spot undirected cycles.
                    nx.find_cycle(self.graph,orientation='ignore')
                except nx.exception.NetworkXNoCycle:
                    pass
                else:
                    #backtrack
                    # node should stay, edge should be removed
                    #NOTE with non-deterministic, the probability of a cycle decreases
                    self.graph.remove_edge(tuple_[0],tuple_[1])
                    # if a node results in a cycle, store, and check before adding an edge,
                    # so as the same node is not added-removed multiple times
                    self.cycle_nodes.update({tuple_[1]:True})
                    continue

                # we check it here only for incremental improvement
                if non_deterministic:
                    modify=np.random.choice([True,False],p=[0.7,0.3])
                    if modify:
                        probability_of_effectiveness=round(np.random.uniform(0.8,0.9),2)
                        change_percentage=1-probability_of_effectiveness
                        node_features=torch.Tensor(list(map(lambda i:i* \
                            np.random.uniform(1-change_percentage,1+change_percentage),node_features)))

                # if incremental_improvement:
                #     if not 1+cost_callable(node_features,self.target_features) <=diff:
                #         #backtrack
                #         self.graph.remove_node(tuple_[1])
                #         continue

                if non_deterministic and modify:
                    self.graph.nodes[tuple_[1]]['features']=node_features
                    unique_id=''.join(random.choices(string.ascii_letters + string.digits, k=3))
                    self.graph.nodes[tuple_[1]]['label']=f"{tuple_[1]}:{probability_of_effectiveness}:{unique_id}"
                    rename={tuple_[1]:f"{tuple_[1]}:{probability_of_effectiveness}:{unique_id}"}
                    nx.relabel_nodes(self.graph,rename,copy=False)
                    self.frontier_que.append(self.graph.nodes[f"{tuple_[1]}:{probability_of_effectiveness}:{unique_id}"])

                else:
                    self.graph.nodes[tuple_[1]]['features']=node_features
                    self.graph.nodes[tuple_[1]]['label']=tuple_[1]
                    self.frontier_que.append(self.graph.nodes[tuple_[1]])

        return self.create_relationships(cost_callable)


    def make_graphs_stage_independent(self,cost_method=3):
        """
        Perform create_relationship() fucntionality for each of the testing instances.
        Visualize, make sure there are no isolated nodes, then calculate all possible
        shortest paths with Djikstra's method. Without cycles, there should be exactly
        1 shortest path for each graph.

        Raises:
            Exception: Raise exception if there are isolated nodes in any graph.

        Returns:
            _type_: List[nx.Digraph]
        """
        self.counter={}
        self.make_models()
        good_indices=(self.model3.output==1).nonzero().flatten()
        self.model3.feature_tensors=self.model3.feature_tensors.index_select(0,good_indices)

        #NOTE test_dat_graphs is not used currenty
        test_data_graphs=[]
        self.target_features=torch.mean(self.model3.feature_tensors,dim=0)
        self.costs_thresholds=Costs_and_Thresholds(self.model3.feature_tensors)
        # sum of standard deviations across features

        threshold_callable=eval(f"self.costs_thresholds.threshold_v{cost_method}")
        cost_callable=eval(f"self.costs_thresholds.cost_v{cost_method}")
        self.threshold_value=1+threshold_callable()
        visualize=True
        for i,test_x in enumerate(self.model1.test_data):
            if torch.all(test_x==-1):
                continue
            # que for storing nodes yet to be explored
            self.frontier_que=deque()
            self.visited_nodes={}
            self.cycle_nodes={}
            self.graph=nx.DiGraph(goal_node=None,intermediary_goal_node=None)
            self.graph.add_node(f"start:{i}",features=test_x,label=f"start:{i}")
            self.start_node=self.graph.nodes[f"start:{i}"]
            #NOTE this is specifically for the cases when cycles are allowed
            self.frontier_que.append(self.start_node)
            test_data_graphs.append(self.create_relationships(cost_callable))
            logger.info(f"{self.diagnosis}\n {i}\n is_tree:{nx.is_tree(self.graph)}")
            if visualize:
                # visualize once per diagnosis
                try:
                    nx.find_cycle(self.graph)
                except nx.exception.NetworkXNoCycle:
                    # if no cycle
                    self.visualize_tree(self.graph)
                else:
                    # if cycle
                    plt.figure()
                    plt.title(self.diagnosis)
                    pos = nx.spring_layout(self.graph, k=10*1/np.sqrt(len(self.graph.nodes())), iterations=20)
                    nx.draw(self.graph,pos=pos,with_labels=True,node_color='#FFFFFF')
                    plt.show()

                visualize=False

            isolated_nodes=list(nx.isolates((self.graph)))
            if not isolated_nodes:
                print(f"{i}th iteration successful.")
            elif len(isolated_nodes)>1:
                #NOTE when there is only start node as isolated it is OK
                print(f"Isolated nodes found for {i}th instance.")
                logger.info(f"{self.diagnosis}\n {i}\n {isolated_nodes}")
            #NOTE we are mostly interest in the tree version as in graph version
            #we allow directed cycles.
            self.compute_path(self.graph)

        return self

    def compute_path(self,graph,method='independent'):
        # start_node number represent the testing instance_id
        start_node=list(graph.nodes)[0]
        end_node=None

        if method!='independent':
            end_node=sorted(list(graph.nodes))[-1]
            self.end_nodes=[end_node]
            #TODO random choice is a better approach however results in different counter values for each run
            # end_nodes=self.end_nodes=[node for node in list(graph.nodes) if "t3" in node]
            # end_node=random.choice(end_nodes)
            assert has_path(graph, start_node,end_node),\
                    "There is no path between start and end nodes."
            #NOTE for each graph there are multiple shortest paths if
            # weight is not provided (weight=1 by default).
            #NOTE even if multiple shortest paths exist, astar returns only 1.
            print(self.diagnosis)
            print('djikstra shortest paths.')
            djikstra_paths=list(nx.all_shortest_paths(graph,start_node,end_node,weight='weight'))
            print(djikstra_paths)

            print('astar shortest path.')
            astar_paths=list(nx.astar_path(graph,start_node,end_node,heuristic=self.astar_heuristic_v2))
            self.increment_counter(astar_paths)
            print(astar_paths)
            return djikstra_paths,astar_paths

        else:
            if graph.graph['intermediary_goal_node']:
                # both end_node obtaining methods are equivalent
                end_node=graph.graph['intermediary_goal_node']['label']
            elif graph.graph['goal_node']:
                # both end_node obtaining methods are equivalent
                end_node=graph.graph['goal_node']['label']

            if end_node:
                    assert has_path(graph, start_node,end_node),\
                        "There is no path between start and end nodes."
                    #NOTE this shows that without undirected cycles there is only 1 shortest path
                    # we thus search without performing a heuristic search.
                    #NOTE for the tree version, weight argument can be None as there is only 1 path
                    #NOTE for the graph version, if we search without weights (regarding path cost as 1),
                    # there can be multiple shortest paths
                    shortest_paths=list(nx.all_shortest_paths(graph,start_node,end_node,weight='weight'))
                    self.increment_counter(shortest_paths[0])
                    print(shortest_paths)

    #NOTE heurstics is admissible but imo cannot be proved, data plays a role
    def astar_heuristic_v1(self,node,end_node):
        """
        Args:
            node (str): node in the graph
            end_node (str): end node of the graph

        Returns:
            float:  heuristic value
        """
        node_stage,node_feature_id=list(map(int,re.findall("\d+",node)))
        end_node_stage,end_node_feature_id=list(map(int,re.findall("\d+",end_node)))

        #dynamically get variables
        node_features=getattr(self,f"model{node_stage}").feature_tensors[node_feature_id]
        end_node_features=getattr(self,f"model{end_node_stage}").feature_tensors[end_node_feature_id]
        #NOTE if the node is 1 step away from the goal, heuristic and actual cost will be the same,logical``
        heuristic_cost=1+sqrt(mean_squared_error(node_features,end_node_features))

        self.check_admissibility(node,end_node,heuristic_cost)

        return heuristic_cost

    def astar_heuristic_v2(self,node,end_node):
        """
        Admissible heuristic based on depth and distribution similarity.
        Provides the same results as djikstra.

        Args:
            node (str): node in the graph
            end_node (str): end node of the graph

        Returns:
            float: heuristic value
        """

        node_stage,node_feature_id=list(map(int,re.findall("\d+",node)))
        end_node_stage,end_node_feature_id=list(map(int,re.findall("\d+",end_node)))

        #dynamically get variables
        node_features=getattr(self,f"model{node_stage}").feature_tensors[node_feature_id]
        end_node_features=getattr(self,f"model{end_node_stage}").feature_tensors[end_node_feature_id]

        #lower p value, higher the cost
        weight=1-kstest(node_features,end_node_features).pvalue
        depth=end_node_stage-node_stage
        #NOTE, heuristic cost can be 0 because of depth (t3 node and there is no path
          # our proof is for the cases when there is path.) and when there is identical p value of 1
        # (in the proof noted that can be 0.)
        heuristic_cost=weight*depth

        self.check_admissibility(node,end_node,heuristic_cost)

        return heuristic_cost

    def check_admissibility(self,node,end_node,heuristic_cost):
        """
        Check if the heuristic is admissible for a given node.

        Args:
            node str: start node
            end_node str: end node
            heuristic_cost float: value of the heuristic function for the given node
        """

        try:
            shortest_path_length=nx.shortest_path_length(self.graph,node,end_node,weight='weight')
        except nx.exception.NetworkXNoPath:
            shortest_path_length=None


        if shortest_path_length:
            #NOTE we say that the heurisics is a good one, but is not admissble
            if not heuristic_cost<=shortest_path_length:
                print(f'not admissible for {node} and {end_node}')
            # print(f"heuristic_cost for ({node}):{heuristic_cost}\n \
            #         shortest_path_length for ({node}):{shortest_path_length}")
        else:
            #acceptable
            # print('No path between node and end_node, same depth.')
            pass

    def visualize_tree(self,graph):
        # pos = hierarchy_pos(graph,root)
        pos=topo_pos(graph)
        plt.title(f"{self.diagnosis}")
        nx.draw(graph, pos,with_labels=True,node_color='#FFFFFF')
        plt.show()

    def evaluate(self,n_clusters=3):
        labels_list=[]
        for data in [self.model1.feature_tensors,self.model2.feature_tensors,self.model3.feature_tensors]:
            kmeans=KMeans(n_clusters=n_clusters).fit(data)
            labels_list.append(kmeans.labels_.reshape(-1,1))

        #shift the labels
        labels_list[1]=labels_list[1]+5
        labels_list[2]=labels_list[2]+10
        labels_array=np.concatenate((*labels_list,),axis=1)

        test_items_count=len(self.astar_paths_list)
        clusters_from_path_algo=np.zeros((test_items_count,3))
        for i,l in enumerate(self.astar_paths_list):
            for j,node in enumerate(l):
                if "start" in node:
                    continue
                column,row= map(int,re.findall("\d+",node))
                #timesteps are ahead of indices by +1, so they need to be shift back
                clusters_from_path_algo[i][j-1]=int(labels_array[row][column-1])
        #NOTE 'ALTERED MENTAL STATUS' has no 3rd timestep nodes, so algo_clusters always remains 0, resulting in low accuracy
        actual_test_clusters=labels_array[-test_items_count:]
        logger.info("clusters-array from algorithm.")
        logger.info(clusters_from_path_algo)
        correct_percentage=round(100*sum((actual_test_clusters==clusters_from_path_algo).flatten())/np.prod(actual_test_clusters.shape),2)
        print(f"Correct cluster allocation percentage with {n_clusters} clusters is: {correct_percentage}%")
        logger.info(f"Correct cluster allocation % with {n_clusters} clusters is: {correct_percentage}%")
        return self

    def increment_counter(self,shortest_path_list):
        for node in shortest_path_list[1:-1]:
            self.counter[node]=self.counter.get(node,0)+1

    def return_counter(self):
        """
        Number of times each intermediary node appears as part of a solution path.
        """
        print("Counter dict.")
        self.counter_dict=dict(reversed(sorted(self.counter.items(),key=lambda dict_:dict_[1])))
        print(self.counter_dict)
        return self


    def visualize_counters(self):
        x,y=zip(*self.counter_dict.items())
        _, ax = plt.subplots()
        bars = ax.bar(x[:5], y[:5])
        ax.bar_label(bars)
        ax.set_xlabel('Nodes')
        ax.set_ylabel('Counts')
        ax.set_title('Betweenness Values for Nodes')
        plt.show()
        return self


    def visualize_cosine_similarities(self,ax):
         #NOTE works only with deterministic features
        if self.cosine_sim_dict:
            x,y=zip(*self.cosine_sim_dict.items())
            bars = ax.bar(x[:5], y[:5])
            ax.bar_label(bars,padding=-15)
            ax.set_title(f"{self.diagnosis}")


    def cosine_similarity_experiment1(self):
        """
        Calculate cosine similarity between intermediary node features and target features.
        """
        self.cosine_sim_dict={}
        end_node_ids=[]
        if hasattr(self,"end_nodes"):
            for end_node in self.end_nodes:
                end_node_ids.append(int(re.findall("\d+",end_node)[1]))

            target_feature_vectors=self.model3.feature_tensors_limited.index_select(0,torch.IntTensor(end_node_ids))
            target_feature_vector=torch.mean(target_feature_vectors,dim=0)

            for node in self.counter_dict:
                stage, row_id=map(int,re.findall("\d+",node))
                node_feature_vector=getattr(self,f"model{stage}").feature_tensors[row_id]
                cosine_similarity=self.calculate_cosine_similarity(node_feature_vector,target_feature_vector)
                self.cosine_sim_dict.update({node:round(cosine_similarity,4)})

        print('Cosine similarities.')
        print(self.cosine_sim_dict)
        return self


    def cosine_similarity_experiment2(self):
        self.cosine_sim_dict={}
        target_feature_vector=self.target_features
        for node in self.counter_dict:
            try:
                #NOTE works only with deterministic features
                stage, row_id=map(int,re.findall("\d+",node))
            except:
                self.cosine_sim_dict={}
                break
            node_feature_vector=getattr(self,f"model{stage}").feature_tensors[row_id]
            cosine_similarity=self.calculate_cosine_similarity(node_feature_vector,target_feature_vector)
            self.cosine_sim_dict.update({node:round(cosine_similarity,4)})

        print('Cosine similarities.')
        print(self.cosine_sim_dict)
        return self


    def calculate_cosine_similarity(self,tensor_a,tensor_b):
        return np.dot(tensor_a,tensor_b)/(norm(tensor_a)*norm(tensor_b))

    def __call__(self):
        self.make_graphs().return_counter().\
            cosine_similarity_experiment1().visualize_counters()

        # self.make_graphs_stage_independent().return_counter().\
        #     cosine_similarity_experiment2().visualize_counters()

        return self