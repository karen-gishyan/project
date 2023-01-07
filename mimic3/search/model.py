import re
import os
import sys
import torch
import networkx as nx
from networkx.algorithms.traversal import dfs_tree
from networkx.algorithms.shortest_paths import has_path,astar_path, shortest_path
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from utils import hierarchy_pos, topo_pos
from collections import deque
from math import sqrt

path=os.path.dirname(os.path.dirname(__file__))
sys.path.append(path)

dir_=os.path.dirname(__file__)
os.chdir(dir_)

from cluster.model import DistanceModel
from helpers import configure_logger

logger=configure_logger()


#NOTE: explore why nx has graph not a tree :low priority
#NOTE: there may be a need to construct multiple paths depending on different end_nodes.
class Graph:
    def __init__(self,diagnosis):
        self.diagnosis=diagnosis


    def make_models(self):
        models=[]
        for t in [1,2,3]:
            models.append(DistanceModel(diagnosis=self.diagnosis,timestep=t))

        self.model1,self.model2,self.model3=models
        self.model1.average_feature_time_series().train_test()
        self.model2.average_feature_time_series()
        self.model3.average_feature_time_series()

    def make_graphs(self,n_childs=5):
        """
        One graph is made for each testing instance.
        Each node has n_child nodes based on root mean_score_error similarity weight.

        Logic: Base Case:
                         If a node is at depth 5 or more, return the graph
                 Recursive Case:
                If the nodes features are within a threshold to target's features,
                add to goal_nodes list, continue execution
                Find closest childs for a given node across three stages.
                Calculate the cost from parent to child by:
                    1. How similar it's features are to parent
                    2. How similar it's features are to target
                    3. Weight this similarities equally.
                    By 90-10 probability, decide if the child will keep existing historical
                    features or they will be updated.
                    If updated,define an drug effectiveness measure rangin from (0.8 to 0.9)
                    For each child node:
                        Update child's features based on the effectiviness probability.
                    Add the child node and features to the node que.
                    Recursively call above functionality for each child node in the node que.
        """
        self.make_models()
        # third timestep should have only the good indices, only one of them is goal currently
        good_indices=(self.model3.output==1).nonzero().flatten()
        self.model3.feature_tensors=self.model3.feature_tensors.index_select(0,good_indices)

        # for empty values having 0 or -1 makes no difference as long as it is the
        # same for all train and test instances
        test_data_graphs=[]
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
                    score=1+sqrt(sqrt(mean_squared_error(test_x,train_x)))
                similarity_scores.append((f"start:{i}",f"t1:{j}",score))

            stage1_top_closest=list(sorted(similarity_scores,key=lambda i:i[2])[:n_childs])
            self.graph.add_weighted_edges_from(stage1_top_closest)

            for tuple_ in stage1_top_closest:
                similarity_scores=[]
                node=int(re.findall("\d+",tuple_[1])[1])
                # remove node's to prevent (node,node) score calculation
                stage2_data=torch.cat((self.model2.feature_tensors[:node],self.model2.feature_tensors[node+1:]))
                for j, train_x in enumerate(stage2_data):
                    if torch.all(train_x==-1):
                        score=2**len(train_x)
                    else:
                        score=1+sqrt(mean_squared_error(test_x,train_x))
                    similarity_scores.append((f"t1:{node}",f"t2:{j}",score))

                stage2_top_closest=list(sorted(similarity_scores,key=lambda i:i[2])[:n_childs])
                self.graph.add_weighted_edges_from(stage2_top_closest)
                for tuple_ in stage2_top_closest:
                    similarity_scores=[]
                    node=int(re.findall("\d+",tuple_[1])[1])
                    stage3_data=torch.cat((self.model3.feature_tensors[:node],self.model3.feature_tensors[node+1:]))
                    for j, train_x in enumerate(stage3_data):
                        if torch.all(train_x==-1):
                            score=2**len(train_x)
                        else:
                            score=1+sqrt(mean_squared_error(test_x,train_x))
                        similarity_scores.append((f"t2:{node}",f"t3:{j}",score))

                    stage3_top_closest=list(sorted(similarity_scores,key=lambda i:i[2])[:n_childs])
                    self.graph.add_weighted_edges_from(stage3_top_closest)

            test_data_graphs.append(self.graph)

        return test_data_graphs

    #NOTE: initially tested with 'no cycles' and 'incremental_improvement' all four combinations
    def create_relationships(self,n_childs=3,allow_cycles=False,incremental_improvement=True):
        try:
              node=self.frontier_que.popleft()
              self.explored_nodes.append(node['label'])
        except IndexError:
            # reason: all nodes in the frontier_que have no children (nothing else to explore),
            # because each new child will result in a cycle formation.
            print('Explored without finding a solution.')
            return self.graph
        features=node['features']
        diff=sqrt(mean_squared_error(features,self.target_features))

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
                if diff<sqrt(mean_squared_error(self.start_node['features'],self.target_features)):
                    print("Reached Maximum exploration depth with acceptable features.")
                    self.graph.graph['intermediary_goal_node']=node
                    return self.graph
            if tree_depth>=5:
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
            print('Found a goal state with desired features.')

            return self.graph
        else:
            similarity_scores=[]
            #stage 1
            for j,train_x in enumerate(self.model1.train_data):
                if torch.all(train_x==-1):
                    # fixed penalty of value of 1024
                    score=2**len(train_x)
                else:
                    #+1 for the lowest cost to be 1 instead of 0
                    score=1+sqrt(mean_squared_error(features,train_x))
                similarity_scores.append((f"{node['label']}",f"t1:{j}",score))

            #stage2
            for j,train_x in enumerate(self.model2.feature_tensors):
                if torch.all(train_x==-1):
                    # fixed penalty of value of 1024
                    score=2**len(train_x)
                else:
                    #+1 for the lowest cost to be 1 instead of 0
                    score=1+sqrt(mean_squared_error(features,train_x))
                similarity_scores.append((f"{node['label']}",f"t2:{j}",score))

            #stage3
            for j,train_x in enumerate(self.model3.feature_tensors):
                if torch.all(train_x==-1):
                    # fixed penalty of value of 1024
                    score=2**len(train_x)
                else:
                    #+1 for the lowest cost to be 1 instead of 0
                    score=1+sqrt(mean_squared_error(features,train_x))
            top_closest=list(sorted(similarity_scores,key=lambda i:i[2])[:n_childs])
            for i,tuple_ in enumerate(top_closest):
                t,int_node=map(lambda i:int(i),re.findall("\d+",tuple_[1]))
                if t==1:
                    features=self.model1.feature_tensors[int_node]
                    close_target_score=sqrt(mean_squared_error(features,self.target_features))
                elif t==2:
                    features=self.model2.feature_tensors[int_node]
                    close_target_score=sqrt(mean_squared_error(features,self.target_features))
                else:
                    features=self.model3.feature_tensors[int_node]
                    close_target_score=sqrt(mean_squared_error(features,self.target_features))

                # convert to a list to be able to assign and convert back to tuple
                top_closest[i]=list(top_closest[i])
                # 50% of the cost is how close it is to parent (child to parent),
                # the other 50% how close it is to target (child to target).
                top_closest[i][2]=0.5*tuple_[2]+0.5*close_target_score
                top_closest[i]=tuple(top_closest[i])

                # self.graph.add_weighted_edges_from(top_closest)

                # we add iteratively and not with bulk to allow removing an edge
                # if a cycle is formed
                child_nodes=[]
                for i in top_closest:
                    self.graph.add_edge(i[0],i[1],weight=i[2])
                    if not allow_cycles:
                        try:
                            # even an undirected cycle is not allowed
                            nx.find_cycle(self.graph,orientation='ignore')
                        except nx.exception.NetworkXNoCycle:
                            # if no cycle, add to child_nodes
                            child_nodes.append(i[1])
                            pass
                        else:
                            self.graph.remove_edge(i[0],i[1])
                    else:
                        child_nodes.append(i[1])


            # not efficient to create then again loop through each node
            for i,key in enumerate(child_nodes):
                # 90 % of the cases features are the child features
                # child_nodes=self.graph.nodes
                if np.random.choice([True,False],p=[0.9,0.1]):
                    self.graph.nodes[key]['features']=features

                    #NOTE: this label may not be used
                    self.graph.nodes[key]['label']=key
                    if not key in self.label_que:
                        self.label_que.append(key)
                        self.frontier_que.append(self.graph.nodes[key])
                # 10% of the cases we assign an effectiveness measure, and obtain
                # new features of the node based on this logic.
                else:
                    probability_of_effectiveness=round(np.random.uniform(0.8,0.9),2)
                    change_percentage=1-probability_of_effectiveness
                    # decide how much each feature will change
                    # e.g if probability_of_effectiveness = 0.8, each feature will deviate
                    # by +- 20%.
                    features=torch.Tensor(list(map(lambda i:i* \
                        np.random.uniform(1-change_percentage,1+change_percentage),features)))
                    #NOTE: with cycles, same node features may be modified multiple times
                    self.graph.nodes[key]['features']=features
                    if incremental_improvement:
                        # NOTE: this is a strong logical change based on incremental state
                        # improvement. If the child features are not at least as similar as
                        # the patients one, then do not add the child node.
                        if not sqrt(mean_squared_error(features,self.target_features))<=diff:
                            if not allow_cycles:
                                #NOTE: do not remove the node with no cycles
                                # for this reason:
                                # node A is good, keep add to frontier
                                # later encounter, node A is bad with modified features,
                                # remove node A, but frontier already contains node A,
                                # error will be raised.
                                #TODO think if this can be removed for the case when no cycles
                                self.graph.remove_node(key)
                            continue
                    self.graph.nodes[key]['label']=f"{key}:{probability_of_effectiveness}"
                    rename={key:f"{key}:{probability_of_effectiveness}"}
                    nx.relabel_nodes(self.graph,rename,copy=False)
                    if not key in self.label_que:
                        self.label_que.append(key)
                        self.frontier_que.append(self.graph.nodes[f"{key}:{probability_of_effectiveness}"])

            return self.create_relationships()


    def make_graphs_stage_independent(self):
        self.make_models()
        good_indices=(self.model3.output==1).nonzero().flatten()
        self.model3.feature_tensors=self.model3.feature_tensors.index_select(0,good_indices)

        # equivalent to random selection no specific logic
        self.target_features=self.model3.feature_tensors[0]
        test_data_graphs=[]
        self.threshold_value=20
        vis_for_diagnosis=True
        #NOTE changing recursion limit stops the program but does not raise an error
        for i,test_x in enumerate(self.model1.test_data):
            if torch.all(test_x==-1):
                continue
            # que for storing nodes yet to be explored
            self.frontier_que=deque()
            self.label_que=deque()
            self.explored_nodes=[]
            self.graph=nx.DiGraph(goal_node=None,intermediary_goal_node=None)
            self.graph.add_node(f"start:{i}",features=test_x,label=f"start:{i}")
            self.start_node=self.graph.nodes[f"start:{i}"]
            self.frontier_que.append(self.start_node)
            #NOTE this is specifically for the cases when cycles are allowed
            self.label_que.append(f"start:{i}")
            test_data_graphs.append(self.create_relationships())
            if vis_for_diagnosis:
                # visualize once per diagnosis
                try:
                    nx.find_cycle(self.graph)
                except nx.exception.NetworkXNoCycle:
                    # if no cycle
                    self.visualize_tree(self.graph,self.start_node)
                else:
                    # if cycle
                    nx.draw(self.graph,with_labels=True)
                    plt.show()
                vis_for_diagnosis=False

            print(f"{i} successful.")

        return test_data_graphs


    def set_start_and_end(self,graph,method='independent'):
        # start_node number represent the testing instance_id
        start_node=list(graph.nodes)[0]

        end_node=None
        if method!='independent':
        #NOTE: end_node logic may need to be more complex
            end_node=sorted(list(graph.nodes))[-1]
        else:
            if graph.graph['intermediary_goal_node']:
                end_node=sorted(list(graph.nodes))[-1]
            elif graph.graph['goal_node']:
                end_node=graph.graph['goal_node']['label']

        if end_node:
                assert has_path(graph, start_node,end_node),\
                    "There is no path between start and end nodes."
        return start_node,end_node

    def depth_first_search(self,graph):
        """
        Useful for checking tree connectivity.
        """
        print("Depth First Search.")
        tree=list(dfs_tree(graph))
        print(tree)

    def astar_path(self,graph,start_node,end_node,**kwargs):
        path=list(astar_path(graph,start_node,end_node,**kwargs))

        t=f"'astar path' method :{path}"
        print(self.diagnosis,f"\n{t}")
        heuristic=kwargs.get('heuristic')
        if heuristic:
            logger.info(f"With heuristic \n{t}")
        else:
            logger.info(t)

        return path

    def shortest_path(self,graph,start_node,end_node,**kwargs):
        """
        method is 'dijkstra or 'bellman-ford'.
        """

        method=kwargs.get('method') if kwargs.get('method') else 'dijkstra'
        # kwargs are fixed, any non-existent kwarg will raise an error
        path=list(shortest_path(graph, start_node,end_node,**kwargs))
        print(self.diagnosis)
        print(f"{method} 'shortest path' method: {path}")
        return path

    def astar_heuristic(self,start_node,end_node):
        """
        Optimal heuristic between a node at depth i and a final depth j is (j-i)
        for f(n)=h(n)+g(n), assuming f(n) is a distance based measure.
        """
        #TODO path stays the same with this heuristic
        # start_node is not used as part of the heuristics calculation,
        # otherwise would be the equal to 'end_node_depth'
        start_node_depth=int(re.findall("\d+",start_node)[0])
        end_node_depth=int(re.findall("\d+",end_node)[0])
        return 1+(end_node_depth-start_node_depth)

    def visualize_tree(self,graph,root):
        # pos = hierarchy_pos(graph,root)
        pos=topo_pos(graph)
        plt.title(f"{self.diagnosis}")
        nx.draw(graph, pos,with_labels=True)
        plt.show()

    def straight_line_heuristic(self):
        """
        The mse between each node features and target node features will
        be an admissible heuristic. Analogous to a straight line heuristic.
        """
        pass

    def subvector_heuristic(self):
        """
        Calculate and store the subvector mse (e.g. with 5 features).
        For each node do a search(astar, shortest) to the end goal using the subvector mse,
        obtain store this value the value.
        Use this value as a heuristic value between each node and the goal state.
        """
        pass

    def __call__(self):
        # test_graphs=self.make_graphs()
        #NOTE for independent graph construction, searching after construction
        # may not be needed.
        test_graphs2=self.make_graphs_stage_independent()
        astar_paths=[]
        shortest_paths_dijkstra=[]
        shortest_paths_bellman_ford=[]
        for graph in test_graphs2:
            # start_node,end_node=self.set_start_and_end(graph,method='base')
            start_node,end_node=self.set_start_and_end(graph)
            if end_node:
                astar_paths.append(self.astar_path(graph,start_node,end_node,heuristic=self.astar_heuristic))
            # shortest_paths_dijkstra.append(self.shortest_path(graph,start_node,end_node))
            # shortest_paths_bellman_ford.append(self.shortest_path(graph,start_node,end_node,method='bellman-ford'))

            ## only the first patient's graph for each diagnosis
            ## ! if cycle exists, tree visualization will not work
            # if len(astar_paths)==1:
            #     self.visualize_tree(graph,start_node)
