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
            visualize=True
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
            if visualize:
                self.visualize_tree(self.graph)
                visualize=False
            self.compute_path(self.graph,method='ordinary')

        return test_data_graphs

    #NOTE: tested with 'no cycles' and 'incremental_improvement' all four combinations
    #BUG in rare cases, in one of the diagnosis there are disconnected nodes
    def create_relationships(self,n_childs=5,allow_cycles=False,incremental_improvement=False):
        """
        For each testing instance with admission features, the goal is
        to construct a graph which will store the path of the optimal treatment.
        A treatment is optimal if it results in features which are withing certain
        distance away from the target features.
        The graph construction starts from the start node with initial features,
        which is added to the frontier que (que storing the nodes to be explored).
        We recursively call create_relationships(), and each call explores a single
        child node to see whether the node's features satsify one of the base
        case criterias. If they do satisfy, the graph is returned, else the algorithm
        continues (continuation means adding child nodes to the que based on similarity to be
        further explored). Each edge represents a treatment, and the node's features
        are the result of the treatment.

        Base cases:
            1. (No solution case): The frontier que is empty as there are no more child nodes
            to be explored. This case is reached frequently when we do not allow cycles, meaning
            each new node wil result in a cycle, thus is not added to the que. Here the graph
            is returned, and we fail to output a path where the end node has desired features.
            2. (No solution case): If the tree reaches a depth of 5 (five treatments) and still
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
            7. From the data available for three stages, obtain three most similar node's
                as child nodes, add to child nodes list.
                7.1 If cycles not allowed, and if a given child node results in a cycle (undirected),
                remove from the child nodes list.
            8. Add edges between node and child nodes from child nodes list.
            9. # allow cycles can be True or False, incremental_improvement can be True or False.
               # we explore 4 combinations.
            10. With probability drawn from a normal distribution, decide if the child nodes
                features take existing features (90% probability) or change (10% probability).
                10.1 If they do not change:
                    10.1.1: If incremental improvement is True, check if the child node's features
                    are better than parent's features.
                        If not, remove the child node and continue to the next node.
                    10.1.2: Assign node's features and add the node to the frontier que.
                10.2 If they change:
                    10.2.1:  Based on probability (between 80% and 90%) decide how similar they
                    will be to the original child node's features.
                    10.2.2: Calculate new features.
                    10.2.3: Repeat the process defined in step 10.1.1
                    10.2.4: Assign node's features and add the node to the frontier que.
                    10.2.5: Relabel the node based on the probability of change.

            11. Perform recursion.
        """

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
                        #FIXME no path error when we allow undirected cycles
                        # orientation='ignore' means even the edges which result
                        # in undirected cycles are not added
                        nx.find_cycle(self.graph,orientation='ignore')
                    except nx.exception.NetworkXNoCycle:
                        # if no cycle, add to child_nodes
                        child_nodes.append(i[1])
                        pass
                    else:
                        #NOTE i[1] should stay, only edge should be removed as it is
                        # the node to which we check for the cycle
                        self.graph.remove_edge(i[0],i[1])
                        #NOTE for debugging
                        # self.visualize_tree(self.graph)
                else:
                    child_nodes.append(i[1])

        for i,key in enumerate(child_nodes):
            # 90 % of the cases features are the child features
            if np.random.choice([True,False],p=[0.9,0.1]):
                if not allow_cycles:
                    if incremental_improvement:
                        if not sqrt(mean_squared_error(features,self.target_features))<=diff:
                            #NOTE node instead of edge should be removed
                            self.graph.remove_node(key)
                            continue

                    self.graph.nodes[key]['features']=features
                    self.graph.nodes[key]['label']=key
                    self.frontier_que.append(self.graph.nodes[key])

                else:
                    if incremental_improvement:
                        # NOTE: label_que contains those node labels which for sure have features.
                        if key in self.label_que:
                            # if the current features are not better than previously explored featured,pass
                            if not sqrt(mean_squared_error(features,self.target_features))<=\
                                sqrt(mean_squared_error(self.graph.nodes[key]['features'],self.target_features)):
                                continue

                    # if not incremental, node features can change depending on the most recent parent node
                    self.graph.nodes[key]['features']=features
                    self.graph.nodes[key]['label']=key

                    if not key in self.label_que:
                        self.label_que.append(key)
                        self.frontier_que.append(self.graph.nodes[key])

            else:
                probability_of_effectiveness=round(np.random.uniform(0.8,0.9),2)
                change_percentage=1-probability_of_effectiveness
                # decide how much each feature will change
                # e.g if probability_of_effectiveness = 0.8, each feature will deviate
                # by +- 20%.
                features=torch.Tensor(list(map(lambda i:i* \
                    np.random.uniform(1-change_percentage,1+change_percentage),features)))
                #NOTE: with cycles, same node features may be modified multiple times
                if not allow_cycles:
                    if incremental_improvement:
                        if not sqrt(mean_squared_error(features,self.target_features))<=diff:
                            self.graph.remove_node(key)
                            continue

                    self.graph.nodes[key]['features']=features
                    self.graph.nodes[key]['label']=f"{key}:{probability_of_effectiveness}"
                    rename={key:f"{key}:{probability_of_effectiveness}"}
                    nx.relabel_nodes(self.graph,rename,copy=False)
                    self.frontier_que.append(self.graph.nodes[f"{key}:{probability_of_effectiveness}"])

                else:
                    if incremental_improvement:
                        # if has been explored
                        if key in self.label_que:
                            if not sqrt(mean_squared_error(features,self.target_features))\
                                <=sqrt(mean_squared_error(self.graph.nodes[key]['features'],self.target_features)):
                                continue

                    self.graph.nodes[key]['features']=features
                    self.graph.nodes[key]['label']=f"{key}:{probability_of_effectiveness}"
                    rename={key:f"{key}:{probability_of_effectiveness}"}
                    nx.relabel_nodes(self.graph,rename,copy=False)
                    try:
                        #NOTE: this is related to relabeling
                        # (t1:10) is in the label que, becomes (t1:10:0.85)
                        # then in the incremental improvement block, (t1:10) will exist
                        # without features, as it has been relabeled.
                        #TODO: worth thinking about improvement
                        self.label_que.remove(key)
                    except:
                        pass
                    if not f"{key}:{probability_of_effectiveness}" in self.label_que:
                        self.label_que.append(f"{key}:{probability_of_effectiveness}")
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
        visualize=True
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
            if visualize:
                # visualize once per diagnosis
                try:
                    nx.find_cycle(self.graph)
                except nx.exception.NetworkXNoCycle:
                    # if no cycle
                    self.visualize_tree(self.graph)
                else:
                    # if cycle
                    nx.draw(self.graph,with_labels=True)
                    plt.show()

                visualize=False

            isolated_nodes=self.check_isolates(self.graph)
            if not isolated_nodes:
                print(f"{i}th iteration successful.")
            else:
                raise Exception(f"Isolate nodes found for {i}th instance.")
            self.compute_path(self.graph)

        return test_data_graphs

    def compute_path(self,graph,method='independent'):
        # start_node number represent the testing instance_id
        start_node=list(graph.nodes)[0]
        end_node=None

        if method!='independent':
            end_node=sorted(list(graph.nodes))[-1]
            assert has_path(graph, start_node,end_node),\
                    "There is no path between start and end nodes."
            self.astar_path(graph,start_node,end_node)

        else:
            if graph.graph['intermediary_goal_node']:
                # both end_node obtaining methods are equivalent
                end_node=graph.graph['intermediary_goal_node']['label']
                # end_node=self.explored_nodes[-1]
            elif graph.graph['goal_node']:
                # both end_node obtaining methods are equivalent
                end_node=graph.graph['goal_node']['label']
                # end_node=self.explored_nodes[-1]

            if end_node:
                    assert has_path(graph, start_node,end_node),\
                        "There is no path between start and end nodes."
                    #NOTE this shows that without undirected cycles there is only 1 shortest path
                    # we thus search with default='djikstra', without performing a heuristic search.
                    print(list(nx.all_shortest_paths(graph,start_node,end_node)))


    def check_isolates(self,graph):
        print('Isolated nodes')
        print(list(nx.isolates(graph)))

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

    def visualize_tree(self,graph):
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
        # self.make_graphs()
        self.make_graphs_stage_independent()
