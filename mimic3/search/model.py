import re
import os
import sys
import torch
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.algorithms.traversal import dfs_tree
from networkx.algorithms.shortest_paths import has_path,astar_path, shortest_path
from sklearn.metrics import mean_squared_error, median_absolute_error
import matplotlib.pyplot as plt


path=os.path.dirname(os.path.dirname(__file__))
sys.path.append(path)

dir_=os.path.dirname(__file__)
os.chdir(dir_)

from cluster.model import DistanceModel


class Graph:
    def __init__(self,diagnosis):
        self.diagnosis=diagnosis

    def make_graphs(self,n_childs=5):
        """
        One graph is made for each testing instance.
        """
        models=[]
        for t in [1,2,3]:
            models.append(DistanceModel(diagnosis=self.diagnosis,timestep=t))

        model1,model2,model3=models
        model1.average_feature_time_series().train_test()
        model2.average_feature_time_series()
        model3.average_feature_time_series()

        self.n_test=len(model1.test_data)
        #TODO fix -1 issue
        test_data_graphs=[]
        for i,test_x in enumerate(model1.test_data):
            # one graph for each testing instance
            self.graph=nx.DiGraph()
            similarity_scores=[]
            for j,train_x in enumerate(model1.train_data):
                score=mean_squared_error(test_x,train_x)
                similarity_scores.append((f"start:{i}",f"t1:{j}",score))

            stage1_top_closest=list(sorted(similarity_scores,key=lambda i:i[2])[:n_childs])
            self.graph.add_weighted_edges_from(stage1_top_closest)

            for tuple_ in stage1_top_closest:
                similarity_scores=[]
                node=int(re.findall("\d+",tuple_[1])[1])
                # remove node's to prevent (node,node) score calculation
                stage2_data=torch.cat((model2.feature_tensors[:node],model2.feature_tensors[node+1:]))
                for j, train_x in enumerate(stage2_data):
                    score=mean_squared_error(model2.feature_tensors[node],train_x)
                    similarity_scores.append((f"t1:{node}",f"t2:{j}",score))

                stage2_top_closest=list(sorted(similarity_scores,key=lambda i:i[2])[:n_childs])
                self.graph.add_weighted_edges_from(stage2_top_closest)
                for tuple_ in stage2_top_closest:
                    similarity_scores=[]
                    node=int(re.findall("\d+",tuple_[1])[1])
                    stage3_data=torch.cat((model3.feature_tensors[:node],model3.feature_tensors[node+1:]))
                    for j, train_x in enumerate(stage3_data):
                        score=mean_squared_error(model3.feature_tensors[node],train_x)
                        similarity_scores.append((f"t2:{node}",f"t3:{j}",score))

                    stage3_top_closest=list(sorted(similarity_scores,key=lambda i:i[2])[:n_childs])
                    self.graph.add_weighted_edges_from(stage3_top_closest)

            test_data_graphs.append(self.graph)
            #TODO some weights are 0 but are not identical

        return test_data_graphs

    def set_start_and_end(self,graph):
        # start_node number represent the testing instance_id
        start_node=list(graph.nodes)[0]
        #TODO there should be an end_node logic
        end_node=sorted(list(graph.nodes))[-1]
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
        print(self.diagnosis)
        print(f"'astar path' method :{path}")
        return path

    def shortest_path(self,graph,start_node,end_node,**kwargs):
        """
        method is 'dijkstra or 'bellman-ford'.
        """
        path=list(shortest_path(graph, start_node,end_node,**kwargs))
        print(self.diagnosis)
        print(f"'shortest path' method: {path}")
        return path

    def visualize_tree(self):
        raise NotImplementedError

    def __call__(self):
        test_graphs=self.make_graphs()
        astar_paths=[]
        shortest_paths=[]
        for graph in test_graphs:
            start_node,end_node=self.set_start_and_end(graph)
            astar_paths.append(self.astar_path(graph,start_node,end_node))
            # shortest_paths.append(self.shortest_path(graph,start_node,end_node,method='bellman-ford'))

        assertion_test="Number of paths does not match to the number of testing data instances."
        if astar_paths:
            assert len(astar_paths)==self.n_test,assertion_test
        if shortest_paths:
             assert len(shortest_paths)==self.n_test,assertion_test
