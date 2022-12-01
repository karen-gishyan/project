
import re
import os
import sys
import torch
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.algorithms.traversal import dfs_tree, bfs_tree
from networkx.algorithms.shortest_paths import astar_path
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
        self.graph=nx.DiGraph()

    def make_graph(self,n_childs=5):
        models=[]
        for t in [1,2,3]:
            models.append(DistanceModel(diagnosis=self.diagnosis,timestep=t))

        model1,model2,model3=models
        model1.average_feature_time_series().train_test()
        model2.average_feature_time_series()
        model3.average_feature_time_series()

        #TODO fix -1 issue
        for i,test_x in enumerate(model1.test_data):
            similarity_scores=[]
            for j,train_x in enumerate(model1.train_data):
                score=mean_squared_error(test_x,train_x)
                similarity_scores.append((f"start:{i}",f"t1:{j}",score))

            stage1_top_closest=list(sorted(similarity_scores,key=lambda i:i[2])[:n_childs])
            self.graph.add_weighted_edges_from(stage1_top_closest)

            for tuple_ in stage1_top_closest:
                similarity_scores=[]
                node=int(re.findall("\d+",tuple_[1])[1])
                stage2_data=torch.cat((model2.feature_tensors[:node],model2.feature_tensors[node+1:]))
                for j, train_x in enumerate(stage2_data):
                    score=mean_squared_error(model2.feature_tensors[node],train_x)
                    similarity_scores.append((f"t1:{node}",f"t2:{j}",score))

                stage2_top_closest=list(sorted(similarity_scores,key=lambda i:i[2])[:n_childs])
                self.graph.add_weighted_edges_from(stage1_top_closest)
                for tuple_ in stage2_top_closest:
                    similarity_scores=[]
                    node=int(re.findall("\d+",tuple_[1])[1])
                    stage3_data=torch.cat((model3.feature_tensors[:node],model3.feature_tensors[node+1:]))
                    for j, train_x in enumerate(stage3_data):
                        score=mean_squared_error(model3.feature_tensors[node],train_x)
                        similarity_scores.append((f"t2:{node}",f"t3:{j}",score))

                    stage3_top_closest=list(sorted(similarity_scores,key=lambda i:i[2])[:n_childs])
                    self.graph.add_weighted_edges_from(stage3_top_closest)


            #TODO some weights are 0 but are not identical
            # position=graphviz_layout(self.graph, prog='dot')
            # nx.draw(self.graph,with_labels=True)
            # plt.show()

            break
        return self

    #TODO final path should include 3 levels
    def depth_first_search(self):
        print("Depth First Search.")
        print(list(dfs_tree(self.graph)))
        return self

    def breadth_first_search(self):
        start_node=list(self.graph.nodes)[0]
        print("Breadth First Search")
        print(list(bfs_tree(self.graph,start_node,depth_limit=3)))
        return self

    def astar_search(self):
        #TODO not reachable issue
        start_node=list(self.graph.nodes)[0]
        end_node=list(self.graph.nodes)[-5] # sample
        print("AStar Search.")
        print(list(astar_path(self.graph,start_node,end_node)))
        return self


graph=Graph("SEPSIS").make_graph().breadth_first_search()
