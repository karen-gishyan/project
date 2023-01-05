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


path=os.path.dirname(os.path.dirname(__file__))
sys.path.append(path)

dir_=os.path.dirname(__file__)
os.chdir(dir_)

from cluster.model import DistanceModel
from helpers import configure_logger

logger=configure_logger()


#TODO explore why nx has graph not a tree :low priority
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
		Each node has n_child nodes based on mean_score_error similarity weight.

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
		TODO: add cycle logic
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
			similarity_scores=[]
			for j,train_x in enumerate(self.model1.train_data):
				#score is 0 when two vectors are exactly the same (e.g.all values for each vector is -1).
				if torch.all(train_x==-1):
					# fixed penalty of value of 1024
					score=2**len(train_x)
				else:
					#+1 for the lowest cost to be 1 instead of 0
					#NOTE: may need to change to rmse
					score=1+mean_squared_error(test_x,train_x)
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
						score=1+mean_squared_error(test_x,train_x)
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
							score=1+mean_squared_error(test_x,train_x)
						similarity_scores.append((f"t2:{node}",f"t3:{j}",score))

					stage3_top_closest=list(sorted(similarity_scores,key=lambda i:i[2])[:n_childs])
					self.graph.add_weighted_edges_from(stage3_top_closest)

			test_data_graphs.append(self.graph)

		return test_data_graphs

	#TODO divide testing into two stages (allow_cycles=True and False), in both cases should work.
	def create_relationships(self,n_childs=3,allow_cycles=False):
		#BUG (not-high priority), popleft() causes an infinite recursion
		node=self.node_que.pop()
		features=node['features']
		diff=mean_squared_error(features,self.target_features)

		# max depth is three
		try:
			# len of 2 means a depth of 1, for that reason we subtract
			tree_depth=len(shortest_path(self.graph,self.start_node['label'],node['label']))-1
			# FIXME: think how to prevent infinite recursion (skip graph, modify logic) when
            # allow cycels is False, some cases there is no issue, while in some there is
			if tree_depth>=3:
				return self.graph
		except:
			pass

		if diff<-self.threshold_value:
			#FIXME: base case may be reached without goal nodes.
   			#FIXME 	Threshhold may need to be incorporated in the base-case-logic
      		# (incorporation will increase the chance of maximum recursion error)
			self.graph.graph['goal_nodes']=self.graph['goal_nodes'].append(node)
		else:
			similarity_scores=[]
			#stage 1
			for j,train_x in enumerate(self.model1.train_data):
				if torch.all(train_x==-1):
					# fixed penalty of value of 1024
					score=2**len(train_x)
				else:
					#+1 for the lowest cost to be 1 instead of 0
					score=1+mean_squared_error(features,train_x)
				similarity_scores.append((f"{node['label']}",f"t1:{j}",score))

			#stage2
			for j,train_x in enumerate(self.model2.feature_tensors):
				if torch.all(train_x==-1):
					# fixed penalty of value of 1024
					score=2**len(train_x)
				else:
					#+1 for the lowest cost to be 1 instead of 0
					score=1+mean_squared_error(features,train_x)
				similarity_scores.append((f"{node['label']}",f"t2:{j}",score))

			#stage3
			for j,train_x in enumerate(self.model3.feature_tensors):
				if torch.all(train_x==-1):
					# fixed penalty of value of 1024
					score=2**len(train_x)
				else:
					#+1 for the lowest cost to be 1 instead of 0
					score=1+mean_squared_error(features,train_x)
				similarity_scores.append((f"{node['label']}",f"t3:{j}",score))

			top_closest=list(sorted(similarity_scores,key=lambda i:i[2])[:n_childs])
			for i,tuple_ in enumerate(top_closest):
				t,int_node=map(lambda i:int(i),re.findall("\d+",tuple_[1]))
				if t==1:
					features=self.model1.feature_tensors[int_node]
					close_target_score=mean_squared_error(features,self.target_features)
				elif t==2:
					features=self.model2.feature_tensors[int_node]
					close_target_score=mean_squared_error(features,self.target_features)
				else:
					features=self.model3.feature_tensors[int_node]
					close_target_score=mean_squared_error(features,self.target_features)

				# convert to a list to be able to assign and convert back to tuple
				top_closest[i]=list(top_closest[i])
				# 50% of the cost is how close it is to parent (child to parent),
                # the other 50% how close it is to target (child to target).
				top_closest[i][2]=0.5*tuple_[2]+0.5*close_target_score
				top_closest[i]=tuple(top_closest[i])
				child_nodes=[i[1] for i in top_closest]
				# self.graph.add_weighted_edges_from(top_closest)

				# we add iteratively and not with bulk to allow removing an edge
				# if a cycle is formed
				for i in top_closest:
					self.graph.add_edge(i[0],i[1],weight=i[2])
					if not allow_cycles:
						try:
							# even an undirected cycle is not allowed
							nx.find_cycle(self.graph,orientation='ignore')
						except nx.exception.NetworkXNoCycle:
							pass
						else:
							self.graph.remove_edge(i[0],i[1])

			# not efficient to create then again loop through each node
			for i,key in enumerate(child_nodes):
				# 90 % of the cases features are the child features
				# child_nodes=self.graph.nodes
				if np.random.choice([True,False],p=[0.9,0.1]):
					self.graph.nodes[key]['features']=features

					#NOTE: this label may not be used
					self.graph.nodes[key]['label']=key
					self.node_que.append(self.graph.nodes[key])
				# 10% of the cases we assign an effectiveness measure, and obtain
				# new features of the node based on this logic.
				else:
					probability_of_effectiveness=round(np.random.uniform(0.8,0.9),2)
					#FIXME the logic needs to change the feature by probability_of_effectiveness
                    # standard deviation, otherwise currently will reduce each feature by probabilitiy_of_effectiveness.
					self.graph.nodes[key]['features']=probability_of_effectiveness*features
					self.graph.nodes[key]['label']=f"{key}:{probability_of_effectiveness}"
					rename={key:f"{key}:{probability_of_effectiveness}"}
					nx.relabel_nodes(self.graph,rename,copy=False)
					self.node_que.append(self.graph.nodes[f"{key}:{probability_of_effectiveness}"])

			return self.create_relationships()

	def make_graphs_stage_independent(self):
		self.make_models()
		good_indices=(self.model3.output==1).nonzero().flatten()
		self.model3.feature_tensors=self.model3.feature_tensors.index_select(0,good_indices)

		# equivalent to random selection no specific logic
		self.target_features=self.model3.feature_tensors[0]
		test_data_graphs=[]
		self.threshold_value=100
		for i,test_x in enumerate(self.model1.test_data):
			if torch.all(test_x==-1):
				continue
			self.node_que=deque()
			#FIXME goal_nodes is empty in the end
			self.graph=nx.DiGraph(goal_nodes=[])
			self.graph.add_node(f"start:{i}",features=test_x,label=f"start:{i}")
			self.start_node=self.graph.nodes[f"start:{i}"]
			self.node_que.append(self.start_node)
			#NOTE currently works with cycles
			test_data_graphs.append(self.create_relationships())
			if i==0:
				nx.draw(self.graph,with_labels=True)
				plt.show()
			try:
				cycle=nx.find_cycle(self.graph,orientation='ignore')
			except nx.exception.NetworkXNoCycle:
				pass
			else:
				print(f'Cycle:{list(cycle)}')
			print(f"{i} successful.")

		return test_data_graphs


	def set_start_and_end(self,graph):
		# start_node number represent the testing instance_id
		#TODO end_node logic may need to be more complex
		start_node=list(graph.nodes)[0]
		# FIXME: make_graphs_stage_independent(), end_node should be chosen from goal_nodes.
		# end_node=sorted(list(graph.goal_nodes))[-1]
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
		test_graphs2=self.make_graphs_stage_independent()
		astar_paths=[]
		shortest_paths_dijkstra=[]
		shortest_paths_bellman_ford=[]
		for graph in test_graphs2:
			start_node,end_node=self.set_start_and_end(graph)
			astar_paths.append(self.astar_path(graph,start_node,end_node,heuristic=self.astar_heuristic))
			# shortest_paths_dijkstra.append(self.shortest_path(graph,start_node,end_node))
			# shortest_paths_bellman_ford.append(self.shortest_path(graph,start_node,end_node,method='bellman-ford'))

			## only the first patient's graph for each diagnosis
			## ! if cycle exists, tree visualization will not work
			# if len(astar_paths)==1:
			#     self.visualize_tree(graph,start_node)
