import networkx as nx
import random
import torch
from sklearn.metrics import mean_squared_error
from math import sqrt


# this is an external visualization function.
# for reference please see: https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3#:~:text=scroll%20down%20a%20bit%20to%20see%20what%20kind%20of%20output%20the%20code%20produces%5D
def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    # if not nx.is_tree(G):
    #     raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children)!=0:
            dx = width/len(children)
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap,
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos


    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

#external function
# for reference, please see: https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3#:~:text=Very%20simple%20hacky%20topology%2Dbased%20heirachical%20plot.%20Only%20works%20with%20DiGraphs.%20Offsetting%20is%20helpful%20if%20you%20have%20long%20labels%3A
def topo_pos(G):
    """Display in topological order, with simple offsetting for legibility"""
    pos_dict = {}
    for i, node_list in enumerate(nx.topological_generations(G)):
        x_offset = len(node_list) / 2
        y_offset = 0.1
        for j, name in enumerate(node_list):
            pos_dict[name] = (j - x_offset, -i + j * y_offset)

    return pos_dict

class Costs_and_Thresholds:
    def __init__(self,third_stage_features):
        self.third_stage_features=third_stage_features

    def cost_v1(self,node_features,target_features):
        """
        Calculates √(f1-mean1)^2+√(f2-mean2)^2+ --- √(f10-mean10)^2
        to be used later for the comparison.
        √(f1-mean1)^2+√(f2-mean2)^2+ --- √(f10-mean10)^2 <= std1+std2+---+std10

        If the sum of single_feature deviations is less than equal to the sum of standard deviations
        (total accepted deviation), then features are acceptable.

        Args:
            node_features (Tensor): _description_
            target_features (Tensor): either node features or the averaged mean features of third timestep.

        Returns:
            Optional[bool]: return true if the features are acceptable
        """
        diff=torch.sum(torch.sqrt(torch.sub(node_features,target_features)**2))
        return diff

    def threshold_v1(self):
        """
        Sum of standard deviations across features (shows acceptable deviation range.)
        """
        std_target_features=torch.std(self.third_stage_features,dim=0)
        return torch.sum(std_target_features)

    def cost_v2(self,node_features,target_features):
        """
        Calculate the count where the deviation is smaller than the accepted standard
        deviation for each feature.
        """
        std_third_stage_features=torch.mean(self.third_stage_features,dim=0)
        deviation_vector=torch.sqrt(torch.sub(node_features,target_features)**2)
        diff=deviation_vector<std_third_stage_features
        return 10-torch.sum(diff).item()

    def threshold_v2(self,acceptable_count=2):
        """
        Returns the count showing the maximum number of cases where deviation may be bigger
        than the standard deviation.
        """
        #TODO currently the threshold is strict, solutions are not found
        return acceptable_count

    def cost_v3(self, features, target_features):
        """
        Default cost function showing the average standard deviation across features.
        """
        return sqrt(mean_squared_error(features,target_features))

    def threshold_v3(self):
        """
        Calculate average rmse between the third_stage_features.
        """
        total_rmse=0
        for x in self.third_stage_features:
            for y in self.third_stage_features:
                total_rmse+=sqrt(mean_squared_error(x,y))
        return total_rmse/len(self.third_stage_features)**2