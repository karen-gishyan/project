import networkx as nx
import random
import torch

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

class Costs_and_Threshholds:
    def __init__(self,target_features):
        self.target_features=target_features

    def normalize_target_feature_tensors(self):
        """
        Obtain target feature tensors and normalize, also returning mean and standard deviation (std).
        """
        features_min=torch.min(self.target_features,dim=0)[0]
        features_max=torch.max(self.target_features,dim=0)[0]
        #z scaling
        normalized_features=(self.target_features-features_min)/(features_max-features_min)
        mean_normalized=torch.mean(normalized_features,dim=0)
        std_normalized=torch.std(normalized_features,dim=0)

        self.normalized_results={'features_min':features_min,
                'features_max':features_max,
                'vector_mean_normalized':mean_normalized,
                'vector_std_normalized':std_normalized
                }

        return self.normalized_results

    def normalize_node_features(self,node_features):
        """
        Normalize features of the node based on mean, std from target_feature tensors.
        """
        self.normalize_target_feature_tensors()
        # NOTE the scaled dataset does not contain the node_features,
        # so node_feautes may contain values bigger/smaller than 'features_max' and 'features_min'
        # and it may not be scaled between (0 and 1)
        max_=torch.maximum(self.normalized_results['features_max'],node_features)
        min_=torch.minimum(self.normalized_results['features_min'],node_features)
        normalized_node_features=(node_features-min_)\
            /(max_- min_)

        # if all the features are not between 0 and 1
        if not (not all(normalized_node_features>1) and not all(normalized_node_features<0)):
            raise Exception("Features are not between 0 and 1.")

        return normalized_node_features

    def cost_v1(self,node_features):
        """
        node_features:        [f1,f2,...,f10]
        mean_targe_features:  [mean1,mean2,...,mean10]
        std_target_features:  [std1,std2,...,std10]

        Calculates √(f1-mean1)^2+√(f2-mean2)^2+ --- √(f10-mean10)^2
        to be used later for the comparison.
        √(f1-mean1)^2+√(f2-mean2)^2+ --- √(f10-mean10)^2 <= std1+std2+---+std10

        If the sum of single_feature deviations is less than equal to the sum of standard deviations
        (total accepted deviation), then features are acceptable.

        Args:
            node_features (Tensor): _description_
            mean_target_features (Tensor): vector showing the mean of each feature
            std_target_features (Tensor): vector showing the std of each feature

        Returns:
            Optional[bool]: return true if the features are acceptable
        """
        mean_target_features=torch.mean(self.target_features,dim=0)
        diff=torch.sum(torch.sqrt(torch.sub(node_features,mean_target_features)**2))
        return diff

    def cost_v1_normalized(self,node_features):
        normalized_node_features=self.normalize_node_features(node_features)
        diff=torch.sum(torch.sqrt(torch.sub(\
            normalized_node_features,self.normalized_results['vector_mean_normalized'])**2))
        return diff

    def threshhold_v1(self):
        std_target_features=torch.std(self.target_features,dim=0)
        return torch.sum(std_target_features)

    def threshhold_v1_normalized(self):
        #NOTE self.normalize_target_feature_tensors() should already be called
        return torch.sum(self.normalized_results['vector_std_normalized'])

    def cost_v2(self,node_features):
        """
        Calculate how much features are within the acceptable deviation range
        Range is [0,10].

        Args:
            node_features (Tensor): _description_
            mean_target_features (Tensor): vector showing the mean of each feature
            std_target_features (Tensor): vector showing the std of each feature

        Returns:
            Optional[bool]: return count of features that are within the accepted range
        """
        mean_target_features=torch.mean(self.target_features,dim=0)
        std_target_features=torch.mean(self.target_features,dim=0)
        deviation_vector=torch.sqrt(torch.sub(node_features,mean_target_features)**2)
        diff=deviation_vector<std_target_features
        return 10-torch.sum(diff).item()

    def threshhold_v2(self,acceptable_count=8):
        return acceptable_count
