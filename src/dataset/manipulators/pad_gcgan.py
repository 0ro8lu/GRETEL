import networkx as nx
import numpy as np

from src.dataset.manipulators.base import BaseManipulator


class GraphPaddingGCGAN(BaseManipulator):
    
    
    def node_info(self, instance):
        #max_nodes=max(self.dataset.num_nodes_values)
        n_nodes=instance.data.shape[0]
        num_padding= 0 if n_nodes%4==0 and n_nodes>3 else 4-(n_nodes%4)
        instance.data=np.pad(instance.data,((0,num_padding),(0,num_padding)),'constant',constant_values=0)
        instance.node_features=np.pad(instance.node_features,((0,num_padding),(0,0)),'constant',constant_values=2)
        return {}