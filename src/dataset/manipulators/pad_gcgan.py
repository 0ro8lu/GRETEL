import networkx as nx
import numpy as np

from src.dataset.manipulators.base import BaseManipulator


class GraphPaddingGCGAN(BaseManipulator):
    
    
    def node_info(self, instance):
        #max_nodes=max(self.dataset.num_nodes_values)
        n_nodes=instance.data.shape[0]
        mult=4
        num_padding= mult if n_nodes==0 else (mult-(n_nodes%mult))%mult
        instance.data=np.pad(instance.data,((0,num_padding),(0,num_padding)),'constant',constant_values=0)
        instance.node_features=np.pad(instance.node_features,((0,num_padding),(0,0)),'constant',constant_values=2)
        return {}