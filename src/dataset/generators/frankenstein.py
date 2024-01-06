from os import listdir
from os.path import isfile, join

import numpy as np
import networkx as nx

from src.dataset.instances.graph import GraphInstance
from src.dataset.generators.base import Generator

class FRANKENSTEIN(Generator):

    def init(self):
        base_path = self.local_config['parameters']['data_dir']
        # Paths to the files of the "FRANKENSTEIN" dataset
        self.frankenestein_arcs_path = join(base_path, 'FRANKENSTEIN_A.txt')  
        self.frankenestein_graphid_path = join(base_path, 'FRANKENSTEIN_graph_indicator.txt')
        self.frankenestein_graphlabels_path = join(base_path, 'FRANKENSTEIN_graph_labels.txt')
        self.generate_dataset()

    def get_num_instances(self):
        return len(self.dataset.instances)
    
    def generate_dataset(self):
        if not len(self.dataset.instances):
            self.read_adjacency_matrices()
    
    def read_adjacency_matrices(self):
        """
        Reads the dataset from the files
        """
        #gets the exiting arcs in the whole datatset as tuples (node_id,node_id)
        arcs=open(self.frankenestein_arcs_path,'r').readlines()
        arcs_tuples=[]
        for arc in arcs:
            arc_list=arc.split(',')
            arcs_tuples.append(((int(arc_list[0].strip())),(int(arc_list[1].strip()))))

        #gets graphs labels 
        graphs_labels=open(self.frankenestein_graphlabels_path,'r').readlines()
        graph_labels_list=[]
        for label in graphs_labels:
            graph_labels_list.append(int(label.strip()))
        
        #gets adjecency matrixes for each graph in the dataset and add them along with their labels
        with open(self.frankenestein_graphid_path,'r') as f:
            graphids=f.read().strip().split('\n')
            graphids_unique_sorted=sorted(list(set(graphids)))
            graphid_nodeid={}
            for graphid in graphids_unique_sorted:
                position=graphids.index(graphid)
                graphid_nodeid[graphid]=position
            instance_id=0
            for graphid in graphids_unique_sorted:
                matrix_dim=graphids.count(graphid)
                result = np.zeros((matrix_dim,matrix_dim))
                for arc in arcs_tuples:
                    if graphids[arc[0]-1] == graphid:
                        firstnode_position=graphid_nodeid[graphid]
                        result[arc[0]-1-firstnode_position][arc[1]-1-firstnode_position]=1
                label=graph_labels_list[int(graphid)-1]
                self.dataset.instances.append(GraphInstance(instance_id, label=label, data=np.array(result, dtype=np.int32)))
                instance_id+=1

# def read_adjacency_matrices(base_path):
#     frankenestein_arcs_path = join(base_path, 'FRANKENSTEIN_A.txt')  
#     frankenestein_graphid_path = join(base_path, 'FRANKENSTEIN_graph_indicator.txt')
#     frankenestein_graphlabels_path = join(base_path, 'FRANKENSTEIN_graph_labels.txt')
#     """
#     Reads the dataset from the adjacency matrices
#     """
#     #gets the exiting arcs in the whole datatset as tuples (node_id,node_id)
#     arcs=open(frankenestein_arcs_path,'r').readlines()
#     arcs_tuples=[]
#     for arc in arcs:
#         arc_list=arc.split(',')
#         arcs_tuples.append(((int(arc_list[0].strip())),(int(arc_list[1].strip()))))

#     #gets graphs labels 
#     graphs_labels=open(frankenestein_graphlabels_path,'r').readlines()
#     graph_labels_list=[]
#     for label in graphs_labels:
#         graph_labels_list.append(int(label.strip()))
    
#     with open(frankenestein_graphid_path,'r') as f:
#         graphids=f.read().strip().split('\n')
#         graphids_unique_sorted=sorted(list(set(graphids)))
#         graphid_nodeid={}
#         for graphid in graphids_unique_sorted:
#             position=graphids.index(graphid)
#             graphid_nodeid[graphid]=position
#         instance_id=0
#         for graphid in graphids_unique_sorted:
#             matrix_dim=graphids.count(graphid)
#             result = np.zeros((matrix_dim,matrix_dim))
#             for arc in arcs_tuples:
#                 if graphids[arc[0]-1] == graphid:
#                     result[arc[0]-1-graphid_nodeid[graphid]][arc[1]-1-graphid_nodeid[graphid]]=1
#             label=graph_labels_list[int(graphid)-1]
#             # if graphid=='2':
#             print("{0} \n".format(np.array(result, dtype=np.int32)))
#             #da decommentare
#             #self.dataset.instances.append(GraphInstance(instance_id, label=label, data=np.array(result, dtype=np.int32)))


# read_adjacency_matrices('/Users/macjack/Documents/MachineLearning/project/GRETEL/data/datasets/FRANKENSTEIN')
