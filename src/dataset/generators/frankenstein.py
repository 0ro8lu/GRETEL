from os import listdir
from os.path import isfile, join

import numpy as np
import networkx as nx

from src.dataset.instances.graph import GraphInstance
from src.dataset.generators.base import Generator

from zipfile import ZipFile 

class FRANKENSTEIN(Generator):

    def init(self):
        base_path = self.local_config['parameters']['data_dir']
        with ZipFile('data/datasets/FRANKENSTEIN.zip', 'r') as zip: 
            zip.extractall('data/datasets')
        node_map={}
        for i in range(780):
            node_map['env'+str(i)]=i
        self.dataset.node_features_map = node_map
        # Paths to the files of the "FRANKENSTEIN" dataset
        self.frankenestein_arcs_path = join(base_path, 'FRANKENSTEIN_A.txt')  
        self.frankenestein_graphid_path = join(base_path, 'FRANKENSTEIN_graph_indicator.txt')
        self.frankenestein_graphlabels_path = join(base_path, 'FRANKENSTEIN_graph_labels.txt')
        #rimuovere per rollback
        self.frankenestein_nodeattibutes_path = join(base_path, 'FRANKENSTEIN_node_attributes.txt')
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
        
        #rimuovere per rollback
        #gets node attributes
        node_attributes=open(self.frankenestein_nodeattibutes_path,'r').readlines()
        attributes_len=len(node_attributes[0].split(','))
        
        #gets adjecency matrixes for each graph in the dataset and add them along with their labels
        with open(self.frankenestein_graphid_path,'r') as f:
            graphids=f.read().strip().split('\n')
            for i in range(len(graphids)):
                graphids[i]=int(graphids[i].strip())
            graphids_unique_sorted=sorted(list(set(graphids)))
            graphid_nodeposition={}
            for graphid in graphids_unique_sorted:
                position=graphids.index(graphid)
                graphid_nodeposition[graphid]=position
            instance_id=0
            for graphid in graphids_unique_sorted:
                if np.random.randint(0,2):
                    #rimuovere per rollback
                    node_count=graphids.count(graphid)
                    node_atr_graphid=np.zeros((graphids.count(graphid),attributes_len)).astype(dtype=np.float32)
                    matrix_dim=graphids.count(graphid)
                    result = np.zeros((matrix_dim,matrix_dim))
                    firstnode_position=graphid_nodeposition[graphid]
                    for arc in arcs_tuples:
                        if graphids[arc[0]-1] == graphid:
                            result[arc[0]-1-firstnode_position][arc[1]-1-firstnode_position]=1
                    for node in range(1,len(graphids)+1):
                        #rimuovere per rollback        
                        if graphids[node-1] == graphid:
                            single_node_attributes=node_attributes[node-1].split(',')
                            node_array=np.array(single_node_attributes).astype(np.float32)
                            node_atr_graphid[node-1-firstnode_position]=node_array
                    label=max(0,graph_labels_list[int(graphid)-1])
                    self.dataset.instances.append(GraphInstance(instance_id, label=label, data=np.array(result, dtype=np.int32), node_features=node_atr_graphid))
                    #self.dataset.instances.append(GraphInstance(instance_id, label=label, data=np.array(result, dtype=np.int32)))
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


# read_adjacency_matrices('C:\Users\werry\Desktop\GRETEL 2.0\GRETEL\data\datasets\FRANKENSTEIN')
