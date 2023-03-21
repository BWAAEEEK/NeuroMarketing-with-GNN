import pickle
import networkx as nx
from tqdm import tqdm
from pprint import PrettyPrinter
import matplotlib.pyplot as plt

node_list = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
node_idx = {node: idx for idx, node in enumerate(node_list)}
idx_node = {idx: node for idx, node in enumerate(node_list)}

vocab = {"size": len(node_list),
         "itos": idx_node,
         "stoi": node_idx}

graph = nx.Graph()

edge_list = []

for node1 in node_list:
    for node2 in node_list:
        edge_list.append((vocab["stoi"][node1], vocab["stoi"][node2]))

graph.add_edges_from(edge_list)

with open("./brain_graph.pkl", "wb") as f:
    pickle.dump(graph, f)

with open("./vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)
