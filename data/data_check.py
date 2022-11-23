import pickle
import torch
from tqdm import tqdm
from pprint import PrettyPrinter
import networkx as nx
from torch_geometric.data import Data

with open("brain_graph.pkl", "rb") as f:
    graph: nx.Graph = pickle.load(f)

x = torch.tensor(list(graph.nodes()))
edge = torch.tensor(list(graph.edges()))

data = Data(x=x, edge_index=edge.t().contiguous())

print(data.edge_index.t())
print(data.edge_index.t().shape)
