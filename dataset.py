import torch
import networkx as nx
from torch.utils.data import Dataset
from torch_geometric.data import Data

class CustomDataset(Dataset):
    def __init__(self, data: dict, vocab: dict, graph: nx.Graph, run="train"):
        self.data = data[run]
        self.vocab = vocab
        self.graph = graph

    def __len__(self):
        return len(self.data["input"])

    def __getitem__(self, idx):
        feature = self.data["input"][idx]
        edge_list = list(self.graph.edges())
        label = self.data["label"][idx]

        # data = {"x": data.x, "edge_index": data.edge_index}
        # return {"X": data.x, "edge_idx": data.edge_index, "label": label}
        return {"feature": feature, "edge_list": torch.tensor(edge_list), "label": label}

