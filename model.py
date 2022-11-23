from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data
import torch
import torch.nn as nn
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self, config: dict):
        super(GAT, self).__init__()
        self.dropout = nn.Dropout(p=config["dropout"])

        self.conv1 = GATConv(in_channels=config["num_feature"],
                             out_channels=config["hidden"],
                             heads=config["n_heads"],
                             concat=False)

        self.conv2 = GATConv(in_channels=config["hidden"],
                             out_channels=config["hidden"],
                             heads=config["n_heads"],
                             concat=False)

        self.linear = nn.Linear(config["hidden"] * 14, int(config["hidden"] / 2), bias=True)
        self.pred = nn.Linear(int(config["hidden"] / 2), 1, bias=True)

    def forward(self, feature, edge_list):
        output = torch.zeros((feature.shape[0], 1))
        attn_score_mat = torch.zeros((feature.shape[0], 14, 14))
        for idx in range(feature.shape[0]):
            attn_score = torch.zeros((14, 14))
            data = Data(x=feature[idx, :, :], edge_index=edge_list[idx].t())
            # print(feature[idx, :, :].shape)
            # print(edge_list[idx, :, :].shape)
            x = self.conv1(data.x, data.edge_index)
            x = x
            x = F.relu(x)
            x = self.dropout(x)
            x = self.conv2(x, data.edge_index, return_attention_weights=True)
            attn_idx, attn_value = x[1][0], x[1][1]
            x = x[0]
            x = x.view(x.shape[0] * x.shape[1])    # (Batch, hidden * 7)

            attn_idx = attn_idx.t()
            for i, seq in enumerate(attn_idx):
                attn_score[seq[0]][seq[1]] = attn_value[i].item()
            x = self.linear(x)
            x = self.pred(x)

            x = torch.sigmoid(x)
            output[idx] = x
            attn_score_mat[idx] = attn_score

        return output, attn_score_mat