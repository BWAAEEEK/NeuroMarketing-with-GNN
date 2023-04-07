from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import networkx as nx
from dataset import CustomDataset
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy


class GAT(LightningModule):
    def __init__(self, config: dict):
        super(GAT, self).__init__()
        self.dropout = nn.Dropout(p=config["dropout"])

        self.metric = BinaryAccuracy().to(self.device)

        self.train_loss = []
        self.val_loss = []
        self.val_acc = []

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

        self.config = config

    def forward(self, feature, edge_list):
        output = torch.zeros((feature.shape[0], 1))
        attn_score_mat = torch.zeros((feature.shape[0], 14, 14))
        for idx in range(feature.shape[0]):
            attn_score = torch.zeros((14, 14))
            data = Data(x=feature[idx, :, :], edge_index=edge_list[idx].t())

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

        return output.to(self.device), attn_score_mat

    def training_step(self, train_batch, batch_idx):
        output = self.forward(train_batch["feature"], train_batch["edge_list"])
        output = output[0]
        loss = nn.BCELoss()(output, train_batch["label"])

        self.train_loss.append(loss)

        self.log("train_step_loss", loss, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_loss).mean()

        self.log("train_loss", avg_loss)

        self.train_loss.clear()

    def validation_step(self, val_batch, batch_idx):
        output = self.forward(val_batch["feature"], val_batch["edge_list"])
        output = output[0]

        # cal loss
        loss = nn.BCELoss()(output.cuda(), val_batch["label"])

        # cal metrics for checking the performance of model
        acc = self.metric(val_batch["label"], output > 0.5)

        self.val_loss.append(loss)
        self.val_acc.append(acc)

        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_loss).mean()
        avg_acc = torch.stack(self.val_acc).mean()

        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_acc", avg_acc, prog_bar=True)

        self.val_loss.clear()
        self.val_acc.clear()

    def prepare_data(self):
        print("Loading Dataset ...")
        with open("./data/feature.pkl", "rb") as f:
            data = pickle.load(f)

        with open("./data/vocab.pkl", "rb") as f:
            vocab = pickle.load(f)

        with open("./data/brain_graph.pkl", "rb") as f:
            graph: nx.Graph = pickle.load(f)

        self.train_dataset = CustomDataset(data=data, vocab=vocab, graph=graph, run="train")
        self.val_dataset = CustomDataset(data=data, vocab=vocab, graph=graph, run="valid")
        self.test_dataset = CustomDataset(data=data, vocab=vocab, graph=graph, run="test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=self.config["num_workers"])

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=self.config["num_workers"])

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config["batch_size"], num_workers=self.config["num_workers"])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])
    
