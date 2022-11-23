import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from util import EarlyStopping
from tqdm import tqdm
from sklearn.metrics import top_k_accuracy_score
from model import GAT
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, config: dict, model: GAT, train_loader: DataLoader, test_loader: DataLoader, val_loader: DataLoader):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model
        self.optim = Adam(params=model.parameters(), lr=config["learning_rate"])

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.patience = config["patience"]

        self.criterion = nn.BCELoss()

    def train(self, epoch, run="train"):
        self.model.to(self.device)
        loss_list = []
        if run == "train":
            data_iter = tqdm(enumerate(self.train_loader),
                             desc=f"EP:{epoch}_train",
                             total=len(self.train_loader))
            self.model.train()
        else:
            data_iter = tqdm(enumerate(self.val_loader),
                             desc=f"EP:{epoch}_val",
                             total=len(self.val_loader))
            self.model.eval()

        avg_loss = 0.0
        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}
            self.optim.zero_grad()
            output, attn_mat = self.model(data["feature"], data["edge_list"])

            loss = self.criterion(output.to(self.device), data["label"])
            if run == "train":
                loss.backward()
                self.optim.step()

            avg_loss += loss.item()
            loss_list.append(loss.item())
            post_fix = {"avg_loss": avg_loss / (i + 1), "cur_loss": loss.item()}

            data_iter.set_postfix(post_fix)

        return avg_loss / len(data_iter), loss_list

    def eval(self, epoch, run="val"):
        self.model.cpu()
        self.model.eval()

        if run == "test":
            data_iter = tqdm(enumerate(self.test_loader),
                             desc=f"EP:{epoch}_test",
                             total=len(self.test_loader))

        else:
            data_iter = tqdm(enumerate(self.val_loader),
                             desc=f"EP:{epoch}_val",
                             total=len(self.val_loader))

        avg_acc = 0.0

        for i, data in data_iter:
            with torch.no_grad():
                data = {key: value.cpu() for key, value in data.items()}

                output, attn_mat = self.model(data["feature"], data["edge_list"])
                acc = top_k_accuracy_score(data["label"].numpy(), output, k=1, labels=[0, 1])

                avg_acc += acc

                post_fix = {"avg_acc": avg_acc / (i + 1), "cur_acc": acc}

                data_iter.set_postfix(post_fix)

        return avg_acc / len(data_iter)