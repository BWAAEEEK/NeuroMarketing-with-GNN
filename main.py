import argparse
import pickle

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from util import EarlyStopping

import networkx as nx
from model import GAT
from trainer import Trainer
from dataset import CustomDataset
import numpy as np
import random
import time


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--dropout", default=0.6, type=float)
    args.add_argument("--hidden", default=258, type=int)
    args.add_argument("--n_heads", default=1, type=int)
    args.add_argument("--learning_rate", default=0.001, type=float)

    args.add_argument("--batch_size", default=32, type=int)
    args.add_argument("--split", default=0.8, type=float)
    args.add_argument("--patience", default=10, type=int)
    args.add_argument("--epoch", default=100, type=int)
    args.add_argument("--seed", default=42, type=int)

    args = args.parse_args()

    fix_seed(args.seed)

    config = vars(args)  # convert to dictionary

    print("Loading Dataset ...")
    with open("../인공지능 공부/NeuroMarketing/data/feature.pkl", "rb") as f:
        data = pickle.load(f)

    with open("../인공지능 공부/NeuroMarketing/data/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    with open("../인공지능 공부/NeuroMarketing/data/brain_graph.pkl", "rb") as f:
        graph: nx.Graph = pickle.load(f)

    config["num_feature"] = 512
    print("Building DataLoader ...")
    train_dataset = CustomDataset(data=data, vocab=vocab, graph=graph, run="train")
    val_dataset = CustomDataset(data=data, vocab=vocab, graph=graph, run="valid")
    test_dataset = CustomDataset(data=data, vocab=vocab, graph=graph, run="test")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)

    print("Building Model ...")
    model = GAT(config)

    print("Building Trainer ...")
    trainer = Trainer(config, model, train_loader, test_loader, val_loader)

    print("+--------------------------------------------+")
    print("|               Training Start               |")
    print("+--------------------------------------------+")

    writer = SummaryWriter("./runs/A{}_H{}_lr{}_batch{}_drop{}".format(config["n_heads"],
                                                                       config["hidden"],
                                                                       config["learning_rate"],
                                                                       config["batch_size"],
                                                                       config["dropout"]))

    loss_list = []
    model_check_point = 0
    count = 0
    for epoch in range(config["epoch"]):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        train_loss = trainer.train(epoch, run="train")
        val_loss = trainer.train(epoch, run="val")
        loss_list.append(val_loss)

        if loss_list[-2] < loss_list[-1]:
            count += 1

        val_acc = trainer.eval(epoch, run="val")

        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("val_loss", val_loss, epoch)
        writer.add_scalar("val_acc", val_acc, epoch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if count == config["paitience"]:
            print("Early Stopping")
            break
    test_acc = trainer.eval(config["epoch"], "test")
    print('\n\n')
    print(f"|     Test Dataset Accuracy {test_acc}     |")
    print("\n\n")
