import argparse

import torch
from model_lightning import GAT
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
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

    args.add_argument("--gpus", default=1, type=int)
    args.add_argument("--seed", default=42, type=int)

    args = args.parse_args()

    fix_seed(args.seed)

    config = vars(args)  # convert to dictionary
    config["num_feature"] = 512

    print("Building DataLoader & Model ...")
    model = GAT(config)
    print("Done")

    print("Building Trainer ...")
    logger = TensorBoardLogger(save_dir="./runs2", name="A{}_H{}_lr{}_batch{}_drop{}".format(config["n_heads"],
                                                                                             config["hidden"],
                                                                                             config["learning_rate"],
                                                                                             config["batch_size"],
                                                                                             config["dropout"]))

    early_stopping = EarlyStopping("val_loss", patience=config["patience"])
    checkpoint = ModelCheckpoint(dirpath="./output",
                                 filename="{epoch}_{val_acc:.2f}",
                                 monitor="val_acc")
    trainer = Trainer(gpus=config["gpus"], accelerator="gpu", max_epochs=config["epoch"], logger=logger,
                      callbacks=[early_stopping, checkpoint], log_every_n_steps=1)

    print("+--------------------------------------------+")
    print("|               Training Start               |")
    print("+--------------------------------------------+")

    trainer.fit(model)



