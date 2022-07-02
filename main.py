from __future__ import print_function
import os
from os import listdir
from os.path import join
import random
import logging
import time
import copy
from turtle import down

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchinfo import summary
from tqdm.notebook import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix

from models.cvt import CvT, EqCvT
from typing import *
from utils.util import (
    make_directories,
    seed_everything,
    get_device,
    init_logging_handler,
)
from utils.dataset import download_dataset, DeepLenseDataset, visualize_samples
from utils.train import train
from utils.inference import Inference
from argparse import ArgumentParser
from config.data_config import DATASET
from config.eqcvt_config import EQCVT_CONFIG
from utils.augmentation import get_transform_test, get_transform_train
from torch.utils.data import DataLoader

parser = ArgumentParser()
parser.add_argument(
    "--dataset_name",
    metavar="Model_X",
    type=str,
    default="Model_I",
    choices=["Model_I", "Model_II", "Model_III", "Model_IV"],
    help="dataset type for DeepLense project",
)
parser.add_argument(
    "--save", metavar="XXX/YYY", type=str, default="data", help="destination of dataset"
)

parser.add_argument("--cuda", action="store_true")
parser.add_argument("--no-cuda", dest="cuda", action="store_false")
parser.set_defaults(cuda=True)

args = parser.parse_args()


def main():
    dataset_name = args.dataset_name
    dataset_dir = args.save
    use_cuda = args.cuda

    classes = DATASET[f"{dataset_name}"]["classes"]

    train_config = EQCVT_CONFIG
    network_type = train_config["network_type"]
    network_config = train_config["network_config"]
    image_size = train_config["image_size"]
    optimizer_config = train_config["optimizer_config"]
    lr_schedule_config = train_config["lr_schedule_config"]

    make_directories([dataset_dir])

    trainset = DeepLenseDataset(
        dataset_dir,
        "train",
        dataset_name,
        transform=get_transform_train(upsample_size=387, final_size=129),
        download=True,
    )  # get_transform_train()

    testset = DeepLenseDataset(
        dataset_dir,
        "test",
        dataset_name,
        transform=get_transform_test(final_size=129),
        download=True,
    )  # transform_test

    seed_everything(seed=42)
    device = get_device(use_cuda=use_cuda, cuda_idx=0)

    # logging
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_dir = "logger"
    init_logging_handler(log_dir, current_time)

    PATH = os.path.join(f"{log_dir}/checkpoint", f"{network_type}_{current_time}.pt")

    train_loader = DataLoader(
        dataset=trainset, batch_size=train_config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        dataset=testset, batch_size=train_config["batch_size"], shuffle=True
    )

    visualize_samples(dataset=trainset, labels_map=classes)

    num_classes = len(classes)  # number of classes to be classified
    # image size (129x129)
    print(num_classes)
    print(f"Train Data: {len(trainset)}")
    print(f"Val Data: {len(testset)}")

    model = EqCvT(
        channels=train_config["channels"],
        num_classes=num_classes,
        s1_emb_dim=network_config["s1_emb_dim"],  # stage 1 - (same as above)
        s1_emb_kernel=network_config["s1_emb_kernel"],
        s1_emb_stride=network_config["s1_emb_stride"],
        s1_proj_kernel=network_config["s1_proj_kernel"],
        s1_kv_proj_stride=network_config["s1_kv_proj_stride"],
        s1_heads=network_config["s1_heads"],
        s1_depth=network_config["s1_depth"],
        s1_mlp_mult=network_config["s1_mlp_mult"],
        mlp_last=network_config["mlp_last"],
        dropout=network_config["dropout"],
        sym_group=network_config["sym_group"],
        N=network_config["N"],
        image_size=image_size,
        e2cc_mult_1=network_config["e2cc_mult_1"],
    ).to(device)

    # print(v)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Parameter count:", count_parameters(model))
    print("\n", model)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=optimizer_config["lr"],
        betas=optimizer_config["betas"],
        weight_decay=optimizer_config["weight_decay"],
    )

    # scheduler
    step_lr = train_config["lr_schedule_config"]["step_lr"]
    reduce_on_plateau = train_config["lr_schedule_config"]["reduce_on_plateau"]

    scheduler_plateau = ReduceLROnPlateau(
        optimizer,
        "min",
        factor=reduce_on_plateau["factor"],
        patience=reduce_on_plateau["patience"],
        threshold=reduce_on_plateau["threshold"],
        verbose=reduce_on_plateau["verbose"],
    )
    scheduler_step = StepLR(
        optimizer, step_size=step_lr["step_size"], gamma=step_lr["gamma"]
    )

    train(
        epochs=train_config["num_epochs"],
        model=model,
        device=device,
        train_loader=train_loader,
        valid_loader=test_loader,  # change to val-loader
        criterion=criterion,
        optimizer=optimizer,
        use_lr_schedule=train_config["lr_schedule_config"]["use_lr_schedule"],
        scheduler_step=scheduler_step,
        path=PATH,
    )


if __name__ == "__main__":
    main()

