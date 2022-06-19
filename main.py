from __future__ import print_function
import os
from os import listdir
from os.path import join
import random
import logging
import time
import copy
from turtle import down
import gdown

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

from models.cvt import CvT
from typing import *
from utils.util import (
    make_directories,
    seed_everything,
    get_device,
    init_logging_handler,
)
from utils.dataset import download_dataset, DeepLenseDataset, visualize_samples
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

    classes = DATASET["dataset_name"]["classes"]
    image_size = DATASET["dataset_name"]["image_size"]

    train_config = EQCVT_CONFIG

    make_directories([dataset_dir])

    trainset = DeepLenseDataset(
        dataset_dir,
        "train",
        dataset_name,
        transform=get_transform_train(upsample_size=387, final_size=129),
        download=True,
    )  # get_transform_train()

    valset = DeepLenseDataset(
        dataset_dir,
        "val",
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

    PATH = os.path.join("model", f"e2cnn_vit_{current_time}.pt")

    train_loader = DataLoader(
        dataset=trainset, batch_size=train_config["batch_size"], shuffle=True
    )
    valid_loader = DataLoader(
        dataset=valset, batch_size=train_config["batch_size"], shuffle=True
    )

    visualize_samples(dataset=trainset, labels_map=classes)
    n_classes = len(classes)  # number of classes to be classified
    # image size (129x129)


if __name__ == "__main__":
    main()

