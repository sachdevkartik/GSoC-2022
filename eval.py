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

parser.add_argument(
    "--checkpoint",
    metavar="XX.pth",
    type=str,
    default="model/ConvTransformer_2022-06-07-23-17-07.pt",
    help="checkpoint of pretrained model",
)

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

    best_model = EqCvT(
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

    # summary(best_model, input_size=(train_config["batch_size"], 1, 129, 129))

    MODEL_PATH = "model/ConvTransformer_2022-06-07-23-17-07.pt"
    best_model.load_state_dict(torch.load(MODEL_PATH))

    # print(v)
    def count_parameters(best_model):
        return sum(p.numel() for p in best_model.parameters() if p.requires_grad)

    print("Parameter count:", count_parameters(best_model))
    print("\n", best_model)

    # inference
    infer_obj = Inference(
        best_model,
        test_loader,
        device,
        num_classes,
        testset,
        dataset_name,
        labels_map=classes,
        image_size=image_size,
        channels=train_config["channels"],
        destination_dir="data",
    )

    infer_obj.infer_plot_roc()
    infer_obj.generate_plot_confusion_matrix()
    infer_obj.test_equivariance()


if __name__ == "__main__":
    main()

