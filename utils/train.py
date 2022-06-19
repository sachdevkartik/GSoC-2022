"""temporary trainer"""
from __future__ import print_function
import logging
import copy
import torch
from tqdm.notebook import tqdm
from typing import *


def train(
    epochs,
    model,
    device,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    use_lr_schedule,
    scheduler_step,
    path,
):
    all_train_loss = []
    all_val_loss = []
    all_train_accuracy = []
    all_val_accuracy = []
    all_test_accuracy = []
    all_epoch_loss = []

    best_accuracy = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        tr_loss_epoch = []
        running_loss = 0

        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = epoch_loss / len(train_loader)
        all_epoch_loss.append(epoch_loss)

        correct = 0

        with torch.no_grad():
            model.eval()
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = v(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc
                epoch_val_loss += val_loss

            epoch_val_accuracy = epoch_val_accuracy / len(valid_loader)
            epoch_val_loss = epoch_val_loss / len(valid_loader)
            all_val_loss.append(epoch_val_loss)

        all_val_accuracy.append(epoch_val_accuracy.item() * 100)
        logging.debug(
            f"Epoch : {epoch+1} - LR {optimizer.param_groups[0]['lr']:.8f} - loss : {epoch_loss:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f} \n"
        )

        if use_lr_schedule:
            # scheduler_plateau.step(epoch_val_loss)
            scheduler_step.step()

        if epoch_val_accuracy > best_accuracy:
            best_accuracy = epoch_val_accuracy
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), path)
