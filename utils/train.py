"""temporary trainer"""
from __future__ import print_function
import logging
import copy
import torch
from tqdm import tqdm
from typing import *
import wandb


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
    config,
    dataset_name,
    log_freq=100,
):
    wandb.init(
        config=config, group=dataset_name, job_type="train", mode="disabled"
    )  # ,
    wandb.watch(model, criterion, log="all", log_freq=log_freq)

    steps = 0
    all_train_loss = []
    all_val_loss = []
    all_train_accuracy = []
    all_val_accuracy = []
    all_test_accuracy = []
    all_epoch_loss = []

    best_accuracy = 0
    model.to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        tr_loss_epoch = []
        running_loss = 0

        for data, label in tqdm(train_loader):
            # for step, (data, label) in loop:
            steps += 1
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if use_lr_schedule:
                # scheduler_plateau.step(epoch_val_loss)
                scheduler_step.step()

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

                val_output = model(data)
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

        # logging frequency = each epoch
        log_dict = {
            "epoch": epoch,
            "steps": steps,
            "train/loss": loss,
            "val/loss": epoch_val_loss,
            "val/accuracy": epoch_val_accuracy,
        }
        wandb.log(log_dict, step=steps)

        if epoch_val_accuracy > best_accuracy:
            best_accuracy = epoch_val_accuracy
            best_model = copy.deepcopy(model)
            wandb.run.summary["best_accuracy"] = epoch_val_accuracy
            wandb.run.summary["best_epoch"] = epoch
            wandb.run.summary["best_step"] = steps
            wandb.save(path)
            torch.save(best_model.state_dict(), path)
