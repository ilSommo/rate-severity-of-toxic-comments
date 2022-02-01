import time
import os
import copy
import numpy as np
from tqdm import tqdm

import torch
from torch import nn, optim

import wandb

from torch.utils.data import Dataset
from rate_severity_of_toxic_comments.dataset import build_dataloaders
from rate_severity_of_toxic_comments.model import create_model
from rate_severity_of_toxic_comments.metrics import *


def train_loop(dataloader, model, loss_fn, optimizer, device, idx_epoch, log_interval=100, pairwise_dataset=False, use_wandb=True):
    """
    Executes the training loop on the given parameters. Logs metrics on TensorBoard.
    """
    model.train()
    total_metrics = {}
    total_loss = 0.0
    running_loss = 0.0
    cumul_batches = 0
    dataset_size = 0

    for idx_batch, data in tqdm(enumerate(dataloader)):
        if not pairwise_dataset:
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['target'].to(device, dtype=torch.long)
            batch_size = ids.size(0)

            scores = model(ids, mask)
            scores = scores.to(torch.float32)
            targets = targets.to(torch.float32)
            loss = loss_fn(scores, targets)
        else:
            more_toxic_ids = data['more_toxic_ids'].to(
                device, dtype=torch.long)
            more_toxic_mask = data['more_toxic_mask'].to(
                device, dtype=torch.long)
            less_toxic_ids = data['less_toxic_ids'].to(
                device, dtype=torch.long)
            less_toxic_mask = data['less_toxic_mask'].to(
                device, dtype=torch.long)
            targets = data['target'].to(device, dtype=torch.long)
            batch_size = more_toxic_ids.size(0)
            # TODO Check output size
            more_toxic_outputs = model(more_toxic_ids, more_toxic_mask)
            less_toxic_outputs = model(less_toxic_ids, less_toxic_mask)
            loss = loss_fn(more_toxic_outputs, less_toxic_outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += (loss.item() * batch_size)
        running_loss += loss.item()
        cumul_batches += 1
        dataset_size += batch_size

        if idx_batch % log_interval == 0 and idx_batch > 0 and use_wandb:
            wandb.log({"Train Running Loss": running_loss / cumul_batches})
            running_loss = 0
            cumul_batches = 0

    total_metrics["train_loss"] = total_loss / dataset_size

    return total_metrics


def test_loop(dataloader, model, loss_fn, device, idx_epoch, log_interval=100, pairwise_dataset=False, use_wandb=True):
    """
    Executes a test loop on the given paramters. Returns metrics and votes.
    """
    model.eval()
    total_metrics = {}
    total_loss = 0.0
    running_loss = 0.0
    cumul_batches = 0
    dataset_size = 0

    with torch.no_grad():
        for idx_batch, data in enumerate(dataloader):
            if not pairwise_dataset:
                ids = data["ids"].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                targets = data['target'].to(device, dtype=torch.long)
                batch_size = ids.size(0)

                scores = model(ids, mask)
                scores = scores.to(torch.float32)
                targets = targets.to(torch.float32)
                loss = loss_fn(scores, targets)
            else:
                more_toxic_ids = data['more_toxic_ids'].to(
                    device, dtype=torch.long)
                more_toxic_mask = data['more_toxic_mask'].to(
                    device, dtype=torch.long)
                less_toxic_ids = data['less_toxic_ids'].to(
                    device, dtype=torch.long)
                less_toxic_mask = data['less_toxic_mask'].to(
                    device, dtype=torch.long)
                targets = data['target'].to(device, dtype=torch.long)
                batch_size = more_toxic_ids.size(0)

                more_toxic_outputs = model(more_toxic_ids, more_toxic_mask)
                less_toxic_outputs = model(less_toxic_ids, less_toxic_mask)
                loss = loss_fn(more_toxic_outputs, less_toxic_outputs, targets)

            total_loss += (loss.item() * batch_size)
            running_loss += loss.item()
            cumul_batches += 1
            dataset_size += batch_size

            if idx_batch % log_interval == 0 and idx_batch > 0 and use_wandb:
                wandb.log(
                    {"Validation Running Loss": running_loss / cumul_batches})
                running_loss = 0
                cumul_batches = 0

    total_metrics["valid_loss"] = total_loss / dataset_size

    return total_metrics


def run_training(run_mode, training_data: Dataset,
                 val_data: Dataset,
                 training_params, model_params, support_bag,
                 seed, use_wandb, use_gpu,
                 verbose: bool = True,
                 log_interval: int = 100) -> dict:
    """
    Executes the full train test loop with the given parameters
    """

    run = None
    if use_wandb:
        run = wandb.init(project="rate-comments",
                        entity="toxicity",
                        config={
                             "model": model_params, 
                             "training": training_params,
                             "seed": seed,
                             "run_mode": run_mode
                        },
                        job_type='Train',
                        # group="", TODO?
                        tags=[run_mode])

        wandb.run.name = run_mode + "-" + wandb.run.id
        wandb.run.save()

    device = torch.device("cuda" if torch.cuda.is_available()
                          and use_gpu else "cpu")
    loss_fn = nn.MSELoss()

    train_dataloader, val_dataloader = build_dataloaders([training_data, val_data], batch_sizes=(
        training_params["train_batch_size"], training_params["valid_batch_size"]))

    model = create_model(run_mode, training_params, model_params, support_bag)
    model.to(device)

    train_loop_stats = TrainLoopStatisticsManager(model, early_stopping_patience=3, verbose=verbose, use_wandb=use_wandb)

    optimizer = optim.Adam(model.parameters(
    ), lr=training_params["learning_rate"], weight_decay=training_params['L2_regularization'])

    if use_wandb:
        wandb.watch(model, log_freq=log_interval)

    loop_start = time.time()

    num_epochs = training_params["epochs"]
    for epoch in range(1, num_epochs + 1):
        time_start = time.time()

        metrics_train = train_loop(train_dataloader, model, loss_fn, optimizer, device, epoch,
                                   log_interval=log_interval, pairwise_dataset=False, use_wandb=use_wandb)
        metrics_val = test_loop(val_dataloader, model, loss_fn, device,
                                epoch, pairwise_dataset=False, use_wandb=use_wandb)

        time_end = time.time()

        train_loop_stats.registerEpoch(
            metrics_train, metrics_val, training_params["learning_rate"], epoch, time_start, time_end)

        if train_loop_stats.early_stop:
            print("Early Stopping")
            break

    loop_end = time.time()
    time_loop = loop_end - loop_start
    if verbose:
        print(f'Time for {num_epochs} epochs (s): {(time_loop):.3f}')

    model.load_state_dict(train_loop_stats.best_model_wts)
    model_filename = run_mode + "-" + \
        time.strftime("%Y%m%d-%H%M%S")+".pth"
    torch.save(model.state_dict(), os.path.join(
        "res", "models", model_filename))

    if use_wandb:
        torch.save(model.state_dict(), os.path.join(
            wandb.run.dir, model_filename))
        run.finish()
    return model, train_loop_stats.getLossHistory()
