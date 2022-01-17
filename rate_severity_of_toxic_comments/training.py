import time
from typing import Counter
import os

import torch
from torch import nn


def train_loop(dataloader, model, loss_fn, optimizer, device, idx_epoch, log_interval=10):
    """
    Executes the training loop on the given parameters. Logs metrics on TensorBoard.
    """
    model.train()
    total_metrics = {}
    total_loss = 0.0
    cumul_batches = 0
    dataset_size = 0

    for idx_batch, data in enumerate(dataloader):
        more_toxic_ids = data['more_toxic_ids'].to(device, dtype=torch.long)
        more_toxic_mask = data['more_toxic_mask'].to(device, dtype=torch.long)
        less_toxic_ids = data['less_toxic_ids'].to(device, dtype=torch.long)
        less_toxic_mask = data['less_toxic_mask'].to(device, dtype=torch.long)
        targets = data['target'].to(device, dtype=torch.long)

        batch_size = more_toxic_ids.size(0)

        more_toxic_outputs = model(more_toxic_ids, more_toxic_mask)
        less_toxic_outputs = model(less_toxic_ids, less_toxic_mask)
        loss = loss_fn(more_toxic_outputs, less_toxic_outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += (loss.item() * batch_size)
        cumul_batches += 1
        dataset_size += batch_size

        epoch_loss = total_loss / cumul_batches

        if idx_batch % log_interval == 0 and idx_batch > 0:  # TODO: Iterative/Cumulative logs?
            # TODO: writer.add_scalar("key", val) -> wandb.log({"key": val})
            # global_step = idx_batch + (idx_epoch * len(dataloader))
            # writer.add_scalar('Metrics/Accuracy_Un_Train_IT', cumul_metrics["accuracy"], global_step)
            # cumul_metrics = normalize_metrics(cumul_metrics, cumul_batches)
            # cumul_f1_score = f1_score(cumul_metrics["precision"], cumul_metrics["recall"])
            # writer.add_scalar('Metrics/Loss_Train_IT', loss, global_step)
            # writer.add_scalar('Metrics/Accuracy_Train_IT', cumul_metrics["accuracy"], global_step)
            # writer.add_scalar('Metrics/Precision_Train_IT', cumul_metrics["precision"], global_step)
            # writer.add_scalar('Metrics/Recall_Train_IT', cumul_metrics["recall"], global_step)
            # writer.add_scalar('Metrics/F1_Train_IT', cumul_f1_score, global_step)

            # cumul_metrics.clear()
            # cumul_batches = 0
            pass

    total_metrics["loss"] = total_loss / dataset_size

    return total_metrics


def test_loop(dataloader, model, loss_fn, device):
    """
    Executes a test loop on the given paramters. Returns metrics and votes.
    """
    model.eval()
    total_metrics = {}
    total_loss = 0.0
    cumul_batches = 0
    dataset_size = 0

    with torch.no_grad():
        for idx_batch, data in enumerate(dataloader):
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
            cumul_batches += 1
            dataset_size += batch_size

    total_metrics["loss"] = total_loss / dataset_size

    return total_metrics


def run_training(train_dataloader: torch.utils.data.DataLoader,
                 val_dataloader: torch.utils.data.DataLoader,
                 model: nn.Module,
                 loss_fn,
                 optimizer: torch.optim,
                 device,
                 num_epochs: int,
                 log_interval: int,
                 training_label=None,
                 verbose: bool = True) -> dict:
    """
    Executes the full train test loop with the given parameters
    """
    loop_start = time.time()

    log_dir = os.path.join("logs", "fact_checker")
    if training_label is not None:
        log_dir = os.path.join(log_dir, training_label)
    # writer = tensorboard.writer.SummaryWriter(log_dir=log_dir)

    for epoch in range(1, num_epochs + 1):
        time_start = time.time()

        metrics_train = train_loop(
            train_dataloader, model, loss_fn, optimizer, device, epoch, log_interval=log_interval)
        metrics_val = test_loop(val_dataloader, model, loss_fn, device)

        time_end = time.time()

        lr = optimizer.param_groups[0]['lr']

        if verbose:
            print(f'Epoch: {epoch} '
                  f' Lr: {lr:.8f} '
                  f' | Time one epoch (s): {(time_end - time_start):.4f} '
                  f' \n Train - '
                  f' Loss: [{metrics_train["loss"]:.4f}] '
                  f' \n Val   - '
                  f' Loss: [{metrics_val["loss"]:.4f}] '
                  )

        # #TODO Tensorboard -> WandB
        # writer.add_scalars('Metrics/Losses', {"Train": metrics_train["loss"], "Val": metrics_val["loss"]}, epoch)
        # writer.add_scalars('Metrics/Accuracy', {"Train": metrics_train["accuracy"], "Val": metrics_val["accuracy"]}, epoch)
        # writer.add_scalars('Metrics/Precision', {"Train": metrics_train["precision"], "Val": metrics_val["precision"]}, epoch)
        # writer.add_scalars('Metrics/Recall', {"Train": metrics_train["recall"], "Val": metrics_val["recall"]}, epoch)
        # writer.add_scalars('Metrics/F1', {"Train": metrics_train["f1_score"], "Val": metrics_val["f1_score"]}, epoch)
        # writer.add_scalars('Metrics/Maj Accuracy', {"Train": majority_metrics_train["accuracy"], "Val": majority_metrics_val["accuracy"]}, epoch)
        # writer.add_scalars('Metrics/Maj Precision', {"Train": majority_metrics_train["precision"], "Val": majority_metrics_val["precision"]}, epoch)
        # writer.add_scalars('Metrics/Maj Recall', {"Train": majority_metrics_train["recall"], "Val": majority_metrics_val["recall"]}, epoch)
        # writer.add_scalars('Metrics/Maj F1', {"Train": majority_metrics_train["f1_score"], "Val": majority_metrics_val["f1_score"]}, epoch)
        # writer.flush()

    loop_end = time.time()
    time_loop = loop_end - loop_start
    if verbose:
        print(f'Time for {num_epochs} epochs (s): {(time_loop):.3f}')

    # TODO: Return best model
