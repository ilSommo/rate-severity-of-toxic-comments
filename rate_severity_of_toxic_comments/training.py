import time
import os

import torch
from torch import nn

import wandb

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
        
        if idx_batch % log_interval == 0 and idx_batch > 0: #TODO: Iterative/Cumulative logs?
            pass

    total_metrics["train_loss"] = total_loss / dataset_size

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

    total_metrics["valid_loss"] = total_loss / dataset_size

    return total_metrics

def run_training(train_dataloader: torch.utils.data.DataLoader, 
                  val_dataloader: torch.utils.data.DataLoader, 
                  model: nn.Module, 
                  loss_fn, 
                  optimizer: torch.optim,
                  device,
                  num_epochs: int, 
                  log_interval: int, 
                  verbose: bool=True) -> dict:
    """
    Executes the full train test loop with the given parameters
    """
    wandb.watch(model, log_freq=log_interval)
    loop_start = time.time()

    log_dir = os.path.join("logs", "fact_checker")

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
                  f' Loss: [{metrics_train["train_loss"]:.4f}] '
                  f' \n Val   - '
                  f' Loss: [{metrics_val["valid_loss"]:.4f}] '
            )
        
        wandb.log(metrics_train.update(metrics_val))
    
    loop_end = time.time()
    time_loop = loop_end - loop_start
    if verbose:
        print(f'Time for {num_epochs} epochs (s): {(time_loop):.3f}')

    # TODO: Return best model
