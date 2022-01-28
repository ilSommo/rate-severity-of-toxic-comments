import time
import os
import copy
import numpy as np

import torch
from torch import nn, optim

import wandb

from rate_severity_of_toxic_comments.dataset import build_dataloaders
from rate_severity_of_toxic_comments.model import create_model

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
                more_toxic_ids = data['more_toxic_ids'].to(device, dtype=torch.long)
                more_toxic_mask = data['more_toxic_mask'].to(device, dtype=torch.long)
                less_toxic_ids = data['less_toxic_ids'].to(device, dtype=torch.long)
                less_toxic_mask = data['less_toxic_mask'].to(device, dtype=torch.long)
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
                wandb.log({"Validation Running Loss": running_loss / cumul_batches})
                running_loss = 0
                cumul_batches = 0

    total_metrics["valid_loss"] = total_loss / dataset_size

    return total_metrics

def run_training(training_data: torch.utils.data.Dataset, 
                  val_data: torch.utils.data.Dataset,
                  log_interval: int, 
                  config,
                  verbose: bool=True) -> dict:
    """
    Executes the full train test loop with the given parameters
    """
    num_epochs = config["epochs"]

    if config["wandb"]:
        run = wandb.init(project="rate-comments",
        entity="toxicity",
        config=config,
        job_type='Train',
        # group="", TODO?
        tags=[config["run_mode"]])

        wandb.run.name = config["run_mode"] + "-" + wandb.run.id
        wandb.run.save()

    device = torch.device("cuda" if torch.cuda.is_available() and config["use_gpu"] else "cpu")
    # loss_fn = nn.MarginRankingLoss(margin=CONFIG['margin'])
    loss_fn = nn.MSELoss()

    train_dataloader, val_dataloader = build_dataloaders([training_data, val_data], batch_sizes=(config["train_batch_size"], config["valid_batch_size"]))

    model = create_model(config)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    if config["wandb"]:
        wandb.watch(model, log_freq=log_interval)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    loss_history = {
        "train": [],
        "valid": [],
    }

    loop_start = time.time()

    for epoch in range(1, num_epochs + 1):
        time_start = time.time()

        metrics_train = train_loop(train_dataloader, model, loss_fn, optimizer, device, epoch, log_interval=log_interval, pairwise_dataset=False, use_wandb=config["wandb"])
        metrics_val = test_loop(val_dataloader, model, loss_fn, device, epoch, pairwise_dataset=False, use_wandb=config["wandb"])

        time_end = time.time()

        all_metrics = metrics_train
        all_metrics.update(metrics_val)

        loss_history['train'].append(all_metrics["train_loss"])
        loss_history['valid'].append(all_metrics["valid_loss"])
        lr = optimizer.param_groups[0]['lr']

        if verbose:
            print(f'Epoch: {epoch} '
                  f' Lr: {lr:.8f} '
                  f' | Time one epoch (s): {(time_end - time_start):.4f} '
                  f' \n Train - '
                  f' Loss: [{all_metrics["train_loss"]:.4f}] '
                  f' \n Val   - '
                  f' Loss: [{all_metrics["valid_loss"]:.4f}] '
            )
        
        if config["wandb"]:
            wandb.log(all_metrics)
        
        if all_metrics["valid_loss"] <= best_epoch_loss:
            best_epoch_loss = all_metrics["valid_loss"]
            best_model_wts = copy.deepcopy(model.state_dict())
    
    loop_end = time.time()
    time_loop = loop_end - loop_start
    if verbose:
        print(f'Time for {num_epochs} epochs (s): {(time_loop):.3f}')

    model.load_state_dict(best_model_wts)
    model_filename = config["run_mode"]+"-"+time.strftime("%Y%m%d-%H%M%S")+".pth"
    torch.save(model.state_dict(), os.path.join("res", "models", model_filename))

    if config["wandb"]:
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, model_filename))
        run.finish()
    return model, loss_history
    
