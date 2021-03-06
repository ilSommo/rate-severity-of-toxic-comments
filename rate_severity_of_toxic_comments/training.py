__version__ = '1.0.0-rc.1'
__author__ = 'Lorenzo Menghini, Martino Pulici, Alessandro Stockman, Luca Zucchini'


import os
import time

import torch
from torch import nn, optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import Dataset
from tqdm import tqdm
import wandb

from rate_severity_of_toxic_comments.dataset import build_dataloaders
from rate_severity_of_toxic_comments.metrics import *
from rate_severity_of_toxic_comments.model import create_model


def run_training(
        run_mode,
        training_data: Dataset,
        val_data: Dataset,
        training_params,
        model_params,
        support_bag,
        seed,
        use_wandb,
        use_gpu,
        verbose: bool = True,
        log_interval=100):
    """
    Executes the full train test loop with the given parameters.

    Parameters
    ----------
    run_mode : str
        Run mode.
    training_data : torch.utils.data.dataset.Dataset
        Training dataset.
    val_data : torch.utils.data.dataset.Dataset
        Validation dataset.
    training_params : dict
        Training parameters.
    model_params : dict
        Model parameters.
    support_bag : dict
        Configuration parameters.
    seed : int
        Seed for random state.
    use_wandb : bool
        Weights & Biases flag.
    use_gpu : bool
        GPU use flag.
    verbose : bool, default True
        Verbosity flag.
    log_interval : int, deafult 100
        Logging interval.

    Returns
    -------
    model : torch.nn.modules.module.Module
        Neural network model.
    loss_history : dict
        Loss history dictionary.

    """
    run = None
    if use_wandb:
        run = wandb.init(project='rate-comments',
                         entity='toxicity',
                         job_type='Train',
                         tags=[run_mode])
        wandb.run.name = run_mode + '-' + wandb.run.id
        wandb.config.update({
            'model': model_params,
            'training': training_params,
            'seed': seed,
            'run_mode': run_mode
        })
        for key in wandb.config.keys():
            if '.' in key:
                portions = key.split('.')
                param_type, remaining = portions[0], portions[1:]
                target_dict = None
                if param_type == 'model':
                    target_dict = model_params
                elif param_type == 'training':
                    target_dict = training_params
                for subkey in remaining[:-1]:
                    target_dict = target_dict[subkey]
                target_dict[remaining[-1]] = wandb.config[key]
        wandb.run.save()
    device = torch.device('cuda' if torch.cuda.is_available()
                          and use_gpu else 'cpu')
    train_dataset_params = training_params['dataset']
    if train_dataset_params['type'] == 'scored':
        loss_fn = nn.MSELoss()
    elif train_dataset_params['type'] == 'pairwise':
        loss_fn = nn.MarginRankingLoss(
            margin=train_dataset_params['loss_margin'])
    train_batch_size = training_params['train_batch_size']
    valid_batch_size = training_params['valid_batch_size']
    lr = training_params['learning_rate']
    l2_reg = training_params['L2_regularization']
    grad_clipping = training_params['gradient_clipping']
    train_dataloader, val_dataloader = build_dataloaders(
        [training_data, val_data], batch_sizes=(train_batch_size, valid_batch_size))
    model = create_model(run_mode, training_params, model_params, support_bag)
    model.to(device)
    train_loop_stats = TrainLoopStatisticsManager(
        model, early_stopping_patience=3, verbose=verbose, use_wandb=use_wandb)
    if training_params['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(
        ), lr=lr, weight_decay=l2_reg)
    elif training_params['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(
        ), lr=lr, weight_decay=l2_reg)
    if use_wandb:
        wandb.watch(model, log_freq=log_interval)
    loop_start = time.time()
    num_epochs = training_params['epochs']
    for epoch in range(1, num_epochs + 1):
        time_start = time.time()
        metrics_train = train_loop(
            train_dataloader,
            model,
            loss_fn,
            optimizer,
            device,
            grad_clipping,
            log_interval=log_interval,
            dataset_type=train_dataset_params['type'],
            use_wandb=use_wandb)
        metrics_val = test_loop(
            val_dataloader,
            model,
            loss_fn,
            device,
            log_interval=log_interval,
            dataset_type=train_dataset_params['type'],
            use_wandb=use_wandb)
        time_end = time.time()
        train_loop_stats.registerEpoch(
            metrics_train, metrics_val, lr, epoch, time_start, time_end)
        if train_loop_stats.early_stop:
            print('Early Stopping')
            break
    loop_end = time.time()
    time_loop = loop_end - loop_start
    if verbose:
        print(f'Time for {num_epochs} epochs (s): {(time_loop):.3f}')
    model.load_state_dict(train_loop_stats.best_model_wts)
    model_filename = run_mode + '-' + \
        time.strftime('%Y%m%d-%H%M%S') + '.pth'
    torch.save(model.state_dict(), os.path.join(
        'res', 'models', model_filename))
    if use_wandb:
        torch.save(model.state_dict(), os.path.join(
            wandb.run.dir, model_filename))
        run.finish()
    loss_history = train_loop_stats.getLossHistory()
    return model, loss_history


def test_loop(
        dataloader,
        model,
        loss_fn,
        device,
        log_interval,
        dataset_type,
        use_wandb=True):
    """
    Executes the testing loop on the given parameters.

    Parameters
    ----------
    dataloader : torch.utils.data.dataloader.DataLoader
        Dataloader.
    model : torch.nn.modules.module.Module
        Neural network model.
    loss_fn : torch.nn.modules.loss._Loss
        Loss function.
    device : torch.device
        Testing device.
    log_interval : int
        Logging interval.
    dataset_type : str
        Dataset type.
    use_wandb : bool, default True
        Weights & Biases flag.

    Returns
    -------
    total_metrics : dict
        Dictionary with testing metrics.

    """
    model.eval()
    total_metrics = {}
    total_loss = 0.0
    total_accuracy = 0.0
    running_loss = 0.0
    cumul_batches = 0
    dataset_size = 0
    total_scores = []
    binarization_targets = []
    with torch.no_grad():
        for idx_batch, data in tqdm(
                enumerate(dataloader), total=len(dataloader)):
            if dataset_type == 'scored':
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                targets = data['target'].to(device, dtype=torch.float32)
                preprocessing_metrics = data['preprocessing_metric'].to(
                    device, dtype=torch.float32)
                batch_size = ids.size(0)
                scores = model(ids, mask, preprocessing_metrics)
                scores = scores.to(torch.float32)
                loss = loss_fn(scores, targets)
            elif dataset_type == 'pairwise':
                more_toxic_ids = data['more_toxic_ids'].to(
                    device, dtype=torch.long)
                more_toxic_mask = data['more_toxic_mask'].to(
                    device, dtype=torch.long)
                more_toxic_metric = data['more_toxic_metric'].to(
                    device, dtype=torch.long)
                less_toxic_ids = data['less_toxic_ids'].to(
                    device, dtype=torch.long)
                less_toxic_mask = data['less_toxic_mask'].to(
                    device, dtype=torch.long)
                less_toxic_metric = data['less_toxic_metric'].to(
                    device, dtype=torch.long)
                targets = data['target'].to(device, dtype=torch.long)
                batch_size = more_toxic_ids.size(0)
                more_toxic_outputs = model(
                    more_toxic_ids, more_toxic_mask, more_toxic_metric)
                less_toxic_outputs = model(
                    less_toxic_ids, less_toxic_mask, less_toxic_metric)
                loss = loss_fn(more_toxic_outputs, less_toxic_outputs, targets)
                more_toxic_scores = more_toxic_outputs.to(
                    torch.float32).tolist()
                less_toxic_scores = less_toxic_outputs.to(
                    torch.float32).tolist()
                total_scores = more_toxic_scores + less_toxic_scores
                total_accuracy += (more_toxic_outputs >
                                   less_toxic_outputs).sum().item()
            elif dataset_type == 'binarized':
                ids = data['text_ids'].to(device, dtype=torch.long)
                mask = data['text_mask'].to(device, dtype=torch.long)
                targets = data['target'].to(device, dtype=torch.long)
                preprocessing_metrics = data['text_metric'].to(
                    device, dtype=torch.float32)
                batch_size = ids.size(0)
                scores = model(ids, mask, preprocessing_metrics)
                scores = scores.to(torch.float32)
                total_scores += scores.tolist()
                targets = targets.to(torch.bool)
                binarization_targets += targets.tolist()
                loss = loss_fn(scores, targets)
            total_loss += (loss.item() * batch_size)
            running_loss += loss.item()
            cumul_batches += 1
            dataset_size += batch_size
            if idx_batch % log_interval == 0 and idx_batch > 0 and use_wandb:
                wandb.log(
                    {'Validation Running Loss': running_loss / cumul_batches})
                running_loss = 0
                cumul_batches = 0
    total_metrics['valid_loss'] = total_loss / dataset_size
    total_metrics['scores'] = total_scores
    if total_accuracy > 0:
        total_metrics['valid_accuracy'] = total_accuracy / dataset_size
    if len(binarization_targets) > 0:
        total_metrics['binarization_targets'] = binarization_targets
    return total_metrics


def train_loop(
        dataloader,
        model,
        loss_fn,
        optimizer,
        device,
        gradient_clipping,
        log_interval,
        dataset_type,
        use_wandb=True):
    """
    Executes the training loop on the given parameters.

    Parameters
    ----------
    dataloader : torch.utils.data.dataloader.DataLoader
        Dataloader.
    model : torch.nn.modules.module.Module
        Neural network model.
    loss_fn : torch.nn.modules.loss._Loss
        Loss function.
    device : torch.device
        Training device.
    gradient_clipping : float
        Maximum norm of the gradients.
    log_interval : int
        Logging interval.
    dataset_type : str
        Dataset type.
    use_wandb : bool, default True
        Weights & Biases flag.

    Returns
    -------
    total_metrics : dict
        Dictionary with training metrics.

    """
    model.train()
    total_metrics = {}
    total_loss = 0.0
    total_accuracy = 0.0
    running_loss = 0.0
    cumul_batches = 0
    dataset_size = 0
    for idx_batch, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        if dataset_type == 'scored':
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['target'].to(device, dtype=torch.float32)
            preprocessing_metric = data['preprocessing_metric'].to(
                device, dtype=torch.float32)
            batch_size = ids.size(0)
            scores = model(ids, mask, preprocessing_metric)
            scores = scores.to(torch.float32)
            loss = loss_fn(scores, targets)
        elif dataset_type == 'pairwise':
            more_toxic_ids = data['more_toxic_ids'].to(
                device, dtype=torch.long)
            more_toxic_mask = data['more_toxic_mask'].to(
                device, dtype=torch.long)
            more_toxic_metric = data['more_toxic_metric'].to(
                device, dtype=torch.long)
            less_toxic_ids = data['less_toxic_ids'].to(
                device, dtype=torch.long)
            less_toxic_mask = data['less_toxic_mask'].to(
                device, dtype=torch.long)
            less_toxic_metric = data['less_toxic_metric'].to(
                device, dtype=torch.long)
            targets = data['target'].to(device, dtype=torch.long)
            batch_size = more_toxic_ids.size(0)
            more_toxic_outputs = model(
                more_toxic_ids, more_toxic_mask, more_toxic_metric)
            less_toxic_outputs = model(
                less_toxic_ids, less_toxic_mask, less_toxic_metric)
            loss = loss_fn(more_toxic_outputs, less_toxic_outputs, targets)
            total_accuracy = (
                more_toxic_outputs > less_toxic_outputs).sum().item()
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()
        total_loss += (loss.item() * batch_size)
        running_loss += loss.item()
        cumul_batches += 1
        dataset_size += batch_size
        if idx_batch % log_interval == 0 and idx_batch > 0 and use_wandb:
            wandb.log({'Train Running Loss': running_loss / cumul_batches})
            running_loss = 0
            cumul_batches = 0
    total_metrics['train_loss'] = total_loss / dataset_size
    if total_accuracy > 0:
        total_metrics['train_accuracy'] = total_accuracy / dataset_size
    return total_metrics
