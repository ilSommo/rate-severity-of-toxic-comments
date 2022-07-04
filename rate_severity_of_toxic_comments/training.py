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
    if train_dataset_params['type'] == 'regression':
        loss_fn = nn.MSELoss()
    elif train_dataset_params['type'] == 'ranking':
        loss_fn = nn.MarginRankingLoss(
            margin=train_dataset_params['loss_margin'])
    train_batch_size = training_params['train_batch_size']
    valid_batch_size = training_params['valid_batch_size']
    lr = training_params['learning_rate']
    l2_reg = training_params['L2_regularization']
    grad_clipping = training_params['gradient_clipping']
    train_dataloader, val_dataloader = build_dataloaders(
        [training_data, val_data], batch_sizes=(train_batch_size, valid_batch_size), weighted_samplings=(training_params["weighted_sampling"], False))
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
        train_loop_stats.register_epoch(
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
    loss_history = train_loop_stats.get_loss_history()
    return model, loss_history


def test_loop(
        dataloader,
        model,
        loss_fn,
        device,
        log_interval,
        dataset_type,
        verbose=False,
        use_wandb=True,
        collect_predictions=False):
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
    epoch_metrics : dict
        Dictionary with testing metrics.

    """
    model.eval()
    epoch_metrics = {}
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    batch_loss = 0.0
    cumul_batches = 0
    dataset_size = 0

    if collect_predictions:
        epoch_metrics["predictions"] = []

    with torch.no_grad():
        for idx_batch, data in tqdm(
                enumerate(dataloader), total=len(dataloader), disable=not verbose):
            if dataset_type == 'regression':
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                targets = data['target'].to(device, dtype=torch.float32)
                preprocessing_metrics = data['preprocessing_metric'].to(
                    device, dtype=torch.float32)
                batch_size = ids.size(0)
                
                scores = model(ids, mask, preprocessing_metrics)
                scores = scores.to(torch.float32)
                
                loss = loss_fn(scores, targets)

                if collect_predictions:
                    epoch_metrics["predictions"] += [{
                        "idx": data["idx"][i].item(),
                        "target": targets[i].item(),
                        "prediction": scores[i].item(),
                        "error": abs(targets[i].item() - scores[i].item())
                    } for i in range(batch_size)]
            elif dataset_type == 'ranking':
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
                
                epoch_accuracy += (more_toxic_outputs >
                                   less_toxic_outputs).sum().item()
                if collect_predictions:
                    epoch_metrics["predictions"] += [{
                        "idx": data["idx"][i].item(),
                        "more_toxic": more_toxic_scores[i].item(),
                        "less_toxic": less_toxic_scores[i].item(),
                        "prediction": more_toxic_scores[i].item() + less_toxic_scores[i].item(),
                        "error": less_toxic_scores[i].item() - more_toxic_scores[i].item()
                    } for i in range(batch_size)]
            elif dataset_type == 'classification':
                ids = data['text_ids'].to(device, dtype=torch.long)
                mask = data['text_mask'].to(device, dtype=torch.long)
                targets = data['target'].to(device, dtype=torch.long)
                preprocessing_metrics = data['text_metric'].to(
                    device, dtype=torch.float32)
                batch_size = ids.size(0)
                
                scores = model(ids, mask, preprocessing_metrics)
                scores = scores.to(torch.float32)
                targets = targets.to(torch.bool)
                loss = loss_fn(scores, targets)
                
                if collect_predictions:
                    epoch_metrics["predictions"] += [{
                        "idx": data["idx"][i].item(),
                        "target": targets[i].item(),
                        "prediction": scores[i].item(),
                        "error": abs(targets[i].item() - scores[i].item())
                    } for i in range(batch_size)]
            
            epoch_loss += (loss.item() * batch_size)
            batch_loss += loss.item()
            cumul_batches += 1
            dataset_size += batch_size
            
            if idx_batch % log_interval == 0 and idx_batch > 0 and use_wandb:
                wandb.log(
                    {'Validation Running Loss': batch_loss / cumul_batches})
                batch_loss = 0
                cumul_batches = 0
    
    epoch_metrics['valid_loss'] = epoch_loss / dataset_size
    if epoch_accuracy > 0:
        epoch_metrics['valid_accuracy'] = epoch_accuracy / dataset_size
    return epoch_metrics


def train_loop(
        dataloader,
        model,
        loss_fn,
        optimizer,
        device,
        gradient_clipping,
        log_interval,
        dataset_type,
        verbose=True,
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
    epoch_metrics : dict
        Dictionary with training metrics.

    """
    model.train()
    epoch_metrics = {}
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    batch_loss = 0.0
    cumul_batches = 0
    dataset_size = 0
    
    for idx_batch, data in tqdm(enumerate(dataloader), total=len(dataloader), disable=not verbose):
        if dataset_type == 'regression':
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['target'].to(device, dtype=torch.float32)
            preprocessing_metric = data['preprocessing_metric'].to(
                device, dtype=torch.float32)
            batch_size = ids.size(0)
            
            scores = model(ids, mask, preprocessing_metric)
            scores = scores.to(torch.float32)
            loss = loss_fn(scores, targets)
        elif dataset_type == 'ranking':
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
            
            epoch_accuracy = (
                more_toxic_outputs > less_toxic_outputs).sum().item()
        
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()
        
        epoch_loss += (loss.item() * batch_size)
        batch_loss += loss.item()
        cumul_batches += 1
        dataset_size += batch_size
        
        if idx_batch % log_interval == 0 and idx_batch > 0 and use_wandb:
            wandb.log({'Train Running Loss': batch_loss / cumul_batches})
            batch_loss = 0
            cumul_batches = 0
    
    epoch_metrics['train_loss'] = epoch_loss / dataset_size
    if epoch_accuracy > 0:
        epoch_metrics['train_accuracy'] = epoch_accuracy / dataset_size
    return epoch_metrics
