import wandb
import copy
import numpy as np


class TrainLoopStatisticsManager:

    def __init__(self, model, early_stopping_patience=7, verbose=True, wandb=None):
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose
        self.wandb = wandb
        self.model = model
        self.counter = 0
        self.best_val_loss = None
        self.train_loss_history = []
        self.val_loss_history = []
        self.early_stop = False,
        self.best_model_wts = copy.deepcopy(self.model.state_dict())

    def registerEpoch(self, metrics_train: dict, metrics_val: dict, lr, epoch, time_start, time_end, early_stop_delta_sensibility=0.01):

        all_metrics = metrics_train
        all_metrics.update(metrics_val)

        valid_loss = metrics_val["valid_loss"]
        train_loss = metrics_train["train_loss"]
        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(valid_loss)

        if self.wandb != None:
            wandb.log(all_metrics)

        if self.verbose:
            print(f'Epoch: {epoch} '
                  f' Lr: {lr:.8f} '
                  f' | Time one epoch (s): {(time_end - time_start):.4f} '
                  f' \n Train - '
                  f' Loss: [{train_loss:.4f}] '
                  f' \n Val   - '
                  f' Loss: [{valid_loss:.4f}] '
                  )

        if self.best_val_loss == None or valid_loss <= self.best_val_loss:
            self.best_val_loss = valid_loss
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
        else:
            self._checkEarlyStop(
                valid_loss, delta_sensibility=early_stop_delta_sensibility)

    def getLossHistory(self):
        return {
            "train": self.train_loss_history,
            "valid": self.val_loss_history,
        }

    def _checkEarlyStop(self, val_loss, delta_sensibility):

        if val_loss < self.best_val_loss - delta_sensibility:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.early_stopping_patience}')
            if self.counter >= self.early_stopping_patience:
                self.early_stop = True
        else:
            self.counter = 0
