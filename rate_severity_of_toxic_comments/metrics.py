__version__ = '1.0.0-rc.1'
__author__ = 'Lorenzo Menghini, Martino Pulici, Alessandro Stockman, Luca Zucchini'


import copy
import torch
import wandb
import matplotlib.pyplot as plt


class TrainLoopStatisticsManager:
    """
    Class managing the statistics of the train loop.

    Attributes
    ----------
    model : torch.nn.modules.module.Module
        Neural network model.
    early_stopping_patience : int
        Early stopping parameter.
    verbose : bool
        Verbosity flag.
    use_wandb : bool
        Weights & Biases flag.
    counter : int
        Early stopping counter.
    best_val_loss : float
        Best validation loss value.
    train_loss_history : list
        History of training loss.
    val_loss_history : list
        History of validation loss.
    early_stop : bool
        Early stopping flag.
    best_model_wts : torch.nn.modules.module.Module
        Best neural network model.

    Methods
    -------
    __init__(self, model, early_stopping_patience=7, verbose=True, use_wandb=False)
        Initializes the manager.
    registerEpoch(self, metrics_train: dict, metrics_val: dict, lr, epoch, time_start, time_end, early_stop_delta_sensibility=0.01)
        Registers an epoch.
    getLossHistory(self)
        Returns the loss history.
    _checkEarlyStop(self, val_loss, delta_sensibility)
        Checks for early stopping conditions.

    """

    def __init__(
            self,
            model,
            early_stopping_patience=7,
            verbose=True,
            use_wandb=False):
        """
        Initializes the manager.

        Parameters
        ----------
        model : torch.nn.modules.module.Module
            Neural network model.
        early_stopping_patience : int, default 7
            Early stopping parameter.
        verbose : bool, default True
            Verbosity flag.
        use_wandb : bool, default False
            Weights & Biases flag.

        """
        self.model = model
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose
        self.use_wandb = use_wandb
        self.counter = 0
        self.best_val_loss = None
        self.train_loss_history = []
        self.val_loss_history = []
        self.early_stop = False
        self.best_model_wts = copy.deepcopy(self.model.state_dict())

    def registerEpoch(
            self,
            metrics_train: dict,
            metrics_val: dict,
            lr,
            epoch,
            time_start,
            time_end,
            early_stop_delta_sensibility=0.01):
        """
        Registers an epoch.

        Parameters
        ----------
        metrics_train : dict
            Training metrics.
        metrics_val : dict
            Validation metrics.
        lr : float
            Learning rate.
        epoch : int
            Epoch number.
        time_start : float
            Epoch starting time.
        time_end : float
            Epoch ending time.
        early_stop_delta_sensibility : float, default 0.01
            Early stopping sensibility.

        """
        all_metrics = metrics_train
        all_metrics.update(metrics_val)
        valid_loss = metrics_val['valid_loss']
        train_loss = metrics_train['train_loss']
        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(valid_loss)
        if self.use_wandb:
            wandb.log(all_metrics)
        if self.verbose:
            print(f'Epoch: {epoch} '
                  f' Lr: {lr:.8f} '
                  f' | Time one epoch (s): {(time_end - time_start):.4f} '
                  f' \n Train - '
                  f' Loss: [{train_loss:.4f}] '
                  f' \n Val   - '
                  f' Loss: [{valid_loss:.4f}] ')
        if self.best_val_loss is None or valid_loss <= self.best_val_loss:
            self.best_val_loss = valid_loss
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
        else:
            self._checkEarlyStop(
                valid_loss, delta_sensibility=early_stop_delta_sensibility)

    def getLossHistory(self):
        """
        Returns the loss history.

        Returns
        -------
        loss_history : dict
            Loss history.

        """
        loss_history = {
            "train": self.train_loss_history,
            "valid": self.val_loss_history,
        }
        return loss_history

    def _checkEarlyStop(self, val_loss, delta_sensibility):
        """
        Checks for early stopping conditions.

        Parameters
        ----------
        val_loss : float
            Validation loss.
        delta_sensibility : floag
            Early stopping sensibility.

        """
        if val_loss < self.best_val_loss - delta_sensibility:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.early_stopping_patience}')
            if self.counter >= self.early_stopping_patience:
                self.early_stop = True
        else:
            self.counter = 0


def f1_score(precision: torch.Tensor, recall: torch.Tensor) -> torch.Tensor:
    """
    Computes the F1 score given the tensor representing the `precision` and `recall` 
    """
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else torch.tensor(0)


def compute_metrics(predicted_label, label):
    """
    Returns a dictionary containing accuracy, precision and recall metrics.
    """
    predicted_label = torch.tensor(
        list(map(lambda v: 1 if v > 0.5 else 0, predicted_label)))
    positive = torch.sum(label).float()
    predicted_positive = torch.sum(predicted_label).float()
    true_positive = torch.sum(
        (predicted_label == label) * predicted_label).float()

    accuracy = torch.sum(predicted_label == label).float() / label.size(0)
    precision = true_positive / \
        predicted_positive if predicted_positive > 0 else torch.tensor(0)
    recall = true_positive / positive if positive > 0 else torch.tensor(0)
    return {
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1_score(precision, recall).item()
    }


def plot_metrics(metrics):

    plt.bar(range(len(metrics)), list(metrics.values()), align='center')
    plt.xticks(range(len(metrics)), list(metrics.keys()))
    # # for python 2.x:
    # plt.bar(range(len(D)), D.values(), align='center')  # python 2.x
    # plt.xticks(range(len(D)), D.keys())  # in python 2.x
    plt.show()
