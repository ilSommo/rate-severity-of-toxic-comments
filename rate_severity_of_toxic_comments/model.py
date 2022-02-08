__version__ = '1.0.0-rc'
__author__ = 'Lorenzo Menghini, Martino Pulici, Alessandro Stockman, Luca Zucchini'


import torch
from torch.nn import LSTM, GRU, Embedding, Module, Dropout, ReLU, Linear, Sigmoid
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel


OUTPUT_CLASSES = 1
AVAILABLE_ARCHITECTURES = ['LSTM', 'GRU', 'BiDi']


class PretrainedModel(Module):
    """
    Class containing a pretrained neural network model.

    Attributes
    ----------
    model : torch.nn.modules.module.Module
        Neural network model.
    sig : torch.nn.modules.activation.Sigmoid
        Sigmoid layer.
    fc : torch.nn.modules.linear.Linear
        Linear layer.

    Methods
    -------
    __init__(self, model_name, dropout, output_features)
        Initializes the model.
    forward(self, ids, mask, _)
        Defines the computation performed at every call.

    """

    def __init__(self, model_name, dropout, output_features):
        """
        Initializes the model.

        Parameters
        ----------
        model_name : str
            Name of the model.
        dropout : float
            Dropout parameter.
        output_features : int
            Number of ouput features.

        """
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        if dropout is not None:
            self.drop = Dropout(p=dropout)
        self.sig = Sigmoid()
        self.fc = Linear(output_features, OUTPUT_CLASSES)

    def forward(self, ids, mask, _):
        """
        Defines the computation performed at every call.

        Parameters
        ----------
        ids : torch.Tensor
            Tensor of ids.
        mask : torch.Tensor
            Tensor of masks.

        Returns
        -------
        x : torch.Tensor
            Tensor to return.

        """
        out = self.model(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=False)
        cls_token = out.last_hidden_state[:, 0, :]
        outputs = self.fc(cls_token)
        x = self.sig(outputs).squeeze()
        return x


class RecurrentModel(Module):
    """
    Class containing a recurrent neural network model.

    Attributes
    ----------
    embedding : torch.nn.modules.sparse.Embedding
        Embedding lookup table.
    recurrent : torch.nn.modules.rnn.RNNBase
        Recurrent Neural Network.
    preprocessing_metric : bool
        Preprocessing metric flag.
    drop : torch.nn.modules.dropout.Dropout
        Dropout layer.
    relu : torch.nn.modules.activation.ReLU
        ReLU activation layer.
    sig : torch.nn.modules.activation.Sigmoid
        Sigmoid layer.
    fc : torch.nn.modules.linear.Linear
        Linear layer.


    Methods
    -------
    __init__(self, embedding_matrix, dropout, hidden_dim, architecture, preprocessing_metric)
        Initializes the model.
    forward(self, ids, mask, preprocessing_metric)
        Defines the computation performed at every call.

    """

    def __init__(
            self,
            embedding_matrix,
            dropout,
            hidden_dim,
            architecture,
            preprocessing_metric):
        """
        Initializes the model.

        Parameters
        ----------
        embedding_matrix : numpy.ndarray
            Embedding matrix.
        dropout : float
            Dropout parameter.
        hidden_dim : int
            Hidden dimension.
        architecture : str
            Name of architecture.
        preprocessing_metric : bool
            Preprocessing metric flag.

        """
        super().__init__()
        _, embedding_dim = embedding_matrix.shape
        self.embedding = Embedding.from_pretrained(
            torch.tensor(embedding_matrix))
        if architecture == 'LSTM':
            self.recurrent = LSTM(embedding_dim, hidden_dim,
                                  batch_first=True)
        elif architecture == 'GRU':
            self.recurrent = GRU(embedding_dim, hidden_dim,
                                 batch_first=True)
        elif architecture == 'BiDi':
            self.recurrent = LSTM(embedding_dim, hidden_dim,
                                  batch_first=True, bidirectional=True)
            hidden_dim = hidden_dim * 2
        self.preprocessing_metric = preprocessing_metric
        self.drop = Dropout(p=dropout)
        self.relu = ReLU()
        self.sig = Sigmoid()
        self.fc = Linear(
            hidden_dim +
            int(preprocessing_metric),
            OUTPUT_CLASSES)

    def forward(self, ids, mask, preprocessing_metric):
        """
        Defines the computation performed at every call.

        Parameters
        ----------
        ids : torch.Tensor
            Tensor of ids.
        mask : torch.Tensor
            Tensor of masks.
        preprocessing_metric : torch.Tensor
            Tensor of preprocessing metric values.

        Returns
        -------
        x : torch.Tensor
            Tensor to return.

        """
        embedded = self.embedding(ids)
        lengths = torch.count_nonzero(mask, dim=1)
        batch_lengths = lengths.to('cpu')
        embedded = pack_padded_sequence(
            embedded, batch_lengths, batch_first=True, enforce_sorted=False)
        rec_out, _ = self.recurrent(embedded)
        rec_out = pad_packed_sequence(
            rec_out, batch_first=True)
        drop_out = self.drop(rec_out[0])
        x = torch.mean(drop_out, dim=-2)
        if self.preprocessing_metric:
            x = torch.cat((x, preprocessing_metric[:, None]), dim=1)
        x = self.relu(x)
        x = self.fc(x)
        x = self.sig(x).squeeze()
        return x


def create_model(run_mode, train_params, model_params, support_bag):
    """
    Returns a model.

    Parameters
    ----------
    run_mode : str
        Run mode.
    train_params : dict
        Training parameters.
    model_params : dict
        Model parameters.
    support_bag : dict
        Configuration parameters.

    Returns
    -------
    model : torch.nn.modules.module.Module
        Model to return.

    """
    if run_mode == 'recurrent':
        model = RecurrentModel(
            support_bag['embedding_matrix'],
            train_params['dropout'],
            model_params['hidden_dim'],
            model_params['architecture'],
            model_params['preprocessing_metric'])
    elif run_mode == 'pretrained':
        model = PretrainedModel(
            model_params['model_name'],
            train_params['dropout'],
            model_params['output_features'])
    return model
