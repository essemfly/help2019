import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from numpy import log


class LSTM(nn.Module):
    def __init__(self, input_size=73, hidden_size=64, num_layers=1, num_labels=1, batch_size=64, positive_prob=0.0059,
                 device='cpu'):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device
        # 0 or 1 classification
        self.num_labels = num_labels

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)

        # output layer which projects back to label space
        self.hidden_to_label = nn.Linear(self.hidden_size, self.num_labels, bias=True)
        self.hidden_to_label.bias.data.fill_(-log((1 - positive_prob) / positive_prob))

    def init_hidden_cell(self):
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        cell = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        hidden = hidden.to(self.device)
        cell = cell.to(self.device)
        return (Variable(hidden), Variable(cell))

    def forward(self, X, X_lengths):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        self.hidden_cell = self.init_hidden_cell()
        # ---------------------
        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        # X = self.word_embedding(X)
        # We've already embedded our X

        # ---------------------
        # 2. Run through RNN
        # TRICK 2 ********************************
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True, enforce_sorted=False)

        # now run through LSTM
        X, self.hidden_cell = self.lstm(X, self.hidden_cell)

        # undo the packing operation
        # X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        output = self.hidden_to_label(self.hidden_cell[0][-1])
        # ---------------------
        # 4. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, nb_lstm_units) -> (batch_size, seq_len, nb_tags)
        # X = F.log_softmax(X, dim=1)

        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, nb_tags)
        # X = X.view(batch_size, seq_len, self.nb_tags)

        # Y_hat = X
        # return Y_hat
        return output


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        # Do not use label smoothing. (We assume that each label is either 0 or 1.)
        # BCE_loss = -log(pt) where pt is sigmoid(logits).
        # focal_loss = -(1-pt)**gamma * log(pt)
        targets = targets.float()
        BCE_loss = F.binary_cross_entropy_with_logits(inputs.view(-1), targets.view(-1), reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()
