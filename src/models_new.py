import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .constants import model_config, hyperparams
from math import log


class NicuLSTM(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        drop_prob = model_config['drop_prob'] if model_config['num_layers'] > 1 else 0.0
        self.lstm = nn.LSTM(input_size=model_config['measure_dim'], hidden_size=model_config['embedd_dim'],
                            batch_first=True, num_layers=model_config['num_layers'], dropout=drop_prob)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(model_config['num_layers'], batch_size, model_config['embedd_dim'])
        hidden = hidden.to(self.device)
        return Variable(hidden)

    def _init_cell(self, batch_size):
        cell = torch.zeros(model_config['num_layers'], batch_size, model_config['embedd_dim'])
        cell = cell.to(self.device)
        return Variable(cell)

    def forward(self, x, x_len):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        hidden = self._init_hidden(x.size()[0])
        cell = self._init_cell(x.size()[0])

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
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)

        # now run through LSTM
        x, _ = self.lstm(x, (hidden, cell))

        # undo the packing operation
        x, x_len = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=hyperparams['max_seq_len'])
        return x, x_len


class NicuEmbeddings(nn.Module):
    """Construct the embeddings from measurement hidden state and positions"""

    def __init__(self, device='cpu'):
        super().__init__()
        self.t_embedding = nn.Embedding(hyperparams['max_seq_len'] + 1, model_config['embedd_dim'], padding_idx=0)
        self.m_embedding = NicuLSTM(device=device)
        self.device = device
        self.layernorm = nn.LayerNorm(model_config['embedd_dim'])
        self.dropout = nn.Dropout(model_config['drop_prob'])

    def _init_time_idx(self, batch_size):
        time_idx = torch.zeros(batch_size, hyperparams['max_seq_len']).long()
        time_idx = time_idx.to(self.device)
        return Variable(time_idx)

    def forward(self, x, x_len):
        x, x_len = self.m_embedding(x, x_len)
        time_idx = self._init_time_idx(x.size()[0])
        for i, seq_len in enumerate(x_len.tolist()):
            if seq_len < hyperparams['max_seq_len']:
                time_idx.data[i, :seq_len] = torch.arange(hyperparams['max_seq_len'] - seq_len + 1,
                                                          hyperparams['max_seq_len'] + 1)
                # time_idx.data[i, :seq_len] = torch.arange(1, seq_len + 1)
            else:
                time_idx.data[i] = torch.arange(1, hyperparams['max_seq_len'] + 1)
        embeddings = self.t_embedding(time_idx) + x
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class NicuEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(model_config['embedd_dim'],
                                               model_config['num_heads'], dropout=model_config['drop_prob'])
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(model_config['embedd_dim'], model_config['ffn_dim'])
        self.linear2 = nn.Linear(model_config['ffn_dim'], model_config['embedd_dim'])

        self.layernorm1 = nn.LayerNorm(model_config['embedd_dim'])
        self.layernorm2 = nn.LayerNorm(model_config['embedd_dim'])
        self.dropout1 = nn.Dropout(model_config['drop_prob'])
        self.dropout2 = nn.Dropout(model_config['drop_prob'])

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.layernorm1(src)
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout2(src2)
        src = self.layernorm2(src)
        return src


class NicuClassifier(nn.Module):
    def __init__(self, prior_prob=None):
        super().__init__()
        self.linear = nn.Linear(model_config['embedd_dim'], model_config['num_labels'])
        if prior_prob is not None:
            self.linear.bias.data.fill_(-log((1 - prior_prob) / prior_prob))

    def forward(self, src):
        src = src.sum(dim=1)
        return self.linear(src)


class NicuModel(nn.Module):
    def __init__(self, device='cpu', prior_prob=None):
        super().__init__()
        self.embedding = NicuEmbeddings(device=device)
        self.encoder1 = NicuEncoder()
        self.encoder2 = NicuEncoder()
        self.classifier = NicuClassifier(prior_prob)

    def forward(self, x, x_len):
        return self.classifier(self.encoder2(self.encoder1(self.embedding(x, x_len))))


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
