"""
The MIT License (MIT)
Originally created sometime in 2019.
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PretrainedFeatures(nn.Module):
    """
    Store a collection of features in a 2D matrix, that can be indexed and
    projected via an FC layer in a space with specific dimensionality.
    """
    def __init__(self, pretrained_features, embed_size):
        super(PretrainedFeatures, self).__init__()
        self.pretrained_feats = nn.Parameter(pretrained_features, requires_grad=False)
        self.fc = nn.Linear(self.pretrained_feats.shape[1], embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

    def __getitem__(self, index):
        assert index.ndim == 2
        res = torch.index_select(self.pretrained_feats, 0, index.flatten())
        res = res.view(index.size(0), index.size(1), res.size(1))
        return res

    def forward(self, index, dropout_prob=0.0, pre_drop=True):
        """ Apply dropout-fc-relu-dropout and return the specified by the index features.
        :param index: B x K
        :param dropout_prob:
        :param pre_drop: Boolean, if True it drops-out the pretrained feature before projection.
        :return: B x K x feat_dim
        """
        x = self[index]
        assert x.ndim == 3
        res = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            if pre_drop:
                x_i = F.dropout(x_i, dropout_prob, self.training)
            x_i = F.relu(self.fc(x_i), inplace=True)
            x_i = self.bn(x_i)
            x_i = F.dropout(x_i, dropout_prob, self.training)
            res.append(x_i)
        res = torch.stack(res, 1)
        return res


class LanguageEncoder(nn.Module):
    """
    Currently it reads the tokens via an LSTM initialized on a specific context feature and
    return the last output of the LSTM.
    https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
    """
    def __init__(self, n_hidden, embedding_dim, vocab_size, padding_idx=0):
        super(LanguageEncoder, self).__init__()
        # Whenever the embedding sees the padding index
        # it'll make the whole vector zeros
        self.padding_idx = padding_idx
        self.word_embedding = nn.Embedding(vocab_size,
                                           embedding_dim=embedding_dim,
                                           padding_idx=padding_idx)
        self.rnn = nn.LSTM(embedding_dim, n_hidden, batch_first=True)

    def forward(self, padded_tokens, init_feats=None, drop_out_rate=0.5):
        w_emb = self.word_embedding(padded_tokens)
        w_emb = F.dropout(w_emb, drop_out_rate, self.training)
        len_of_sequence = (padded_tokens != self.padding_idx).sum(dim=1)
        x_packed = pack_padded_sequence(w_emb, len_of_sequence, enforce_sorted=False, batch_first=True)

        context_size = 1
        if init_feats is not None:
            context_size = init_feats.shape[1]

        batch_size = len(padded_tokens)
        res = []
        for i in range(context_size):
            init_i = init_feats[:, i].contiguous()
            init_i = torch.unsqueeze(init_i, 0)    # rep-mat if multiple LSTM cells.
            rnn_out_i, _ = self.rnn(x_packed, (init_i, init_i))
            rnn_out_i, dummy = pad_packed_sequence(rnn_out_i, batch_first=True)
            lang_feat_i = rnn_out_i[torch.arange(batch_size), len_of_sequence - 1]
            res.append(lang_feat_i)
        return res
