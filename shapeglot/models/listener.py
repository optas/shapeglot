"""
The MIT License (MIT)
Originally created sometime in 2019.
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import torch
from torch import nn


class Listener(nn.Module):
    def __init__(self, language_encoder, image_encoder, mlp_decoder, pc_encoder=None):
        super(Listener, self).__init__()
        self.language_encoder = language_encoder
        self.image_encoder = image_encoder
        self.pc_encoder = pc_encoder
        self.logit_encoder = mlp_decoder

    def forward(self, item_ids, padded_tokens, dropout_rate=0.5):
        visual_feats = self.image_encoder(item_ids, dropout_rate)
        lang_feats = self.language_encoder(padded_tokens, init_feats=visual_feats)

        if self.pc_encoder is not None:
            pc_feats = self.pc_encoder(item_ids, dropout_rate, pre_drop=False)
        else:
            pc_feats = None

        logits = []
        for i, l_feats in enumerate(lang_feats):
            if pc_feats is not None:
                feats = torch.cat([l_feats, pc_feats[:, i]], 1)
            else:
                feats = l_feats

            logits.append(self.logit_encoder(feats))
        return torch.cat(logits, 1)
