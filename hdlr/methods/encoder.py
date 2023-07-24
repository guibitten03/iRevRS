# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNN(nn.Module):
    '''
    for review and summary encoder
    '''

    def __init__(self, filters_num, k1, k2, padding=True):
        super(CNN, self).__init__()

        if padding:
            self.cnn = nn.Conv2d(1, filters_num, (k1, k2), padding=(int(k1 / 2), 0))
        else:
            self.cnn = nn.Conv2d(1, filters_num, (k1, k2))

    def multi_attention_pooling(self, x, qv):
        '''
        x: 704 * 100 * 224
        qv: 5 * 100
        '''
        att_weight = torch.matmul(x.permute(0, 2, 1), qv.t())  # 704 * 224 * 5
        att_score = F.softmax(att_weight, dim=1) * np.sqrt(att_weight.size(1))  # 704 * 224 *5
        x = torch.bmm(x, att_score)  # 704 * 100 * 5
        x = x.view(-1, x.size(1) * x.size(2))  # 704 * 500
        return x

    def attention_pooling(self, x, qv):
        '''
        x: 704 * 224 * 100
        qv: 704 * 100
        '''
        att_weight = torch.matmul(x, qv)
        att_score = F.softmax(att_weight, dim=1)
        x = x * att_score

        return x.sum(1)

    def forward(self, x, max_num, review_len, pooling="MAX", qv=None):
        '''
        eg. user
        x: (32, 11, 224, 300)
        multi_qv: 5 * 100
        qv: 32, 11, 100
        '''
        x = x.view(-1, review_len, self.cnn.kernel_size[1])
        x = x.unsqueeze(1)
        x = F.relu(self.cnn(x)).squeeze(3)
        if pooling == 'multi_att':
            assert qv is not None
            x = self.multi_attention_pooling(x, qv)
            x = x.view(-1, max_num, self.cnn.out_channels * qv.size(0))
        elif pooling == "att":
            x = x.permute(0, 2, 1)
            qv = qv.t()
            x = self.attention_pooling(x, qv)
            x = x.view(-1, max_num, self.cnn.out_channels)
        else:
            x = F.max_pool1d(x, x.size(2)).squeeze(2)  # B, F
            x = x.view(-1, max_num, self.cnn.out_channels)

        return x
