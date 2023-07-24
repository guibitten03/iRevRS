# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class HDRL(nn.Module):
    '''
    HDRL recommend system
    '''
    def __init__(self, opt, uori='user'):
        super(HDRL, self).__init__()
        self.opt = opt

        if uori == 'user':
            id_num = self.opt.item_num
        else:
            id_num = self.opt.user_num

        self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)
        self.id_embedding = nn.Embedding(self.opt.user_num, self.opt.item_num)
        self.id_embedding.weight.requires_grad = False

        self.mlp_layers = nn.Sequential(
            nn.Linear(id_num, id_num // 2),
            nn.ReLU(),
            nn.Linear(id_num // 2, id_num // 4),
            nn.ReLU(),
            nn.Linear(id_num // 4, self.opt.r_filters_num),
            nn.LayerNorm(self.opt.r_filters_num),
        )

        # single query vector in sentence level
        self.s_query_v = nn.Parameter(torch.randn(1, self.opt.r_filters_num))
        self.w_query_v = nn.Parameter(torch.randn(1, self.opt.r_filters_num))

        self.attention_linear = nn.Linear(self.opt.attention_size, 1)

        self.fc_layer = nn.Linear(self.opt.r_filters_num, self.opt.attention_size)

        self.r_encoder = CNN(self.opt.r_filters_num, self.opt.kernel_size, self.opt.word_dim)

        self.dropout = nn.Dropout(self.opt.drop_out)

        self.init_word_emb()
        
        self.init_model_weight()

    def forward(self, review, review_len, max_num, index, att_method, rm_merge, review_weight, uori):
        # ------------------- word embedding -----------------------------
        review = self.word_embs(review)

        if self.opt.use_word_drop:
            review = self.dropout(review)

        # -----------------------matrix ----------------------------------
        if uori == 'user':
            matrix_vevtor = self.id_embedding(index)
        else:
            matrix_vevtor = (self.id_embedding.weight[:, index]).t()
        Matrix_vevtor = self.mlp_layers(matrix_vevtor)

        # ------------------- cnn for review -----------------------------
        r_fea = self.r_encoder(review, max_num, review_len, pooling='MAX', qv=self.w_query_v)

        # -------------------------attention method-----------------------
        if att_method == 'add':
            s_query_v = self.s_query_v + Matrix_vevtor
        elif att_method == 'dot':
            s_query_v = self.s_query_v * Matrix_vevtor
        elif att_method == 'matrix':
            s_query_v = Matrix_vevtor
        elif att_method == 'normal':
            s_query_v = self.s_query_v

        att_weight = torch.bmm(r_fea, s_query_v.unsqueeze(2))

        # -------------------------review weight method-------------------
        if review_weight == 'mean':
            att_score = torch.mean(att_weight, 1)
            r_fea = F.relu(r_fea) * att_score.unsqueeze(2)
        elif review_weight == 'softmax':
            att_score = (F.softmax(att_weight, 1))
            r_fea = F.relu(r_fea) * att_score
        r_fea = r_fea.sum(1)

        # --------------------------aggregate method----------------------
        if rm_merge == 'cat':
            all_feature = torch.cat((r_fea, Matrix_vevtor), 1)
        elif rm_merge == 'dot':
            all_feature = r_fea * Matrix_vevtor
        elif rm_merge == 'add':
            all_feature = Matrix_vevtor + r_fea

        return all_feature

    def init_word_emb(self):
        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.word_embs.weight.data.copy_(w2v.cuda())
            else:
                self.word_embs.weight.data.copy_(w2v)
        else:
            nn.init.xavier_normal_(self.word_embs.weight)

        # ----------------Matrix init method--------------#
        ratingMatrix = torch.from_numpy(np.load(self.opt.ratingMatrix_path))
        self.id_embedding.weight.data.copy_(ratingMatrix.cuda())

    def init_model_weight(self):
        nn.init.xavier_uniform_(self.r_encoder.cnn.weight)
        nn.init.uniform_(self.r_encoder.cnn.bias, a=-0.1, b=0.1)

        nn.init.xavier_normal_(self.s_query_v)
        nn.init.xavier_normal_(self.w_query_v)

        nn.init.xavier_uniform_(self.mlp_layers[0].weight)
        nn.init.constant_(self.mlp_layers[0].bias, 0.1)

        nn.init.xavier_uniform_(self.mlp_layers[2].weight)
        nn.init.constant_(self.mlp_layers[2].bias, 0.1)

        nn.init.xavier_uniform_(self.mlp_layers[4].weight)
        nn.init.constant_(self.mlp_layers[4].bias, 0.1)


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
