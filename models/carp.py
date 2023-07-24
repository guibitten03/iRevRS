# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class CARP(nn.Module):
    '''
    CARP
    '''
    def __init__(self, opt, uori='user'):
        super(CARP, self).__init__()
        
        self.opt = opt
        
        self.user_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)  # vocab_size * 300
        self.item_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)  # vocab_size * 300
        
        # Aplicar uma relu
        self.user_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim))
        self.item_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim))
        
        self.dropout = nn.Dropout(self.opt.drop_out)


    def forward(self, datas):
        _, _, uids, iids, _, _, user_doc, item_doc = datas
        
        u_doc = self.user_word_embs(user_doc)
        i_doc = self.item_word_embs(item_doc)
        
        u_fea = F.relu(self.user_cnn(u_doc.unsqueeze(1)))
        i_fea = F.relu(self.item_cnn(i_doc.unsqueeze(1)))
        
        u_fea = u_fea * torch.sigmoid()

        

    def reset_para(self):

        for cnn in [self.user_cnn, self.item_cnn]:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        for fc in [self.user_fc_linear, self.item_fc_linear]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.constant_(fc.bias, 0.1)

        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.user_word_embs.weight.data.copy_(w2v.cuda())
                self.item_word_embs.weight.data.copy_(w2v.cuda())
            else:
                self.user_word_embs.weight.data.copy_(w2v)
                self.item_word_embs.weight.data.copy_(w2v)
        else:
            nn.init.uniform_(self.user_word_embs.weight, -0.1, 0.1)
            nn.init.uniform_(self.item_word_embs.weight, -0.1, 0.1)


class ViewPointSelfAttention(nn.Module):
    def __init__(self):
        super(ViewPointSelfAttention, self).__init__()
        
        self.u_viewpoint_emb = 