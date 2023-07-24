import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.cuda.empty_cache()


class ANR(nn.Module):
    def __init__(self, opt):
        super(ANR, self).__init__()

        self.opt = opt
        self.num_fea = 1

        self.net = Net(opt)

    def forward(self, datas):
        # _, _, uids, iids, user_item2id, item_user2id, user_doc, item_doc = datas
        ratings = self.net(datas)
        return ratings


class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        self.opt = opt

        self.uid_user_doc = nn.Embedding(opt.user_num, opt.doc_len)
        self.uid_user_doc.weight.requires_grad = False

        self.iid_item_doc = nn.Embedding(opt.item_num, opt.doc_len)
        self.iid_item_doc.weight.requires_grad = False

        self.word_emb = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)
        self.word_emb.weight.requires_grad = False

        self.arl = ARL(self.opt)

        self.aie = AIE(self.opt)

        self.rating_pred = ANR_RatingPred(opt.user_num, opt.item_num)

    def forward(self, datas):

        _, _, uids, iids, user_item2id, item_user2id, user_doc, item_doc = datas

        # Input
        batch_userDoc = self.uid_user_doc(uids)
        batch_itemDoc = self.iid_item_doc(iids)

        # Embedding Layer
        batch_userDocEmbed = self.word_emb(batch_userDoc.long())
        batch_itemDocEmbed = self.word_emb(batch_itemDoc.long())

        userAspAttn, userAspDoc = self.arl(batch_userDocEmbed)
        itemAspAttn, itemAspDoc = self.arl(batch_itemDocEmbed)

        userCoAttn, itemCoAttn = self.aie(userAspDoc, itemAspDoc)

        rating_pred = self.ANR_RatingPred(
            userAspDoc, itemAspDoc, userCoAttn, itemCoAttn, uids, iids)

        return rating_pred


'''
Aspect-based Representation Learning (ARL)
'''


class ARL(nn.Module):

    def __init__(self, opt):
        super(ARL, self).__init__()

        # num_aspects = 5
        # ctx_win_size = 3
        # h1 = 10
        # word_embed_dim = 300

        # Aspect Embeddings
        self.aspects_embeddings = nn.Embedding(5, 3 * 10)
        self.aspects_embeddings.weight.requires_grad = True

        # Aspect-Specific Projection Matrices
        self.aspects_projection = nn.Parameter(
            torch.Tensor(5, 300, 10), requires_grad=True)

        # Initialize all weights using random uniform distribution from [-0.01, 0.01]
        self.aspects_embeddings.weight.data.uniform_(-0.01, 0.01)
        self.aspects_projection.data.uniform_(-0.01, 0.01)

    '''
	[Input]		batch_docIn:	bsz x max_doc_len x word_embed_dim
	[Output]	batch_aspRep:	bsz x num_aspects x h1
	'''

    def forward(self, batch_docIn):
        # Cria uma matriz de aspectos 

        # Loop over all aspects
        lst_batch_aspAttn = []
        lst_batch_aspRep = []
        
        for a in range(5):

            # Aspect-Specific Projection of Input Word Embeddings: (bsz x max_doc_len x h1)
            # Faz uma multiplicação de matrizes
            
            print(self.aspects_projection[a])
            
            batch_aspProjDoc = torch.matmul(
                batch_docIn, self.aspects_projection[a])

    # self.args.use_cuda = True

            # Aspect Embedding: (bsz x h1 x 1) after tranposing!
            batch_size = batch_docIn.size()[0]
            # batch_aspEmbed = self.aspects_embeddings( to_var(torch.LongTensor(batch_size, 1).fill_(a), use_cuda = True) )
            batch_aspEmbed = self.aspects_embeddings(
                torch.LongTensor(batch_size, 1).fill_(a), use_cuda=True)
            batch_aspEmbed = torch.transpose(batch_aspEmbed, 1, 2)

            # Window Size (self.args.ctx_win_size) of 1: Calculate Attention based on the word itself!
            if (3 == 1):

                # Calculate Attention: Inner Product & Softmax
                # (bsz x max_doc_len x h1) x (bsz x h1 x 1) -> (bsz x max_doc_len x 1)
                batch_aspAttn = torch.matmul(batch_aspProjDoc, batch_aspEmbed)
                batch_aspAttn = F.softmax(batch_aspAttn, dim=1)

            # Context-based Word Importance
            # Calculate Attention based on the word itself, and the (self.args.ctx_win_size - 1) / 2 word(s) before & after it
            else:

                # Pad the document
                # pad_size = int( (self.args.ctx_win_size - 1) / 2 )
                pad_size = 1

                batch_aspProjDoc_padded = F.pad(
                    batch_aspProjDoc, (0, 0, pad_size, pad_size), "constant", 0)

                # Use "sliding window" using stride of 1 (word at a time) to generate word chunks of ctx_win_size
                # (bsz x max_doc_len x h1) -> (bsz x max_doc_len x (ctx_win_size x h1))
                batch_aspProjDoc_padded = batch_aspProjDoc_padded.unfold(
                    1, 3, 1)

                batch_aspProjDoc_padded = torch.transpose(
                    batch_aspProjDoc_padded, 2, 3)

        # max_doc_len = 500

                batch_aspProjDoc_padded = batch_aspProjDoc_padded.contiguous().view(-1, 500, 3 * 10)

                # Calculate Attention: Inner Product & Softmax
                # (bsz x max_doc_len x (ctx_win_size x h1)) x (bsz x (ctx_win_size x h1) x 1) -> (bsz x max_doc_len x 1)
                batch_aspAttn = torch.matmul(
                    batch_aspProjDoc_padded, batch_aspEmbed)

                batch_aspAttn = F.softmax(batch_aspAttn, dim=1)

            # Weighted Sum: Broadcasted Element-wise Multiplication & Sum over Words
            # (bsz x max_doc_len x 1) and (bsz x max_doc_len x h1) -> (bsz x h1)
            batch_aspRep = batch_aspProjDoc * \
                batch_aspAttn.expand_as(batch_aspProjDoc)

            batch_aspRep = torch.sum(batch_aspRep, dim=1)

            # Store the results (Attention & Representation) for this aspect
            lst_batch_aspAttn.append(torch.transpose(batch_aspAttn, 1, 2))
            lst_batch_aspRep.append(torch.unsqueeze(batch_aspRep, 1))

        # Reshape the Attentions & Representations
        # batch_aspAttn:	(bsz x num_aspects x max_doc_len)
        # batch_aspRep:		(bsz x num_aspects x h1)
        batch_aspAttn = torch.cat(lst_batch_aspAttn, dim=1)
        batch_aspRep = torch.cat(lst_batch_aspRep, dim=1)

        # Returns the aspect-level attention over document words, and the aspect-based representations
        return batch_aspAttn, batch_aspRep


'''
Aspect Importance Estimation (AIE)
'''


class AIE(nn.Module):

    def __init__(self, opt):

        super(AIE, self).__init__()

        self.opt = opt

    # h2 = 50

        # Matrix for Interaction between User Aspect-level Representations & Item Aspect-level Representations
        # This is a learnable (h1 x h1) matrix, i.e. User Aspects - Rows, Item Aspects - Columns
        self.W_a = nn.Parameter(torch.Tensor(10, 10), requires_grad=True)

        # User "Projection": A (h2 x h1) weight matrix, and a (h2 x 1) vector
        self.W_u = nn.Parameter(torch.Tensor(50, 10), requires_grad=True)
        self.w_hu = nn.Parameter(torch.Tensor(50, 1), requires_grad=True)

        # Item "Projection": A (h2 x h1) weight matrix, and a (h2 x 1) vector
        self.W_i = nn.Parameter(torch.Tensor(50, 10), requires_grad=True)
        self.w_hi = nn.Parameter(torch.Tensor(50, 1), requires_grad=True)

        # Initialize all weights using random uniform distribution from [-0.01, 0.01]
        self.W_a.data.uniform_(-0.01, 0.01)

        self.W_u.data.uniform_(-0.01, 0.01)
        self.w_hu.data.uniform_(-0.01, 0.01)

        self.W_i.data.uniform_(-0.01, 0.01)
        self.w_hi.data.uniform_(-0.01, 0.01)

    '''
	[Input]		userAspRep:		bsz x num_aspects x h1
	[Input]		itemAspRep:		bsz x num_aspects x h1
	'''

    def forward(self, userAspRep, itemAspRep):

        userAspRepTrans = torch.transpose(userAspRep, 1, 2)
        itemAspRepTrans = torch.transpose(itemAspRep, 1, 2)

        '''
		Affinity Matrix (User Aspects x Item Aspects), i.e. User Aspects - Rows, Item Aspects - Columns
		'''
        affinityMatrix = torch.matmul(userAspRep, self.W_a)

        affinityMatrix = torch.matmul(affinityMatrix, itemAspRepTrans)

        # Non-Linearity: ReLU
        affinityMatrix = F.relu(affinityMatrix)

        # ===================================================================== User Importance (over Aspects) =====================================================================
        H_u_1 = torch.matmul(self.W_u, userAspRepTrans)
        H_u_2 = torch.matmul(self.W_i, itemAspRepTrans)
        H_u_2 = torch.matmul(H_u_2, torch.transpose(affinityMatrix, 1, 2))
        H_u = H_u_1 + H_u_2

        # Non-Linearity: ReLU
        H_u = F.relu(H_u)

        # User Aspect-level Importance
        userAspImpt = torch.matmul(torch.transpose(self.w_hu, 0, 1), H_u)

        # User Aspect-level Importance: (bsz x 1 x num_aspects) -> (bsz x num_aspects x 1)
        userAspImpt = torch.transpose(userAspImpt, 1, 2)

        userAspImpt = F.softmax(userAspImpt, dim=1)

        # User Aspect-level Importance: (bsz x num_aspects x 1) -> (bsz x num_aspects)
        userAspImpt = torch.squeeze(userAspImpt, 2)

        # ===================================================================== User Importance (over Aspects) =====================================================================

        # ===================================================================== Item Importance (over Aspects) =====================================================================
        H_i_1 = torch.matmul(self.W_i, itemAspRepTrans)
        H_i_2 = torch.matmul(self.W_u, userAspRepTrans)
        H_i_2 = torch.matmul(H_i_2, affinityMatrix)
        H_i = H_i_1 + H_i_2

        # Non-Linearity: ReLU
        H_i = F.relu(H_i)

        # Item Aspect-level Importance
        itemAspImpt = torch.matmul(torch.transpose(self.w_hi, 0, 1), H_i)

        # Item Aspect-level Importance: (bsz x 1 x num_aspects) -> (bsz x num_aspects x 1)
        itemAspImpt = torch.transpose(itemAspImpt, 1, 2)

        itemAspImpt = F.softmax(itemAspImpt, dim=1)

        # Item Aspect-level Importance: (bsz x num_aspects x 1) -> (bsz x num_aspects)
        itemAspImpt = torch.squeeze(itemAspImpt, 2)

        # ===================================================================== Item Importance (over Aspects) =====================================================================

        return userAspImpt, itemAspImpt


'''
Aspect-Based Rating Predictor, using Aspect-based Representations & the estimated Aspect Importance
'''


class ANR_RatingPred(nn.Module):

    def __init__(self, num_users, num_items):

        super(ANR_RatingPred, self).__init__()

        self.num_users = num_users
        self.num_items = num_items

    # dropout rate = 0.5

        # Dropout for the User & Item Aspect-Based Representations
        if (0.5 > 0.0):
            self.userAspRepDropout = nn.Dropout(p=0.5)
            self.itemAspRepDropout = nn.Dropout(p=0.5)

        # Global Offset/Bias (Trainable)
        self.globalOffset = nn.Parameter(torch.Tensor(1), requires_grad=True)

        # User Offset/Bias & Item Offset/Bias
        self.uid_userOffset = nn.Embedding(self.num_users, 1)
        self.uid_userOffset.weight.requires_grad = True

        self.iid_itemOffset = nn.Embedding(self.num_items, 1)
        self.iid_itemOffset.weight.requires_grad = True

        # Initialize Global Bias with 0
        self.globalOffset.data.fill_(0)

        # Initialize All User/Item Offset/Bias with 0
        self.uid_userOffset.weight.data.fill_(0)
        self.iid_itemOffset.weight.data.fill_(0)

    '''
	[Input]	userAspRep:		bsz x num_aspects x h1
	[Input]	itemAspRep:		bsz x num_aspects x h1
	[Input]	userAspImpt:	bsz x num_aspects
	[Input]	itemAspImpt:	bsz x num_aspects
	[Input]	batch_uid:		bsz
	[Input]	batch_iid:		bsz
	'''

    def forward(self, userAspRep, itemAspRep, userAspImpt, itemAspImpt, batch_uid, batch_iid):

        # User & Item Bias
        batch_userOffset = self.uid_userOffset(batch_uid)
        batch_itemOffset = self.iid_itemOffset(batch_iid)

        # ======================================== Dropout for the User & Item Aspect-Based Representations ========================================
        if (0.5 > 0.0):

            userAspRep = self.userAspRepDropout(userAspRep)
            itemAspRep = self.itemAspRepDropout(itemAspRep)

        # ======================================== Dropout for the User & Item Aspect-Based Representations ========================================

        lstAspRating = []

        # (bsz x num_aspects x h1) -> (num_aspects x bsz x h1)
        userAspRep = torch.transpose(userAspRep, 0, 1)
        itemAspRep = torch.transpose(itemAspRep, 0, 1)

        for k in range(5):

            user = torch.unsqueeze(userAspRep[k], 1)
            item = torch.unsqueeze(itemAspRep[k], 2)

            aspRating = torch.matmul(user, item)

            aspRating = torch.squeeze(aspRating, 2)

            lstAspRating.append(aspRating)

        # List of (bsz x 1) -> (bsz x num_aspects)
        rating_pred = torch.cat(lstAspRating, dim=1)

        # Multiply Each Aspect-Level (Predicted) Rating with the Corresponding User-Aspect Importance & Item-Aspect Importance
        rating_pred = userAspImpt * itemAspImpt * rating_pred

        # Sum over all Aspects
        rating_pred = torch.sum(rating_pred, dim=1, keepdim=True)

        # Include User Bias & Item Bias
        rating_pred = rating_pred + batch_userOffset + batch_itemOffset

        # Include Global Bias
        rating_pred = rating_pred + self.globalOffset

        return rating_pred
