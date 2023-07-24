# -*- encoding: utf-8 -*-
import time
import random
import math
import fire
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from dataset import AmazonData
from models import Model
import methods
import config
import pandas as pd


from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from plus_metrics import *

def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def collate_fn(batch):
    data, label = zip(*batch)
    return data, label

def get_word_para():
    u_word = ['user_net.word_embs.weight', 'user_net.char_embs.weight']
    i_word = ['item_net.word_embs.weight', 'item_net.char_embs.weight']
    return u_word + i_word

def get_para(model):
    u_linear = ['user_net.review_linear.0.weight', 'user_net.summary_linear.0.weight',
                    'user_net.id_linear.0.weight', 'user_net.attention_linear.0.weight']
    i_linear = ['item_net.review_linear.0.weight', 'item_net.summary_linear.0.weight',
                    'item_net.id_linear.0.weight', 'item_net.attention_linear.0.weight']
    linear_weight = u_linear + i_linear
    # id_emb = ['user_net.id_embedding.weight', 'item_net.id_embedding.weight']

    return linear_weight    # , id_emb_weight

def unpack_input(opt, x):

    uids, iids = list(zip(*x))
    uids = list(uids)
    iids = list(iids)

    user_reviews = opt.user_list[uids]
    # user_item2id = opt.user2itemid_dict[uids]  # 检索出该user对应的item id

    item_reviews = opt.item_list[iids]
    # item_user2id = opt.item2userid_dict[iids]  # 检索出该item对应的user id

    train_data = [user_reviews, item_reviews, uids, iids]
    train_data = list(map(lambda x: torch.LongTensor(x).cuda(), train_data))
    return train_data


def train(**kwargs):

    if 'dataset' not in kwargs:
        opt = getattr(config, 'Toys_and_Games_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)
    # 2 model
    model = Model(opt, getattr(methods, opt.model))
    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:

            model = nn.DataParallel(model, device_ids=opt.gpu_ids)

    if opt.load_ckp:
        assert len(opt.ckp_path) > 0
        model.load(opt.ckp_path)

    # 3 data
    train_data = AmazonData(opt.data_root, mode="Train")
    train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=collate_fn)
    test_data = AmazonData(opt.data_root, mode="Val")
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, collate_fn=collate_fn)
    print('{}: train data: {}; test data: {}'.format(now(), len(train_data), len(test_data)))

    # 4 optimiezer
    # optimizer = optim.Adadelta(model.parameters(), rho=0.95, eps=1e-6, weight_decay=opt.weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
    if opt.fine_tune:
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)

    # training
    print("start training....")
    min_loss = 1e+20
    best_res = 1e+10
    mse_func = nn.MSELoss()
    for epoch in range(opt.num_epochs):
        total_loss = 0.0
        model.train()
        scheduler.step(epoch)
        print("{} Epoch {}: start".format(now(), epoch))
        for idx, (train_datas, scores) in enumerate(train_data_loader):
            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)
            train_datas = unpack_input(opt, train_datas)
            optimizer.zero_grad()
            output = model(train_datas)
            loss = mse_func(output, scores)
            total_loss += loss.item() * len(scores)
            
            # loss = loss / 2.0  # tf.nn.l2loss
            loss = torch.sqrt(loss)
            loss.backward()
            optimizer.step()
            
            # if idx % opt.print_step == 0 and idx > 0:
            #     print("\t{}, {} step finised;".format(now(), idx))
            #     predict_loss, test_mse = predict(model, test_data_loader, opt, use_gpu=opt.use_gpu)
            #     if predict_loss < min_loss:
            #         # model.save(name=opt.dataset, opt=opt.print_opt)
            #         min_loss = predict_loss
            #         print("\tmodel save")
            #     if test_mse < best_res:
            #         best_res = test_mse
            
            

        print("{};epoch:{};total_loss:{}".format(now(), epoch, total_loss))
        mse = total_loss * 1.0 / len(train_data)
        print("{};train reslut: mse: {}; rmse: {}".format(now(), mse, math.sqrt(mse)))
        predict_loss, test_mse, mae = predict(model, test_data_loader, opt, use_gpu=opt.use_gpu)
        if predict_loss < min_loss:
            model.save(name=opt.dataset, opt=opt.print_opt)
            min_loss = predict_loss
            print("model save")
        if test_mse < best_res:
            best_res = test_mse

    print("----"*20)
    print(f"{now()} {opt.dataset} {opt.print_opt} best_res:  {best_res}")
    print("----"*20)
    
def test(**kwargs):
    
    if 'dataset' not in kwargs:
        opt = getattr(config, 'Toys_and_Games_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)
    
    # assert(len(opt.pth_path) > 0)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)
    # 2 model
    model = Model(opt, getattr(methods, opt.model))
    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:

            model = nn.DataParallel(model, device_ids=opt.gpu_ids)

    if opt.load_ckp:
        assert len(opt.ckp_path) > 0
        model.load(opt.ckp_path)
        
    model.load(opt.pth_path)
    print(f"load model: {opt.pth_path}")
    test_data = AmazonData(opt.data_root, mode="Test")
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, collate_fn=collate_fn)
    print(f"{now()}: test in the test datset")
    predict_loss, test_mse, test_mae = predict(model, test_data_loader, opt)


def predict(model, test_data_loader, opt, use_gpu=True):
        acc = 0.0
        f1 = 0.0
        precision = 0.0
        recall = 0.0
        sur = 0.0
        div = 0.0
        
        batchs = 0
    
        total_loss = 0.0
        total_maeloss = 0.0
        model.eval()
        step = 0
        for idx, (test_data, scores) in enumerate(test_data_loader):
            if use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)
            test_data = unpack_input(opt, test_data)
            output = model(test_data)
            loss = torch.sum((output-scores)**2)
            total_loss = loss.item()
            step += len(scores)
            
            
            
            mae_loss = torch.sum(abs(output-scores))
            total_maeloss = mae_loss.item()
            
            trunc_output = torch.round(output).int()
            
            list_out = trunc_output.tolist()
            list_scores = scores.tolist()
            f1 = f1_score(list_scores, list_out, average="macro")
            acc = accuracy_score(list_scores, list_out)
            precision += precision_score(list_scores, list_out, average="macro")
            recall += recall_score(list_scores, list_out, average="macro")
            sur = serendipity(list_scores, list_out, 4.0)
            div = diversity(list_scores, list_out)
            
            df = pd.read_csv("results.csv")
            df.loc[df.shape[0]] = [(total_loss/len(scores)), (total_maeloss/len(scores)), acc, f1, sur, div]
            df.to_csv("results.csv", index=False)
            
            batchs += 1
            
        acc = acc * 1.0 / batchs
        f1 = f1 * 1.0 / batchs
        precision = precision * 1.0 / batchs
        recall = recall * 1.0 / batchs

        sur = sur * 1.0 / batchs
        div = div * 1.0 / batchs
            
        mse = total_loss * 1.0 / step
        mae = total_maeloss * 1.0 / step
        
        print("\t{};test reslut: mse: {}; rmse: {}; mae: {}; acc: {}; pre: {}; re: {}; f1: {}; Ser: {}; Div: {}".format(
            now(), mse, math.sqrt(mse), mae, acc, precision, recall, f1, sur, div))
        model.train()
        return total_loss, mse, mae


if __name__ == "__main__":
    fire.Fire()
