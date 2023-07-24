import os
import numpy as np
import random
from torch.utils.data import Dataset
seed = 99
random.seed(seed)
np.random.seed(seed)


class AmazonData(Dataset):

    def __init__(self, root_path, mode="Train"):
        if mode == "Train":
            path = os.path.join(root_path, 'train/npy/')
            print('loading train data')
            self.data = np.load(path + 'Train.npy', encoding='bytes')
            self.scores = np.load(path + 'Train_Score.npy')
        if mode == "Test":
            path = os.path.join(root_path, 'test/npy/')
            print('loading test data')
            self.data = np.load(path + 'Test.npy', encoding='bytes')
            self.scores = np.load(path + 'Test_Score.npy')
        if mode == "Val":
            path = os.path.join(root_path, 'val/npy/')
            print('loading test data')
            self.data = np.load(path + 'Val.npy', encoding='bytes')
            self.scores = np.load(path + 'Val_Score.npy')
        
        self.x = list(zip(self.data, self.scores))
        print('loading finish')

    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]

    def __len__(self):
        return len(self.x)
