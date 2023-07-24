import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AAMR(nn.Module):
    def __init__(self, opt):
        super(AAMR, self).__init__()