import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from networks.embedding import *
from networks.encoder import *
from networks.selector import *
from networks.classifier import *
from .Model import Model

class PCNN_FF(Model):
    def __init__(self, params):
        super(PCNN_FF, self).__init__(params)
        self.encoder = PCNN(params)
        self.selector = FeedForward(params, params['hidden_size']*3)
