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
    def __init__(self, config):
        super(PCNN_FF, self).__init__(config)
        self.encoder = PCNN(config)
        self.selector = FeedForward(config, config.hidden_size*3)
