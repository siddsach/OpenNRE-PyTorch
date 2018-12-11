import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from networks.encoder import *
from networks.selector import *
from .models import Model

class BERT_ATT(Model):
    super(BERT, self).__init__()
    self.embedding = PassEmbedding(config)
    self.encoder = BertEncoder(config)
    self.selector = Attention(config, config.bert_size)
