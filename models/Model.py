import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from networks.embedding import *
from networks.encoder import *
from networks.selector import *

class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.params = params
        if params['drop_encoding'] is not None:
            self.drop = nn.Dropout(params['drop_encoding'])
        self.embedding = Embedding(params)
        self.encoder = None
        self.selector = None
    def forward(self, word, pos1, pos2, chars=None, mask=None):
        embedding = self.embedding(word, pos1, pos2, chars)
        sen_embedding = self.encoder(embedding, mask)
        if self.params['drop_encoding']:
            sen_embedding = self.drop(sen_embedding)
        logits = self.selector(sen_embedding)
        return logits


