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
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.embedding = Embedding(config)
        self.encoder = None
        self.selector = None
    def forward(self, word, pos1, pos2, chars, mask, scope, attention_query, label):
        embedding = self.embedding(word, pos1, pos2, chars)
        sen_embedding = self.encoder(embedding, mask)
        logits = self.selector(sen_embedding, scope, attention_query, label)
        return logits

    def test(self, word, pos1, pos2, chars, mask, scope):
        embedding = self.embedding(word, pos1, pos2, chars)
        sen_embedding = self.encoder(embedding, mask)
        return self.selector.test(sen_embedding, scope)
