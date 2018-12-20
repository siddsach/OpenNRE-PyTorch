import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class _CNN(nn.Module):
    def __init__(self, params):
        super(_CNN, self).__init__()
        self.params = params
        self.in_channels = 1
        self.in_height = self.params['max_length']
        self.in_width = self.params['word_size'] + 2 * self.params['pos_size'] + self.params['char_size']
        self.kernel_size = (self.params['window_size'], self.in_width)
        self.out_channels = self.params['hidden_size']
        self.stride = (1, 1)
        self.padding = (1, 0)
        self.cnn = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
    def forward(self, embedding, mask=None):
        return self.cnn(embedding)

class _PiecewisePooling(nn.Module):
    def __init(self):
        super(_PiecewisePooling, self).__init__()

    def forward(self, x, mask, hidden_size):
        mask = torch.unsqueeze(mask, 1).float()
        x, _ = torch.max(mask + x, dim = 2)
        x = x - 100
        return x.view(-1, hidden_size * 3)

class _MaxPooling(nn.Module):
    def __init__(self):
        super(_MaxPooling, self).__init__()
    def forward(self, x, hidden_size):
        x, _ = torch.max(x, dim = 2)
        return x.view(-1, hidden_size)

class PCNN(nn.Module):
    def __init__(self, params):
        super(PCNN, self).__init__()
        self.params = params
        self.cnn = _CNN(params)
        self.pooling = _PiecewisePooling()
        self.activation = nn.ReLU()
        #self.drop = nn.Dropout(p=self.params['drop_prob)
    def forward(self, embedding, mask):
        embedding = torch.unsqueeze(embedding, dim = 1)
        x = self.cnn(embedding)
        x = self.pooling(x, mask, self.params['hidden_size'])
        return self.activation(x)

class CNN(nn.Module):
    def __init__(self, params):
        super(CNN, self).__init__()
        self.params = params
        self.cnn = _CNN(config)
        self.pooling = _MaxPooling()
        self.activation = nn.ReLU()
        self.drop = nn.Dropout(p=self.params['drop_prob'])
    def forward(self, embedding, mask=None):
        embedding = torch.unsqueeze(embedding, dim = 1)
        x = self.cnn(embedding)
        x = self.pooling(x, self.params['hidden_size'])
        return self.drop(self.activation(x))

class MyRNN(nn.Module):
    def __init__(self, params):
        super(MyRNN, self).__init__()
        self.params = params
        self.rnn = nn.GRU(input_size=self.params['embed_size'], hidden_size=self.params['hidden_size'], batch_first=True)
        self.drop = nn.Dropout(p=self.params['drop_prob'])

    def forward(self, embedding, mask):
        _, hidden = self.rnn(embedding)
        return torch.squeeze(hidden)

class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.config = config
        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, word):
        bert_ids = self.convert_to_bert(word)
        _, pooled_output = self.bert_encoder(bert_ids)
        return pooled_output

    def convert_to_bert(self, word):
        return [bert2id[x] for x in word]

