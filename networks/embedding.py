import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.config = config
        if config.embed_pos:
            self.pos_embedding = PositionEmbedding(config)
        if config.embed_char:
            self.char_embedding = CharacterEmbedding(config)

        self.word_embedding = nn.Embedding(self.config.num_words, self.config.word_size)
        if self.config.data_word_vec is not None:
            self.word_embedding.weight.data.copy_(self.config.data_word_vec)

    def forward(self, word, pos1, pos2, chars):
        word_emb = self.word_embedding(word)
        if self.config.embed_pos:
            pos_emb = self.pos_embedding(pos1, pos2)
        else:
            print('DONT EMBED POS')
            pos_emb = None
        if self.config.embed_char:
            char_emb = self.char_embedding(chars)
        else:
            char_emb = None
        l = [word_emb, pos_emb, char_emb]
        all_embeds = [x for x in l if x is not None]
        embedding = torch.cat(all_embeds, dim = 2)
        return embedding

class PassEmbedding(nn.Module):
    def __init__(self, config):
        self.config

    def forward(self, word, pos1, pos2, chars):
        return word


class PositionEmbedding(nn.Module):
    def __init__(self, config):
        super(PositionEmbedding, self).__init__()
        self.config = config
        self.pos1_embedding = nn.Embedding(self.config.pos_num, self.config.pos_size, padding_idx = 0)
        self.pos2_embedding = nn.Embedding(self.config.pos_num, self.config.pos_size, padding_idx = 0)
        self.init_pos_weights()

    def init_pos_weights(self):
        nn.init.xavier_uniform(self.pos1_embedding.weight.data)
        if self.pos1_embedding.padding_idx is not None:
            self.pos1_embedding.weight.data[self.pos1_embedding.padding_idx].fill_(0)
        nn.init.xavier_uniform(self.pos2_embedding.weight.data)
        if self.pos2_embedding.padding_idx is not None:
            self.pos2_embedding.weight.data[self.pos2_embedding.padding_idx].fill_(0)

    def forward(self, pos1, pos2):
        pos1_emb = self.pos1_embedding(pos1)
        pos2_emb = self.pos2_embedding(pos2)
        embedding = torch.cat((pos1_emb, pos2_emb), dim = 2)
        return embedding

class CharacterEmbedding(nn.Module):
    def __init__(self, config):
        super(CharacterEmbedding, self).__init__()
        self.config = config
        self.in_channels = 10
        self.char_embed = nn.Embedding(self.config.num_chars, self.in_channels)
        self.kernel_size = self.config.char_window_size
        self.out_channels = self.config.char_size
        self.padding = self.kernel_size
        self.cnn = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, padding=self.padding)
        self.pooling = nn.MaxPool1d(self.config.max_word_length)
        self.activation = nn.ReLU()
        self.drop = nn.Dropout(p=self.config.drop_prob)

    def forward(self, chars):
        mask = chars != 0
        embed = chars.view(chars.size(0)*chars.size(1), chars.size(2))
        mask = mask.view(chars.size(0)*chars.size(1), chars.size(2))
        embed = self.char_embed(embed).transpose(1, 2)
        embed.masked_fill_(mask.unsqueeze(1),0)
        embed = self.cnn(embed)
        if torch.cuda.is_available():
            front_mask = torch.ones((mask.size(0), self.padding-1)).long().cuda()
            back_mask = torch.zeros((mask.size(0), self.padding-1)).long().cuda()
        else:
            front_mask = torch.ones((mask.size(0), self.padding-1)).long()
            back_mask = torch.zeros((mask.size(0), self.padding-1)).long()
        mask = torch.cat([front_mask, mask.long(), back_mask], dim=1)
        embed = embed.masked_fill_(mask.byte().unsqueeze(1),0)
        embed = torch.max(embed, dim=2)[0].reshape(chars.size(0), chars.size(1), -1)
        embed = self.activation(embed)
        return embed

