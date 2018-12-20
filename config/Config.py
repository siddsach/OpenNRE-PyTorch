#coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torchtext import data
import numpy as np
import os
import time
import datetime
import json
import sys
import sklearn.metrics
from tqdm import tqdm
from process_mobius import load_dataset

NEG_LABEL = 0
POS_LABEL = 2

DEFAULTS = {'embed_pos':True,
            'embed_char':True,
            'max_length': 230,
            'hidden_size': 230,
            'pretrained_wordvec': None,
            'char_window_size': 3,
            'max_word_length': 50,
            'max_epoch': 15,
            'opt_method':'SGD',
            'pos_size': 5,
            'char_size': 100,
            'max_epoch': 15,
            'learning_rate': 0.005,
            'weight_decay': 1e-5,
            'drop_prob': 0.5,
            'batch_size': 75,
            'window_size': 3,
            'use_gpu': True,
            'is_training': True,
            'test_epoch': 1,
            'save_epoch': 1}


def to_var(x):
    if torch.cuda.is_available():
        return Variable(torch.from_numpy(x).cuda())
    else:
        return Variable(torch.from_numpy(x))

class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.total = 0
    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1
    def get(self):
        if self.total == 0:
            return 0.0
        else:
            return float(self.correct) / self.total
    def clear(self):
        self.correct = 0
        self.total = 0

class Config(object):
    def __init__(self, hyperparams=None):
        self.acc_NA = Accuracy()
        self.acc_not_NA = Accuracy()
        self.acc_total = Accuracy()
        # Not using bag heuristic for now.
        #self.use_bag = True

        # Set paths
        #self.data_path = '/efs/sid/mobius_data/mimic/output'
        #self.data_path = '/Users/sidsachdeva/roam/data/mimic'
        self.data_path = 'output'

        self.params = {}
        # Set hyperparams
        for name, value in DEFAULTS.items():
            if hyperparams is None or name not in hyperparams:
                self.params[name] =  value
            else:
                self.params[name] =  hyperparams[name]

        # If we don't use embeddings, embed size is 0
        if not self.params['embed_pos']:
            self.pos_size = 0
        if not self.params['embed_char']:
            self.char_size = 0

        self.checkpoint_dir = './checkpoint'
        self.test_result_dir = './test_result'
        self.pretrain_model = None
        self.trainModel = None
        self.testModel = None
        self.optimizer = None
        self.loss = nn.CrossEntropyLoss()

    def load_data(self):
        # Load Data
        print('Loading data...')
        self.train_data, self.test_data = load_dataset(self.data_path)

        #if self.pretrained_wordvec is not None:
        #    self.train_data.fields['text'].vocab.load_vectors(self.pretrained_wordvec, cache='/efs/sid/mobius_data/vectors')
        if self.params['pretrained_wordvec'] is not None:
            self.train_data.fields['text'].vocab.load_vectors(self.pretrained_wordvec, cache='vectors')

        self.params['pos_num'] = max(len(self.train_data.fields['pos1_rel'].vocab),
                           len(self.train_data.fields['pos2_rel'].vocab),
                           len(self.test_data.fields['pos1_rel'].vocab),
                           len(self.test_data.fields['pos2_rel'].vocab))

        # Get vocab
        self.data_word_vec = self.train_data.fields['text'].vocab.vectors
        self.params['num_words'] = len(self.train_data.fields['text'].vocab)
        self.params['num_chars'] = len(self.train_data.fields['chars'].vocab)
        self.params['num_classes'] = len(self.train_data.fields['relation'].vocab)

        self.params['word_size'] = self.data_word_vec.shape[1] if self.data_word_vec is not None else 300

        # Set embeddings given vocab
        embed_size = self.params['word_size']
        if self.params['embed_pos']:
            embed_size += (self.params['pos_size'] * 2)
        if self.params['embed_char']:
            embed_size += self.params['char_size']
        self.params['embed_size'] = embed_size

    def get_iterator(self, dataset):
        return data.BucketIterator(dataset, batch_size=self.params['batch_size'], shuffle=False)

    def set_train_model(self, model):
        print("Initializing training model...")
        self.model = model
        self.trainModel = self.model(params = self.params)
        if self.pretrain_model != None:
            self.trainModel.load_state_dict(torch.load(self.pretrain_model))
        if torch.cuda.is_available():
            self.trainModel.cuda()
        if self.data_word_vec is not None:
            self.trainModel.embedding.load_vectors(self.data_word_vec)
        if self.optimizer != None:
            pass
        elif self.params['opt_method'] == "Adagrad" or self.params['opt_method'] == "adagrad":
            self.optimizer = optim.Adagrad(self.trainModel.parameters(), lr = self.params['learning_rate'], lr_decay = self.lr_decay, weight_decay = self.params['weight_decay'])
        elif self.params['opt_method'] == "Adadelta" or self.params['opt_method'] == "adadelta":
            self.optimizer = optim.Adadelta(self.trainModel.parameters(), lr = self.params['learning_rate'], weight_decay = self.params['weight_decay'])
        elif self.params['opt_method'] == "Adam" or self.params['opt_method'] == "adam":
            self.optimizer = optim.Adam(self.trainModel.parameters(), lr = self.params['learning_rate'], weight_decay = self.params['weight_decay'])
        else:
            self.optimizer = optim.SGD(self.trainModel.parameters(), lr = self.params['learning_rate'], weight_decay = self.params['weight_decay'])
        print("Finish initializing")

    def get_bag_vars(self, batch):
        raise NotImplementedError

    def get_train_batch(self):
        return next(iter(self.train_iterator))

    def train_one_step(self, batch):
        # TODO: Add length option (4th arg)
        words, length = batch.text
        mask = get_mask(words, batch.pos1, batch.pos2, length)
        self.optimizer.zero_grad()
        if torch.cuda.is_available():
            logits = self.trainModel(words.cuda(),
                                     batch.pos1_rel.cuda(),
                                     batch.pos2_rel.cuda(),
                                     batch.chars.cuda(),
                                     mask.cuda(),
                                     batch.relation.cuda())
            loss = self.loss(logits, batch.relation.cuda())
        else:
            logits = self.trainModel(words,
                                     batch.pos1_rel,
                                     batch.pos2_rel,
                                     batch.chars,
                                     mask,
                                     batch.relation)
            loss = self.loss(logits, batch.relation)
        _, output = torch.max(logits, dim = 1)
        loss.backward()
        self.optimizer.step()
        output = [int(x.item()) for x in output]
        gold = [batch.relation[i].data.item() for i in range(len(output))]
        for label, pred in zip(gold, output):
            if label == NEG_LABEL:
                self.acc_NA.add(pred == label)
            else:
                self.acc_not_NA.add(pred == label)
            self.acc_total.add(pred == label)
        return loss.data[0]


    def test_one_step(self, batch):
        # TODO: Add length option (4th arg)
        words, length = batch.text
        mask = self.get_mask(words, batch.pos1, batch.pos2, length)
        if torch.cuda.is_available():
            logits = self.trainModel(words.cuda(),
                                     batch.pos1_rel.cuda(),
                                     batch.pos2_rel.cuda(),
                                     batch.relation.cuda(),
                                     batch.chars.cuda(),
                                     mask.cuda())
        else:
            logits = self.trainModel(words,
                                     batch.pos1_rel,
                                     batch.pos2_rel,
                                     batch.relation,
                                     batch.chars,
                                     mask)
        _, output = torch.max(logits, dim = 1)
        output = [x.item() for x in output]
        gold = [batch.relation[i].data.item() for i in range(len(output))]
        return gold, output

    def metric(self, gold, pred):
        p, r, f1, s = sklearn.metrics.precision_recall_fscore_support(
                        gold, pred, pos_label=POS_LABEL, average='micro')
        return {'precision': p, 'recall':r, 'f1': f1, 'support':s}

    def train(self):
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        best_f1 = 0.0
        best_scores = None
        best_breakdown = None
        best_epoch = 0
        for epoch in range(self.params['max_epoch']):
            print('Epoch ' + str(epoch) + ' starts...')
            self.acc_NA.clear()
            self.acc_not_NA.clear()
            self.acc_total.clear()
            train_iter = self.get_iterator(self.train_data)
            for i, batch in enumerate(train_iter):
                if i > 2:
                    break
                loss = self.train_one_step(batch)
                time_str = datetime.datetime.now().isoformat()
                sys.stdout.write("\n\nepoch %d step %d time %s | loss: %f, NA accuracy: %f, not NA accuracy: %f, total accuracy: %f\r" % (epoch, i, time_str, loss, self.acc_NA.get(), self.acc_not_NA.get(), self.acc_total.get()))
                sys.stdout.flush()
            if (epoch + 1) % self.params['save_epoch'] == 0:
                print('Epoch ' + str(epoch) + ' has finished')
                print('Saving model...')
                dir_path = os.path.join(self.checkpoint_dir, self.model.__name__)
                model_path = os.path.join(dir_path, 'models')
                os.makedirs(model_path, exist_ok=True)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                name = 'Epoch-' + str(epoch)
                torch.save(self.trainModel.state_dict(), os.path.join(model_path, name))
                params_path = os.path.join(model_path, 'params.json')
                json.dump(self.params, open(params_path, 'w'))
                print('Have saved model to ' + dir_path)
            #if (epoch + 1) % self.params['test_epoch'] == 0:
            #    print('Testing...')
            #    self.testModel = self.trainModel
            #    scores, breakdown = self.test_one_epoch()
            #    if scores['f1'] > best_f1:
            #        best_scores = scores
            #        best_breakdown = breakdown
            self.load_data()
        return {'scores': best_scores, 'breakdown':best_breakdown}



    def test_one_epoch(self):
        labels, preds, rel_types = [], [], []
        test_iter = self.get_iterator(self.test_data)
        for i, batch in enumerate(test_iter):
            batch_labels, batch_preds = self.test_one_step(batch)
            labels += batch_labels
            preds += batch_preds
            rel_types += [x.item() for x in batch.rel_type]
        breakdowns = [{'label':l, 'pred':p, 'type':r} for l, p, r in zip(labels, preds, rel_types)]
        scores = self.metric(labels, preds)
        return scores, breakdowns

def get_mask(word, pos1, pos2, length):
    pos1 = pos1.unsqueeze(1)
    pos2 = pos2.unsqueeze(1)
    pos = torch.cat((pos1, pos2), dim=1)
    pos_min = torch.min(pos, dim=1)[0].view(pos1.size(0), 1)
    pos_max = torch.max(pos, dim=1)[0].view(pos1.size(0), 1)
    inds = torch.arange(0, word.size(1), dtype=torch.long).expand(word.size(0), -1)
    pos_min = pos_min.expand_as(inds)
    pos_max = pos_max.expand_as(inds)
    mask = torch.zeros(word.size(0), word.size(1), 3, dtype=torch.long)
    mask[:, :, 0] = (inds.le(pos_min)) * 100
    mask[:, :, 1] = (inds.gt(pos_min) * inds.le(pos_max)) * 100
    length = length.unsqueeze(1).expand_as(inds)
    if length is not None:
        g = inds.gt(pos_max)
        l = inds.le(length)
        mask[:, :, 2] = (g * l) * 100
    else:
        mask[:, :, 2] = (inds.gt(pos_max)) * 100
    return mask

