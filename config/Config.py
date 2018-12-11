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

NA_LABEL_INDEX = 0

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
    def __init__(self):
        self.acc_NA = Accuracy()
        self.acc_not_NA = Accuracy()
        self.acc_total = Accuracy()
        # Not using bag heuristic for now.
        #self.use_bag = True

        # Set paths
        self.data_path = './tmp'
        self.train_path = os.path.join(self.data_path, 'train')
        self.test_path = os.path.join(self.data_path, 'test')

        # Set hyperparams

        # Bool options
        self.use_gpu = True
        self.embed_pos = True
        self.embed_char = False
        self.is_training = True

        self.max_length = 120
        self.num_classes = 4
        self.hidden_size = 230
        if self.embed_pos:
            self.pos_size = 5
        else:
            self.pos_size = 0
        if self.embed_char:
            self.char_size = 100
        else:
            self.char_size = 0
        self.char_window_size = 3
        self.max_word_length = 50
        self.max_epoch = 15
        self.opt_method = 'SGD'
        self.optimizer = None
        self.learning_rate = 0.5
        self.weight_decay = 1e-5
        self.drop_prob = 0.5
        self.checkpoint_dir = './checkpoint'
        self.test_result_dir = './test_result'
        self.save_epoch = 1
        self.test_epoch = 1
        self.pretrain_model = None
        self.trainModel = None
        self.testModel = None
        self.batch_size = 75
        self.window_size = 3
        self.epoch_range = None
        self.loss = nn.CrossEntropyLoss()


    def set_data_path(self, data_path):
        self.data_path = data_path
    def set_max_length(self, max_length):
        self.max_length = max_length
        self.pos_num = 2 * self.max_length
    def set_num_classes(self, num_classes):
        self.num_classes = num_classes
    def set_hidden_size(self, hidden_size):
        self.hidden_size = hidden_size
    def set_window_size(self, window_size):
        self.window_size = window_size
    def set_pos_size(self, pos_size):
        self.pos_size = pos_size
    def set_word_size(self, word_size):
        self.word_size = word_size
    def set_max_epoch(self, max_epoch):
        self.max_epoch = max_epoch
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    def set_opt_method(self, opt_method):
        self.opt_method = opt_method
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay
    def set_drop_prob(self, drop_prob):
        self.drop_prob = drop_prob
    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
    def set_test_epoch(self, test_epoch):
        self.test_epoch = test_epoch
    def set_save_epoch(self, save_epoch):
        self.save_epoch = save_epoch
    def set_pretrain_model(self, pretrain_model):
        self.pretrain_model = pretrain_model
    def set_is_training(self, is_training):
        self.is_training = is_training
    def set_use_bag(self, use_bag):
        self.use_bag = use_bag
    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
    def set_epoch_range(self, epoch_range):
        self.epoch_range = epoch_range

    def load_train_data(self):
        # Load Data
        print('Loading data...')
        self.train_data = load_dataset(self.train_path)
        self.train_iter = data.BucketIterator(self.train_data, batch_size=self.batch_size, shuffle=False)
        self.pos_num = max(len(self.train_data.fields['pos1'].vocab), len(self.train_data.fields['pos2'].vocab))
        #self.test_data = load_dataset(self.test_path)
        #self.test_iter = data.BucketIterator(self.test_data, batch_size=self.batch_size)

        # Get vocab
        self.data_word_vec = self.train_data.fields['text'].vocab.vectors
        self.num_words = len(self.train_data.fields['text'].vocab)
        self.num_chars = len(self.train_data.fields['chars'].vocab)
        self.word_size = self.data_word_vec.shape[1] if self.data_word_vec is not None else 300

        # Set embeddings given vocab
        self.embed_size = self.word_size
        if self.embed_pos:
            self.embed_size += (self.pos_size * 2)
        if self.embed_char:
            self.embed_size += self.char_size

    def set_train_model(self, model):
        print("Initializing training model...")
        self.model = model
        self.trainModel = self.model(config = self)
        if self.pretrain_model != None:
            self.trainModel.load_state_dict(torch.load(self.pretrain_model))
        if torch.cuda.is_available():
            self.trainModel.cuda()
        if self.optimizer != None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(self.trainModel.parameters(), lr = self.learning_rate, lr_decay = self.lr_decay, weight_decay = self.weight_decay)
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(self.trainModel.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(self.trainModel.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        else:
            self.optimizer = optim.SGD(self.trainModel.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        print("Finish initializing")

    def set_test_model(self, model):
        print("Initializing test model...")
        self.model = model
        self.testModel = self.model(config = self)
        self.testModel.cuda()
        self.testModel.eval()
        print("Finish initializing")

    def get_bag_vars(self, batch):
        raise NotImplementedError

    def get_train_batch(self):
        return next(iter(self.train_iterator))


    #def get_train_batch(self, batch):
    #    input_scope = np.take(self.data_train_scope, self.train_order[batch * self.batch_size : (batch + 1) * self.batch_size], axis = 0)
    #    index = []
    #    scope = [0]
    #    for num in input_scope:
    #        index = index + list(range(num[0], num[1] + 1))
    #        scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
    #    self.batch_word = self.data_train_word[index, :]
    #    self.batch_pos1 = self.data_train_pos1[index, :]
    #    self.batch_pos2 = self.data_train_pos2[index, :]
    #    self.batch_chars = self.data_train_chars[index, :]
    #    self.batch_mask = self.data_train_mask[index, :]
    #    self.batch_label = np.take(self.data_train_label, self.train_order[batch * self.batch_size : (batch + 1) * self.batch_size], axis = 0)
    #    self.batch_attention_query = self.data_query_label[index]
    #    self.batch_scope = scope

    #def get_test_batch(self, batch):
    #    input_scope = self.data_test_scope[batch * self.batch_size : (batch + 1) * self.batch_size]
    #    index = []
    #    scope = [0]
    #    for num in input_scope:
    #        index = index + list(range(num[0], num[1] + 1))
    #        scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
    #    self.batch_word = self.data_test_word[index, :]
    #    self.batch_pos1 = self.data_test_pos1[index, :]
    #    self.batch_pos2 = self.data_test_pos2[index, :]
    #    self.batch_chars = self.data_test_chars[index, :]
    #    self.batch_mask = self.data_test_mask[index, :]
    #    self.batch_scope = scope

    @staticmethod
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
        print('pos')
        print(pos_max.shape)
        print('i')
        print(inds.shape)
        print('l')
        length = length.unsqueeze(1).expand_as(inds)
        if length is not None:
            g = inds.gt(pos_max)
            l = inds.le(length)
            mask[:, :, 2] = (g * l) * 100
        else:
            mask[:, :, 2] = (inds.gt(pos_max)) * 100
        return mask

    def train_one_step(self, batch):
        # TODO: Add length option (4th arg)
        words, length = batch.text
        #mask = self.get_mask(words, batch.pos1, batch.pos2, length)
        self.optimizer.zero_grad()
        print('POS1')
        print(batch.pos1.shape)
        logits = self.trainModel(words,
                                 batch.pos1,
                                 batch.pos2,
                                 batch.relation,
                                 batch.chars)
                                 #mask)
        loss = self.loss(logits, batch.relation)
        _, output = torch.max(logits, dim = 1)
        loss.backward()
        self.optimizer.step()
        for i, prediction in enumerate(output):
            if batch.relation[i] == NA_LABEL_INDEX:
                self.acc_NA.add(prediction == batch.relation[i])
            else:
                b = prediction == batch.relation[i]
                self.acc_not_NA.add(b)
            self.acc_total.add(prediction == batch.relation[i])
        return loss.data[0]

    #def test_one_step(self):
    #    word = to_var(self.batch_word)
    #    chars = to_var(self.batch_chars)
    #    pos1 = to_var(self.batch_pos1)
    #    pos2 = to_var(self.batch_pos2)
    #    mask = to_var(self.batch_mask)
    #    scope = self.batch_scope
    #    return self.testModel.test(word, pos1, pos2, chars, mask, scope)

    def train(self):
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        best_auc = 0.0
        best_p = None
        best_r = None
        best_epoch = 0
        for epoch in range(self.max_epoch):
            print('Epoch ' + str(epoch) + ' starts...')
            self.acc_NA.clear()
            self.acc_not_NA.clear()
            self.acc_total.clear()
            for i, batch in enumerate(self.train_iter):
                loss = self.train_one_step(batch)
                time_str = datetime.datetime.now().isoformat()
                sys.stdout.write("\n\nepoch %d step %d time %s | loss: %f, NA accuracy: %f, not NA accuracy: %f, total accuracy: %f\r" % (epoch, i, time_str, loss, self.acc_NA.get(), self.acc_not_NA.get(), self.acc_total.get()))
                sys.stdout.flush()
            if (epoch + 1) % self.save_epoch == 0:
                print('Epoch ' + str(epoch) + ' has finished')
                print('Saving model...')
                path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-' + str(epoch))
                torch.save(self.trainModel.state_dict(), path)
                print('Have saved model to ' + path)
            #if (epoch + 1) % self.test_epoch == 0:
            #    self.testModel = self.trainModel
            #    auc, pr_x, pr_y = self.test_one_epoch()
            #    if auc > best_auc:
            #        best_auc = auc
            #        best_p = pr_x
            #        best_r = pr_y
            #        best_epoch = epoch
        print("Finish training")
        #print("Best epoch = %d | auc = %f" % (best_epoch, best_auc))
        print("Storing best result...")
        if not os.path.isdir(self.test_result_dir):
            os.mkdir(self.test_result_dir)
        np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_x.npy'), best_p)
        np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_y.npy'), best_r)
        print("Finish storing")
    def test_one_epoch(self):
        test_score = []
        for batch in tqdm(range(self.test_batches)):
            self.get_test_batch(batch)
            batch_score = self.test_one_step()
            test_score = test_score + batch_score
        test_result = []
        for i in range(len(test_score)):
            for j in range(1, len(test_score[i])):
            #for j in range(1, self.data_test_label.shape[1]):
                dts = self.data_test_label[i][j]
                ts = test_score[i][j]
                test_result.append([dts, ts])
        test_result = sorted(test_result, key = lambda x: x[1])
        test_result = test_result[::-1]
        pr_x = []
        pr_y = []
        correct = 0
        for i, item in enumerate(test_result):
            correct += item[0]
            pr_y.append(float(correct) / (i + 1))
            pr_x.append(float(correct) / self.total_recall)
        auc = sklearn.metrics.auc(x = pr_x, y = pr_y)
        print("auc: ", auc)
        return auc, pr_x, pr_y
    #def test(self):
    #    best_epoch = None
    #    best_auc = 0.0
    #    best_p = None
    #    best_r = None
    #    for epoch in self.epoch_range:
    #        path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-' + str(epoch))
    #        if not os.path.exists(path):
    #            continue
    #        print("Start testing epoch %d" % (epoch))
    #        self.testModel.load_state_dict(torch.load(path))
    #        auc, p, r = self.test_one_epoch()
    #        if auc > best_auc:
    #            best_auc = auc
    #            best_epoch = epoch
    #            best_p = p
    #            best_r = r
    #        print("Finish testing epoch %d" % (epoch))
    #    print("Best epoch = %d | auc = %f" % (best_epoch, best_auc))
    #    print("Storing best result...")
    #    if not os.path.isdir(self.test_result_dir):
    #        os.mkdir(self.test_result_dir)
    #    np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_x.npy'), best_p)
    #    np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_y.npy'), best_r)
    #    print("Finish storing")
