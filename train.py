import config
import models
import numpy as np
import os
import time
import datetime
import json
from sklearn.metrics import average_precision_score
import sys
import os
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, default = 'pcnn_ff', help = 'name of the model')
args = parser.parse_args()
model = {
	'pcnn_att': models.PCNN_ATT,
	'pcnn_one': models.PCNN_ONE,
	'pcnn_ave': models.PCNN_AVE,
	'cnn_att': models.CNN_ATT,
	'cnn_one': models.CNN_ONE,
	'cnn_ave': models.CNN_AVE,
	'rnn_one': models.RNN_ONE,
	'rnn_ave': models.RNN_AVE,
	'rnn_ff': models.RNN_FF,
	'cnn_ff': models.CNN_FF,
	'pcnn_ff': models.PCNN_FF
}
embed_pos = [True, False]
embed_char = [True, False]
drop_embed = [True, False]
drop_encode = [True, False]
lrs = [0.01, 0.005, 0.001, 0.0005]
wds = [1e-3,1e-4,1e-5,1e-6,1e-7]
results = {}
for lr in lrs:
    for wd in wds:
        for p in embed_pos:
            for c in embed_char:
                for model_name in ['pcnn_ff', 'cnn_ff', 'rnn_ff']:
                    try:
                        con = config.Config(lr, wd, p, c)
                        con.load_data()
                        con.set_max_epoch(15)
                        con.set_train_model(model[model_name])
                        results['lr-{},wd-{},pos-{},char-{},model-{}'.format(lr, wd, p, c, model_name)]= con.train()
                    except:
                        pass
json.dump(results, open('results.json', 'w'))
