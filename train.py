import config
import models
import numpy as np
import os
import time
import datetime
import json
from sklearn.metrics import average_precision_score
from itertools import product
import sys
import os
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, default = 'pcnn_ff', help = 'name of the model')
parser.add_argument('--tune', type = bool, default = False, help = 'Whether or not to tune hyperparams')
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

if args.tune:
    # Hyperparams to tune on

    CHOICES = {'embed_pos': [True, False],
    'embed_char' : [True, False]}
    #'drop_embed' : [None, 0.2, 0.5, 0.8],
    #'drop_encode' : [None, 0.2, 0.5, 0.8],
    #'lrs' : [0.01, 0.005, 0.001, 0.0005],
    #'wds' : [1e-3,1e-4,1e-5,1e-6,1e-7]}

    results = {}

    knobs = list(CHOICES.keys())
    levels = [CHOICES[k] for k in knobs]
    runs = product(*levels)
    for run in runs:
        params = {name:val for name, val in zip(knobs, run)}
        con = config.Config(hyperparams=params)
        con.load_data()
        con.set_train_model(model['pcnn_ff'])
        scores = con.train()
        key =','.join(['{}-{}'.format(name, val) for name, val in params.items()])
        results[key] = scores
        json.dump(results, open('results.json', 'w'))
else:
        con = config.Config()
        con.load_data()
        con.set_train_model(model['pcnn_ff'])
        con.train()

