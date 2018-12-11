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
parser.add_argument('--model_name', type = str, default = 'cnn_ff', help = 'name of the model')
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
con = config.Config()
con.set_max_epoch(15)
con.load_train_data()
#con.load_test_data()
con.set_train_model(model[args.model_name])
con.train()
