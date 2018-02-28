# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import tflearn
import tensorflow as tf
from tflearn.data_utils import load_csv

csv.field_size_limit(sys.maxsize)
# Data loading and preprocessing
#data,labels = load_csv

#top5 = tflearn.metrics.Top_k(k=5)


first_layer = tflearn.input_data(shape=[None, 22284])
first_layer = tflearn.fully_connected(first_layer, 8000,regularizer='L2')

second_layer = tflearn.fully_connected(first_layer, 3000,activation='relu',regularizer='L1')

third_layer = tflearn.fully_connected(second_layer, 1000,activation='relu',regularizer='L1')
fourth_layer = tflearn.fully_connected(third_layer,300,activation='relu',regularizer='L1')
fourth_layer = tflearn.dropout(fourth_layer, 0.8)
output_layer = tflearn.fully_connected(fourth_layer, 193,activation='sigmoid')
regression_layer = tflearn.regression(output_layer,optimizer='Momentum', learning_rate=0.01,
                         loss='categorical_crossentropy', metric='accuracy',to_one_hot = True, n_classes=193)

model = tflearn.DNN(regression_layer, tensorboard_verbose=3)
model.fit(data, labels, n_epoch=100, validation_set=0.4, batch_size = 256, run_id="ai_network", show_metric=True)



#model.save

