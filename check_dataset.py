#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Прямой проход без хеширования промежуточных данных.

v2 - added saver.
"""
	
# https://github.com/tensorflow/tensorflow/issues/22837#issuecomment-428327601

import tensorflow as tf
import numpy as np
import math
import sys, os

from dataset_factory import GoodsDataset
#from goods_tf_records import GoodsTfrecordsDataset

# tf.enable_eager_execution()
import settings
from settings import IMAGE_SIZE
from utils.timer import timer

#--
#------------
# dataset
goods_dataset = GoodsDataset(settings.dataset_list, settings.labels_list, 
settings.IMAGE_SIZE, settings.train_batch, settings.valid_batch, settings.multiply, 
settings.valid_percentage)

train_dataset = goods_dataset.get_train_dataset()
valid_dataset = goods_dataset.get_valid_dataset()

graph = tf.Graph()  # сreate a new graph

with graph.as_default():
	
	iterator_train = train_dataset.make_one_shot_iterator()
	next_element_train = iterator_train.get_next()
	iterator_valid = valid_dataset.make_one_shot_iterator()
	next_element_valid = iterator_valid.get_next()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		train_count = 0
		while True:				
			try:
				features, labels = sess.run(next_element_train)
				train_count += 1
				if train_count % 20 == 0:
					print('train_count', train_count)				
			except tf.errors.OutOfRangeError:
				print("End of training dataset. Count={} batches".format(train_count))
				break	

		valid_count = 0
		while True:				
			try:
				features, labels = sess.run(next_element_valid)
				valid_count += 1	
				if valid_count % 20 == 0:
					print('valid_count', valid_count)								
			except tf.errors.OutOfRangeError:
				print("End of validation dataset. Count={} batches".format(valid_count))
				break	

		print('\nTrain={}, valid={}'.format(train_count, valid_count))