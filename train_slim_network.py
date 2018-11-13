#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Прямой проход без хеширования промежуточных данных.
"""
	
# https://github.com/tensorflow/tensorflow/issues/22837#issuecomment-428327601

import tensorflow as tf
import numpy as np
import math

#import models.inception_v3 as inception
from tensorflow.contrib.slim.nets import inception
slim = tf.contrib.slim

from dataset_factory import GoodsDataset
#from goods_tf_records import GoodsTfrecordsDataset

# tf.enable_eager_execution()
import settings
from utils.timer import timer

#tf.enable_eager_execution()

#from settings import IMAGE_SIZE
IMAGE_SIZE = (299, 299)
num_classes = settings.num_classes
print('num_classes:', num_classes)



# dataset
goods_dataset = GoodsDataset("dataset-181018.list", "dataset-181018.labels", 
settings.IMAGE_SIZE, settings.train_batch, settings.valid_batch, settings.multiply, 
settings.valid_percentage)

train_dataset = goods_dataset.get_train_dataset()
valid_dataset = goods_dataset.get_valid_dataset()

num_epochs = 100
train_steps_per_epoch = 1157
valid_steps_per_epoch = 77
train_dataset = train_dataset.repeat()
valid_dataset = valid_dataset.repeat()

"""
def model_function(next_element):
	x, y = next_element
	logits, end_points = inception.inception_v3(
		x, num_classes=settings.num_classes, is_training=True)
	loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
	return logits, loss
"""

# Create a new graph
graph = tf.Graph() # no necessiry

with graph.as_default():
	
	iterator_train = train_dataset.make_one_shot_iterator()
	next_element_train = iterator_train.get_next()
	iterator_valid = valid_dataset.make_one_shot_iterator()
	next_element_valid = iterator_valid.get_next()

	#iterator_train = train_dataset.make_initializable_iterator()
	#x, y = next_element_train

	#x = tf.placeholder(tf.float32, [None, 784]) # Placeholder for input.
	#y = tf.placeholder(tf.float32, [None, 10])  # Placeholder for labels.
	#x_images = tf.reshape(x, [-1,28,28,1])
	#x_images = tf.image.resize_images(x_images, [299, 299])	

	#input_tensor = keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
	
	x = tf.placeholder(tf.float32, [None, 299, 299, 3], name='input')
	y = tf.placeholder(tf.float32, [None, num_classes], name='y')

	logits, end_points = inception.inception_v3(
		x, num_classes=num_classes, is_training=True)

	loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
	train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
	correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
	acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # top-1	
	acc_top6 = tf.nn.in_top_k(logits, tf.argmax(y,1), 6)
		
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		
		for epoch in range(num_epochs):
			print('\nEPOCH {0}'.format(epoch))

			# valid
			timer('valid, epoch {0}'.format(epoch))
			valid_acc_list = []
			valid_acc_top6_list = []			

			for i in range(valid_steps_per_epoch):
				
				try:
					features, labels = sess.run(next_element_valid)
					valid_acc, valid_acc_top6 = sess.run([acc, acc_top6], feed_dict={x: features, y: labels})
					valid_acc_list.append(valid_acc)
					valid_acc_top6_list.append(valid_acc_top6)
					if i%10 == 0:					
						print('valid_acc_list:', valid_acc_list)
						print('valid_acc_top6_list:', valid_acc_top6_list)
						#print('epoch={} i={}: valid_acc={:.4f} [top6={:.4f}]'.\
						#	format(epoch, i, np.mean(valid_acc_list), np.mean(valid_acc_top6_list)))
				except tf.errors.OutOfRangeError:
					print("End of valid dataset.")
					break
			timer()


			timer('train, epoch {0}'.format(epoch))
			train_acc_list = []
			train_acc_top6_list = []

			for i in range(train_steps_per_epoch):
				
				try:
					features, labels = sess.run(next_element_train)
					#print(i, labels[0])
					sess.run(train_op, feed_dict={x: features, y: labels})
					
					train_acc, train_acc_top6 = sess.run([acc, acc_top6], feed_dict={x: features, y: labels})
					#train_acc = acc.eval(feed_dict={x: features, y: labels})
					train_acc_list.append(train_acc)
					train_acc_top6_list.append(train_acc_top6)						
					if i%100 == 0:
						print('epoch={} i={}: train_acc={:.4f} [top6={:.4f}]'.\
							format(epoch, i, np.mean(train_acc_list), np.mean(train_acc_top6_list)))
					
					#if i%100 == 0:
					#	train_acc, train_acc_top6 = sess.run([acc, acc_top6], feed_dict={x: features, y: labels})

				except tf.errors.OutOfRangeError:
					print("End of training dataset.")
					break	


			print('EPOCH {}: train_acc={:.4f} [top6={:.4f}]; valid_acc={:.4f} [top6={:.4f}]\n'.\
				format(epoch, np.mean(train_acc_list), np.mean(train_acc_top6_list),
					np.mean(valid_acc_list), np.mean(valid_acc_top6_list)))
