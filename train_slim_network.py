#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Прямой проход без хеширования промежуточных данных.
"""
	
# https://github.com/tensorflow/tensorflow/issues/22837#issuecomment-428327601

import tensorflow as tf
import numpy as np

#import models.inception_v3 as inception
from tensorflow.contrib.slim.nets import inception
slim = tf.contrib.slim

from dataset_factory import GoodsDataset
#from goods_tf_records import GoodsTfrecordsDataset

# tf.enable_eager_execution()
import settings

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


def model_function(next_element):

	x, y = next_element

	logits, end_points = inception.inception_v3(
		x, num_classes=settings.num_classes, is_training=True)
	loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)

	return logits, loss


# Create a new graph
graph = tf.Graph() # no necessiry

with graph.as_default():

	iterator_train = train_dataset.make_one_shot_iterator()
	next_element_train = iterator_train.get_next()

	#x = tf.placeholder(tf.float32, [None, 784]) # Placeholder for input.
	#y = tf.placeholder(tf.float32, [None, 10])  # Placeholder for labels.
	#x_images = tf.reshape(x, [-1,28,28,1])
	#x_images = tf.image.resize_images(x_images, [299, 299])	

	#input_tensor = keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
	
	#x =  tf.placeholder(tf.float32, [None, 299, 299, 3])

	"""
	y = tf.placeholder(tf.float32, [None, 10])
	
	x = next_batch_train
	logits, end_points = inception.inception_v3(
		x, num_classes=10, is_training=True)

	loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
	train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
	correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
	acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # top-1	
	acc_top6 = tf.nn.in_top_k(logits, tf.argmax(y,1), 6)
	"""

	logits, loss = model_function(next_element_train)
	train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
	correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
	acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # top-1	
	acc_top6 = tf.nn.in_top_k(logits, tf.argmax(y,1), 6)

	NUM_EPOCH = 100
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(NUM_EPOCH):
			print('\nEPOCH {0}'.format(epoch))	
			
			
			#for images, labels in enumerate(train_dataset):
			#	print(labels)


			i = 0
			while True:
				i += 1
				try:
					#batch = sess.run(next_batch_train)
					#features = batch[0]
					#labels = batch[1]
					#print(labels)
					sess.run(train_op)
					train_acc = acc.eval(feed_dict={x: batch[0], y: batch[1]})
					print('epoch={0} i={1} train_acc={2:.4f}'.format(epoch, i, train_acc))

				except tf.errors.OutOfRangeError:
					print("End of training dataset.")
					break	

			