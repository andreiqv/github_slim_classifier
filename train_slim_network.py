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

tf.enable_eager_execution()

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
iterator_train = train_dataset.make_one_shot_iterator()
next_batch_train = iterator_train.get_next()


# Create a new graph
graph = tf.Graph() # no necessiry

with graph.as_default():

	#x = tf.placeholder(tf.float32, [None, 784]) # Placeholder for input.
	#y = tf.placeholder(tf.float32, [None, 10])  # Placeholder for labels.
	#x_images = tf.reshape(x, [-1,28,28,1])
	#x_images = tf.image.resize_images(x_images, [299, 299])	

	#input_tensor = keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
	"""
	x =  tf.placeholder(tf.float32, [None, 299, 299, 3])
	y = tf.placeholder(tf.float32, [None, 10])
	
	logits, end_points = inception.inception_v3(
		x, num_classes=10, is_training=True)
	"""

	NUM_EPOCH = 100
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(NUM_EPOCH):
			print('\nEPOCH {0}'.format(epoch))	
			
			
			#for images, labels in enumerate(train_dataset):
			#	print(labels)


			
			while True:
				try:
					batch = sess.run(next_batch_train)
					#features = batch[0]
					#labels = batch[1]
					print(batch)

				except tf.errors.OutOfRangeError:
					print("End of training dataset.")
					break	

			