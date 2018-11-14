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

#tf.enable_eager_execution()

#--
# Select network
#import models.inception_v3 as inception
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.slim.nets import resnet_v1, resnet_v2
from tensorflow.contrib.slim.nets import vgg

slim = tf.contrib.slim
#net = inception.inception_v2
net = resnet_v2.resnet_v2_101
#net = vgg.vgg_19
net_model_name = 'test'
print('Network name:', net_model_name)
#IMAGE_SIZE = (299, 299) 

num_classes = settings.num_classes
print('num_classes:', num_classes)
print('IMAGE_SIZE:', IMAGE_SIZE)

#--
# for saving results
results_filename = '_results_{}.txt'.format(net_model_name)
f_res = open(results_filename, 'wt')
dir_for_pb = 'pb'
dir_for_checkpoints = 'checkpoints'
checkpoint_name = net_model_name
os.system('mkdir -p {}'.format(dir_for_pb))
os.system('mkdir -p {}'.format(dir_for_checkpoints))

# dataset
goods_dataset = GoodsDataset("dataset-181018.list", "dataset-181018.labels", 
settings.IMAGE_SIZE, settings.train_batch, settings.valid_batch, settings.multiply, 
settings.valid_percentage)

train_dataset = goods_dataset.get_train_dataset()
valid_dataset = goods_dataset.get_valid_dataset()

num_epochs = 10
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
	
	x = tf.placeholder(tf.float32, [None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3], name='input')
	y = tf.placeholder(tf.float32, [None, num_classes], name='y')

	logits, end_points = net(x, num_classes=num_classes, is_training=True)
	logits = tf.reshape(logits, [-1, num_classes])

	loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
	train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
	correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
	acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # top-1 - mean value	
	acc_top6 = tf.nn.in_top_k(logits, tf.argmax(y,1), 6)  # list values for batch.
		
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		
		for epoch in range(num_epochs):
			print('\nEPOCH {0}'.format(epoch))

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
					
					#print(type((train_acc)))
					#print('train_acc:', train_acc)
					#print(type((train_acc_top6)))
					#print('train_acc_top6:', train_acc_top6)

					train_acc_list.append(train_acc)
					train_acc_top6_list.append(np.mean(train_acc_top6))
					if i % 100 == 0:
						print('epoch={} i={}: train_acc={:.4f} [top6={:.4f}]'.\
							format(epoch, i, np.mean(train_acc_list), np.mean(train_acc_top6_list)))
					
					#if i%100 == 0:
					#	train_acc, train_acc_top6 = sess.run([acc, acc_top6], feed_dict={x: features, y: labels})

				except tf.errors.OutOfRangeError:
					print("End of training dataset.")
					break	


			# valid
			timer('valid, epoch {0}'.format(epoch))
			valid_acc_list = []
			valid_acc_top6_list = []			

			for i in range(valid_steps_per_epoch):
				
				try:
					features, labels = sess.run(next_element_valid)
					valid_acc, valid_acc_top6 = sess.run([acc, acc_top6], feed_dict={x: features, y: labels})
					#print(type((valid_acc)))
					#print('valid_acc:', valid_acc)
					#print(type((valid_acc_top6)))
					#print('valid_acc_top6:', valid_acc_top6)
					valid_acc_list.append(valid_acc)
					valid_acc_top6_list.append(np.mean(valid_acc_top6))
					if i % 10 == 0:
						print('epoch={} i={}: valid_acc={:.4f} [top6={:.4f}]'.\
							format(epoch, i, np.mean(valid_acc_list), np.mean(valid_acc_top6_list)))
				except tf.errors.OutOfRangeError:
					print("End of valid dataset.")
					break
			timer()

			mean_train_acc = np.mean(train_acc_list)
			mean_train_acc_top6 = np.mean(train_acc_top6_list)
			mean_valid_acc = np.mean(valid_acc_list)
			mean_valid_acc_top6 = np.mean(valid_acc_top6_list)
			res = 'EPOCH {}: train_acc={:.4f} [top6={:.4f}]; valid_acc={:.4f} [top6={:.4f}]\n'.\
				format(epoch, mean_train_acc, mean_train_acc_top6,
					mean_valid_acc, mean_valid_acc_top6)
			print(res)
			f_res.write(res + '\n')

		
		# save_checkpoints	
		saver = tf.train.Saver()		
		saver.save(sess, './saved_model/{0}'.format(checkpoint_name))  

		# SAVE GRAPH TO PB
		graph = sess.graph			
		tf.graph_util.remove_training_nodes(graph.as_graph_def())
		# tf.contrib.quantize.create_eval_graph(graph)
		# tf.contrib.quantize.create_training_graph()
		output_node_names = [OUTPUT_NODE]
		output_graph_def = tf.graph_util.convert_variables_to_constants(
			sess, graph.as_graph_def(), output_node_names)
		# save graph:		
		pb_file_name = '{}_acc={:.4f}_[{:.4f}].pb'.format(net_model_name, mean_valid_acc, mean_valid_acc_top6)
		tf.train.write_graph(output_graph_def, dir_for_model, pb_file_name, as_text=False)	
			


f_res.close()

"""
Inception-v3.  (910 + 31 sec / epoch)  299x299.
EPOCH 0: train_acc=0.1611 [top6=0.3261]; valid_acc=0.1425 [top6=0.3247]
EPOCH 1: train_acc=0.1795 [top6=0.3852]; valid_acc=0.1840 [top6=0.3927]
EPOCH 5: train_acc=0.3400 [top6=0.7028]; valid_acc=0.2802 [top6=0.5678]
EPOCH 10: train_acc=0.6066 [top6=0.9249]; valid_acc=0.3785 [top6=0.7643]
EPOCH 20: train_acc=0.8325 [top6=0.9869]; valid_acc=0.6048 [top6=0.9237]
EPOCH 25: train_acc=0.8812 [top6=0.9933]; valid_acc=0.6762 [top6=0.9581]
EPOCH 30: train_acc=0.9187 [top6=0.9973]; valid_acc=0.6758 [top6=0.9500]
EPOCH 31: train_acc=0.9229 [top6=0.9972]; valid_acc=0.6625 [top6=0.9525]

"""