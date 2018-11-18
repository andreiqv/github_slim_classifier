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
#import models.inception_v3 as inception
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.slim.nets import resnet_v1, resnet_v2
from tensorflow.contrib.slim.nets import vgg
from tensorflow.contrib.slim.nets import alexnet
from nets import mobilenet_v1
from nets.mobilenet import mobilenet_v2
from nets.nasnet import nasnet
slim = tf.contrib.slim

#-----------------
# Select network

from nets import simple_fc
net_model_name = 'simple_fc'
net = simple_fc.fc

#net = inception.inception_v3
#net = inception.inception_v4
#net = resnet_v2.resnet_v2_50
#net = vgg.vgg_19
#net = mobilenet_v1.mobilenet_v1
#net = mobilenet_v2.mobilenet_v2_035
#net = nasnet.build_nasnet_mobile


#net_model_name = 'alexnet_v2'
#net = alexnet.alexnet_v2

#net_model_name = 'resnet_v2_152'
#net = resnet_v2.resnet_v2_152

#--------------

OUTPUT_NODE = 'softmax'
num_classes = settings.num_classes
print('num_classes:', num_classes)
print('IMAGE_SIZE:', IMAGE_SIZE) #IMAGE_SIZE = (299, 299) 
print('Network name:', net_model_name)

#--
# for saving results
results = {'epoch':[], 'train_loss':[], 'valid_loss':[], 'train_acc':[],\
	'valid_acc':[], 'train_top6':[], 'valid_top6':[]}
results_filename = '_results_{}.txt'.format(net_model_name)
f_res = open(results_filename, 'wt')
dir_for_pb = 'pb'
dir_for_checkpoints = 'checkpoints'
checkpoint_name = net_model_name
os.system('mkdir -p {}'.format(dir_for_pb))
os.system('mkdir -p {}'.format(dir_for_checkpoints))


#--
# plotting
SHOW_PLOT = True
import _thread
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
fig.suptitle(net_model_name, fontsize=16)

def plot_figure(results, ax1, ax2):
	ax1.cla()
	ax1.plot(results['epoch'], results['train_loss'])
	ax1.plot(results['epoch'], results['valid_loss'])
	ax1.legend(['train_loss', 'valid_loss'], loc='upper right')
	ax1.grid(color='g', linestyle='-', linewidth=0.2)
	ax1.set_ylim(0, 3.0)
	ax2.cla()
	ax2.plot(results['epoch'], results['train_top6'])
	ax2.plot(results['epoch'], results['valid_top6'])
	ax2.legend(['train_top6', 'valid_top6'], loc='upper left')
	ax2.grid(color='g', linestyle='-', linewidth=0.2)
	ax2.set_ylim(0.8, 1.0)
	#plt.show()
	outfile = '_results/{}.png'.format(net_model_name)
	plt.savefig(outfile)


#------------
# dataset
goods_dataset = GoodsDataset(settings.dataset_list, settings.labels_list, 
settings.IMAGE_SIZE, settings.train_batch, settings.valid_batch, settings.multiply, 
settings.valid_percentage)

train_dataset = goods_dataset.get_train_dataset()
valid_dataset = goods_dataset.get_valid_dataset()

num_epochs = 400
epochs_checkpoint = 2 # saving checkpoints and pb-file 
#train_steps_per_epoch = 1157
#valid_steps_per_epoch = 77
train_steps_per_epoch = 115
valid_steps_per_epoch = 12
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

graph = tf.Graph()  # сreate a new graph

with graph.as_default():
	
	iterator_train = train_dataset.make_one_shot_iterator()
	next_element_train = iterator_train.get_next()
	iterator_valid = valid_dataset.make_one_shot_iterator()
	next_element_valid = iterator_valid.get_next()

	#iterator_train = train_dataset.make_initializable_iterator()
	#x, y = next_element_train

	x = tf.placeholder(tf.float32, [None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3], name='input')
	y = tf.placeholder(tf.float32, [None, num_classes], name='y')

	logits, end_points = net(x, num_classes=num_classes, is_training=True)
	logits = tf.reshape(logits, [-1, num_classes])
	output = tf.nn.softmax(logits, name=OUTPUT_NODE)

	loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
	train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
	correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
	acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # top-1 - mean value	
	acc_top6 = tf.nn.in_top_k(logits, tf.argmax(y,1), 6)  # list values for batch.
		
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		
		for epoch in range(num_epochs):
			print('\nEPOCH {}/{}'.format(epoch, num_epochs))

			timer('train, epoch {0}'.format(epoch))
			train_loss_list, train_acc_list, train_top6_list = [], [], []

			for i in range(train_steps_per_epoch):
				
				try:
					features, labels = sess.run(next_element_train)
					#print(i, labels[0])
					sess.run(train_op, feed_dict={x: features, y: labels})
					
					#train_acc, train_acc_top6 = sess.run([acc, acc_top6], feed_dict={x: features, y: labels})
					train_loss, train_acc, train_top6 = sess.run([loss, acc, acc_top6], feed_dict={x: features, y: labels})

					train_loss_list.append(np.mean(train_loss))
					train_acc_list.append(train_acc)
					train_top6_list.append(np.mean(train_top6))

					if i % 100 == 0:
						print('epoch={} i={}: train loss={:.4f}, acc={:.4f}, top6={:.4f}'.\
							format(epoch, i, np.mean(train_loss_list), 
							np.mean(train_acc_list), np.mean(train_top6_list)))
					
				except tf.errors.OutOfRangeError:
					print("End of training dataset.")
					break	


			# valid
			timer('valid, epoch {0}'.format(epoch))
			valid_loss_list = []
			valid_acc_list = []
			valid_top6_list = []			

			for i in range(valid_steps_per_epoch):
				
				try:
					features, labels = sess.run(next_element_valid)
					valid_loss, valid_acc, valid_top6 = sess.run([loss, acc, acc_top6], feed_dict={x: features, y: labels})

					valid_loss_list.append(np.mean(valid_loss))
					valid_acc_list.append(valid_acc)
					valid_top6_list.append(np.mean(valid_top6))
					if i % 10 == 0:
						print('epoch={} i={}: valid acc={:.4f}, top6={:.4f}'.\
							format(epoch, i, np.mean(valid_acc_list), np.mean(valid_top6_list)))
				except tf.errors.OutOfRangeError:
					print("End of valid dataset.")
					break
			
			timer()

			# result for each epoch
			mean_train_loss = np.mean(train_loss_list)
			mean_valid_loss = np.mean(valid_loss_list)
			mean_train_acc = np.mean(train_acc_list)
			mean_valid_acc = np.mean(valid_acc_list)
			mean_train_top6 = np.mean(train_top6_list)
			mean_valid_top6 = np.mean(valid_top6_list)
			res = 'EPOCH {}: TRAIN loss={:.4f} acc={:.4f} top6={:.4f}; VALID loss={:.4f} acc={:.4f} top6={:.4f}\n'.\
				format(epoch, mean_train_loss, mean_train_acc, mean_train_top6,
					mean_valid_loss, mean_valid_acc, mean_valid_top6)
			print(res)
			f_res.write(res)
			f_res.flush()

			results['epoch'].append(epoch)
			results['train_loss'].append(mean_train_loss)
			results['valid_loss'].append(mean_valid_loss)
			results['train_acc'].append(mean_train_acc)
			results['valid_acc'].append(mean_valid_acc)
			results['train_top6'].append(mean_train_top6)
			results['valid_top6'].append(mean_valid_top6)			
			if SHOW_PLOT:
				plot_figure(results, ax1, ax2)
				#_thread.start_new_thread(plot_figure, ())

			if epoch % epochs_checkpoint == 0 and epoch > 1:
				# save_checkpoints	
				saver = tf.train.Saver()		
				saver.save(sess, './{}/{}'.\
					format(dir_for_checkpoints, checkpoint_name))  

				# SAVE GRAPH TO PB
				graph = sess.graph			
				tf.graph_util.remove_training_nodes(graph.as_graph_def())
				# tf.contrib.quantize.create_eval_graph(graph)
				# tf.contrib.quantize.create_training_graph()

				output_node_names = [OUTPUT_NODE]
				output_graph_def = tf.graph_util.convert_variables_to_constants(
					sess, graph.as_graph_def(), output_node_names)
				# save graph:		
				pb_file_name = '{}_(ep={}_acc={:.4f}_top6={:.4f}).pb'.format(net_model_name, epoch, mean_valid_acc, mean_valid_top6)
				tf.train.write_graph(output_graph_def, dir_for_pb, pb_file_name, as_text=False)	
			
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

vgg_19: (1653.3062 sec. + 39) 224x224
EPOCH 0: train_acc=0.1567 [top6=0.3003]; valid_acc=0.1644 [top6=0.3174]
EPOCH 1: train_acc=0.1674 [top6=0.3843]; valid_acc=0.1653 [top6=0.3898]


------
"""