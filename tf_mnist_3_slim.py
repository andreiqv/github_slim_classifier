#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" A one-hidden-layer-MLP MNIST-classifier. """
from __future__ import absolute_import,  division, print_function
import tensorflow as tf
# Import the training data (MNIST)
import sys
import numpy as np
np.set_printoptions(precision=4, suppress=True)

import models.inception_v3 as inception
#from tensorflow.contrib.slim.nets import inception
slim = tf.contrib.slim

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/mnist", one_hot=True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_valid = mnist.validation.images
y_valid = mnist.validation.labels
x_test = mnist.test.images
y_test = mnist.test.labels
print('The train data size:', len(y_train))
print('The validation data size:', len(y_valid))
print('The test data size:', len(y_test))

BATCH_SIZE = 10
SAMPLE_SIZE = 500
NUM_STEPS = 1000

# some functions

def weight_variable(shape, name=None):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name=name)

def conv2d(x, W, name=None):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name=name)

def max_pool_2x2(x, name=None):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME',  name=name) 


# Create a new graph
graph = tf.Graph() # no necessiry

with graph.as_default():

	# 1. Construct a graph representing the model.
	x = tf.placeholder(tf.float32, [None, 784]) # Placeholder for input.
	y = tf.placeholder(tf.float32, [None, 10])  # Placeholder for labels.
	
	NN = 1 # type of neural network

	if NN == 1:  # SLIM inception

		x_images = tf.reshape(x, [-1,28,28,1])
		x_images = tf.image.resize_images(x_images, [299, 299])	
		logits, end_points = inception.inception_v3(
			x_images, num_classes=10, is_training=True)

	if NN == 2:  # SLIM simple NN

		print('use slim')
		def lenet(images):
			net = slim.conv2d(images, 20, [5,5], scope='conv1')
			net = slim.max_pool2d(net, [2,2], scope='pool1')
			net = slim.conv2d(net, 50, [5,5], scope='conv2')
			net = slim.max_pool2d(net, [2,2], scope='pool2')
			net = slim.flatten(net, scope='flatten3')
			net = slim.fully_connected(net, 500, scope='fc4')
			net = slim.fully_connected(net, 10, activation_fn=None, scope='fc5')
			#net = tf.contrib.layers.softmax(net)
			return net

		x_images = tf.reshape(x, [-1,28,28,1])	
		logits = lenet(x_images)	

	elif NN == 3:  # Full-connected

		W1 = tf.Variable(tf.random_uniform([784, 100])) # 784x100 weight matrix.
		b1 = tf.Variable(tf.zeros([100]))				# 100-element bias vector.
		layer1 = tf.nn.relu(tf.matmul(x, W1) + b1)   # Output of hidden layer.
	
		W2 = tf.Variable(tf.random_uniform([100, 10]))	# 100x10 weight matrix.
		b2 = tf.Variable(tf.zeros([10])) 				# 10-element bias vector.
		layer2 = tf.matmul(layer1, W2) + b2 		# Output of linear layer.

		logits = layer2

	elif NN == 4:	# CNN

		x_image = tf.reshape(x, [-1,28,28,1])

		# conv layer 1
		W1 = weight_variable([5,5,1,32], name='W1')  # 32 features, 5x5
		b1 = bias_variable([32], name='b1')
		h1 = tf.nn.relu(conv2d(x_image, W1, name='conv1') + b1)
		p1 = max_pool_2x2(h1, name='pool1')
		print('p1 =', p1)
		# 14 x 14
		
		# conv layer 2
		W2 = weight_variable([5,5,32,64], name='W2')  
		b2 = bias_variable([64], name='b2')
		h2 = tf.nn.relu(conv2d(p1, W2, name='conv2') + b2)
		p2 = max_pool_2x2(h2, name='pool2')
		print('p2 =', p2)    # 4*4*64
		# 7 x 7

		# fully-connected layer
		p2_flat = tf.reshape(p2, [-1, 7*7*64])
		W3 = weight_variable([7*7*64, 1024], name='W3')
		b3 = bias_variable([1024], name='b3')
		h3 = tf.nn.relu(tf.matmul(p2_flat, W3) + b3)
		print('h3 =', h3)

		# output layer
		W4 = weight_variable([1024, 10], name='W4')  
		b4 = bias_variable([10], name='b4')		  
		logits = tf.matmul(h3, W4) + b4   
		print('h3 =', h3)
		print('W4 =', W4)
		print('b4 =', b4)

	# 2. Add nodes that represent the optimization algorithm.
	#loss = tf.nn.softmax_cross_entropy_with_logits_v2(layer_2, y)

	print('output =', logits)
	print('y =', y)
	#sys.exit()

	loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
	train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)

	correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# 3. Execute the graph on batches of input data.
	with tf.Session() as sess:						# Connect to the TF runtime.
		init = tf.global_variables_initializer()
		sess.run(init)		# Randomly initialize weights.

		model_path = 'models/inception_v3.ckpt'
		variables_to_restore = slim.get_variables_to_restore(exclude=['fc8'])
		restorer = tf.train.Saver(variables_to_restore)
		restorer.restore(sess,model_path)

		for step in range(NUM_STEPS):				# Train iteratively for NUM_STEPS.			
			x_data, y_data = mnist.train.next_batch(BATCH_SIZE) # Load one batch of input data
			sess.run(train_op, {x: x_data, y: y_data}) 	 # Perform one training step.

			if step % 10 == 0:
				#train_accuracy = accuracy.eval(feed_dict = {x:x_data, y:y_data})
				train_accuracy = accuracy.eval(
					feed_dict={x:x_train[:SAMPLE_SIZE], y:y_train[:SAMPLE_SIZE]})
				validation_accuracy = accuracy.eval(
					feed_dict={x:x_valid[:SAMPLE_SIZE], y:y_valid[:SAMPLE_SIZE]})
				#x_valid_1000, y_valid_1000 = mnist.validation.next_batch(size)
				loss_value = loss.eval(
					feed_dict={x:x_train[:SAMPLE_SIZE], y:y_train[:SAMPLE_SIZE]}).mean() 
				print('step {:3}: train_loss={:0.3f}, train_acc={:0.3f}, valid_acc={:0.3f}'.\
					format(step, loss_value, train_accuracy, validation_accuracy))

				# sess.run(b2)	
				#print(accuracy.eval(feed_dict = {x : mnist.test.images[0:10], y : mnist.test.labels[0:10]}))

		# Save the comp. graph
		x_data, y_data = mnist.train.next_batch(BATCH_SIZE)		
		writer = tf.summary.FileWriter("output", sess.graph)
		print(sess.run(train_op, {x: x_data, y: y_data}))
		writer.close()	
		
		# Test of model
		test_accuracy = accuracy.eval(feed_dict={x:x_test, y:y_test})
		print('Test_accuracy={0:0.4f}'.format(test_accuracy))

		# Inference
		batch = mnist.test.next_batch(BATCH_SIZE)
		softmax = tf.nn.softmax(logits)
		output = softmax.eval(feed_dict = {x:batch[0]})
		predict = [np.argmax(output[i]) for i in range(BATCH_SIZE)]	
		target = [np.argmax(batch[1][i]) for i in range(BATCH_SIZE)]

		for i in range(BATCH_SIZE):
			print('{0} -> {1} -- {2}'.format(output[i], predict[i], target[i]))

		# Saver
		saver = tf.train.Saver()		
		saver.save(sess, './save_model/my_test_model')	






"""
Possible errors:

ValueError: Only call `softmax_cross_entropy_with_logits` with named arguments
- 
Use `tf.global_variables_initializer` instead.

"""