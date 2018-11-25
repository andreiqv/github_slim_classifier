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
import argparse

# tf.enable_eager_execution()
import settings
from settings import IMAGE_SIZE
from utils.timer import timer
from augment import images_augment


#-----------------
# Select network

import keras
#from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

from keras import models
from keras import layers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.applications import VGG16, inception_v3
from keras.applications.inception_v3 import InceptionV3


input_tensor = keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
conv_base = InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)
#conv_base = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

x = conv_base.output
x = Flatten()(x)
#x = Dense(1024, activation="relu")(x)
#x = Dropout(0.5)(x)
#x = Dense(1024, activation="relu")(x)
#predictions = Dense(settings.num_classes, activation="softmax")(x)

predictions = layers.Dense(settings.num_classes, activation='softmax')(x)
#model = Model(input=conv_base.input, output=predictions)  ??????????????
model = Model(inputs=conv_base.input, outputs=predictions)
print('model.trainable_weights:', len(model.trainable_weights))
#conv_base.trainable = False

num_layers = len(model.layers)
print('num_layers:', num_layers)
num_last_trainable_layers = 60
for layer in model.layers[:num_layers-num_last_trainable_layers]:
    layer.trainable = False

print('model.trainable_weights:', len(model.trainable_weights))
#  if num_last_trainable_layers = 60
#  model.trainable_weights: 190
#  model.trainable_weights: 35


#--------------

OUTPUT_NODE = 'softmax'
num_classes = settings.num_classes
print('num_classes:', num_classes)
print('IMAGE_SIZE:', IMAGE_SIZE) #IMAGE_SIZE = (299, 299) 

"""
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
"""


#------------
# dataset
from dataset_factory import GoodsDataset
#from dataset_factory_imgaug import GoodsDatasetImgaug as GoodsDataset

goods_dataset = GoodsDataset(settings.dataset_list, settings.labels_list, 
settings.IMAGE_SIZE, settings.train_batch, settings.valid_batch, settings.multiply, 
settings.valid_percentage)

train_dataset = goods_dataset.get_train_dataset()
valid_dataset = goods_dataset.get_valid_dataset()

num_epochs = 500		
epochs_checkpoint = 20 # interval for saving checkpoints and pb-file 
train_steps_per_epoch = 724 #1157
valid_steps_per_epoch = 78  #77

#train_dataset = train_dataset.repeat()
#valid_dataset = valid_dataset.repeat()


"""
def model_function(next_element):
	x, y = next_element
	logits, end_points = inception.inception_v3(
		x, num_classes=settings.num_classes, is_training=True)
	loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
	return logits, loss
"""


def createParser ():
	"""	ArgumentParser """
	parser = argparse.ArgumentParser()
	#parser.add_argument('-r', '--restore', dest='restore', action='store_true')
	parser.add_argument('-rc', '--restore_checkpoint', default=None, type=str, help='Restore from checkpoints')
	return parser


if __name__ == '__main__':

	parser = createParser()
	arguments = parser.parse_args(sys.argv[1:])	

	#graph = tf.Graph()  # сreate a new graph

	
	# ----------------------------------------------
	# Training

	def top_6(y_true, y_pred):    
	    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=6)

	#from keras.utils import multi_gpu_model
	#model = multi_gpu_model(model, gpus=2)

	model.compile(loss='categorical_crossentropy', #loss='binary_crossentropy',
				#optimizer='adagrad', 
				optimizer=optimizers.RMSprop(lr=0.01),
				metrics=['accuracy'])

	#train_steps_per_epoch = math.ceil(train_generator.n / train_generator.batch_size)
	#validation_steps = math.ceil(valid_generator.n / valid_generator.batch_size)
	#print('train data size:', train_generator.n)
	#print('train steps per epoch:', train_steps_per_epoch)
	#print('valid data size:', valid_generator.n)
	#print('validation_steps:', validation_steps)

	history = model.fit_generator(train_dataset,
		steps_per_epoch=train_steps_per_epoch,
		epochs=num_epochs,
		validation_data=valid_dataset,
		validation_steps=valid_steps_per_epoch)
