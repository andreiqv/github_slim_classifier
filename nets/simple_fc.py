import tensorflow as tf
slim = tf.contrib.slim
from settings import IMAGE_SIZE

def fc(inputs, num_classes=1000, is_training=True):

	x = tf.reshape(inputs, [-1, IMAGE_SIZE[0]*IMAGE_SIZE[1]*3])
	logits = slim.fully_connected(x, num_classes, scope='fc/fc_1')
	end_points = ['none']
	return logits, end_points

