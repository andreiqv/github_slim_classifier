import tensorflow as tf
import math
import settings
from settings import IMAGE_SIZE
      
def images_augment(images):
	"""
	images = tf.image.random_flip_left_right(images)
	images = tf.image.random_flip_up_down(images)
	"""
	
	# angle = tf.random_uniform(shape=(1,), minval=0, maxval=90)
	# images = tf.contrib.image.rotate(images, angle * math.pi / 180, interpolation='BILINEAR')

	# Rotation and transformation
	# print(images.shape)  # = (?, 299, 299, ?)
	print('images.shape:', images.shape)      
	w, h = IMAGE_SIZE
	a = max(w, h)
	d = math.ceil(a * (math.sqrt(2) - 1) / 2)
	print('paddings d =', d)
	paddings = tf.constant([[0, 0], [d, d], [d, d], [0, 0]])
	images = tf.pad(images, paddings, "SYMMETRIC")
	#images = tf.image.resize_image_with_crop_or_pad(images, w+d, h+d)
	print('images.shape:', images.shape)
	angle = tf.random_uniform(shape=(1,), minval=0, maxval=settings.rotation_max_angle)	    
	images = tf.contrib.image.rotate(images, angle * math.pi / 180, interpolation='BILINEAR')
	
	#images = tf.image.crop_to_bounding_box(images, d, d, s+d, s+d)
		        
	# Transformation
	#transform1 = tf.constant([1.0, 0.2, -30.0, 0.2, 1.0, 0.0, 0.0, 0.0], dtype=tf.float32)	
	# transform is  vector of length 8 or tensor of size N x 8
	# [a0, a1, a2, b0, b1, b2, c0, c1]	
	a0 = tf.constant([1.0])
	a1 = tf.random_uniform(shape=(1,), minval=0.0, maxval=settings.transform_maxval)
	a2 = tf.constant([-30.0])
	b0 = tf.random_uniform(shape=(1,), minval=0.0, maxval=settings.transform_maxval)
	b1 = tf.constant([1.0])
	b2 = tf.constant([-30.0])
	c0 = tf.constant([0.0])
	c1 = tf.constant([0.0])
	transform1 = tf.concat(axis=0, values=[a0, a1, a2, b0, b1, b2, c0, c1])
	#transform = tf.tile(tf.expand_dims(transform1, 0), [batch, 1])
	#print('Added transformations:', transform)
	images = tf.contrib.image.transform(images, transform1)	
	images = tf.image.resize_image_with_crop_or_pad(images, h, w)
	# ---	
	zoom = 1.1
	w_crop = math.ceil(w / zoom)
	h_crop = math.ceil(h / zoom)
	#batch_size = int(images.shape[0])
	#print(images.shape)
	batch_size = tf.size(images) / (3*h*w)
	images = tf.random_crop(images, [batch_size, h_crop, w_crop, 3])

	images = tf.image.resize_images(images, [h, w])	
	# ---
	# end of Rotation and Transformation block   

	
	"""
	# small delta:	
	images = tf.image.random_hue(images, max_delta=0.02)
	images = tf.image.random_contrast(images, lower=0.9, upper=1.2)
	images = tf.image.random_brightness(images, max_delta=0.05)
	images = tf.image.random_saturation(images, lower=1.0, upper=1.2)
	"""
	
	images = tf.image.random_hue(images, max_delta=0.05)
	images = tf.image.random_contrast(images, lower=0.9, upper=1.5)
	images = tf.image.random_brightness(images, max_delta=0.1)
	images = tf.image.random_saturation(images, lower=1.0, upper=1.5)	
	
	
	# add noise:
	noise = tf.random_normal(shape=tf.shape(images), mean=0.0, stddev=0.1, dtype=tf.float32)
	#images = tf.add(images, noise)

	#images = tf.image.per_image_standardization(images)
	#images = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images) 
	
	images = tf.minimum(images, 1.0)
	images = tf.maximum(images, 0.0)

	images.set_shape([None, None, None, 3])
	return images