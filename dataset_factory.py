#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Старый вариант factory без crop, rotate и transforms.
"""

import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import sys
import sklearn
import math

import settings
from settings import IMAGE_SIZE

# tfe = tf.contrib.eager
# slim = tf.contrib.slim

#tf.enable_eager_execution()


def plot_random_nine(images, labels, names=[]):
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    idx = np.arange(0, int(images.shape[0]))
    #np.random.shuffle(idx)
    idx = idx[:9]

    for i, ax in enumerate(axes.flat):
        original = images[idx[i]]
        label = np.argmax(labels[idx[i], :])

        np_image = np.uint8(original * 255)  # [..., [0,1,2]]
        im = Image.fromarray(np_image).resize((140, 120), Image.BILINEAR)
        #fnt = ImageFont.truetype('Alice-Regular.ttf', 12)
        draw = ImageDraw.Draw(im)
        #draw.text((5, 5), str(label), font=fnt, fill=(255, 255, 255, 128))
        #draw.text((5, 5), names[label], font=fnt, fill=(255, 255, 255, 128))
        del draw

        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])

    """
    if True:    
        w, h = IMAGE_SIZE
        zoom = 1.5
        w_crop = math.ceil(w / zoom)
        h_crop = math.ceil(h / zoom)
        images = tf.random_crop(images, [settings.train_batch, h_crop, w_crop, 3])
        images = tf.image.resize_images(images, [h, w])

        fig, axes = plt.subplots(3, 3)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        idx = np.arange(0, int(images.shape[0]))
        #np.random.shuffle(idx)
        idx = idx[:9]
        for i, ax in enumerate(axes.flat):
            original = images[idx[i]]
            label = np.argmax(labels[idx[i], :])

            np_image = np.uint8(original * 255)  # [..., [0,1,2]]
            im = Image.fromarray(np_image).resize((140, 120), Image.BILINEAR)
            #fnt = ImageFont.truetype('Alice-Regular.ttf', 12)
            draw = ImageDraw.Draw(im)
            #draw.text((5, 5), str(label), font=fnt, fill=(255, 255, 255, 128))
            #draw.text((5, 5), names[label], font=fnt, fill=(255, 255, 255, 128))
            del draw
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
    """

    plt.show()
    
    


class GoodsDataset:
    def __init__(self, path_list,
                 labels_list,
                 image_size,
                 train_batch,
                 valid_batch,
                 multiply,
                 valid_percentage) -> None:
        super().__init__()
        self.path_list = path_list
        self.image_size = image_size
        self.labels_list = labels_list
        self.train_batch = train_batch
        self.valid_batch = valid_batch
        self.multiply = multiply
        self.valid_percentage = valid_percentage

        self.load_labels()
        self.load_images()

    def load_labels(self):
        self.labels = {}
        i = 0
        with open(self.labels_list, "r") as l_f:
            for line in l_f:
                line = line.replace("\n", "").strip()

                self.labels[line] = i
                i += 1

    def load_images(self):
        train_image_paths = np.array([])
        valid_image_paths = np.array([])

        train_image_labels = []
        valid_image_labels = []

        images_dict = {}

        with open(self.path_list, "r") as pl:
            for line in pl:
                line = line.strip()
                line = line.replace("\n", "")
                plu_id = line.split("/")[-2]

                
                def add_line_to_images_dict(_id, line, img_dict):
                    if _id in img_dict:
                        img_dict[_id].append(line)
                    else:
                        img_dict[_id] = [line]                	

                
                #similar_goods = [{'38','413','36','17'},\
                #    {'407','31','404','44','313','35'},\
                #    {'37', '46', '424', '40', '4', '103'}]

                similar_goods = []
                    
                flag = False
                for goods in similar_goods:
                    if plu_id in goods:
                        flag = True
                        for _id in goods:
                            add_line_to_images_dict(_id, line, images_dict)
                
                if not flag:   # as usually
                    add_line_to_images_dict(plu_id, line, images_dict)

                """    
                if plu_id not in images_dict:
                    images_dict[plu_id] = [line]
                else:
                    images_dict[plu_id].append(line)
                """

                    
        self.classes_count = len(images_dict.keys())

        for plu_id in images_dict.keys():
            
            # SORTED:
            #print('Dataset_order: {}'.format(settings.dataset_order))
            if settings.dataset_order == 'sort':
                images_dict[plu_id] = sorted(images_dict[plu_id])
            # or RANDOM with fix random_state   
            elif settings.dataset_order == 'shuffle':
                sklearn.utils.shuffle(images_dict[plu_id], random_state=15)
            else:
                raise Exception('Bad value of dataset_order.')
            
            images_dict[plu_id] = np.array(images_dict[plu_id])
            valid_mask = np.zeros(len(images_dict[plu_id]), dtype=bool)

            index = self.labels[plu_id]
            one_hot = np.eye(self.classes_count, self.classes_count)[index]
            one_hot = np.tile(one_hot, (len(images_dict[plu_id]), 1))

            if len(images_dict[plu_id]) >= 20:
                idx_count = int(len(images_dict[plu_id]) * self.valid_percentage)
                idx = np.arange(len(images_dict[plu_id]))
                # np.random.shuffle(idx)
                idxs = idx[:idx_count]

                valid_mask[idxs] = True
                valid_images = images_dict[plu_id][valid_mask]

                valid_image_paths = np.append(valid_image_paths, valid_images)
                if valid_image_labels == []:
                    valid_image_labels = one_hot[valid_mask]
                else:
                    valid_image_labels = np.concatenate([valid_image_labels, one_hot[valid_mask]])

            if train_image_labels == []:
                train_image_labels = one_hot[~valid_mask]
            else:
                train_image_labels = np.concatenate([train_image_labels, one_hot[~valid_mask]])
            train_image_paths = np.append(train_image_paths, images_dict[plu_id][~valid_mask])

        print("train dataset", train_image_labels.shape[0])
        print("valid dataset", valid_image_labels.shape[0])
        randomize = np.arange(valid_image_labels.shape[0])
        np.random.shuffle(randomize)

        self.valid_image_labels = valid_image_labels[randomize]
        self.valid_image_paths = valid_image_paths[randomize]

        randomize = np.arange(train_image_labels.shape[0])
        np.random.shuffle(randomize)

        self.train_image_labels = train_image_labels[randomize]
        self.train_image_paths = train_image_paths[randomize]

    def _parse_function(self, image_path, label):
        image_string = tf.read_file(image_path)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [self.image_size[1], self.image_size[0]],
                                               method=tf.image.ResizeMethod.BICUBIC)
        image = tf.cast(image_resized, tf.float32) / tf.constant(255.0)

        return image, label,

    def _augment_dataset(self, dataset, multiply, batch):
        
        dataset = dataset.repeat(multiply).batch(batch)

        def _random_distord(images, labels):
            images = tf.image.random_flip_left_right(images)
            images = tf.image.random_flip_up_down(images)
            # angle = tf.random_uniform(shape=(1,), minval=0, maxval=90)
            # images = tf.contrib.image.rotate(images, angle * math.pi / 180, interpolation='BILINEAR')

            if settings.do_transformations:
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
                zoom = settings.crop_zoom
                w_crop = math.ceil(w / zoom)
                h_crop = math.ceil(h / zoom)
                #batch_size = int(images.shape[0])
                #print(images.shape)
                batch_size = tf.size(images) / (3*h*w)
                images = tf.random_crop(images, [batch_size, h_crop, w_crop, 3])
                

                images = tf.image.resize_images(images, [h, w])            
                # ---
                # end of Rotation and Transformation block   


            images = tf.image.random_hue(images, max_delta=0.1)
            images = tf.image.random_contrast(images, lower=0.6, upper=1.7)
            images = tf.image.random_brightness(images, max_delta=0.12)
            images = tf.image.random_saturation(images, lower=0.7, upper=1.7)
            """
            images = tf.image.random_hue(images, max_delta=0.05)
            images = tf.image.random_contrast(images, lower=0.9, upper=1.5)
            images = tf.image.random_brightness(images, max_delta=0.1)
            images = tf.image.random_saturation(images, lower=1.0, upper=1.5)
            """

            # add noise:
            noise = tf.random_normal(shape=tf.shape(images), mean=0.0, stddev=0.03, dtype=tf.float32)
            images = tf.add(images, noise)

            #images = tf.image.per_image_standardization(images)
            #images = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images)                    
            images = tf.minimum(images, 1.0)
            images = tf.maximum(images, 0.0)

            images.set_shape([None, None, None, 3])
            return images, labels

        dataset = dataset.map(_random_distord, num_parallel_calls=8)
        #dataset = dataset.batch(batch)

        return dataset

    def get_train_dataset(self):
        with tf.device("/device:CPU:0"):
        #with tf.device("/device:GPU:1"):        
            dataset = tf.data.Dataset.from_tensor_slices((self.train_image_paths, self.train_image_labels))
            dataset = dataset.map(self._parse_function, num_parallel_calls=8)
            dataset = self._augment_dataset(dataset, self.multiply, self.train_batch)
        return dataset

    def get_valid_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.valid_image_paths, self.valid_image_labels))
        dataset = dataset.map(self._parse_function).prefetch(2)

        return dataset.batch(self.valid_batch)

    def get_images_for_label(self, label):
        def _filter(im, lbl):
            l = tf.argmax(lbl)
            return tf.math.equal(l, tf.constant(label, dtype=tf.int64))

        dataset = tf.data.Dataset.from_tensor_slices((self.valid_image_paths, self.valid_image_labels))
        dataset = dataset.map(self._parse_function)
        dataset = dataset.filter(_filter)

        return dataset


    @staticmethod
    def generate_labels_list(dataset_paths, labels_list_output):
        labels = []
        with open(dataset_paths, "r") as d_p:
            for line in d_p:
                line = line.strip()
                line = line.replace("\n", "")
                plu_id = line.split("/")[-2]
                labels.append(plu_id)
        labels = set(labels)
        labels = sorted(labels)
        with open(labels_list_output, "w") as l_f:
            for label in labels:
                l_f.write(label + "\n")


if __name__ == '__main__':

    #tf.enable_eager_execution()

    arguments = argparse.ArgumentParser(description="create labels list for dataset paths")
    arguments.add_argument("--data", type=str, default=None,
                           help="a list of paths of images")
    arguments.add_argument("--labels", type=str, default=None,
                           help="name of the output labels list")
    arguments = arguments.parse_args()

    if arguments.data and arguments.labels:
        GoodsDataset.generate_labels_list(arguments.data, arguments.labels)
        print('the list of labels was created.')
        sys.exit()


    # labels_list = []
    # with open("131018.labels", "r") as labels_file:
    #     for line in labels_file:
    #         labels_list.append(line.strip())
    #
    #goods_dataset = GoodsDataset("dataset.list", "output/se_classifier_161018.list", (299, 299), 32, 32, 5, 0.1)
    goods_dataset = GoodsDataset("dataset-181018.list", "dataset-181018.labels", 
        settings.IMAGE_SIZE, settings.train_batch, settings.valid_batch, settings.multiply, 
        settings.valid_percentage)
    #
    for i, (images, labels) in enumerate(goods_dataset.get_train_dataset()):

        plot_random_nine(images, labels)

        """
        w, h = IMAGE_SIZE
        zoom = 1.2
        w_crop = math.ceil(w / zoom)
        h_crop = math.ceil(h / zoom)
        crop_images = tf.random_crop(images, [1, h_crop, w_crop, 3])
        crop_images = tf.image.resize_images(crop_images, [h, w])  
        plot_random_nine(crop_images, labels)
		"""

        if i > 2: 
        	sys.exit(0)
          #plot_random_nine(images, labels, labels_list)
    #     q = 2


