import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import sklearn
import sys
import math
import imgaug as ia
from imgaug import augmenters as iaa

# tfe = tf.contrib.eager
#tf.enable_eager_execution()


# slim = tf.contrib.slim


def plot_random_nine(images, labels, names):
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    idx = np.arange(0, int(images.shape[0]))
    np.random.shuffle(idx)
    idx = idx[:9]

    for i, ax in enumerate(axes.flat):
        original = images[idx[i]]
        label = np.argmax(labels[idx[i], :])

        np_image = np.uint8(original * 255)  # [..., [0,1,2]]
        im = Image.fromarray(np_image).resize((140, 120), Image.BILINEAR)
        fnt = ImageFont.truetype('aux/Alice-Regular.ttf', 12)
        draw = ImageDraw.Draw(im)
        draw.text((5, 5), str(label) + names[label], font=fnt, fill=(255, 255, 255, 128))
        del draw

        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
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

                if plu_id not in images_dict:
                    images_dict[plu_id] = [line]
                else:
                    images_dict[plu_id].append(line)

        self.classes_count = len(images_dict.keys())

        for plu_id in images_dict.keys():
            # images_dict[plu_id] = sorted(images_dict[plu_id])
            sklearn.utils.shuffle(images_dict[plu_id], random_state=15)

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
        img_size = self.image_size

        def _random_distord(images, labels):
            images = tf.image.random_flip_left_right(images)
            images = tf.image.random_flip_up_down(images)

            print('images.shape:', images.shape)
            w, h = img_size
            a = max(w, h)
            d = math.ceil(a * (math.sqrt(2) - 1) / 2)
            print('d =', d)
            paddings = tf.constant([[0, 0], [d, d], [d, d], [0, 0]])
            images = tf.pad(images, paddings, "SYMMETRIC")

            print('images.shape:', images.shape)
            angle = tf.random_uniform(shape=(1,), minval=0, maxval=90)
            images = tf.contrib.image.rotate(images, angle * math.pi / 180, interpolation='BILINEAR')
            a0 = tf.constant([1.0])
            a1 = tf.random_uniform(shape=(1,), minval=0.0, maxval=0.2)
            a2 = tf.constant([-30.0])
            b0 = tf.random_uniform(shape=(1,), minval=0.0, maxval=0.2)
            b1 = tf.constant([1.0])
            b2 = tf.constant([-30.0])
            c0 = tf.constant([0.0])
            c1 = tf.constant([0.0])
            transform1 = tf.concat(axis=0, values=[a0, a1, a2, b0, b1, b2, c0, c1])
            images = tf.contrib.image.transform(images, transform1)
            images = tf.image.resize_image_with_crop_or_pad(images, w, h)

            images = tf.image.random_hue(images, max_delta=0.05)
            images = tf.image.random_contrast(images, lower=0.9, upper=1.5)
            images = tf.image.random_brightness(images, max_delta=0.1)
            images = tf.image.random_saturation(images, lower=1.0, upper=1.5)

            images = tf.minimum(images, 1.0)
            images = tf.maximum(images, 0.0)
            images.set_shape([None, None, None, 3])
            return images, labels

        dataset = dataset.map(_random_distord, num_parallel_calls=8)

        return dataset

    def get_train_dataset(self):
        with tf.device("/device:CPU:0"):
            dataset = tf.data.Dataset.from_tensor_slices((self.train_image_paths, self.train_image_labels))
            dataset = dataset.map(self._parse_function, num_parallel_calls=8)
            dataset = self._augment_dataset(dataset, self.multiply, self.train_batch)
        return dataset

    def get_valid_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.valid_image_paths, self.valid_image_labels))
        dataset = dataset.map(self._parse_function, num_parallel_calls=8)
        dataset = dataset.batch(self.valid_batch)
        return dataset

    def get_images_for_label(self, label):
        def _filter(im, lbl):
            l = tf.argmax(lbl)
            return tf.math.equal(l, tf.constant(label, dtype=tf.int64))

        dataset = tf.data.Dataset.from_tensor_slices((self.valid_image_paths, self.valid_image_labels))
        dataset = dataset.map(self._parse_function, num_parallel_calls=12)
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


class GoodsDatasetImgaug(GoodsDataset):

    def _augment_dataset(self, dataset, multiply, batch):
        dataset = dataset.repeat(multiply).batch(batch)
        img_size = self.image_size

        def _random_distord(images, labels):

            #--------
            # color augmentation
            images = tf.image.random_hue(images, max_delta=0.01)
            images = tf.image.random_contrast(images, lower=0.9, upper=1.1)
            images = tf.image.random_brightness(images, max_delta=0.02)
            images = tf.image.random_saturation(images, lower=1.0, upper=1.1)
            #--------            
            images = np.array(images)
            labels = np.array(labels)
            sometimes = lambda aug: iaa.Sometimes(0.5, aug)

            seq = iaa.Sequential([
                iaa.Crop(px=(0, 5)),  # crop images from each side by 0 to 16px (randomly chosen)
                iaa.Fliplr(0.5),  # horizontally flip 50% of the images
                iaa.Flipud(0.2),  # vertically flip 20% of all images
                iaa.GaussianBlur(sigma=(0, 3.0)),  # blur images with a sigma of 0 to 3.0
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    # translate by -20 to +20 percent (per axis)
                    rotate=(-45, 45),  # rotate by -45 to +45 degrees
                    shear=(-16, 16),  # shear by -16 to +16 degrees
                    order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # sometimes(iaa.Superpixels(p_replace=(0, 0.25), n_segments=125)),
                iaa.SomeOf((0, 2), [
                    iaa.PerspectiveTransform(scale=0.08),
                    iaa.PiecewiseAffine(scale=0.045),
                    iaa.PerspectiveTransform(0.075),
                ]),
                iaa.SomeOf((0, 1), [
                    # iaa.Pepper(0.03),
                    # iaa.Salt(0.015),
                    iaa.Dropout(0.01, per_channel=True),
                    iaa.CoarseDropout(p=0.1, size_percent=0.1, per_channel=True)
                ]),
                sometimes(iaa.Emboss(1, strength=0.25))
            ])

            images_aug = seq.augment_images(images)

            return images_aug, labels

        dataset = dataset.map(
            lambda images, labels:
            tuple(tf.contrib.eager.py_func(
                _random_distord, [images, labels], [tf.float32, tf.float64])
            )
        )

        return dataset


if __name__ == '__main__':
    # labels_list = []
    # with open("131018.labels", "r") as labels_file:
    #     for line in labels_file:
    #         labels_list.append(line.strip())
    #
    # goods_dataset = GoodsDataset("dataset.list", "output/se_classifier_161018.list", (299, 299), 32, 32, 5, 0.1)
    #
    # for i, (images, labels) in enumerate(goods_dataset.get_train_dataset()):
    #     plot_random_nine(images, labels, labels_list)
    #     q = 2

    arguments = argparse.ArgumentParser(description="create labels list for dataset paths")

    arguments.add_argument("--data", type=str, default="dataset.list",
                           help="a list of paths of images")

    arguments.add_argument("--labels", type=str, default="labels.list",
                           help="name of the output labels list")

    arguments = arguments.parse_args()

    GoodsDataset.generate_labels_list(arguments.data, arguments.labels)
