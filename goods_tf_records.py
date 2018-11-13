import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from dataset_factory_old import GoodsDataset
import time
import numpy as np

import settings
from settings import IMAGE_SIZE

#tf.enable_eager_execution()



def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def generate_tf_records(dataset, output_file, model):
    time_start = time.time()
    tt = 0.0

    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    with tf.python_io.TFRecordWriter(output_file, options) as record_writer:
        for i, (images, labels) in enumerate(dataset):
            output = model.predict(images)
            # labels = np.argmax(labels, axis=1)
            for q in range(output.shape[0]):
                label = np.array(labels[q])
                features = tf.train.Features(
                    feature={
                        "bottleneck_len": _int64_feature([output[q].shape[0]]),
                        "bottleneck": _float_feature(output[q].tolist()),
                        "label_len": _int64_feature([label.shape[0]]),
                        "label": _float_feature(label.tolist())
                    }
                )
                example = tf.train.Example(features=features)
                record_writer.write(example.SerializeToString())

            time_end = time.time()
            tt += 1.0 / (time_end - time_start)
            print(i, "{:.2f} fps".format(tt * output.shape[0] / (i + 1)))
            time_start = time.time()


def generate_tf_records_inceptionv3_top_60(dataset, output_file, output_layer_number):
    # mixed8 output # 248 (8,8,1280)
    # mixed9 output # 279 (8,8,2048)
    input_tensor = keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg',
                             input_tensor=input_tensor)
    intermediate_layer_model = keras.Model(inputs=base_model.input,
                                           outputs=base_model.layers[output_layer_number].output)

    time_start = time.time()
    tt = 0.0

    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    with tf.python_io.TFRecordWriter(output_file, options) as record_writer:
        for i, (images, labels) in enumerate(dataset):
            output = intermediate_layer_model.predict(images)
            # labels = np.argmax(labels, axis=1)
            for q in range(output.shape[0]):
                flat_output = np.reshape(output[q], (output[q].size))
                label = np.array(labels[q])

                features = tf.train.Features(
                    feature={
                        "bottleneck": _float_feature(flat_output.tolist()),
                        "label_len": _int64_feature([label.shape[0]]),
                        "label": _float_feature(label.tolist())
                    }
                )
                example = tf.train.Example(features=features)
                record_writer.write(example.SerializeToString())

            time_end = time.time()
            tt += 1.0 / (time_end - time_start)
            print(i, "{:.2f} fps".format(tt / (i + 1)))
            time_start = time.time()


class GoodsTfrecordsDataset:
    
    def __init__(self, train_file, valid_file, train_batch_size, valid_batch_size, compression=None) -> None:
        self.train_file = train_file
        self.valid_file = valid_file
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.compression = compression

        self.train_set = self.load_tfrecords_file(self.train_file)
        self.valid_set = self.load_tfrecords_file(self.valid_file)

    def load_tfrecords_file(self, file_name):
        def _convert_tfrecord_to_tensor(example_proto):
            # https://stackoverflow.com/questions/41921746/tensorflow-varlenfeature-vs-fixedlenfeature
            features = {
                "bottleneck_len": tf.FixedLenFeature(shape=[], dtype=tf.int64),
                'bottleneck': tf.VarLenFeature(dtype=tf.float32),
                'label_len': tf.FixedLenFeature(shape=[], dtype=tf.int64),
                'label': tf.VarLenFeature(dtype=tf.float32),
            }
            parsed_features = tf.parse_single_example(example_proto, features)

            label_len = tf.cast(parsed_features["label_len"], tf.int32)
            label_tensor = tf.sparse_tensor_to_dense(parsed_features["label"])
            label_tensor = tf.reshape(label_tensor, (label_len,))

            bottleneck_len = tf.cast(parsed_features["bottleneck_len"], tf.int32)
            bottleneck = tf.sparse_tensor_to_dense(parsed_features["bottleneck"])
            bottleneck = tf.reshape(bottleneck, (bottleneck_len,))
            # bottleneck.set_shape([1536])
            # label_tensor.set_shape([148])

            return bottleneck, label_tensor

        dataset = tf.data.TFRecordDataset(file_name, compression_type=self.compression)

        dataset = dataset.map(_convert_tfrecord_to_tensor, num_parallel_calls=8)
        return dataset


class InceptionV3Top60tfrecordsDataset:

    def __init__(self, train_file, valid_file, bottleneck_shape=(8, 8, 1280)) -> None:
        self.train_file = train_file
        self.valid_file = valid_file
        self.bottleneck_shape = bottleneck_shape

        self.train_set = self.load_tfrecords_file(self.train_file)
        self.valid_set = self.load_tfrecords_file(self.valid_file)

    def load_tfrecords_file(self, file_name):
        def _convert_tfrecord_to_tensor(example_proto):
            # https://stackoverflow.com/questions/41921746/tensorflow-varlenfeature-vs-fixedlenfeature
            features = {
                'bottleneck': tf.VarLenFeature(dtype=tf.float32),
                'label_len': tf.FixedLenFeature(shape=[], dtype=tf.int64),
                'label': tf.VarLenFeature(dtype=tf.float32),
            }
            parsed_features = tf.parse_single_example(example_proto, features)

            label_len = tf.cast(parsed_features["label_len"], tf.int32)
            label_tensor = tf.sparse_tensor_to_dense(parsed_features["label"])
            label_tensor = tf.reshape(label_tensor, (label_len,))

            bottleneck = tf.sparse_tensor_to_dense(parsed_features["bottleneck"])
            bottleneck = tf.reshape(bottleneck, self.bottleneck_shape)
            # bottleneck.set_shape([1536])
            # label_tensor.set_shape([148])

            return bottleneck, label_tensor

        dataset = tf.data.TFRecordDataset(file_name, buffer_size=10 * 1024 * 1024, 
            compression_type='ZLIB', num_parallel_reads=8) # added  num_parallel_calls

        dataset = dataset.map(_convert_tfrecord_to_tensor, num_parallel_calls=8)
        return dataset


if __name__ == '__main__':

    tf.enable_eager_execution()

    dataset = GoodsDataset("dataset-181018.list", "dataset-181018.labels", (IMAGE_SIZE[0], IMAGE_SIZE[1]),
                           settings.train_batch, settings.valid_batch, 
                           settings.multiply, settings.valid_percentage)

    dt_train = dataset.get_train_dataset()
    dt_valid = dataset.get_valid_dataset()

    #input_tensor = keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    # base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg',
    #                          input_tensor=input_tensor)

    # base_model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg',
    #                                input_tensor=input_tensor)

    # generate_tf_records(dt_train,
    #                     "tf_records/inceptionv4_bottlenecks_train_dataset181018.zlib.tfr",
    #                     base_model)
    #
    # generate_tf_records(dt_valid,
    #                     "tf_records/inceptionv4_bottlenecks_valid_dataset181018.zlib.tfr",
    #                     base_model)

    dir_prefix = "/home/andrei/work/_tf_records"
    generate_tf_records_inceptionv3_top_60(dataset.get_train_dataset().prefetch(4),
                                           dir_prefix + "/inceptionv3_top60_train_dataset1810.zlib.tfr", 
                                           output_layer_number=248)
    generate_tf_records_inceptionv3_top_60(dataset.get_valid_dataset().prefetch(4),
                                           dir_prefix + "/inceptionv3_top60_valid_dataset1810.zlib.tfr", 
                                           output_layer_number=248)

    # generate_tf_records(dataset.get_valid_dataset(),
    #                     "tf_records/goods_classifier_valid_noflip_hue_dataset1810.tfrecords")
    # goods_dataset = GoodsTfrecordsDataset("tf_records/goods_classifier_train_dataset1810.tfrecords",
    #                                       "tf_records/goods_classifier_valid_dataset1810.tfrecords",
    #                                       32, 32)
    #
    #
    # goods_dataset = InceptionV3Top30frecordsDataset(
    #     dir_prefix + "/goods_classifier_train_v3_top30_dataset1810.tfrecords",
    #     dir_prefix + "/goods_classifier_valid_v3_top30_dataset1810.tfrecords")

    # for i, (bottleneck, label) in enumerate(goods_dataset.valid_set.batch(32).prefetch(4)):
    #     q = 2
