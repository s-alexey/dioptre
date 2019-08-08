import glob
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordWriter

from dioptre.data.base import DataSource


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    if isinstance(value, float):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    if isinstance(value, int):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def to_example(image, text):
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    raw_image = image.tobytes()

    height, width, *channels = image.shape

    if channels:
        channels = channels[0]
    else:
        channels = 1

    features = {
        'height': _int64_feature([height]),
        'width': _int64_feature([width]),
        'channels': _int64_feature([channels]),
        'image': _bytes_feature(raw_image),
        'text': _bytes_feature(text.encode()),
    }

    return tf.train.Example(features=tf.train.Features(feature=features))


def parse_fn(example):
    feature_description = {
        'height': tf.io.FixedLenFeature((), tf.int64, -1),
        'width': tf.io.FixedLenFeature((), tf.int64, -1),
        'channels': tf.io.FixedLenFeature((), tf.int64, -1),
        'image': tf.io.FixedLenFeature((), tf.string, ''),
        'text': tf.io.FixedLenFeature((), tf.string, '')
    }
    parsed = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_raw(parsed['image'], tf.uint8)
    image = tf.cast(image, tf.float32) / 255.
    image = tf.reshape(image, [parsed['height'], parsed['width'], parsed['channels']])
    return image, parsed['text']


def serialize(iterator, output_file):
    with TFRecordWriter(output_file) as writer:
        for img, text in iterator:
            writer.write(to_example(image=img, text=text).SerializeToString())


class TFRecordReader(DataSource):
    def __init__(self, target=None):
        self.target = target

    def detect_files(self):
        if os.path.isdir(self.target):
            return glob.glob(os.path.join(self.target, '*.tfrecord*'))

        return glob.glob(self.target)

    def make_dataset(self):
        import tensorflow as tf
        dataset = tf.data.TFRecordDataset(self.detect_files())

        return dataset.map(map_func=parse_fn)
