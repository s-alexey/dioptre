import os

import tensorflow as tf

from .base import DataSource


class MJSynthReader(DataSource):
    """Reads data from MJSynth dataset.

    Arguments:
        filename: path to a file with images paths
        image_height: pad images to this height
    """

    def __init__(self, filename, image_height=31):
        self.dirname = os.path.dirname(filename)
        self.filename = filename
        self.image_height = image_height

    def make_dataset(self):
        dataset = tf.data.TextLineDataset(self.filename)

        dataset = dataset.map(lambda x: tf.strings.split(x)[0])
        dataset = dataset.map(lambda x: (self.dirname + x, tf.strings.split(x, '_')[-2]))
        dataset = dataset.map(lambda f, l: (tf.image.decode_jpeg(tf.io.read_file(f)), l))
        if self.image_height != 31:
            dataset = dataset.map(lambda f, l: (tf.image.resize_with_crop_or_pad(
                f, target_height=self.image_height, target_width=tf.shape(f)[1]), l))

        dataset = dataset.map(lambda f, l: (f / 255 - .5, l))

        return dataset
