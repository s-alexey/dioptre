import tensorflow as tf
from dioptre.utils import to_sparse


def character_error_rate(truth: tf.SparseTensor, prediction: tf.SparseTensor):
    """Calculate character error rate.

    CER is the ratio between the number of mistakes (edit distance) and the text length.

    """
    truth, prediction = to_sparse(truth), to_sparse(prediction)

    edit_distances = tf.edit_distance(truth, prediction, normalize=False)

    distance = tf.reduce_sum(edit_distances)
    text_length = tf.shape(truth.values)[0]

    error_rate = tf.truediv(distance, tf.cast(text_length, tf.float32))

    return error_rate
