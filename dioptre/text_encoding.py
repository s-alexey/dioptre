import tensorflow as tf
from tensorflow.python.ops import lookup_ops


class TextEncoder:
    """Encode a string tensor as an array of characters numeric encoding.

    Arguments:
        alphabet: iterable of unicode characters

    """

    def __init__(self, alphabet: str):
        self.alphabet = alphabet

        chars = tf.constant([x.encode() for x in self.alphabet])
        # chars = tf.constant(tf.strings.unicode_split(alphabet, input_encoding='UTF-8'))
        self._encode_table = lookup_ops.index_table_from_tensor(chars)
        self._decode_table = lookup_ops.index_to_string_table_from_tensor(chars)

    def encode(self, text: tf.Tensor):
        chars = tf.strings.unicode_split(text, input_encoding='UTF-8')

        if text.shape.rank == 2:
            # in batch mode `unicode_split` outputs ragged tensor
            chars = chars.values

        return self._encode_table.lookup(chars)

    def decode(self, labels: tf.SparseTensor):
        text = self._decode_table.lookup(tf.cast(labels, tf.int64))
        if not isinstance(text, tf.Tensor):
            text = tf.sparse.to_dense(text, default_value=b'')
        return tf.strings.reduce_join(text, axis=1)
