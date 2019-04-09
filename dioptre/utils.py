import tensorflow as tf
from dioptre.text_encoding import TextEncoder


def to_sparse(tensor):
    """Ensures `tensor` is a sparse tensor."""

    if isinstance(tensor, tf.SparseTensor):
        return tensor

    indices = tf.where(tf.not_equal(tensor, 0))
    values = tf.gather_nd(tensor, indices)
    shape = tf.shape(tensor, out_type=tf.int64)
    return tf.SparseTensor(indices, values, shape)


def batch_dataset(alphabet, image_shape, dataset, batch_size=32, bucket_boundaries=None, padded=True):
    # add image widths and text length
    dataset = dataset.map(lambda i, t: (i, tf.shape(i)[1], t, tf.strings.length(t, unit='UTF8_CHAR')))

    # ensure it created outside map function
    encoder = TextEncoder(alphabet)

    dataset = dataset.map(
        lambda img, width, text, length: (img, width, encoder.encode(text), length))

    output_shapes = (image_shape,) + dataset.output_shapes[1:]

    if bucket_boundaries:
        if isinstance(batch_size, int):
            batch_size = [batch_size] * (len(bucket_boundaries) + 1)

        dataset = dataset.apply(
            tf.data.experimental.bucket_by_sequence_length(lambda i, w, label, length: w,
                                                           bucket_boundaries=bucket_boundaries,
                                                           bucket_batch_sizes=batch_size,
                                                           padded_shapes=output_shapes)
        )

    elif padded:
        dataset = dataset.padded_batch(batch_size=batch_size,
                                       padded_shapes=output_shapes)
    else:
        dataset = dataset.batch(batch_size)

    return dataset
