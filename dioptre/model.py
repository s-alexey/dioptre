import tensorflow as tf

from dioptre.text_encoding import TextEncoder


def _shape_transformation(size, padding, stride, dilation=1):
    """Calculate output shape transformation of convolution or pooling layer.

    Output:
      a, b: transformation parameters in form of f(shape) = (shape + a) // b
    """
    if padding == 'same':
        padding = size - 1
    else:
        padding = 0

    return padding - dilation * (size - 1) - 1 + stride, stride


def output_width(layers, input_width):
    """Calculate output width after passing through sequential convolutional network.

    This function is used after batching variable-width images for
    computing true with of visual features sequence length.
    """
    a, b = 0, 1
    for layer in layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            a1, b1 = _shape_transformation(layer.kernel_size[1],
                                           padding=layer.padding,
                                           stride=layer.strides[1],
                                           dilation=layer.dilation_rate[1])
            a = a + a1 * b
            b = b1 * b

        if isinstance(layer, tf.keras.layers.MaxPool2D):
            a1, b1 = _shape_transformation(layer.pool_size[1],
                                           padding=layer.padding,
                                           stride=layer.strides[1])
            a = a + a1 * b
            b = b1 * b

    return (input_width + a) // b


class LineRecognizer(tf.keras.Model):
    """Line recognition model that consists of convolution network that extract visual features
    and a recurrent network for seq2seq transformation of this features into text."""

    def __init__(self, alphabet, convolutional, recurrent, dense_activation, skip_connection=False,
                 image_shape=(32, None, 1), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alphabet = alphabet
        self.image_shape = image_shape

        self.convolutional = tf.keras.Sequential(convolutional)
        self.recurrent = tf.keras.Sequential(recurrent)
        self.dense_activation = dense_activation
        self.dense = tf.keras.layers.Dense(len(alphabet) + 1, activation=dense_activation)

        self.encoder = TextEncoder(alphabet)
        self.skip_connection = skip_connection

    def _logits(self, images, widths, training=True):
        x = tf.cast(images, tf.float32)

        x = self.convolutional(x, training=training)

        # get rid of height dimension
        if x.shape[1] == 1:
            x = tf.squeeze(x, axis=1)
        else:
            # [None, height, width, filters] -> [None, width, height * filters]
            x = tf.concat(tf.unstack(x, axis=1), axis=2)

        # transpose to time-major
        x = transposed = tf.transpose(x, [1, 0, 2])

        x = self.recurrent(x)

        if self.skip_connection:
            x = tf.concat([transposed, x], axis=2)

        logits = self.dense(x)

        logits_length = output_width(self.convolutional.layers, widths)

        return logits, logits_length

    def decode(self, logits, logits_length):
        output, _ = tf.nn.ctc_beam_search_decoder(logits, logits_length)
        return self.encoder.decode(output[0])

    def call(self, image, width):
        return self._logits(image, width)
