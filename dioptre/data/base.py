from typing import NamedTuple

import numpy as np


class Example(NamedTuple):
    image: np.ndarray
    text: str

    def _ipython_display_(self):
        from matplotlib import pyplot as plt
        if self.image.shape[2] == 1:
            image = self.image[:, :, 0]
        else:
            image = self.image

        plt.imshow(image, cmap='gray', interpolation=None)
        plt.title(self.text)


class DataSource:
    """Base class for training and validation example providers."""

    def make_dataset(self):
        """Creates a `tf.data.Dataset` instances of (image, text) tensors.
        Images has tf.float32 type and scaled within [-0.5, 0.5] bounds.
        """
        raise NotImplementedError
