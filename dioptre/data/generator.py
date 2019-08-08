import random
from typing import Tuple, Generator

from PIL import Image
import albumentations
import numpy as np

from dioptre.data import rendering
from dioptre.data.base import DataSource, Example


def resize(img, height):
    return img.resize((int(img.width * (height / img.height)), height), Image.ANTIALIAS)


def load_augmentation(config):
    if isinstance(config, dict):
        config = config.copy()
        name = config.pop('name')
        cls = getattr(albumentations, name)
        if 'transforms' in config:
            config['transforms'] = [load_augmentation(c) for c in config['transforms']]
        return cls(**config)

    if isinstance(config, str):
        return getattr(albumentations, config)()

    if isinstance(config, list):
        name, params = config
        cls = getattr(albumentations, name)
        if 'transforms' in config:
            params['transforms'] = [load_augmentation(c) for c in params['transforms']]
        return cls(**params)


class DataGenerator(DataSource):
    """Class for generating <image, text> examples.

    Inherited classes should overwrite `generate_text` for more realistic text generation.

    Arguments:
        alphabet: charset for random text generation
        fonts: either path to a directory with fonts or a list of font filenames
        augmentation: composition of `albumentations`' augmentations
        image_height: height of generated images

    Note:
         images are generated in [-.5, .5] scale
    """
    def __init__(self, alphabet: str, fonts, augmentation, image_height=32):
        self.alphabet = alphabet

        if isinstance(fonts, str):
            fonts = rendering.find_fonts(fonts, image_height)
        else:
            fonts = [rendering.load_font(font, size=image_height) for font in fonts]

        self.fonts = fonts

        if isinstance(augmentation, dict):
            augmentation = load_augmentation(augmentation)

        self.augmentation = augmentation
        self.image_height = image_height

    def iterate_text(self) -> Generator[str, None, None]:
        raise NotImplementedError

    def sample_font(self):
        return random.choice(self.fonts)

    def render(self, text, font=None):
        image = rendering.render(text=text, font=font)
        image = self.augmentation(image=np.array(resize(image, self.image_height)))['image']

        if len(image.shape) == 2:
            image = np.expand_dims(image, 2)

        image = image / 255. - .5
        return Example(image, text)

    def _make_iterator(self):
        for text in self.iterate_text():
            yield self.render(text, font=self.sample_font())

    def make_dataset(self):
        import tensorflow as tf
        return tf.data.Dataset.from_generator(self._make_iterator, (tf.float32, tf.string),
                                              (tf.TensorShape([self.image_height, None, 1]), tf.TensorShape([])))


class RandomDataGenerator(DataGenerator):
    """Class for meaningless text generator.

    Arguments:
        alphabet: charset for random text generation
        length_range: numeric bounds for text length

    """
    def __init__(self, alphabet: str, length_range: Tuple[int, int], **kwargs):
        super().__init__(alphabet=alphabet, **kwargs)
        self.length_range = length_range

    def iterate_text(self):
        while True:
            length = random.randint(*self.length_range)
            yield ''.join(random.choices(self.alphabet, k=length))
