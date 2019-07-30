import random
from typing import Tuple, NamedTuple

from PIL import Image
import yaml
import albumentations
import numpy as np

from dioptre import rendering


def resize(img, height):
    return img.resize((int(img.width * (height / img.height)), height), Image.ANTIALIAS)


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


class DataGenerator:
    """Tool for generating training <image, text> examples.

    Inherited classes should overwrite `generate_text` for more realistic text generation.

    Arguments:
        alphabet: charset for random text generation
        length_range: numeric bounds for text length
        fonts: either path to a directory with fonts or a list of font filenames
        augmentation: composition of `albumentations`' augmentations
        image_height: height of generated images

    Note:
         images are generated in [-.5, .5] scale
    """
    def __init__(self, alphabet: str, length_range: Tuple[int, int], fonts, augmentation, image_height=32):
        self.alphabet = alphabet
        self.length_range = length_range

        if isinstance(fonts, str):
            fonts = rendering.find_fonts(fonts, image_height)
        else:
            fonts = [rendering.load_font(font, size=image_height) for font in fonts]

        self.fonts = fonts

        if isinstance(augmentation, dict):
            augmentation = load_augmentation(augmentation)

        self.augmentation = augmentation
        self.image_height = image_height

    def generate_text(self):
        length = random.randint(*self.length_range)
        text = ''.join(random.choices(self.alphabet, k=length))
        return text

    def sample_font(self):
        return random.choice(self.fonts)

    def generate(self) -> Example:
        text = self.generate_text()
        font = self.sample_font()
        image = rendering.render(text=text, font=font)
        image = self.augmentation(image=np.array(resize(image, self.image_height)))['image']

        if len(image.shape) == 2:
            image = np.expand_dims(image, 2)

        image = image / 255. - .5
        return Example(image, text)

    @classmethod
    def load(cls, file, model=None):
        with open(file) as fp:
            kwargs = yaml.safe_load(fp)
            if model:
                kwargs['image_height'] = model.image_height
                kwargs['alphabet'] = model.alphabed
            return cls(**kwargs)

    def _make_iterator(self):
        while True:
            yield self.generate()

    def to_dataset(self):
        """Creates an `tf.data.Dataset`."""
        import tensorflow as tf
        return tf.data.Dataset.from_generator(self._make_iterator, (tf.float32, tf.string),
                                              (tf.TensorShape([self.image_height, None, 1]), tf.TensorShape([])))
