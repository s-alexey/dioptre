import random
import os
from typing import Tuple, NamedTuple

from PIL import Image
import yaml
import albumentations
import numpy as np

from dioptre import rendering


def resize(img, height):
    return img.resize((int(img.width * (height / img.height)), height), Image.ANTIALIAS)


class Example(NamedTuple):
    image: Image.Image
    text: str


def load(config):
    if isinstance(config, dict):
        config = config.copy()
        name = config.pop('name')
        cls = getattr(albumentations, name)
        if 'transforms' in config:
            config['transforms'] = [load(c) for c in config['transforms']]
        return cls(**config)

    if isinstance(config, str):
        return getattr(albumentations, config)()

    if isinstance(config, list):
        name, params = config
        cls = getattr(albumentations, name)
        if 'transforms' in config:
            params['transforms'] = [load(c) for c in params['transforms']]
        return cls(**params)


class DataGenerator:
    def __init__(self, alphabet: str, length: Tuple, fonts, augmentation, height=32):
        self.alphabet = alphabet
        self.length = length
        if isinstance(fonts, str):
            fonts = rendering.find_fonts(fonts, height)
        self.fonts = fonts

        if isinstance(augmentation, dict):
            augmentation = load(augmentation)

        self.augmentation = augmentation
        self.height = height

    def get_params(self):
        return {
            'alphabet': self.alphabet,
            'length': self.length,
            'augmentation': self.augmentation,
            'fonts': [' '.join(f.getname()) for f in self.fonts]
        }

    def __call__(self):
        while True:
            yield self.generate()

    def generate(self) -> Example:
        length = random.randint(*self.length)
        text = ''.join(random.choices(self.alphabet, k=length))
        font = random.choice(self.fonts)
        image = rendering.render(text=text, font=font)
        image = self.augmentation(image=np.array(resize(image, self.height)))['image']
        if len(image.shape) == 2:
            image = np.expand_dims(image, 2)

        image = image / 255. - .5
        return Example(image, text)

    def save(self, path):
        if os.path.isdir(path):
            path = os.path.join(path, 'generator.yaml')

        with open(path, 'wt') as fp:
            yaml.dump(self.get_params(), fp)

    @classmethod
    def load(cls, file, model=None):
        with open(file) as fp:
            kwargs = yaml.safe_load(fp)
            if model:
                kwargs['height'] = model.image_height
                kwargs['alphabet'] = model.alphabed
            return cls(**kwargs)

    def to_dataset(self):
        import tensorflow as tf
        return tf.data.Dataset.from_generator(self, (tf.float32, tf.string),
                                              (tf.TensorShape([self.height, None, 1]), tf.TensorShape([])))
