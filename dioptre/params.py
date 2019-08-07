import os
import string
import logging

import yaml
import param

logger = logging.getLogger(__name__)


class BaseConfig(param.Parameterized):

    def __init__(self, **kwargs):
        kwargs = {
            key: value if not isinstance(getattr(type(self), key), BaseConfig)
            else type(getattr(type(self), key))(**value)
            for key, value in kwargs.items()}
        super().__init__(**kwargs)

    @classmethod
    def load(cls, source):
        if os.path.isdir(source):
            source = os.path.join(source, cls.name.lower() + '.yaml')

        if not os.path.isfile(source):
            logger.warning('Missing config file `%s`, default parameters are used.', source)
            return cls()

        with open(source) as fp:
            return cls(**yaml.safe_load(fp))

    def save(self, target):
        if os.path.isdir(target):
            target = os.path.join(target, type(self).name + '.yaml')

        with open(target, 'wt') as fp:
            values = self.get_param_values()
            values.pop('name')
            yaml.safe_dump(values, fp, allow_unicode=True)


class Model(BaseConfig):
    alphabet = param.String(
        default=string.ascii_letters,
        doc='A string that contains all the characters.',
    )
    image_height = param.Integer(32, bounds=(7, None))
    cnn = param.List()
    rnn = param.List()

    default_conv_activation = param.String('relu')
    dense_activation = param.String('relu')
    skip_connection = param.Boolean(False)

    def create(self):
        from dioptre.model import LineRecognizer
        from tensorflow.keras import layers

        cnn = []
        for name, params in self.cnn:
            if name == 'Conv2D':
                params.setdefault('activation', self.default_conv_activation)
            elif name == 'BatchNormalization':
                params['axis'] = 3

            cnn.append(getattr(layers, name)(**params))

        rnn = []
        for name, config in self.rnn:
            config = config.copy()
            bidirectional = config.pop('bidirectional', False)
            layer = getattr(layers, name)(**config, time_major=True, return_sequences=True)

            if bidirectional:
                layer = layers.Bidirectional(layer)

            rnn.append(layer)

        return LineRecognizer(self.alphabet,
                              convolutional=cnn,
                              recurrent=rnn,
                              dense_activation=self.dense_activation,
                              skip_connection=self.skip_connection)


class Feeding(BaseConfig):
    type = param.String('generator')
    params = param.Dict({})

    def create(self):
        if self.type == 'generator':
            from dioptre.data_generator import DataGenerator
            return DataGenerator(**self.params)

        elif self.type == 'tfrecords':
            from dioptre.tfrecords import TFRecordReader
            return TFRecordReader(**self.params)

        else:
            raise ValueError(f'Unknown feeding type `{self.type}`')


class BatchConfig(BaseConfig):
    size = param.Parameter(32)
    bucket_boundaries = param.List()
    padded = param.Boolean(True)


class Training(BaseConfig):
    batch = BatchConfig()

    seed = param.Number(0)

    learning_rate = param.Number(bounds=(0, 1))
    learning_rate_decay_step = param.Integer(None, bounds=(0, None))
    learning_rate_decay_rate = param.Number(None, bounds=(0, None))

    optimizer = param.String('Adam')
    optimizer_params = param.Dict({})

    epochs = param.Integer(100)
    steps_per_epoch = param.Integer(100)

    l2_regularization = param.Number(0.0001, inclusive_bounds=(0, .01))


class Config(param.Parameterized):
    training = Training()
    model = Model()
    feeding = Feeding()

    @classmethod
    def load(cls, directory):
        return cls(training=Training.load(directory),
                   model=Model.load(directory),
                   feeding=Feeding.load(directory))
