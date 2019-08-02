# diOpTRe 

This repository contain implementation of end-to-end neural network for **op**tical **t**ext **re**cognition 
described in an [arXiv paper](https://arxiv.org/abs/1507.05717) and implemented in various framework 
(like tesseract and ocropy).

In the repository you can find tools for generation of artificial <image, text> data, network evaluation and 
result visualization. 


In order to train a new model with artificial data one should complete several steps: 

* define alphabet, fonts and augmentation (using [albumentations](https://albumentations.readthedocs.io/en/latest/)): 
```python
import albumentations as alb

alphabet = ' ' + string.ascii_letters

fonts = ['fonts/calibri.ttf', 'fonts/cambria.ttc']

augmentation = alb.Compose([
        alb.OneOf([
            alb.IAAAdditiveGaussianNoise(),
            alb.GaussNoise(),
        ], p=0.2),
        alb.OneOf([
            alb.MotionBlur(p=0.2),
            alb.MedianBlur(blur_limit=3, p=0.1),
            alb.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        alb.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.01, rotate_limit=5, p=0.2),
        alb.OneOf([
            alb.OpticalDistortion(p=0.3),
            alb.GridDistortion(p=0.3),
            alb.IAAPiecewiseAffine(p=0.3),
        ], p=0.9),
    ], p=.9)
```

* create a `DataGenerator` and `tf.data.Dataset`:

```python
from dioptre.data_generator import DataGenerator
generator = DataGenerator(alphabet=' ' + string.ascii_letters, 
                          length_range=(3, 15), 
                          fonts=fonts, 
                          augmentation=augmentation, 
                          image_height=32)
dataset = generator.to_dataset()
```

* create a network that consist of two major pars: convolutional and recurrent networks. 
In the paper mentioned before was used a slightly deeper network, but it seems like shorter ones 
are sufficient for most real-word tasks.

```python
import tensorflow as tf

from dioptre.model import LineRecognizer


model = LineRecognizer(
    alphabet=alphabet,
    convolutional=[
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 1), padding='valid'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu'),
        tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='valid', activation='relu'),
        tf.keras.layers.BatchNormalization(),
    ],
    recurrent=[
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(256, time_major=True, return_sequences=True)
        )
    ],
    dense_activation='relu',
    image_shape=(32, None, 1),
)
```

* run training loop

```python
padded_dataset = dioptre.utils.batch_dataset(dataset=dataset, model=model, 
                                             batch_size=64, padded=True)
                                             
@tf.function
def train_step(image, width, labels, length):
    with tf.GradientTape() as tape:
        logits, logit_length = model(image, width)
        loss = tf.reduce_mean(tf.nn.ctc_loss(
            labels=labels, label_length=length,
            logits=logits, logit_length=logit_length,
            blank_index=-1,
            logits_time_major=True))

        variables = model.trainable_variables

        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return loss, logits, logit_length

optimizer = tf.keras.optimizers.Adam(learning_rate=.005)

cer_metric = dioptre.metrics.CharacterErrorRate()
ctc_metric = tf.keras.metrics.Mean()

summary_writer = tf.summary.create_file_writer('logs')
with summary_writer.as_default():
    for i, (image, width, labels, length) in enumerate(padded_dataset.take(15000), start=1): 
        loss, logits, new_width = train_step(image, width, labels, length)

        prediction, _ = tf.nn.ctc_greedy_decoder(logits, new_width)

        ctc_metric.update_state(loss)
        cer_metric.update_state(labels, prediction[0])

        if i % 250 == 0:
            ctc_loss = ctc_metric.result().numpy()
            cer_loss = cer_metric.result().numpy()
            tf.summary.scalar('Loss/CTC', ctc_loss, step=i)
            tf.summary.scalar('Loss/CER', cer_loss, step=i)
            print('step {:3}: ctc={:.03f} cer={:.03}'.format(i, ctc_loss, cer_loss))
            cer_metric.reset_states()
            ctc_metric.reset_states()

```