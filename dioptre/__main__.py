import os
import argparse
import shutil

import tensorflow as tf

from dioptre.metrics import CharacterErrorRate
from dioptre.params import Config
from dioptre.utils import batch_dataset


def clean_dir(directory):
    for file in os.listdir(directory):
        if not file.endswith('.yaml') and not os.path.isdir(file):
            os.remove(os.path.join(directory, file))


def copy(source, target):
    os.makedirs(target, exist_ok=True)

    for file in os.listdir(source):
        if file.endswith('.yaml'):
            shutil.copy(os.path.join(source, file), target)


def deep_copy(source, target):
    os.makedirs(target, exist_ok=True)

    for file in os.listdir(source):
        shutil.copy(os.path.join(source, file), target)


def train(directory):
    # Experiment specific configuration
    config = Config.load(directory)
    training = config.training

    dataset = config.feeding.create().to_dataset()
    model = config.model.create()
    optimizer = getattr(tf.keras.optimizers, training.optimizer)(**training.optimizer_params)

    @tf.function
    def train_step(image, width, labels, length):
        with tf.GradientTape() as tape:
            logits, new_width = model(image, width)

            loss = tf.reduce_mean(tf.nn.ctc_loss(
                labels=labels, label_length=length,
                logits=logits, logit_length=new_width,
                blank_index=-1,
                logits_time_major=True))

            variables = model.trainable_variables

            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))

            return loss, logits, new_width

    cer_metric = CharacterErrorRate()
    ctc_metric = tf.keras.metrics.Mean()

    tf.random.set_seed(training.seed)

    dataset = batch_dataset(alphabet=config.model.alphabet,
                            image_shape=(config.model.image_height, None, 1),
                            dataset=dataset,
                            batch_size=training.batch.size,
                            bucket_boundaries=training.batch.bucket_boundaries,
                            padded=training.batch.padded)

    for i, (image, width, labels, length) in enumerate(dataset.take(training.steps)):
        loss, logits, new_width = train_step(image, width, labels, length)

        prediction, _ = tf.nn.ctc_greedy_decoder(logits, new_width)

        ctc_metric.update_state(loss)
        cer_metric.update_state(labels, prediction[0])

        if i and i % training.log_every == 0:
            ctc_loss = ctc_metric.result().numpy()
            cer_loss = cer_metric.result().numpy()
            tf.summary.scalar('Loss/CTC', ctc_loss)
            tf.summary.scalar('Loss/CER', cer_loss)
            print('step {: 3}: ctc={:.03f} cer={:.03}'.format(i, ctc_loss, cer_loss))
            cer_metric.reset_states()
            ctc_metric.reset_states()


parser = argparse.ArgumentParser()

parser.add_argument('directory')
parser.add_argument('--cleanup', action='store_true')
parser.add_argument('--copy', type=str, help='copy configuration from other model dir')
parser.add_argument('--deep-copy', type=str, help='copy configuration and checkpoints from other model dir')
parser.add_argument('--train', action='store_true')

args = parser.parse_args()

if args.copy:
    copy(args.copy, args.directory)

elif args.deep_copy:
    deep_copy(args.deep_copy, args.directory)

if args.cleanup:
    clean_dir(args.directory)

if args.train:
    train(args.directory)