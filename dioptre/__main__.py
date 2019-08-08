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

    dataset = config.feeding.create().make_dataset()
    model = config.model.create()
    optimizer = getattr(tf.keras.optimizers, training.optimizer)(**training.optimizer_params)

    cer_metric = CharacterErrorRate()
    ctc_metric = tf.keras.metrics.Mean()

    tf.random.set_seed(training.seed)

    dataset = batch_dataset(model=model,
                            dataset=dataset,
                            batch_size=training.batch.size,
                            bucket_boundaries=training.batch.bucket_boundaries,
                            padded=training.batch.padded)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, directory=directory, max_to_keep=5,
                                         checkpoint_name='training')
    checkpoint.restore(manager.latest_checkpoint)

    @tf.function
    def fit(dataset):
        for image, width, labels, length in dataset:
            with tf.GradientTape() as tape:
                logits, logits_length = model(image, width)

                loss = tf.reduce_mean(tf.nn.ctc_loss(
                    labels=labels, label_length=length,
                    logits=logits, logit_length=logits_length,
                    blank_index=-1,
                    logits_time_major=True))

            variables = model.trainable_variables

            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))

            prediction, _ = tf.nn.ctc_greedy_decoder(logits, logits_length)

            ctc_metric.update_state(loss)
            cer_metric.update_state(labels, prediction[0])

        return ctc_loss.result(), cer_loss.result()

    summary_writer = tf.summary.create_file_writer(directory)
    with summary_writer.as_default():
        for epoch in range(training.epochs):
            ctc_loss, cer_loss = fit(dataset.take(training.steps_per_epoch))
            tf.summary.scalar('Loss/CTC', ctc_loss, step=epoch)
            tf.summary.scalar('Loss/CER', cer_loss, step=epoch)
            print('epoch {: 3}: ctc={:.03f} cer={:.03}'.format(epoch, ctc_loss, cer_loss))
            cer_metric.reset_states()
            ctc_metric.reset_states()
            manager.save(checkpoint_number=epoch)


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
