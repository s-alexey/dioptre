import tensorflow as tf
from dioptre.utils import to_sparse


class CharacterErrorRate(tf.keras.metrics.Metric):
    """Calculates character error rate (CER)."""

    def __init__(self, name='character_error_rate', **kwargs):
        super().__init__(name=name, **kwargs)
        self.count = self.add_weight(name='tp', initializer='zeros', dtype=tf.int32)
        self.errors = self.add_weight(name='tp', initializer='zeros', dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = to_sparse(y_true)
        edit_distances = tf.edit_distance(y_true, to_sparse(y_pred),
                                          normalize=False)
        distance = tf.reduce_sum(edit_distances)
        self.errors.assign_add(tf.cast(distance, dtype=tf.int32))
        self.count.assign_add(tf.shape(y_true.values)[0])

    def result(self):
        return tf.truediv(self.errors, self.count)
