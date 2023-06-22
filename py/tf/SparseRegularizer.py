import tensorflow as tf
from tensorflow.keras import regularizers


class SparseRegularizer(regularizers.Regularizer):
    def __init__(self, intensity=0.1, threshold=0.1, steepness=100, l1=0.001):
        self.intensity = tf.constant(intensity, dtype=tf.float32)
        self.threshold = tf.constant(threshold, dtype=tf.float32)
        self.steepness = tf.constant(steepness, dtype=tf.float32)
        self.y_shift = tf.math.tanh(-self.threshold * self.steepness)
        self.l1_intensity = tf.constant(l1, dtype=tf.float32)

    def __call__(self, x):
        # abs(x - threshold)
        abs_weights = tf.math.abs(x)
        shifted_weights = abs_weights - self.threshold

        # tanh((x - threshold) * steepness) - tanh(-threshold * steepness)
        tanh_weights = tf.math.tanh(shifted_weights * self.steepness) - self.y_shift

        # Add a bit of l1 regularization. This seems to be important to prevent very large weights.
        l1_weight = tf.reduce_mean(abs_weights) * self.l1_intensity

        # Sum over all elements and scale by intensity
        penalty = tf.reduce_mean(tanh_weights) * self.intensity + l1_weight
        return penalty

    def get_config(self):
        return {
            "intensity": float(self.intensity.numpy()),
            "threshold": float(self.threshold.numpy()),
            "steepness": float(self.steepness.numpy()),
            "l1": float(self.l1_intensity.numpy()),
        }
