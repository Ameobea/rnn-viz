import tensorflow as tf


def ameo_grad(x):
    return tf.where(
        (x <= -1.0) & (x > -2.0) | ((x <= 1.0) & (x > 0.0)), 1.0, tf.where((x <= 0.0) & (x > -1.0), -1.0, 0.0)
    )


@tf.custom_gradient
def ameo(x, **kwargs):
    x *= 0.5
    x -= 0.5
    y = tf.where(x <= -1.0, tf.maximum(x + 2.0, 0.0), tf.where(x <= 0.0, -x, tf.minimum(x, 1.0)))
    y = (y - 0.5) * 2.0

    def grad(dy):
        return dy * ameo_grad(x)

    return y, grad


def soft_leaky_ameo_grad(x, leakyness):
    condition1_grad = tf.logical_and(-2 <= x, x <= -1.5)
    condition2_grad = tf.logical_and(-1.5 < x, x <= -0.5)
    condition3_grad = tf.logical_and(-0.5 < x, x <= 0.5)
    condition4_grad = tf.logical_and(0.5 < x, x <= 1)

    grad = tf.where(
        x <= -2,
        leakyness,
        tf.where(
            condition1_grad,
            32 * tf.pow(x + 2, 3),
            tf.where(
                condition2_grad,
                -32 * tf.pow(x, 3) - 96 * tf.pow(x, 2) - 96 * x - 32,
                tf.where(
                    condition3_grad,
                    32 * tf.pow(x, 3),
                    tf.where(condition4_grad, -32 * tf.pow(x, 3) + 96 * tf.pow(x, 2) - 96 * x + 32, leakyness),
                ),
            ),
        ),
    )
    return grad


@tf.custom_gradient
def soft_leaky_ameo(x, leakyness, **kwargs):
    x *= 0.5
    x -= 0.5

    condition1 = tf.logical_and(-2 <= x, x <= -1.5)
    condition2 = tf.logical_and(-1.5 < x, x <= -0.5)
    condition3 = tf.logical_and(-0.5 < x, x <= 0.5)
    condition4 = tf.logical_and(0.5 < x, x <= 1)

    y = tf.where(
        x <= -2,
        leakyness * (x + 2),
        tf.where(
            condition1,
            8 * tf.pow(x + 2, 4),
            tf.where(
                condition2,
                -8 * tf.pow(x, 4) - 32 * tf.pow(x, 3) - 48 * tf.pow(x, 2) - 32 * x - 7,
                tf.where(
                    condition3,
                    8 * tf.pow(x, 4),
                    tf.where(
                        condition4,
                        -8 * tf.pow(x, 4) + 32 * tf.pow(x, 3) - 48 * tf.pow(x, 2) + 32 * x - 7,
                        leakyness * (x - 1) + 1,
                    ),
                ),
            ),
        ),
    )
    y = (y - 0.5) * 2

    def grad(dy):
        return dy * soft_leaky_ameo_grad(x, leakyness), None

    return y, grad


@tf.custom_gradient
def interpolated_ameo(x, factor, leakyness, **kwargs):
    y0_mix = factor
    y1_mix = 1 - factor

    y0 = ameo(x) * y0_mix
    y1 = soft_leaky_ameo(x, leakyness) * y1_mix

    y = y0 + y1

    def grad(dy):
        dy0 = ameo_grad(x) * dy * y0_mix
        dy1 = soft_leaky_ameo_grad(x, leakyness) * dy * y1_mix
        return dy0 + dy1, None, None

    return y, grad


class AmeoActivation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AmeoActivation, self).__init__(**kwargs)

    def call(self, inputs):
        return ameo(inputs)

    def get_config(self):
        config = super(AmeoActivation, self).get_config()
        return config


class SoftLeakyAmeoActivation(tf.keras.layers.Layer):
    def __init__(self, leakyness, **kwargs):
        super(SoftLeakyAmeoActivation, self).__init__(**kwargs)
        self.leakyness = leakyness

    def call(self, inputs):
        return soft_leaky_ameo(inputs, self.leakyness)

    def get_config(self):
        config = super(SoftLeakyAmeoActivation, self).get_config()
        config["leakyness"] = self.leakyness
        return config


class InterpolatedAmeoActivation(tf.keras.layers.Layer):
    def __init__(self, factor, leakyness=0.05, **kwargs):
        super(InterpolatedAmeoActivation, self).__init__(**kwargs)
        self.factor = factor
        self.leakyness = float(leakyness)

    def call(self, inputs):
        return interpolated_ameo(inputs, self.factor, self.leakyness)

    def get_config(self):
        config = super(InterpolatedAmeoActivation, self).get_config()
        config["factor"] = self.factor
        config["leakyness"] = self.leakyness
        return config
