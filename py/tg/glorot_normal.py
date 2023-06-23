import numpy as np


def _compute_fans(shape):
    """Computes the number of input and output units for a weight shape.
    Args:
        shape: Integer shape tuple or TF tensor shape.
    Returns:
        A tuple of integer scalars (fan_in, fan_out).
    """
    if len(shape) < 1:  # Just to avoid errors for constants.
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:  # Standard fully connected layer.
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        # Assuming convolutional kernels (2D, 3D, or more).
        # kernel shape: (..., input_depth, depth)
        receptive_field_size = 1
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return fan_in, fan_out


def glorot_normal(shape, seed=None):
    fan_in, fan_out = _compute_fans(shape)
    scale = 1.0
    scale /= max(1.0, (fan_in + fan_out) / 2.0)

    # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
    stddev = np.sqrt(scale) / 0.87962566103423978

    # We use a truncated normal distribution here to get the best of both
    # worlds of normal and uniform initializations.
    # See https://arxiv.org/abs/1707.09725 for details.
    values = stddev * np.random.normal(loc=0.0, scale=1.0, size=shape).astype(np.float32)

    # Truncate values more than 2 standard deviations away from the mean
    while True:
        too_large = np.abs(values) > 2 * stddev
        if not np.any(too_large):
            break
        values[too_large] = stddev * np.random.normal(
            loc=0.0, scale=1.0, size=np.sum(too_large)
        ).astype(np.float32)

    return values
