from tinygrad.tensor import Tensor
import numpy as np


class SparseRegularizer:
    def __init__(self, intensity=0.1, threshold=0.1, steepness=100, l1=0.001):
        self.intensity = intensity
        self.threshold = threshold
        self.steepness = steepness
        self.y_shift = np.tanh(-self.threshold * self.steepness)
        self.l1_intensity = l1

    def __call__(self, x: Tensor):
        # abs(x - threshold)
        abs_weights = x.abs()
        shifted_weights = abs_weights - self.threshold

        # tanh((x - threshold) * steepness) - tanh(-threshold * steepness)
        tanh_weights = (shifted_weights * self.steepness).tanh() - self.y_shift

        # Add a bit of l1 regularization. This seems to be important to prevent very large weights.
        l1_weight = abs_weights.mean() * self.l1_intensity

        # Sum over all elements and scale by intensity
        penalty = tanh_weights.mean() * self.intensity + l1_weight
        return penalty
