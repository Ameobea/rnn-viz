/**
 * Encourages sparsity by adding a penalty the further the weights are from zero.
 */

import { tidy, type Rank, type Scalar, type Tensor, serialization, scalar } from '@tensorflow/tfjs';
import { Regularizer } from '@tensorflow/tfjs-layers/dist/regularizers';

export class SparseRegularizer extends Regularizer {
  private intensity: number;
  private threshold = 0.1;
  /**
   * The steepness of the sigmoid function used to calculate the penalty.
   */
  private steepness = 100;
  /**
   * Y shift so that the penalty is zero at 0.
   */
  private yShift;

  constructor(intensity: number, threshold = 0.1, steepness = 100) {
    super();
    this.intensity = intensity;
    this.threshold = threshold;
    this.steepness = steepness;
    this.yShift = Math.tanh(-threshold * steepness);
  }

  apply(x: Tensor<Rank>): Scalar {
    return tidy(() => {
      const threshold = scalar(this.threshold);
      const steepness = scalar(this.steepness);
      const intensity = scalar(this.intensity);
      const yShift = scalar(this.yShift);

      // tanh((x - threshold) * intensity) - tanh(-threshold * intensity)
      const absWeights = x.abs();
      const shiftedWeights = absWeights.sub(threshold);
      const tanhWeights = shiftedWeights.mul(steepness).tanh().sub(yShift);

      // Add a small bit of l1 regularization to help guide weights to zero
      const l1Weight = absWeights.mean().mul(0.01);

      // Sum over all elements and scale by intensity
      const penalty: Scalar = (tanhWeights.mean() as Scalar).add(l1Weight).mul(intensity);
      return penalty;

      // // Scale by the number of elements in the tensor
      // const numElements = x.size;
      // const scaledPenalty: Scalar = penalty.div(numElements);
      // // console.log('penalty', scaledPenalty.dataSync()[0]);
      // return scaledPenalty;
    });
  }

  getConfig(): serialization.ConfigDict {
    return { intensity: this.intensity };
  }

  static fromConfig(
    cls: serialization.SerializableConstructor<any>,
    config: serialization.ConfigDict
  ): any {
    return new SparseRegularizer(
      config.intensity as any,
      config.threshold as any,
      config.steepness as any
    );
  }
}
