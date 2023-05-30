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
  private l1Intensity = 0.001;

  constructor(intensity: number, threshold = 0.1, steepness = 100, l1 = 0.001) {
    super();
    this.intensity = intensity;
    this.threshold = threshold;
    this.steepness = steepness;
    this.yShift = Math.tanh(-threshold * steepness);
    this.l1Intensity = l1;
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

      // Add a bit of l1 regularization.  This seems to be important to prevent very large weights.
      //
      // Since the penalty of regularizer doesn't activate until the weights are above the threshold,
      // networks find their way around this by having a very small weight of like 0.02 and then a
      // very large one of like 10.
      //
      // Adding in the l1 penalty helps prevent and keeps the weights smaller overall, allowing for
      // more effective quantization and pruning.
      const l1Weight = absWeights.mean().mul(this.l1Intensity);

      // Sum over all elements and scale by intensity
      const penalty: Scalar = (tanhWeights.mean() as Scalar).mul(intensity).add(l1Weight);
      return penalty;
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
